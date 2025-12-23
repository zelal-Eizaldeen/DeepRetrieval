import re
import pandas as pd
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional
from .base_api_wrapper import BaseAPIWrapper
from .api import _rerank_studies_bm25

class MultiSourceWrapper(BaseAPIWrapper):
    """Wrapper that searches multiple sources and merges results"""

    def __init__(self, wrappers: List[BaseAPIWrapper]):
        self.wrappers = wrappers

    def get_source_name(self) -> str:
        return "multi_source"

    def __call__(self, inputs: Dict[str, Any], api_key: Optional[str] = None, 
                 max_results: int = 2000, included_sources: List[str] = [], **kwargs) -> Tuple[pd.DataFrame, str, int]:
        return self._run(inputs, api_key, max_results, included_sources, **kwargs)

    def _run(self, inputs: Dict[str, Any], api_key: Optional[str], 
             max_results: int, included_sources: List[str] = [], **kwargs) -> Tuple[pd.DataFrame, str, int]:
        all_results = []
        total_count = 0
        search_queries = []
        
        # Get progress callback if provided
        progress_callback = kwargs.get('progress_callback')
        
        # Get the list of sources to search from inputs
        sources_to_search = self._get_sources_to_search(included_sources)
        
        # Filter wrappers based on selected sources
        active_wrappers = [w for w in self.wrappers if w.get_source_name().lower() in sources_to_search]
        
        if not active_wrappers:
            active_wrappers = self.wrappers
        
        # Calculate progress increments
        total_sources = len(active_wrappers)
        base_progress = 15  # Start from 15% (after initialization)
        progress_per_source = (80 - base_progress) / total_sources  # Distribute 65% across sources
        
        # Search each selected source
        for i, wrapper in enumerate(active_wrappers):
            try:
                if progress_callback:
                    current_progress = base_progress + (i * progress_per_source)
                    progress_callback(current_progress, f"Searching {wrapper.get_source_name()}")
                
                # Each source gets the full max_results
                papers_df, query, count = wrapper(
                    inputs=inputs, 
                    api_key=api_key, 
                    max_results=max_results, 
                    **kwargs
                )

                if papers_df is not None and not papers_df.empty:
                    all_results.append(papers_df)
                    total_count += count
                    search_queries.append(query)

            except Exception as e:
                print(f"Error searching {wrapper.get_source_name()}: {e}")
                continue

        if not all_results:
            return pd.DataFrame(), "", 0

        if progress_callback:
            progress_callback(80, "Combining and deduplicating results")

        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)

        # Handle deduplication and source merging
        if len(active_wrappers) > 1:
            combined_df = self._deduplicate_and_merge_results(combined_df)

        if progress_callback:
            progress_callback(90, "Reranking and finalizing results")

        # Rerank and limit to max_results
        if inputs.get("keyword_map") is not None:
            papers_df = _rerank_studies_bm25(combined_df, inputs.get("keyword_map"), max_results)
        else:
            papers_df = combined_df
        papers_df = papers_df.head(max_results)

        # Standardize output
        papers_df = self.standardize_output(papers_df)

        # Build meaningful combined query string for the user
        combined_query_str = self._format_combined_query(active_wrappers, search_queries)

        return papers_df, combined_query_str, total_count

    def _format_combined_query(self, active_wrappers: List[BaseAPIWrapper], search_queries: List[str]) -> str:
        """Format a user-facing combined query string.

        - If a single source was queried, return that query as-is.
        - If multiple sources were queried, prefix each with the source name and join with newlines.
        """
        if not search_queries:
            return ""
        if len(search_queries) == 1 and len(active_wrappers) == 1:
            return search_queries[0]
        # Align wrappers to queries; assume same order as loop above
        parts: List[str] = []
        for wrapper, q in zip(active_wrappers, search_queries):
            source_name = wrapper.get_source_name()
            parts.append(f"{source_name}: {q}")
        return "<SOURCE_SEPARATOR>".join(parts)

    def _deduplicate_and_merge_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate results across sources and merge source information and IDs.
        
        Strategy:
        1) Merge by canonical identifiers when present (DOI, PMID, PMCID, NCT_Number, arXiv_ID) to combine rows and concatenate ID/source columns.
        2) Drop exact duplicates by canonical ID if present.
        3) Fallback: deduplicate by normalized title for residual duplicates.
        """
        if df.empty:
            return df

        # First, identify duplicates and merge information before dropping
        df = self._merge_duplicate_info(df)

        # Merge across canonical identifiers where present to collapse cross-source dupes
        for key_col in ['DOI', 'PMID', 'PMCID', 'NCT_Number', 'arXiv_ID']:
            if key_col in df.columns:
                df = self._merge_group_info(df, key_col)
        
        # Now deduplicate by keeping the first occurrence of each unique paper
        if 'ID' in df.columns:
            # Primary deduplication by ID (most reliable)
            df = df.drop_duplicates(subset=['ID'], keep='first')
        
        # Secondary deduplication by normalized title (fast exact match after stopword/punctuation removal)
        df = self._deduplicate_by_normalized_title(df)

        return df

    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _merge_duplicate_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge information from duplicate papers before deduplication"""
        if df.empty:
            return df
        
        # Create a copy to work with
        df_copy = df.copy()
        
        # Group by DOI first (most reliable identifier)
        if 'ID' in df_copy.columns:
            df_copy = self._merge_group_info(df_copy, 'ID')
        
        return df_copy

    def _merge_group_info(self, df: pd.DataFrame, group_column: str) -> pd.DataFrame:
        """Merge information for papers grouped by a specific column"""
        # Define core columns that should use 'first' aggregation
        core_columns = {
            'Title': 'first',
            'Abstract': 'first',
            'Authors': 'first',
            'Journal': 'first',
            'Year': 'first',
            'Month': 'first',
            'Day': 'first',
            'Publication_Type': 'first',
            'Volume': 'first',
            'Issue': 'first',
            'Pages': 'first',
        }
        
        # Define columns that should be joined (comma-separated)
        join_columns = {
            'Source': lambda x: ','.join(dict.fromkeys([s for v in x for s in str(v).split(',') if str(v).strip() != ''])),
        }
        
        # Define special ID columns that should be joined
        id_columns = ['ID', 'DOI', 'PMID', 'PMCID', 'NCT_Number', 'arXiv_ID']
        
        # Define optional columns that should be preserved (use 'first' non-null value)
        # Includes CTGov-specific metadata fields for extraction/searching
        optional_columns = [
            'Linked_Publications',
            'CTGov_Conditions',
            'CTGov_Interventions',
            'CTGov_Study_Type',
            'CTGov_Phase',
            'CTGov_Allocation',
            'CTGov_Intervention_Model',
            'CTGov_Primary_Purpose',
            'CTGov_Masking',
            'CTGov_Status',
            'CTGov_Enrollment',
            'CTGov_Enrollment_Type',
            'CTGov_Eligibility_Criteria',
            'CTGov_Sex',
            'CTGov_Minimum_Age',
            'CTGov_Maximum_Age',
            'CTGov_Arm_Descriptions',
            'CTGov_Primary_Outcome_Measures',
            'CTGov_Secondary_Outcome_Measures',
            'CTGov_Collaborators',
        ]
        
        # Build aggregation dictionary dynamically
        agg_dict = {}
        
        # Add core columns that exist in the dataframe
        for col, agg_func in core_columns.items():
            if col in df.columns:
                agg_dict[col] = agg_func
        
        # Add join columns that exist in the dataframe
        for col, agg_func in join_columns.items():
            if col in df.columns:
                agg_dict[col] = agg_func
        
        # Add optional columns that exist in the dataframe
        for col in optional_columns:
            if col in df.columns:
                agg_dict[col] = 'first'
        
        # Group by the specified column and aggregate information
        grouped = df.groupby(group_column, dropna=False).agg(agg_dict).reset_index()
        
        # Handle ID columns - merge all IDs from different sources (including duplicates for frontend matching)
        for col in id_columns:
            if col in df.columns:
                def join_unique(vals):
                    parts = []
                    for v in vals:
                        if pd.isna(v) or v is None:
                            continue
                        parts.extend([p for p in str(v).split(',') if str(p).strip() != ''])
                    return ','.join(dict.fromkeys(parts))
                id_merge = df.groupby(group_column, dropna=False)[col].apply(join_unique)
                grouped[col] = grouped[group_column].map(id_merge)
        
        return grouped

    # Lightweight English stopword list for fast title normalization (kept local to avoid heavy deps)
    _STOPWORDS = {
        'a','an','and','the','of','in','on','for','to','with','by','from','at','as','is','are','was','were',
        'be','been','being','or','that','this','these','those','it','its','into','about','over','after','before',
        'between','within','without','also','we','our','their','his','her','they','you','your','i','me','my','mine',
        'but','not','no','yes','there','here','such','than','then','thus','via','per','using','use','based','study',
        'studies','trial','trials','effect','effects','analysis','analyses','report','reports','result','results',
        'method','methods','review','reviews'
    }

    def _normalize_title_for_dedup(self, title: str) -> str:
        """Normalize a title by lowercasing, removing punctuation, dropping stopwords, and collapsing spaces.

        This produces a deterministic key suitable for grouping exact duplicates without O(n^2) comparisons.
        """
        if not isinstance(title, str):
            return ''

        text = title.lower()
        # Remove punctuation and symbols
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ''

        tokens = [t for t in text.split(' ') if t and t not in self._STOPWORDS and len(t) > 2]
        # Keep token order to avoid over-merging distinct titles that share bag-of-words
        return ' '.join(tokens)

    def _deduplicate_by_normalized_title(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate rows whose titles are identical after stopword/punctuation removal.

        This is significantly faster than fuzzy matching and works well for large result sets.
        It also merges IDs and Sources across duplicates using the same logic as other group merges.
        """
        if df.empty or 'Title' not in df.columns:
            return df

        df_copy = df.copy()
        df_copy['__norm_title__'] = df_copy['Title'].fillna('').map(self._normalize_title_for_dedup)

        # If normalization produced nothing (empty), leave those as-is; group only non-empty keys
        has_key_mask = df_copy['__norm_title__'] != ''
        df_with_key = df_copy[has_key_mask]
        df_without_key = df_copy[~has_key_mask].drop(columns=['__norm_title__'])

        if df_with_key.empty:
            return df_copy.drop(columns=['__norm_title__'])

        merged = self._merge_group_info(df_with_key, '__norm_title__')
        # Drop helper column
        merged = merged.drop(columns=['__norm_title__'])

        if df_without_key.empty:
            return merged
        else:
            # Preserve original order as much as possible by concatenating; duplicates have been merged
            return pd.concat([merged, df_without_key], ignore_index=True)

    def _get_sources_to_search(self, sources) -> List[str]:
        """Determine which sources to search based on inputs"""
        # Default to all available sources if none specified
        if sources is None or len(sources) == 0:
            return [w.get_source_name().lower() for w in self.wrappers]
        
        requested_sources = [s.lower() for s in sources]
        
        # Handle both string and list inputs
        if isinstance(requested_sources, str):
            requested_sources = [s.strip() for s in requested_sources.split(',')]
        
        # Validate requested sources against available sources
        available_sources = [w.get_source_name().lower() for w in self.wrappers]
        valid_sources = [s for s in requested_sources if s in available_sources]
        
        if not valid_sources:
            print(f"Warning: No valid sources requested. Available sources: {available_sources}")
            print(f"Requested sources: {requested_sources}")
            return available_sources  # Fallback to all sources
        
        return valid_sources

    def _deduplicate_by_title_similarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate by title similarity using fuzzy matching and merge information"""

        # First, identify similar titles and merge their information
        df = self._merge_similar_titles(df)
        
        # Then remove duplicates
        to_remove = []
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if self._similarity(df.iloc[i]['Title'], df.iloc[j]['Title']) > 0.9:
                    to_remove.append(j)

        return df.drop(df.index[to_remove])

    def _merge_similar_titles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge information from papers with similar titles before deduplication"""
        if df.empty:
            return df
        
        # Create a copy to work with
        df_copy = df.copy()
        
        # Group by title similarity and merge information
        # This is a simplified approach - in practice you might want more sophisticated clustering
        processed_indices = set()
        merged_rows = []
        
        for i in range(len(df_copy)):
            if i in processed_indices:
                continue
                
            current_row = df_copy.iloc[i].copy()
            similar_indices = [i]
            
            # Find similar titles
            for j in range(i + 1, len(df_copy)):
                if j in processed_indices:
                    continue
                    
                if self._similarity(current_row['Title'], df_copy.iloc[j]['Title']) > 0.9:
                    similar_indices.append(j)
                    processed_indices.add(j)
            
            # Merge information from similar titles
            if len(similar_indices) > 1:
                similar_rows = df_copy.iloc[similar_indices]
                
                # Merge sources
                all_sources = []
                for idx in similar_indices:
                    source = df_copy.iloc[idx]['Source']
                    if source:
                        all_sources.extend(source.split(','))
                current_row['Source'] = ','.join(all_sources)
                
                # Merge IDs
                all_ids = []
                for idx in similar_indices:
                    paper_id = df_copy.iloc[idx]['ID']
                    if paper_id:
                        all_ids.extend(paper_id.split(','))
                current_row['ID'] = ','.join(all_ids)
                
                # Merge other ID columns if they exist
                if 'PMID' in df_copy.columns:
                    all_pmids = []
                    for idx in similar_indices:
                        pmid = df_copy.iloc[idx]['PMID']
                        if pmid:
                            all_pmids.extend(pmid.split(','))
                    current_row['PMID'] = ','.join(all_pmids)
                
                if 'arXiv ID' in df_copy.columns:
                    all_arxiv_ids = []
                    for idx in similar_indices:
                        arxiv_id = df_copy.iloc[idx]['arXiv ID']
                        if arxiv_id:
                            all_arxiv_ids.extend(arxiv_id.split(','))
                    current_row['arXiv ID'] = ','.join(all_arxiv_ids)
                
                # Merge optional columns like Linked_Publications
                if 'Linked_Publications' in df_copy.columns:
                    # For Linked_Publications, keep the first non-null value
                    for idx in similar_indices:
                        linked_pubs = df_copy.iloc[idx]['Linked_Publications']
                        if pd.notna(linked_pubs) and linked_pubs:
                            current_row['Linked_Publications'] = linked_pubs
                            break
            
            merged_rows.append(current_row)
            processed_indices.add(i)
        
        return pd.DataFrame(merged_rows)
