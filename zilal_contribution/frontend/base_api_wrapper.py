import uuid
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any

class BaseAPIWrapper(ABC):
    """Base class for all literature search API wrappers"""
    
    @abstractmethod
    def __call__(self, inputs: Dict[str, Any], api_key: Optional[str] = None, 
                 max_results: int = 2000, **kwargs) -> Tuple[pd.DataFrame, str, int]:
        """
        Search for papers using the given inputs.
        
        Returns:
            Tuple of (papers_df, search_query, total_count)
        """
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return the name of this data source"""
        pass
    
    def standardize_output(self, papers_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the output DataFrame to match expected schema"""
        if papers_df is None or papers_df.empty:
            return pd.DataFrame()
        
        # Ensure we have the required columns
        required_columns = ['Title', 'Abstract', 'Year', 'Month', 'Day', 'Authors', 'Journal', 'Publication Type', 'Volume', 'Issue', 'Pages']
        for col in required_columns:
            if col not in papers_df.columns and col.replace(" ", "_") not in papers_df.columns:
                papers_df[col] = ""
        
        # Add source column
        if 'Source' not in papers_df.columns:
            papers_df['Source'] = self.get_source_name()
        
        # Standardize column names (remove spaces, ensure consistency)
        papers_df.columns = papers_df.columns.str.replace(" ", "_")
        
        # Ensure we have consistent ID columns
        if 'PMID' in papers_df.columns:
            papers_df['ID'] = papers_df.pop('PMID')
        elif 'arXiv_ID' in papers_df.columns:
            papers_df['ID'] = papers_df.pop('arXiv_ID')
        elif 'Scholar_ID' in papers_df.columns:
            papers_df['ID'] = papers_df.pop('Scholar_ID')
        elif 'DOI' in papers_df.columns:
            papers_df['ID'] = papers_df.pop('DOI')
        elif 'NCT_Number' in papers_df.columns:
            papers_df['ID'] = papers_df.pop('NCT_Number')
        elif 'PMCID' in papers_df.columns:
            papers_df['ID'] = papers_df.pop('PMCID')
        elif 'ID' not in papers_df.columns:
            # Generate a unique ID if none exists
            papers_df['ID'] = str(uuid.uuid4())
        
        # Preserve optional source-specific metadata columns if present
        # These are used for searching/extraction but not displayed in the UI
        # Examples: Linked_Publications for CTGov, CTGov_* metadata fields
        
        return papers_df

    def lookup_by_id(self, paper_id: str) -> pd.DataFrame:
        """Look up a specific paper by its ID. Override in subclasses for specific implementations."""
        raise NotImplementedError(f"lookup_by_id not implemented for {self.get_source_name()}")

    def lookup_by_title(self, title: str) -> pd.DataFrame:
        """Look up a specific paper by its title. Override in subclasses for specific implementations."""
        raise NotImplementedError(f"lookup_by_title not implemented for {self.get_source_name()}")

    def lookup_by_ids(self, paper_ids: List[str]) -> pd.DataFrame:
        """Default bulk lookup implementation using per-ID lookups.

        Subclasses can override for sources that support efficient bulk queries.
        """
        if paper_ids is None or len(paper_ids) == 0:
            return pd.DataFrame()
        results: List[pd.DataFrame] = []
        for pid in paper_ids:
            try:
                df = self.lookup_by_id(pid)
                if df is not None and not df.empty:
                    results.append(df)
            except Exception:
                # Skip failures for individual IDs to make bulk resilient
                continue
        if len(results) == 0:
            return pd.DataFrame()
        return self.standardize_output(pd.concat(results, ignore_index=True))
