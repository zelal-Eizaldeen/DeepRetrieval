import re
import uuid
import logging
import asyncio
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse

from trialmind.TrialMetaAnalysis.pubmed import PubmedAPIWrapper
from trialmind.TrialMetaAnalysis.arxiv_wrapper import ArxivWrapper
from trialmind.TrialMetaAnalysis.biomedrxiv_wrapper import BiomedRxivWrapper
from trialmind.TrialMetaAnalysis.europe_pmc_wrapper import EuropePMCWrapper
from trialmind.TrialMetaAnalysis.scholar_wrapper import ScholarWrapper
from trialmind.TrialMetaAnalysis.multi_source_wrapper import MultiSourceWrapper
from trialmind.TrialMetaAnalysis.ctgov_wrapper import CTGovPublicationWrapper

from trialmind.TrialMetaAnalysis.api import (
    ScreeningCriteriaGeneration,
    SearchQueryGeneration,
    PICOGeneration,
    FilterGeneration,
)
from trialmind.api.utils import (
    upload_dict_to_s3,
    upload_dict_to_s3_async,
    download_dict_from_s3_async,
    check_s3_file_exists_async,
)

from trialmind import LITERATURE_SEARCH_S3_BUCKET

# Import your new logic
from deepRetrieval_wrapper import (
    generate_search_terms, 
    
)
from trialmind.api.data_models import SearchQueryInputWrapper, SearchQueryInputWrapper, MASearchQueryGenerationBody
# Initialize the API
searchQueryGenAPI = SearchQueryGeneration()
screenCriteriaGenAPI = ScreeningCriteriaGeneration()
pubmedAPIWrapper = PubmedAPIWrapper()
arxivWrapper = ArxivWrapper()
europePMCWrapper = EuropePMCWrapper()
biorxivWrapper = BiomedRxivWrapper("biorxiv")
medrxivWrapper = BiomedRxivWrapper("medrxiv")
scholarWrapper = ScholarWrapper()
ctgovPublicationWrapper = CTGovPublicationWrapper()

multiSourceWrapper = MultiSourceWrapper([
    pubmedAPIWrapper,
    arxivWrapper,
    europePMCWrapper,
    biorxivWrapper,
    medrxivWrapper,
    scholarWrapper,
    ctgovPublicationWrapper
])
picoGenerationAPI = PICOGeneration()
filterGenerationAPI = FilterGeneration()

async def ma_pico_generation(request_body):
    """Generate PICO elements for meta-analysis"""
    try:
        # Run LLM call in thread pool to avoid blocking
        pico = await asyncio.to_thread(
            picoGenerationAPI.run,
            research_topic=request_body.research_topic,
            llm=request_body.llm,
        )
        return JSONResponse(status_code=200, content=pico)
    except Exception as e:
        logging.exception(e)
        return JSONResponse(
            status_code=500, content={"error": "Error generating PICO elements"}
        )



#By Zilal
# async def ma_search_query_generation(request_body):
#     """Generate search query for meta-analysis"""
#     try:
#         # Run LLM call in thread pool to avoid blocking
#         terms = await asyncio.to_thread(
#             searchQueryGenAPI.run,
#             research_topic=request_body.research_topic,
#             population=request_body.population,
#             intervention=request_body.intervention,
#             comparator=request_body.comparator,
#             outcome=request_body.outcome,
#             custom_filters=request_body.custom_filters,
#             llm=request_body.llm,
#             user_request=request_body.user_request,
#             category_name=request_body.category_name,
#         )

#         # For category-specific queries, terms is a list; otherwise it's a dict with dynamic categories
#         if request_body.category_name != "":
#             # Category-specific query returns a list
#             return JSONResponse(
#                 status_code=200,
#                 content=terms,
#             )
#         else:
#             # Main query now returns dynamic categories as a dict
#             # terms is already in the format: {"category1": [...], "category2": [...], ...}
#             return JSONResponse(
#                 status_code=200,
#                 content=terms,
#             )

#     except Exception as e:
#         logging.exception(e)
#         return JSONResponse(
#             status_code=500, content={"error": "Error generating search query"}
#         )

#Added By Zilal
async def ma_search_query_generation(request_body:MASearchQueryGenerationBody):
    """
    Generate search query with a Fallback Mechanism:
    Primary: DeepRetrieval (vLLM)
    Secondary: Legacy SearchQueryGenAPI (OpenAI/GPT)
    """
    
    # 1. Attempt Primary: DeepRetrieval
    try:
        logging.info("Attempting search term generation via DeepRetrieval (vLLM)...")
        
        # Prepare the input for your specialized model
        input_wrapper = SearchQueryInputWrapper(
            research_topic=request_body.research_topic or f"P: {request_body.population}, I: {request_body.intervention}",
            custom_filters=request_body.custom_filters
        )
        
        vllm_url = "http://localhost:8000/v1/chat/completions"
        # vllm_url = "http://localhost:8000/v1/generate-search-terms"
        
        # Call your new async logic
        deep_results = await generate_search_terms(input_wrapper, vllm_url)
        
        return JSONResponse(status_code=200, content=deep_results)

    except Exception as deep_err:
        # 2. Fallback to Legacy SearchQueryGenAPI
        logging.warning(f"DeepRetrieval failed: {deep_err}. Falling back to Legacy API.")
        
        try:
            # Re-use the existing logic you already had in api.py
            terms = await asyncio.to_thread(
                searchQueryGenAPI.run,
                research_topic=request_body.research_topic,
                population=request_body.population,
                intervention=request_body.intervention,
                comparator=request_body.comparator,
                outcome=request_body.outcome,
                custom_filters=request_body.custom_filters,
                llm=request_body.llm,
                user_request=request_body.user_request,
                category_name=request_body.category_name,
            )
            
            # Map legacy format to a consistent response structure
            fallback_response = {
                "wrapper_type": "search_query_generation_output_fallback",
                "categories": [{"name": k, "synonyms": v} for k, v in terms.items()],
                "final_query": " AND ".join([f"({' OR '.join(v)})" for v in terms.values()])
            }
            
            return JSONResponse(status_code=200, content=fallback_response)

        except Exception as legacy_err:
            logging.error(f"Critical Failure: Both Primary and Fallback failed. {legacy_err}")
            return JSONResponse(
                status_code=500, 
                content={"error": "All search generation methods failed."}
            )
#ended by Zilal
#ByZilal
# async def ma_retrieve_papers(request_body, background_tasks):
#     """Retrieve papers for meta-analysis"""
#     try:
#         # Generate task ID
#         task_id = str(uuid.uuid4())
        
#         # Initialize task state in S3
#         task_state = {"status": "pending", "progress": 0, "result": []}
#         await upload_dict_to_s3_async(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json", task_state)

#         # Run paper retrieval in background
#         background_tasks.add_task(
#             run_paper_retrieval_task,
#             task_id,
#             request_body.search_terms,
#             request_body.max_results,
#             request_body.return_clinical_trials_only,
#             request_body.included_journals,
#             request_body.included_sources,
#             request_body.min_date,
#             request_body.max_date,
#             request_body.within_category_operator,
#             request_body.between_category_operator
#         )

#         return JSONResponse(status_code=202, content={"task_id": task_id})
#     except Exception as e:
#         logging.exception(e)
#         return JSONResponse(status_code=500, content={"error": str(e)})

#Added By Zilal
async def ma_retrieve_papers(request_body: MARetrievePapersBody, background_tasks: BackgroundTasks):

    """Retrieve papers for meta-analysis using DeepRetrieval Boolean logic"""
    try:
        task_id = str(uuid.uuid4())
        task_state = {"status": "pending", "progress": 0, "result": []}
        await upload_dict_to_s3_async(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json", task_state)
        # Extract final_query if provided by the frontend
        final_query = getattr(request_body, "final_query", "")
        background_tasks.add_task(
            run_paper_retrieval_task,
            task_id,
            request_body.search_terms,
            request_body.max_results,
            request_body.return_clinical_trials_only,
            request_body.included_journals,
            request_body.included_sources,
            request_body.min_date,
            request_body.max_date,
            request_body.within_category_operator,
            request_body.between_category_operator,
            final_query  # Pass the specialized query
        )

        return JSONResponse(status_code=202, content={"task_id": task_id})
    except Exception as e:
        logging.exception(e)
        return JSONResponse(status_code=500, content={"error": str(e)})
#Ended by Zilal


async def ma_criteria_generation(request_body):
    """Generate criteria for meta-analysis"""
    try:
        # Run LLM call in thread pool to avoid blocking
        criteria = await asyncio.to_thread(
            screenCriteriaGenAPI.run,
            research_topic=request_body.research_topic,
            population=request_body.population,
            intervention=request_body.intervention,
            comparator=request_body.comparator,
            outcome=request_body.outcome,
            custom_filters=request_body.custom_filters,
            llm=request_body.llm,
            num_title_criteria=request_body.num_title_criteria,
            num_abstract_criteria=request_body.num_abstract_criteria,
        )
        criteria = criteria['criteria']

        return JSONResponse(status_code=200, content=criteria)
    except Exception as e:
        logging.exception(e)
        return JSONResponse(status_code=500, content={"error": "An error occurred"})


async def get_retrieve_papers_status(task_id):
    """Get retrieve papers status for meta-analysis"""
    try:
        # Check if task file exists in S3
        if not await check_s3_file_exists_async(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json"):
            return JSONResponse(status_code=404, content="Task not found")

        # Get task state from S3
        task = await download_dict_from_s3_async(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json")

        if task["status"] == "completed":
            # Clean up the task file
            result = task["result"]
            query = task.get("query", "")
            response_content = {
                "result": result,
                "query": query
            }
            # remove_file_from_s3(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json")
            return JSONResponse(status_code=200, content=response_content)
        elif task["status"] == "failed":
            # Clean up the task file
            error = task["error"]
            # remove_file_from_s3(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json")
            return JSONResponse(status_code=500, content={"error": error})
        else:
            # Return progress information
            response_content = {
                "progress": task["progress"],
                "result": task.get("result", [])
            }
            # Add stage info if available
            if "stage_info" in task:
                response_content["stage_info"] = task["stage_info"]
            return JSONResponse(status_code=202, content=response_content)
    except Exception as e:
        logging.exception(e)
        return JSONResponse(status_code=500, content={"error": str(e)})


def parse_term_with_category(term: str) -> tuple[str, str | None]:
    """
    Parse a search term to extract custom category tag.
    
    Example:
        "diabetes[title]" -> ("diabetes", "title")
        "cancer" -> ("cancer", None)
    
    Returns:
        Tuple of (term_without_tag, category_tag_or_none)
    """
    match = re.search(r'(.+?)\[(\w+)\]$', term.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip().lower()
    return term.strip(), None


def preprocess_terms_with_categories(keyword_groups: dict) -> dict:
    """
    Preprocess keyword groups to extract custom category tags from terms.
    Groups terms by their category tags.
    
    Input:
        {"conditions": ["diabetes[title]", "cancer", "hypertension[abstract]"]}
    
    Output:
        {
            "conditions": ["cancer"],
            "conditions__title": ["diabetes"],
            "conditions__abstract": ["hypertension"]
        }
    """
    processed_groups = {}
    
    for group_name, terms in keyword_groups.items():
        # Default group for terms without category tags
        default_terms = []
        # Category-specific groups
        category_terms = {}
        
        for term in terms:
            clean_term, category = parse_term_with_category(term)
            if category:
                # Create a special key for this category
                category_key = f"{group_name}___{category}"
                if category_key not in category_terms:
                    category_terms[category_key] = []
                category_terms[category_key].append(clean_term)
            else:
                default_terms.append(clean_term)
        
        # Add default terms if any
        if default_terms:
            processed_groups[group_name] = default_terms
        
        # Add category-specific terms
        processed_groups.update(category_terms)
    
    return processed_groups

#By Zilal
# async def run_paper_retrieval_task(task_id, search_terms, max_results, return_clinical_trials_only, included_journals, included_sources, min_date, max_date, within_category_operator="OR", between_category_operator="AND"):
#     """Background task for paper retrieval"""
#     try:
#         def update_task_progress(progress, stage_info=""):
#             """Update task progress during paper retrieval"""
#             task_state = {
#                 "status": "running", 
#                 "progress": progress, 
#                 "result": [],
#                 "stage_info": stage_info
#             }
#             upload_dict_to_s3(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json", task_state)

#         async def update_task_progress_async(progress, stage_info=""):
#             """Update task progress during paper retrieval"""
#             task_state = {
#                 "status": "running", 
#                 "progress": progress, 
#                 "result": [],
#                 "stage_info": stage_info
#             }
#             await upload_dict_to_s3_async(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json", task_state)
        
#         # Update initial progress
#         await update_task_progress_async(5, "Initializing search parameters")

#         # terms = {
#         #     "conditions": conditions,
#         #     "treatments": treatments,
#         # }

#         keyword_groups = {
#             kw.name: kw.values
#             for kw in search_terms
#         }
        
#         # Preprocess terms to extract custom category tags
#         keyword_groups = preprocess_terms_with_categories(keyword_groups)

#         api_inputs = {
#             "keyword_map": keyword_groups,
#             "within_operator": within_category_operator,  # How terms combine within a category (OR/AND)
#             "between_operator": between_category_operator,  # How categories combine with each other (AND/OR)
#             "min_date": min_date,
#             "max_date": max_date,
#         }
        
#         # Add journal filtering if included_journals is provided
#         if included_journals and len(included_journals) > 0:
#             api_inputs["journal"] = included_journals
            
#         await update_task_progress_async(15, "Searching sources")
        
#         # Use multi-source wrapper with progress callback (run in thread pool)
#         papers_df, url_query, total_count = await asyncio.to_thread(
#             multiSourceWrapper,
#             inputs=api_inputs, 
#             max_results=max_results,
#             return_clinical_trials_only=return_clinical_trials_only,
#             included_sources=included_sources,
#             progress_callback=update_task_progress
#         )
        
#         if papers_df.empty:
#             raise Exception("No papers found")

#         await update_task_progress_async(95, "Processing results")
        
#         # Convert any non-finite values to None before JSON serialization
#         papers_df = papers_df.replace([np.inf, -np.inf], None)
#         papers_df = papers_df.where(pd.notnull(papers_df), None)
#         papers_df.columns = papers_df.columns.str.replace(" ", "_")
        
#         results = papers_df.to_dict(orient="records")
        
#         # Save completed results to S3
#         task_state = {"status": "completed", "result": results, "query": url_query}
#         await upload_dict_to_s3_async(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json", task_state)
        
#     except Exception as e:
#         # Save error state to S3
#         task_state = {"status": "failed", "error": str(e)}
#         await upload_dict_to_s3_async(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json", task_state) 


#Added by Zilal
async def run_paper_retrieval_task(task_id, search_terms, max_results, return_clinical_trials_only, included_journals, included_sources, min_date, max_date, within_category_operator="OR", between_category_operator="AND", final_query=""):
    """Background task for paper retrieval"""
    try:
        def update_task_progress(progress, stage_info=""):
            """Update task progress during paper retrieval"""
            task_state = {
                "status": "running", 
                "progress": progress, 
                "result": [],
                "stage_info": stage_info
            }
            upload_dict_to_s3(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json", task_state)

        async def update_task_progress_async(progress, stage_info=""):
            """Update task progress during paper retrieval"""
            task_state = {
                "status": "running", 
                "progress": progress, 
                "result": [],
                "stage_info": stage_info
            }
            await upload_dict_to_s3_async(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json", task_state)
        
        # Update initial progress
        await update_task_progress_async(5, "Initializing search parameters")

        # terms = {
        #     "conditions": conditions,
        #     "treatments": treatments,
        # }

        keyword_groups = {
            kw.name: kw.values
            for kw in search_terms
        }
        
        # Preprocess terms to extract custom category tags
        keyword_groups = preprocess_terms_with_categories(keyword_groups)

        api_inputs = {
            "keyword_map": keyword_groups,
            "final_query": final_query, # Pass the DeepRetrieval string #By Zilal
            "within_operator": within_category_operator,  # How terms combine within a category (OR/AND)
            "between_operator": between_category_operator,  # How categories combine with each other (AND/OR)
            "min_date": min_date,
            "max_date": max_date,
        }
        
        # Add journal filtering if included_journals is provided
        if included_journals and len(included_journals) > 0:
            api_inputs["journal"] = included_journals
            
        await update_task_progress_async(15, "Searching sources")
        
        # Use multi-source wrapper with progress callback (run in thread pool)
        papers_df, url_query, total_count = await asyncio.to_thread(
            multiSourceWrapper,
            inputs=api_inputs, 
            max_results=max_results,
            return_clinical_trials_only=return_clinical_trials_only,
            included_sources=included_sources,
            progress_callback=update_task_progress
        )
        
        if papers_df.empty:
            raise Exception("No papers found")

        await update_task_progress_async(95, "Processing results")
        
        # Convert any non-finite values to None before JSON serialization
        papers_df = papers_df.replace([np.inf, -np.inf], None)
        papers_df = papers_df.where(pd.notnull(papers_df), None)
        papers_df.columns = papers_df.columns.str.replace(" ", "_")
        
        results = papers_df.to_dict(orient="records")
        
        # Save completed results to S3
        task_state = {"status": "completed", "result": results, "query": url_query}
        await upload_dict_to_s3_async(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json", task_state)
        
    except Exception as e:
        # Save error state to S3
        task_state = {"status": "failed", "error": str(e)}
        await upload_dict_to_s3_async(LITERATURE_SEARCH_S3_BUCKET, f"{task_id}.json", task_state) 

#Added by Zilal
def build_search_prompt(wrapper: SearchQueryInputWrapper) -> str:
    return f"""
You are an expert clinical literature search engine. 

You will receive a wrapper input in JSON format:

[INPUT_AS_JSON]
{wrapper.model_dump_json(indent=2)}

You must output a highly structured object inside <answer> </answer> tags with:
1. Search categories with lists of clean synonyms (no sentences, only terms)
2. Boolean-ready variants
3. A final compact Boolean query optimized for PubMed

### RULES
- Each category's values must be a **list of synonyms**, not sentences.
- Terms must be short, controlled-vocabulary-style (e.g., "radiology", "diagnostic imaging", "deep learning").
- Remove words like “trial”, “study”, “effects”, “benefits”, “evaluation”.
- Final Boolean query MUST use:
  - OR for synonyms
  - AND for category groups
  - Parentheses around grouped OR terms

### OUTPUT FORMAT (MANDATORY)

<answer>
{{
  "wrapper_type": "search_query_generation_output",
  "categories": [
    {{
      "name": "Population",
      "synonyms": ["term1", "term2", "term3"]
    }},
    {{
      "name": "Intervention",
      "synonyms": ["term1", "term2"]
    }}
  ],
  "boolean_blocks": {{
    "Population": "(term1 OR term2 OR term3)",
    "Intervention": "(term1 OR term2)"
  }},
  "final_query": "(term1 OR term2 OR term3) AND (term1 OR term2)"
}}
</answer>

### NOW GENERATE THE STRUCTURED SPECIFICATIONS.
"""

import requests
import json

def call_deepretrieval_model(prompt: str, vllm_url: str):
    payload = {
        "model": "DeepRetrieval/DeepRetrieval-PubMed-3B-Llama",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512
    }

    resp = requests.post(vllm_url, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
import re

def extract_output_wrapper(text: str) -> dict:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not matches:
        raise ValueError("No <answer> block found")
    return json.loads(matches[-1].strip())


def generate_search_terms(wrapper: SearchQueryInputWrapper, vllm_url: str):
    prompt = build_search_prompt(wrapper)
    raw_output = call_deepretrieval_model(prompt, vllm_url)
    return extract_output_wrapper(raw_output)
