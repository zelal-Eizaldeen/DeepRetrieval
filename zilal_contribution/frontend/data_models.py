from fastapi import UploadFile
from pydantic import BaseModel
from typing import Any, List, Dict, Optional

class CustomFilter(BaseModel):
    name: str
    value: str
    description: Optional[str] = None #Added by Zilal

#Added by Zilal
class SearchQueryInputWrapper(BaseModel):
    """Specific wrapper required by the generate_search_terms logic"""
    wrapper_type: str = "search_query_generation_input"
    research_topic: str
    custom_filters: List[CustomFilter] = []


class SearchQueryOutputWrapper(BaseModel):
    """The data structure returned by the DeepRetrieval vLLM"""
    wrapper_type: str = "search_query_generation_output"
    categories: List[dict]
    final_query: str
#Ended By Zilal
 
class MASearchQueryGenerationBody(BaseModel):
    research_topic: str = ""
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    outcome: str = ""
    custom_filters: List[CustomFilter] = []
    user_request: str = ""
    category_name: str = ""
    llm: str = "gpt-4o"
    final_query: str = "" #Added by Zilal

class MATerms(BaseModel):
    name: str
    values: List[str]

class MARetrievePapersBody(BaseModel):
    search_terms: List[MATerms]
    max_results: int = 100
    return_clinical_trials_only: bool = True
    included_journals: List[str] = []
    included_sources: List[str] = []
    min_date: str = "1980-01-01"
    max_date: str = "2099-01-01"
    within_category_operator: str = "OR"
    between_category_operator: str = "AND"
    #Added by Zilal
    # final_query: str = ""  # New field for DeepRetrieval Boolean query
    #Ended

class MACriteriaGenerationBody(BaseModel):
    research_topic: str
    population: str
    intervention: str
    comparator: str
    outcome: str
    custom_filters: List[CustomFilter] = []
    num_title_criteria: int = 3
    num_abstract_criteria: int = 3
    llm: str = "gpt-4o"

    
class MAPICOGenerationBody(BaseModel):
    research_topic: str
    llm: str = "gpt-4o"


class MAFilterGenerationBody(BaseModel):
    research_topic: str
    filter_type: str  # PICOT, PICOS, SPIDER, SPICE, ECLIPSE
    llm: str = "gpt-4o"
    