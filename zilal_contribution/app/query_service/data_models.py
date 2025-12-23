from typing import List, Optional
from pydantic import BaseModel



class CustomFilter(BaseModel):
    name: str
    value: str
    description: Optional[str] = None


class SearchQueryInputWrapper(BaseModel):
    wrapper_type: str = "search_query_generation_input"
    research_topic: str
    custom_filters: List[CustomFilter]


class SearchQueryOutputWrapper(BaseModel):
    wrapper_type: str = "search_query_generation_output"
    categories: List[dict]
    final_query: str
    
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