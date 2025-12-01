from fastapi import APIRouter
from .query_generator import (
    generate_search_terms, 
    SearchQueryInputWrapper
)

router = APIRouter()

@router.post("/generate-search-terms")
def generate_terms(input_wrapper: SearchQueryInputWrapper):
    result = generate_search_terms(input_wrapper, vllm_url="http://localhost:8000/v1/chat/completions")
    return result


