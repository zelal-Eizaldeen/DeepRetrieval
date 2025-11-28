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


# TODO: 
# from query_service.api import router as query_router
# app.include_router(query_router, prefix="/v1")

#For Testing on EC2-
# curl -X POST http://54.x.x.x:8000/v1/generate-search-terms \
# -H "Content-Type: application/json" \
# -d '{
#   "wrapper_type": "search_query_generation_input",
#   "research_topic": "Effectiveness of rectal analgesia in perineal trauma",
#   "custom_filters": [
#     {"name": "Population", "value": "women postpartum"},
#     {"name": "Intervention", "value": "rectal analgesia"},
#     {"name": "Outcome", "value": "pain relief"}
#   ]
# }'