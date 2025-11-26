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
    
    
def build_search_prompt(wrapper: SearchQueryInputWrapper) -> str:
    filters_text = "\n".join(
        [f"- {f.name}: {f.value}" for f in wrapper.custom_filters]
    )

    return f"""
You are an expert clinical literature search engine. 

You will receive a wrapper input in JSON format:

[INPUT_AS_JSON]
{wrapper.json(indent=2)}

Your job:
1. Interpret research_topic and custom_filters
2. Generate structured search term categories
3. Produce a final boolean-style search query

Return ONLY JSON inside <answer> </answer> tags:
<answer>
{{
  "wrapper_type": "search_query_generation_output",
  "categories": [
    {{"name": "Population", "values": ["term1", "term2"]}},
    {{"name": "Intervention", "values": ["term1", "term2"]}}
  ],
  "final_query": "(term1 AND term2)"
}}
</answer>
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
