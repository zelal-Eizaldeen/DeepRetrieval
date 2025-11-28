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
