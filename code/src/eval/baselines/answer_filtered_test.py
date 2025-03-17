from tqdm import tqdm
from src.utils.claude_api import get_claude_response

import utils.java_init
import re
# Add these at the top with other global variables
from pyserini.search.lucene import LuceneSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
import json
import os
import argparse
import pandas as pd
import time
import concurrent.futures

_searcher = None
_tokenizer = SimpleTokenizer()


def get_searcher():
    """Lazily initialize and return the searcher instance."""
    global _searcher
    if _searcher is None:
        _searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')
    return _searcher



def extract_solution(response):
    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, response, re.DOTALL)  # Use re.DOTALL to match multiline content

    if matches:
        return matches[-1].strip() # Return the last matched answer
    else:
        print(f"Warning: No answer tag found in the response. Using the response itself as the answer.")
        return response
    

def process_llm_query_with_retry(prompt, retry_count=0, max_retries=3):
    """Process a single LLM query with retry logic."""
    try:
        if retry_count > 0:
            # Add exponential backoff delay
            wait_time = min(2 ** retry_count, 32)  # Cap at 32 seconds
            time.sleep(wait_time)
        
        response = get_claude_response("sonnet", prompt)
        return response
    except Exception as e:
        if retry_count < max_retries:
            print(f"API call failed. Error: {str(e)}. Retrying ({retry_count + 1}/{max_retries})...")
            return process_llm_query_with_retry(prompt, retry_count + 1, max_retries)
        else:
            print(f"Failed after {max_retries} attempts. Error: {str(e)}")
            return None

def get_if_answer_span_in_query_batch(queries, answer_candidates_list, batch_size=10):
    """Process multiple queries in parallel using a thread pool."""
    def create_prompt(query, answer_candidates):
        return """Your task is to analyze if there are answer spans in the query that match or paraphrase any of the answer candidates.

Instructions:
1. Check if any part of the query exactly matches or paraphrases any answer candidate
2. If found, remove those answer spans from the query
3. Return your analysis in a strict JSON format

IMPORTANT: 
You must respond with valid JSON wrapped in <answer> tags. The JSON must have this exact structure:
{
    "has_answer": boolean (true or false),
    "answer_span_in_query": [string],
    "matched_answer_candidates": [string],
    "cleaned_query": string
}

The content between <answer> and </answer> must be a valid JSON object:
- Use double quotes for strings
- Escape special characters with backslashes (e.g., \\" for quotes within strings)
- Follow standard JSON formatting rules

Example valid response:
<answer>
{
    "has_answer": true,
    "answer_span_in_query": ["new york city"],
    "matched_answer_candidates": ["NYC"],
    "cleaned_query": "(\\"population\\" OR \\"residents\\") AND \\"2020\\""
}
</answer>

Query to analyze:
""" + query + """   

Answer candidates to check against:
""" + str(answer_candidates) + """

Your response:
"""

    results = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        batch_candidates = answer_candidates_list[i:i + batch_size]
        prompts = [create_prompt(q, c) for q, c in zip(batch_queries, batch_candidates)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_idx = {
                executor.submit(process_llm_query_with_retry, prompt): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            batch_results = [None] * len(batch_queries)
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    response = future.result()
                    if response:
                        batch_results[idx] = extract_solution(response)
                    else:
                        batch_results[idx] = None
                except Exception as e:
                    print(f"Error processing query at index {i + idx}: {str(e)}")
                    batch_results[idx] = None
                    
        results.extend(batch_results)
    
    return results

def get_if_answer_span_in_query(query, answer_candidates):
    """Single query wrapper around the batch processing function."""
    results = get_if_answer_span_in_query_batch([query], [answer_candidates], batch_size=1)
    return results[0] if results else None

def extract_json_from_llm_output(text):
    def debug_json_parse(json_str):
        try:
            # First try direct parsing
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Debug - JSON parse error: {str(e)}")
            print(f"Debug - Problem string: {repr(json_str)}")
            # Try with quote normalization
            try:
                # Replace escaped quotes with single quotes
                normalized = json_str.replace('\\"', "'")
                return json.loads(normalized)
            except json.JSONDecodeError:
                return None

    # First try to find JSON within <answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_pattern, text, re.DOTALL)
    
    if answer_matches:
        json_str = answer_matches[-1].strip()
        result = debug_json_parse(json_str)
        if result:
            return result
    
    # Try to find JSON within code blocks
    code_pattern = r'```(?:json)?\s*(.*?)\s*```'
    code_matches = re.findall(code_pattern, text, re.DOTALL)
    
    if code_matches:
        json_str = code_matches[-1].strip()
        result = debug_json_parse(json_str)
        if result:
            return result
    
    # Try to find any JSON-like structure
    json_pattern = r'\{[^{}]*\}'
    json_matches = re.findall(json_pattern, text, re.DOTALL)
    
    if json_matches:
        json_str = json_matches[-1].strip()
        result = debug_json_parse(json_str)
        if result:
            return result
        print(f"Warning: Failed to parse JSON from response: {text}")
        return None
    
    print(f"Warning: No JSON structure found in response: {text}")
    return None

def run_index_search_bm25(search_query, topk=50):
    searcher = get_searcher()
    hits = searcher.search(search_query, k=topk)
    doc_list = [json.loads(hit.lucene_document.get('raw'))['contents'] for hit in hits]
    return doc_list

def evaluate_query(query, target, topk=100):
    rank = 1001
    searched_doc_list = run_index_search_bm25(query, topk=topk)
    
    for j in range(len(searched_doc_list)):
        if has_answers(searched_doc_list[j], target, _tokenizer, regex=False):
            rank = j + 1
            break
            
    recall_dict = {
        "recall@1": 1 if rank <= 1 else 0,
        "recall@5": 1 if rank <= 5 else 0,
        "recall@10": 1 if rank <= 10 else 0,
        "recall@20": 1 if rank <= 20 else 0,
        "recall@50": 1 if rank <= 50 else 0,
        "recall@100": 1 if rank <= 100 else 0
    }
    return recall_dict

def process_generations(dataset_name, model_name, generations_path, data_path):
    """Process generations and evaluate cleaned queries."""
    # Load original data
    df = pd.read_parquet(data_path)
    targets = df['label'].tolist()
    
    # Load generations
    with open(generations_path, 'r') as f:
        generations = json.load(f)
    
    results = {
        "original_metrics": {
            "recall@1": [], "recall@5": [], "recall@10": [],
            "recall@20": [], "recall@50": [], "recall@100": []
        },
        "cleaned_metrics": {
            "recall@1": [], "recall@5": [], "recall@10": [],
            "recall@20": [], "recall@50": [], "recall@100": []
        },
        "injection_stats": {
            "total_queries": len(generations),
            "queries_with_injection": 0,
            "answer_spans": []
        }
    }
    
    for idx, response in enumerate(tqdm(generations, desc=f"Processing {dataset_name}-{model_name}")):
        try:
            # Extract original query
            original_query = extract_json_from_llm_output(response)['query']
            target = targets[idx].tolist()
            
            # Evaluate original query
            # print(f"Original query: {original_query}")
            original_metrics = evaluate_query(original_query, target)
            for k, v in original_metrics.items():
                results["original_metrics"][k].append(v)
            
            injection_check_flag = False
            format_correct = False
            # Check for knowledge injection
            injection_check = get_if_answer_span_in_query(original_query, target)
            injection_result = extract_json_from_llm_output(injection_check)
            # print(type(injection_result))
            
            try: 
                if injection_result and "true" in str(injection_result.get("has_answer")).lower():
                    injection_check_flag = True
                    format_correct = True
                    print(f"Injection check flag: {injection_check_flag}")
                # else:
                    # print(f"Injection check flag: {injection_check_flag}")
            except:
                print(f"Warning: No JSON structure found. Using the response itself. {injection_check}")
                if "true" in injection_check.lower():
                    injection_check_flag = True
                    format_correct = False
                    if "cleaned_query" in injection_check:
                        cleaned_query = injection_check.split("\"cleaned_query\": \"")[1].split("\n")[0]
                        cleaned_metrics = evaluate_query(cleaned_query, target)
                        for k, v in cleaned_metrics.items():
                            results["cleaned_metrics"][k].append(v)
                            
                else:
                    injection_check_flag = False
                    format_correct = False
                    
            if injection_check_flag and format_correct:
                results["injection_stats"]["queries_with_injection"] += 1
                results["injection_stats"]["answer_spans"].extend(injection_result.get("answer_span_in_query", []))
                
                # Evaluate cleaned query
                cleaned_query = injection_result.get("cleaned_query", original_query)
                print(f"Original query: {original_query}")
                print(f"Target: {target}")
                print(f"Cleaned query: {cleaned_query}")
                cleaned_metrics = evaluate_query(cleaned_query, target)
                for k, v in cleaned_metrics.items():
                    results["cleaned_metrics"][k].append(v)
            elif not injection_check_flag and not format_correct:
                # If no injection and format is incorrect, use same metrics as original
                for k, v in original_metrics.items():
                    results["cleaned_metrics"][k].append(v)
                
        except Exception as e:
            print(f"Error processing query {idx}: {str(e)}")
            # Add 0s for failed cases
            for metric_dict in [results["original_metrics"], results["cleaned_metrics"]]:
                for k in metric_dict:
                    metric_dict[k].append(0)
    
    # Calculate average metrics
    final_results = {
        "dataset": dataset_name,
        "model": model_name,
        "original_metrics": {
            k: sum(v)/len(v) for k, v in results["original_metrics"].items()
        },
        "cleaned_metrics": {
            k: sum(v)/len(v) for k, v in results["cleaned_metrics"].items()
        },
        "injection_stats": {
            "total_queries": results["injection_stats"]["total_queries"],
            "queries_with_injection": results["injection_stats"]["queries_with_injection"],
            "injection_rate": results["injection_stats"]["queries_with_injection"] / results["injection_stats"]["total_queries"],
            "unique_answer_spans": len(set(results["injection_stats"]["answer_spans"])),
            "answer_spans": list(set(results["injection_stats"]["answer_spans"]))
        }
    }
    
    # Save results
    output_dir = "results/answer_filtered"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_{model_name}_results.json")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return final_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["nq_serini", "triviaqa", "squad"])
    parser.add_argument("--models", nargs="+", default=["gpt4o", "claude35"])
    args = parser.parse_args()
    
    all_results = {}
    
    for dataset in args.datasets:
        all_results[dataset] = {}
        data_path = f"data/local_index_search/{dataset}/test.parquet"
        
        for model in args.models:
            generations_path = f"results/generations/{dataset}_{model}_generations.json"
            
            if not os.path.exists(generations_path):
                print(f"Skipping {dataset}-{model}: generations file not found")
                continue
                
            print(f"\nProcessing {dataset}-{model}")
            results = process_generations(dataset, model, generations_path, data_path)
            all_results[dataset][model] = results
    
    # Save overall results
    with open("results/answer_filtered/overall_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()


