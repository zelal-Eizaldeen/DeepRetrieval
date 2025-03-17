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

_searcher = None
_tokenizer = SimpleTokenizer()

PROMPT = """ Your task is to (1) carefully check if there's a answer span in the query that either exactly matches any of the answer candidates or is a very similar paraphrase of any of the answer candidates.
(2) If there is, remove the answer span(s) from the query and return the cleaned query.

You response should be in the following JSON format:
<answer>
{
    "has_answer": "True/False",
    "answer_span_in_query": ["answer_span_1", "answer_span_2", ...],
    "matched_answer_candidates": ["answer_candidate_1", "answer_candidate_2", ...],
    "cleaned_query": "..."
}
</answer>

Note: The content between <answer> and </answer> should be a valid JSON object, e.g., using double quotes for strings, using slashes for special characters.

Here's the query:
{query}

Here's the answer candidates:
{answer_candidates}

Your response:
"""


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
    

def get_if_answer_span_in_query(query, answer_candidates):
    prompt = PROMPT.format(query=query, answer_candidates=answer_candidates)
    response = get_claude_response("sonnet", prompt)
    return extract_solution(response)

def extract_json_from_llm_output(text):
    pattern = r"```json\n([\s\S]+?)\n```"
    matched_jsons = re.findall(pattern, text)
    
    if matched_jsons:
        extracted_json = matched_jsons[-1]  # get the final one
        return json.loads(extracted_json)
    else:
        try:
            pattern = r"\{.*?\}"
            matched_jsons = re.findall(pattern, text, re.DOTALL)
            
            if matched_jsons:
                extracted_json = matched_jsons[-1]  # get the final one
                try:
                    print(f"Extracted JSON: {extracted_json}")
                    return json.loads(extracted_json)
                except:
                    print(f"Warning: No JSON structure found. Using the response itself. {extracted_json}")
                    return None
            else:
                raise ValueError('No JSON structure found.')
        except:
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
            target = targets[idx]
            
            # Evaluate original query
            original_metrics = evaluate_query(original_query, target)
            for k, v in original_metrics.items():
                results["original_metrics"][k].append(v)
            
            # Check for knowledge injection
            injection_check = get_if_answer_span_in_query(original_query, target)
            injection_result = extract_json_from_llm_output(injection_check)
            
            if injection_result and injection_result.get("has_answer") == "True":
                results["injection_stats"]["queries_with_injection"] += 1
                results["injection_stats"]["answer_spans"].extend(injection_result.get("answer_span_in_query", []))
                
                # Evaluate cleaned query
                cleaned_query = injection_result.get("cleaned_query", original_query)
                cleaned_metrics = evaluate_query(cleaned_query, target)
                for k, v in cleaned_metrics.items():
                    results["cleaned_metrics"][k].append(v)
            else:
                # If no injection, use same metrics as original
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
    parser.add_argument("--models", nargs="+", default=["gpt35", "gpt4o", "claude3", "claude35"])
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


