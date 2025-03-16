import sys
import os

# Debug: Print current directory and Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Remove one dirname call to point to Panacea-R1
# Add the project root to Python path
sys.path.insert(0, project_root)  # This will now add Panacea-R1 to the path

import argparse
import pandas as pd
from tqdm import tqdm
import json
import re

import utils.java_init

# Add these at the top with other global variables
from pyserini.search.lucene import LuceneSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from tqdm import tqdm
from src.utils.gpt_azure import gpt_chat_4o, gpt_chat_35_msg
from src.utils.claude_api import get_claude_response


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
        print("[Error] No valid answer tags found")
        return None
    
    
def extract_json_from_llm_output(text):
    pattern = r"```json\n([\s\S]+?)\n```"
    matched_jsons = re.findall(pattern, text)
    
    if matched_jsons:
        extracted_json = matched_jsons[-1]  # get the final one
        return json.loads(extracted_json)['query']
    else:
        try:
            pattern = r"\{.*?\}"
            matched_jsons = re.findall(pattern, text, re.DOTALL)
            
            if matched_jsons:
                extracted_json = matched_jsons[-1]  # get the final one
                try:
                    return json.loads(extracted_json)['query']
                except:
                    return extracted_json.split(": ")[1].strip()
            else:
                raise ValueError('No JSON structure found.')
        except:
            return text
        
def run_index_search_bm25(search_query, topk=50):
    
    searcher = get_searcher()
    
    # Rate limit checking
    hits = searcher.search(search_query, k=topk)
    
    doc_list = [json.loads(hit.lucene_document.get('raw'))['contents'] for hit in hits]
    
    return doc_list

# Do not inject your own knowledge into the query. You should assume you don't know the answer and just use the query itself to find the answer.
def process_prompt(prompt):
    prompt = prompt.replace("""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n""", "")
    prompt = prompt.replace("Note: ", "Note: The content between <answer> and </answer> should be a valid JSON object.")
    return prompt

def compute_filtered_recall(recall_list, error_count, total_queries):
    """
    Compute filtered recall by excluding error cases.
    
    Args:
        recall_list: List of recall values (0s and 1s)
        error_count: Number of errors encountered
        total_queries: Total number of queries processed
    
    Returns:
        tuple: (filtered_recall, raw_recall)
    """
    successful_queries = total_queries - error_count
    if successful_queries == 0:
        return 0.0, 0.0
    
    raw_recall = sum(recall_list) / total_queries
    filtered_recall = sum(recall_list) / successful_queries
    return filtered_recall, raw_recall

def evaluate_model(data_path, model_name, batch_size=8):
    df = pd.read_parquet(data_path)
    
    all_responses = []
    
    inputs = [process_prompt(item[0]['content']) for item in df['prompt'].tolist()]
    targets = df['label'].tolist()
    
    error_count = 0
    total_queries = len(inputs)
    
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    recall_at_20 = []
    recall_at_50 = []
    recall_at_100 = []
    
    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]
        
        if 'gpt4o' in model_name:
            responses = [gpt_chat_4o(prompt=prompt) for prompt in batch_inputs]
        elif 'gpt35' in model_name:
            responses = [gpt_chat_35_msg(prompt=prompt) for prompt in batch_inputs]
        elif 'claude35' in model_name:
            responses = [get_claude_response(llm="sonnet", prompt=prompt) for prompt in batch_inputs]
        elif 'claude3' in model_name:
            responses = [get_claude_response(llm="hiku", prompt=prompt) for prompt in batch_inputs]
            
        all_responses.extend(responses)
        
        for i, response in enumerate(responses):
            try:
                generated_text = response
                idx = batch_start + i
                # convert target from ndarray to list
                
                extracted_solution = extract_solution(generated_text)
                query = extract_json_from_llm_output(extracted_solution)
                
                rank = 1001
                target = targets[idx].tolist()
                searched_doc_list = run_index_search_bm25(query, topk=100)
                
                for j in range(len(searched_doc_list)):
                    if has_answers(searched_doc_list[j], target, _tokenizer, regex=False):
                        rank = j + 1
                        break
                    
                recall_at_1.append(1) if rank <= 1 else recall_at_1.append(0)
                recall_at_5.append(1) if rank <= 5 else recall_at_5.append(0)
                recall_at_10.append(1) if rank <= 10 else recall_at_10.append(0)
                recall_at_20.append(1) if rank <= 20 else recall_at_20.append(0)
                recall_at_50.append(1) if rank <= 50 else recall_at_50.append(0)
                recall_at_100.append(1) if rank <= 100 else recall_at_100.append(0)
                
            except:
                recall_at_1.append(0)
                recall_at_5.append(0)
                recall_at_10.append(0)
                recall_at_20.append(0)
                recall_at_50.append(0)
                recall_at_100.append(0)
                error_count += 1
                print(f"Error: {generated_text}, Error count: {error_count}")
                continue
            
            
        try:
            print(f"Recall@1: {sum(recall_at_1) / len(recall_at_1)}")
            print(f"Recall@5: {sum(recall_at_5) / len(recall_at_5)}")
            print(f"Recall@20: {sum(recall_at_20) / len(recall_at_20)}")
            print(f"Recall@100: {sum(recall_at_100) / len(recall_at_100)}")
        except:
            continue

    with open(f"results/{model_name}_responses.json", "w") as f:
        json.dump(all_responses, f)
    
    print("Error count: ", error_count)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/local_index_search/nq_serini/test.parquet")
    parser.add_argument("--model_name", type=str, default="nq-gpt")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    evaluate_model(args.data_path, args.model_name, args.batch_size)

if __name__ == "__main__":
    main()
