import json
import sys
import os
from tqdm import tqdm
import pdb
sys.path.append('./')

from src.Lucene.scifact.search import PyseriniMultiFieldSearch
from src.Lucene.utils import ndcg_at_k

import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset to evaluate")
    parser.add_argument('--res_path', type=str, default="../results/Qwen-inst-scifact.json", help="Path to the qrels file")
    args = parser.parse_args()

    search_system = PyseriniMultiFieldSearch(index_dir=f"data/local_index_search/{args.dataset}/pyserini_index")

    with open(args.res_path, "r", encoding="utf-8") as file:
        qrel_test = json.load(file)

    # transform qrel_test_dict to list
    test_data = []
    for qid, value in qrel_test.items():
        test_data.append(value)

    ndcg = []
    batch_size = 100

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        queries = [item['generated_text'] for item in batch]
        targets = {item['generated_text']: item['target'] for item in batch} 
        
        results = search_system.batch_search(queries, top_k=10, threads=16)
        
        for query in queries:
            retrieved = [result[0] for result in results.get(query, [])]
            ndcg.append(ndcg_at_k(retrieved, targets[query], 10))
    
    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")