CUDA_VISIBLE_DEVICES=6

# /home/pj20/server-04/LMR/code/checkpoints/pubmed_search/pubmed_search_3b_3000/actor/global_step_100
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/SearchEngine/eval_pubmed.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path data/search_engine/pubmed/test_full.parquet \
    --model_name pubmed_search_3b_zero_shot