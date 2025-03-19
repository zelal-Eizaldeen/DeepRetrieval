export CUDA_VISIBLE_DEVICES=1

# /shared/eng/pj20/lmr_model/pubmed_search_3b_no_reason/actor/global_step_200
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/SearchEngine/eval_pubmed.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path data/search_engine/no_reason/SearchEngine/pmd_no_reason/test_full.parquet \
    --model_name pubmed_search_3b_base_no_reason