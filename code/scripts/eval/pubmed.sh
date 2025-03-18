CUDA_VISIBLE_DEVICES=0

# /home/pj20/server-04/LMR/code/checkpoints/pubmed_search/pubmed_search_3b_3000/actor/global_step_100
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/SearchEngine/eval_pubmed.py \
    --model_path /shared/eng/pj20/lmr_model/pubmed_search_3b_no_reason/actor/global_step_200 \
    --data_path data/search_engine/no_reason/SearchEngine/pmd_no_reason/test_full.parquet \
    --model_name pubmed_search_3b_no_reason