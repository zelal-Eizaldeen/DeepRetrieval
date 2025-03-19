export CUDA_VISIBLE_DEVICES=0

# /shared/eng/pj20/lmr_model/nq_serini_3b_continue/actor/global_step_1150 \
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/BM25/nq.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --model_name nq_base_no_reason \
    --data_path data/local_index_search/no_reason/nq_serini_no_reason/test.parquet