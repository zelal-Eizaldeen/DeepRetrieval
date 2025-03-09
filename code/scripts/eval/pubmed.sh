CUDA_VISIBLE_DEVICES=6

python src/eval/eval_pubmed.py \
    --model_path /shared/eng/pj20/lmr_model/pubmed_search_3b/actor/global_step_1200 \
    --data_path data/search_engine/pubmed/test_full.parquet