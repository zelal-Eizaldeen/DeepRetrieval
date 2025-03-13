CUDA_VISIBLE_DEVICES=0,3

python src/eval/eval_pubmed.py \
    --model_path /shared/eng/pj20/lmr_model/pubmed_search/pubmed_search_3b/actor/global_step_1250 \
    --data_path data/search_engine/pubmed/test_full.parquet