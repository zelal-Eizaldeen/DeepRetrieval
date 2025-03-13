CUDA_VISIBLE_DEVICES=0,3

python src/eval/eval_ctgov.py \
    --model_path /shared/eng/pj20/lmr_model/ctgov_3b_transfer/actor/global_step_100 \
    --data_path data/search_engine/ctgov/test_full.parquet