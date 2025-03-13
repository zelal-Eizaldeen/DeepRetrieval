export CUDA_VISIBLE_DEVICES=7

python src/eval/SearchEngine/eval_ctgov.py \
    --model_path /shared/eng/pj20/lmr_model/ctgov_3b_transfer_1/actor/global_step_100 \
    --data_path data/search_engine/ctgov/test_full.parquet