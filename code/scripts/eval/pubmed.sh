CUDA_VISIBLE_DEVICES=0,3

python src/eval/eval_pubmed.py \
    --model_path /home/pj20/server-04/LMR/code/checkpoints/pubmed_search/pubmed_search_3b_3000/actor/global_step_100 \
    --data_path data/search_engine/pubmed/test_full.parquet