DATASET=$1
DATA_PATH=data/local_index_search/no_reason/$DATASET/sparse/test.parquet
SAVE_DIR=../results/no_reason

python src/eval/BM25/baselines/model_generate_no_reasoning/msmarco_subset/claude.py \
    --data_path $DATA_PATH \
    --dataset $DATASET \
    --save_dir $SAVE_DIR