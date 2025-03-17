DATASET=$1
DATA_PATH=data/local_index_search/no_reason/Dense/$DATASET/test.parquet
SAVE_DIR=../results_dense/no_reason

python src/eval/Dense/baselines/model_generate_no_reasoning/claude.py \
    --data_path $DATA_PATH \
    --dataset $DATASET \
    --save_dir $SAVE_DIR