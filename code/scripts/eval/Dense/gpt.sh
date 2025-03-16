DATASET=$1
DATA_PATH=data/local_index_search/$DATASET/dense/test.parquet

python src/eval/Dense/baselines/model_generate/gpt-4o.py \
    --data_path $DATA_PATH \
    --dataset $DATASET 