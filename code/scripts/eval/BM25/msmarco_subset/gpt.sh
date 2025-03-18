
DATASET=$1
DATA_PATH=data/local_index_search/$DATASET/sparse/test.parquet

python src/eval/BM25/baselines/model_generate/msmarco_subset/gpt-4o.py \
    --data_path $DATA_PATH \
    --dataset $DATASET 