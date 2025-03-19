export CUDA_VISIBLE_DEVICES=1

python -m pyserini.encode \
 input  --corpus /shared/eng/pj20/lmr_model/raw_data/msmarco/collection_jsonl \
        --fields text \
        --delimiter "\n" \
        --shard-id 0 \
        --shard-num 1 \
 output --embeddings indexes/contriever-msmarco-passage-dense-index \
        --to-faiss \
 encoder --encoder facebook/contriever \
        --fields text \
        --batch 16 \
        --pooling mean \
        --dimension 768
        --fp16