# Data Processing Guide for MSMARCO

Reference:
* https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-dense-vector-index
* https://sbert.net/docs/sentence_transformer/pretrained_models.html

1. Download the collection
```bash
mkdir -p collections/msmarco-passage
mkdir indexes

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz -P collections/msmarco-passage


tar xvfz collections/msmarco-passage/collectionandqueries.tar.gz -C collections/msmarco-passage
```

2. Concert to Pyserini format
```bash
python code/data/raw_data/msmarco/convert.py \
 --collection-path ./collections/msmarco-passage/collection.tsv \
 --output-folder ./collections/msmarco-passage/collection_jsonl
```


3. Indexing with the processed collection

* Sparse:

```bash
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input collections/msmarco-passage/collection_jsonl \
    --index indexes/lucene-index-msmarco-passage \
    --generator DefaultLuceneDocumentGenerator \
    --threads 9 \
    --storePositions \
    --storeDocvectors \
    --storeRaw
```

* Dense: `sentence-transformers/all-mpnet-base-v2`: dimension 768, `sentence-transformers/all-MiniLM-L6-v2`: dimension 384

```bash
export CUDA_VISIBLE_DEVICES=3

python -m pyserini.encode \
 input  --corpus collections/msmarco-passage/collection_jsonl \
        --fields text \
        --delimiter "\n" \
        --shard-id 0 \
        --shard-num 1 \
 output --embeddings indexes/mpnet_msmarco_passage_dense_index \
        --to-faiss \
 encoder --encoder sentence-transformers/all-mpnet-base-v2 \
        --fields text \
        --batch 16 \
        --dimension 768
        --fp16
```


```bash
export CUDA_VISIBLE_DEVICES=2

python -m pyserini.encode \
 input  --corpus collections/msmarco-passage/collection_jsonl \
        --fields text \
        --delimiter "\n" \
        --shard-id 0 \
        --shard-num 1 \
 output --embeddings indexes/minilm_msmarco_passage_dense_index \
        --to-faiss \
 encoder --encoder sentence-transformers/all-MiniLM-L6-v2 \
        --fields text \
        --batch 16 \
        --dimension 384
        --fp16
```
