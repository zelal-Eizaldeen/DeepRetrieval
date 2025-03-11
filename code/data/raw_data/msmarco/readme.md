# Data Processing Guide for MSMARCO

Download the collection
```bash
mkdir collections/msmarco-passage

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz -P collections/msmarco-passage


tar xvfz collections/msmarco-passage/collectionandqueries.tar.gz -C .
```

Concert to Pyserini format
```bash
python convert.py \
 --collection-path ./collection.tsv \
 --output-folder ./collection_jsonl
```


Indexing with the processed collection
```bash
python -m pyserini.index.lucene   --collection JsonCollection   --input collection_jsonl   --index indexes/lucene-index-msmarco-passage   --generator DefaultLuceneDocumentGenerator   --threads 9   --storePositions --storeDocvectors --storeRaw
```
