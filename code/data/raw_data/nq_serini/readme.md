# Data Download Guidance


Dataset download:
```bash
python download_nq.py
```

Prebuilt indexes:
```bash
python -c "from pyserini.search.lucene import LuceneSearcher; LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')"
```

BM25 baseline:
```bash
- model_name: BM25-k1_0.9_b_0.4
command: 
    - python -m pyserini.search.lucene --threads ${sparse_threads} --batch-size ${sparse_batch_size} --index wikipedia-dpr-100w --topics nq-test --output $output --bm25 --k1 0.9 --b 0.4
scores:
    - Top5: 44.82
    Top20: 64.02
    Top100: 79.20
    Top500: 86.59
    Top1000: 88.95
```

DPR baseline:
```bash
- model_name: DPR
command: 
    - python -m pyserini.search.faiss --threads ${dense_threads} --batch-size ${dense_batch_size} --index wikipedia-dpr-100w.dpr-single-nq --encoder facebook/dpr-question_encoder-single-nq-base --topics nq-test --output $output
scores:
    - Top5: 68.61
    Top20: 80.58 
    Top100: 86.68
    Top500: 90.91
    Top1000: 91.83
```


Test data (with groundtruth documents):
```bash
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
unzip nq.zip
```




