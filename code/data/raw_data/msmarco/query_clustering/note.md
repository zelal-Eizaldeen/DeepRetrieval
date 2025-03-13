# MS-MARCO-V1

## Introduction

We cluster and split the queries in MS MARCO into several topics.

## Quick start

```bash
python 1_merge_queries.py
python 2_query_embedding.py
python 4_query_clustering.py
# run 5_check_and_name_cluster.ipynb to confirm cluster topic name
python 6_generate_new_query.py
```

## Source

* official: https://microsoft.github.io/msmarco/
* Download: https://microsoft.github.io/msmarco/Datasets.html

## Data format

* collection.tsv: pid, passage
* queries.xxx.tsv: qid, query
* qrels.xxx.tsv: qid, group_id, pid, relevance judge - (qid, 0, pid, 1)