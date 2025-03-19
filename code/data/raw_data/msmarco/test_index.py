import os
from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder
from pyserini.search.lucene import LuceneSearcher



os.environ["CUDA_VISIBLE_DEVICES"] = "3"

index_dir = "/home/azureuser/cloudfiles/code/DeepRetrieval/indexes/contriever-msmarco-passage-dense-index"
query_encoder_name = "facebook/contriever"

query_encoder = AutoQueryEncoder(query_encoder_name, pooling='mean', device='cuda')
dense_searcher = FaissSearcher(index_dir=index_dir, query_encoder=query_encoder)

query = "France"

results = dense_searcher.search(query)
docids = [results[i].docid for i in range(len(results))]

sparse_searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')


print(query)


for docid in docids:
    doc = sparse_searcher.doc(docid).raw()
    print(doc)
    print("--------------------------------")



