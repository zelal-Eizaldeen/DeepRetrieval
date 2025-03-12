from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from src.Lucene.utils import ndcg_at_k

# REPLACE THIS WITH YOUR OWN INDEX PATH
index_dir = "/shared/eng/pj20/lmr_model/raw_data/msmarco/indexes/lucene-index-msmarco-passage"
_searcher = None

# _searcher = FaissSearcher.from_prebuilt_index('msmarco-v1-passage.bge-base-en-v1.5', None)



from pyserini.search.faiss import FaissSearcher
FaissSearcher.list_prebuilt_indexes()
