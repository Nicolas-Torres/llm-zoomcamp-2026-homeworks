from hw_2_vector_search.embedder import Embedder
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import VectorSearch, Index

embedder = Embedder()

# ---------------------------------------------------------------------------
# Q1. How does approximate nearest neighbor search work?
# ---------------------------------------------------------------------------

QUERY_Q1 = "How does approximate nearest neighbor search work?"

v = embedder.encode(text=QUERY_Q1, normalize=True)
v_0 = v[0]

print("\nQ1: Embeeding a query")
print(f" -> Numbers             : {len(v)}")
print(f" -> v[0]                : {v_0}")

