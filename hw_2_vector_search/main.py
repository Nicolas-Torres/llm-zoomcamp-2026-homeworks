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

# ---------------------------------------------------------------------------
# Q2. Cosine similarity
# ---------------------------------------------------------------------------

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)

TARGET_FILE = "02-vector-search/lessons/07-sqlitesearch-vector.md"

target_doc = None

for d in documents:
    if d["filename"] == TARGET_FILE:
        target_doc = d
        break

v_doc = embedder.encode(text=target_doc["content"])
similarity = float(v.dot(v_doc))

print("\nQ2: Cosine similarity")
print(f" -> Uploaded documents  : {len(documents)}")
print(f" -> File                : {TARGET_FILE}")
print(f" -> Similarity          : {similarity}")


# ---------------------------------------------------------------------------
# Q3. Chunking and search by hand
# ---------------------------------------------------------------------------

chunks = chunk_documents(documents, size=2000, step=1000)

contents = [c["content"] for c in chunks]

X = embedder.encode_batch(contents)


print("\nQ3: Chunking and search by hand")

print(f" -> Matriz X            : {X.shape}")

scores = X.dot(v)
best_idx = int(scores.argmax())
best_chunk = chunks[best_idx]
print(f" -> # Chunk             : {best_idx}")
print(f" -> filename            : {best_chunk['filename']}")
print(f" -> score               : {scores[best_idx]:.4f}")


# ---------------------------------------------------------------------------
# Q4. Vector search with minsearch
# ---------------------------------------------------------------------------

vs = VectorSearch(keyword_fields=["filename"])
vs.fit(X, chunks)

QUERY_Q4 = "What metric do we use to evaluate a search engine?"

vector_q4 = embedder.encode(QUERY_Q4)

results_q4 = vs.search(vector_q4, num_results=5)

print("\nQ4: Vector search with minsearch")
print(f" -> Query: {QUERY_Q4!r}")
print("\n -> Top 5 resultados (vector search):")
for i, r in enumerate(results_q4):
    print(f"    [{i}] {r['filename']}  (start={r['start']})")
 
print(f"\n -> Primer resultado  : {results_q4[0]['filename']}")


# ---------------------------------------------------------------------------
# Q5. Text search vs vector search
# ---------------------------------------------------------------------------

text_idx = Index(text_fields=["content"], keyword_fields=["filename"])
text_idx.fit(chunks)

QUERY_Q5 = "How do I store vectors in PostgreSQL?"
v_q5 = embedder.encode(QUERY_Q5)

vector_results_q5 = vs.search(v_q5, num_results=5)
text_results_q5   = text_idx.search(QUERY_Q5, num_results=5)

vector_files_q5 = {r["filename"] for r in vector_results_q5}
text_files_q5   = {r["filename"] for r in text_results_q5}

print("\nQ5: Text search vs vector search")
print(" -> Top 5 - Vector search:")
for i, r in enumerate(vector_results_q5):
    print(f"  [{i}] {r['filename']}")
 
print("\n -> Top 5 - Text search:")
for i, r in enumerate(text_results_q5):
    print(f"  [{i}] {r['filename']}")

only_in_vector = vector_files_q5 - text_files_q5
only_in_text   = text_files_q5   - vector_files_q5
print(f"\n -> In vector but not in text: {only_in_vector}")


# ---------------------------------------------------------------------------
# Q6. Hybrid search (RRF)
# ---------------------------------------------------------------------------

def rrf(result_lists, k=60, num_results=5):
    """
    Reciprocal Rank Fusion.
 
    Args:
        result_lists: lista de listas de dicts, cada una rankeada por score.
        k:            constante de aplanado (default=60, del paper original).
        num_results:  cuántos resultados devolver.
 
    Returns:
        Lista de dicts rankeados por score RRF descendente.
    """
    scores: dict = {}
    docs:   dict = {}
 
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
 
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


QUERY_Q6 = "How do I give the model access to tools?"
v_q6 = embedder.encode(QUERY_Q6)
 
vector_results_q6 = vs.search(v_q6, num_results=5)
text_results_q6   = text_idx.search(QUERY_Q6, num_results=5)
hybrid_results_q6 = rrf([vector_results_q6, text_results_q6])
 

print("\nQ6: Hybrid search (RRF)")

print(f" -> Query: {QUERY_Q6!r}\n")
 
print(" -> Top 5 - Vector search:")
for i, r in enumerate(vector_results_q6):
    print(f"  [{i}] {r['filename']}")
 
print("\n -> Top 5 - Text search:")
for i, r in enumerate(text_results_q6):
    print(f"  [{i}] {r['filename']}")
 
print("\n -> Top 5 - Hybrid (RRF):")
for i, r in enumerate(hybrid_results_q6):
    print(f"  [{i}] {r['filename']}")
 
print(f"\n -> First result RRF: {hybrid_results_q6[0]['filename']}")