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

