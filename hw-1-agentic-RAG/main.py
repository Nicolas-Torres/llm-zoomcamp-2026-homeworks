"""
LLM Zoomcamp 2026 - Homework 1: Agentic RAG
"""

from gitsource import GithubRepositoryDataReader
from minsearch import Index

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

print("Q1: How many lesson pages are in the dataset?")
print(" -> lesson pages:", len(documents))


QUERY = "How does the agentic loop keep calling the model until it stops?"

# Note: documents is a list of dicts with "content" and "filename" keys, e.g.:
# [
#     {
#         "content": "# Lesson 1: Introduction to LLMs\n\nIn this lesson, we will cover the basics of large language models (LLMs)...",
#         "filename": "lessons/01-introduction.md"
#     }
# ]

index = Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)

index.fit(documents)

results = index.search(QUERY, num_results=3)

filename = results[0]["filename"]

print("Q2: Indexing and searching")
print(" -> filename:", filename)