"""
LLM Zoomcamp 2026 - Homework 1: Agentic RAG
"""

from gitsource import GithubRepositoryDataReader

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
