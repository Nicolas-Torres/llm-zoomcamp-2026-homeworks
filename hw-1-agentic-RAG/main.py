"""
LLM Zoomcamp 2026 - Homework 1: Agentic RAG
"""

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from openai import OpenAI

from dotenv import load_dotenv
import json

load_dotenv()

# ---------------------------------------------------------------------------
# Q1. How many lesson pages are in the dataset?
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

print("Q1: How many lesson pages are in the dataset?")
print(" -> lesson pages:", len(documents))

# ---------------------------------------------------------------------------
# Q2. Indexing and searching
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Q3. RAG
# ---------------------------------------------------------------------------

MODEL = "gpt-5.4-mini"

INSTRUCTIONS = """
    Your task is to answer questions from the course participants based on the provided context.
    Use the context to find relevant information and provide accurate answers. If the answer is not found in the context, respond with "I don't know."
""".strip()

PROMPT_TEMPLATE = """
    QUESTION: {question}
    CONTEXT: {context}
""".strip()

class RAG:
    def __init__(self, index, llm_client, model=MODEL):
        self.index = index
        self.llm_client = llm_client
        self.model = model
 
    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)
 
    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(doc["filename"])
            lines.append(doc["content"])
            lines.append("")
        return "\n".join(lines).strip()
 
    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return PROMPT_TEMPLATE.format(question=query, context=context)
 
    def llm(self, prompt):
        messages = [
            {
                "role": "developer",
                "content": INSTRUCTIONS
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
        return self.llm_client.responses.create(model=self.model, input=messages)
 
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response.output_text, response.usage

client = OpenAI()

rag = RAG(index=index, llm_client=client)
answer, usage = rag.rag(QUERY)
tokens_q3 = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)

print("Q3: RAG")
print(" -> input tokens:", tokens_q3)

# ---------------------------------------------------------------------------
# Q4. Chunking
# ---------------------------------------------------------------------------

chunks = chunk_documents(documents, size=2000, step=1000)
print("Q4: Chunking")
print(" -> number of chunks:", len(chunks))

# ---------------------------------------------------------------------------
# Q5. RAG with chunking
# ---------------------------------------------------------------------------

chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
chunk_index.fit(chunks)

rag = RAG(chunk_index, llm_client=client)
answer, usage = rag.rag(QUERY)

tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)

print("Q5: RAG with chunking:")
print(" -> ratio Q3/Q5:", round(tokens_q3 / tokens, 1))

# ---------------------------------------------------------------------------
# Q6. Turning it into an agent
# ---------------------------------------------------------------------------

def search(query: str) -> list:
        """Search the course lessons for relevant passages."""
        results = chunk_index.search(query, num_results=5)

        formatted_results = []

        for result in results:
            clean_data = {
                "filename": result["filename"],
                "content": result["content"]
            }
            formatted_results.append(clean_data)

        return formatted_results
 
tools = [{
    "type": "function",
    "name": "search",
    "description": "Search the course lessons for relevant passages.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}]

agent_instructions = (
    "You're a course teaching assistant. Answer the student's question "
    "using the search tool. Make multiple searches with different keywords "
    "before answering."
)
question = "How does the agentic loop work, and how is it different from plain RAG?"

messages = [
    {"role": "developer", "content": agent_instructions},
    {"role": "user", "content": question},
]

search_calls = 0

MAX_ITERATIONS = 10

for _ in range(MAX_ITERATIONS):
    response = client.responses.create(
        model=MODEL,
        input=messages, 
        tools=tools
    )
    messages += response.output

    function_calls = []

    for element in response.output:
        if element.type == "function_call":
            function_calls.append(element)

    if not function_calls:
        break

    for fc in function_calls:
        args = json.loads(fc.arguments)
        if fc.name == "search":
            search_calls += 1
            result = search(**args)
            messages.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": json.dumps(result),
            })

print("Q6: Turning it into an agent")
print(" -> number of calls:", search_calls)
