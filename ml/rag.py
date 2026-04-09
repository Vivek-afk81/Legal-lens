# ml/rag.py

from ml.chunking import chunk_text
from ml.embeddings import embed_text, embed_query
from ml.retrieval import retrieve_top_k
from transformers import pipeline

# Load LLM once
generator = pipeline("text-generation", model="google/flan-t5-base")


def build_prompt(context, query):
    return f"""
You are a legal assistant.

Answer ONLY from the context.
If answer is not present, say "Not found in document".

Context:
{context}

Question:
{query}

Answer:
"""


def get_answer(document_text, query):
    # Step 1: Chunking
    chunks = chunk_text(document_text)

    # Step 2: Embeddings
    chunk_embeddings = embed_text(chunks)
    query_embedding = embed_query(query)

    # Step 3: Retrieval
    top_chunks, scores = retrieve_top_k(
        query_embedding,
        chunk_embeddings,
        chunks,
        k=2
    )

    # Step 4: Build context
    context = "\n".join(top_chunks)

    # Step 5: Prompt
    prompt = build_prompt(context, query)

    # Step 6: Generate answer
    response = generator(prompt, max_length=200)

    answer = response[0]["generated_text"]

    return {
        "answer": answer,
        "retrieved_chunks": top_chunks,
        "scores": scores
    }