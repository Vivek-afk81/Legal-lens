from ml.chunking import chunk_by_clause, chunk_text
from ml.embeddings import embed_text, embed_query
from ml.retrieval import retrieve_top_k
from transformers import pipeline

_generator = None


def _load_generator():
    global _generator
    if _generator is None:
        _generator = pipeline(
            "text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256
        )
    return _generator


def build_prompt(context: str, query: str) -> str:
    return (
        "You are a legal assistant.\n"
        "Answer the question using ONLY the numbered context below.\n"
        'If the answer is absent, respond with "Not found in document."\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def get_answer(
    document_text: str,
    query: str,
    k: int = 3,
    use_clause_chunking: bool = True,
) -> dict:
    """
    Full RAG pipeline with source citations.

    Returns
    -------
    dict
        answer          : str
        retrieved_chunks: list[str]
        scores          : list[float]
        citations       : list[dict]  — {id, text, score}
        total_chunks    : int
    """
    # 1. Chunk
    chunks = chunk_by_clause(document_text) if use_clause_chunking else chunk_text(document_text)
    if not chunks:
        return {"answer": "No content found.", "retrieved_chunks": [], "scores": [], "citations": [], "total_chunks": 0}

    # 2. Embed
    chunk_embeddings = embed_text(chunks)
    query_embedding  = embed_query(query)

    # 3. Retrieve
    k = min(k, len(chunks))
    top_chunks, scores = retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=k)

    # 4. Build numbered context + citation metadata
    context_lines = [f"[{i+1}] {chunk}" for i, chunk in enumerate(top_chunks)]
    context = "\n".join(context_lines)

    citations = [
        {"id": i + 1, "text": chunk, "score": scores[i]}
        for i, chunk in enumerate(top_chunks)
    ]

    # 5. Generate
    prompt    = build_prompt(context, query)
    generator = _load_generator()
    response  = generator(prompt, max_new_tokens=150, do_sample=False)
    answer    = response[0]["generated_text"].strip()

    return {
        "answer":           answer,
        "retrieved_chunks": top_chunks,
        "scores":           scores,
        "citations":        citations,
        "total_chunks":     len(chunks),
    }