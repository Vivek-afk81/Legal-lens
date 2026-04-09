# ml/retrieval.py

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=2):
    scores = []

    for emb in chunk_embeddings:
        score = cosine_similarity(query_embedding, emb)
        scores.append(score)

    top_indices = np.argsort(scores)[-k:][::-1]

    top_chunks = [chunks[i] for i in top_indices]
    return top_chunks, scores