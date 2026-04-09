# ml/embeddings.py

from sentence_transformers import SentenceTransformer

# Load model once (IMPORTANT)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text_list):
    return model.encode(text_list)

def embed_query(query):
    return model.encode([query])[0]