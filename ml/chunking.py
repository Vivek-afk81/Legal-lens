# ml/chunking.py

def chunk_text(text):
    chunks = text.split(".")
    chunks = [c.strip() for c in chunks if len(c.strip()) > 0]
    return chunks