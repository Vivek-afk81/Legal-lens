# ml/summary.py

from transformers import pipeline

summarizer = pipeline("text-generation", model="google/flan-t5-base")

def summarize(text):
    prompt = f"Summarize the following legal document in simple terms:\n{text}"

    result = summarizer(prompt, max_length=200)
    return result[0]["generated_text"]