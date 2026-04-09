from ml.rag import get_answer
from ml.risk import analyze_risks
from ml.summary import summarize

# sample doc
with open("data/sample.txt") as f:
    text = f.read()

query = "How can the agreement be terminated?"

# RAG
result = get_answer(text, query)

print("\nANSWER:\n", result["answer"])
print("\nCHUNKS:\n", result["retrieved_chunks"])

# Risk
risks = analyze_risks(result["retrieved_chunks"])
print("\nRISKS:\n", risks)

# Summary
print("\nSUMMARY:\n", summarize(text))
print("TEXT:\n", text)
print("TEXT LENGTH:", len(text))