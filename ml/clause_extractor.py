from transformers import pipeline
from ml.chunking import chunk_by_clause

_classifier = None

CLAUSE_TYPES = [
    "termination clause",
    "liability clause",
    "payment clause",
    "confidentiality clause",
    "dispute resolution clause",
    "intellectual property clause",
    "governing law clause",
    "force majeure clause",
    "indemnification clause",
    "warranty clause",
    "general clause",
]


def _load_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _classifier


def classify_clause(clause_text: str) -> dict:
    """
    Classify a single clause.

    Returns
    -------
    dict
        clause_type : str   — top predicted label
        confidence  : float — probability of top label
        top_3       : dict  — {label: score} for top 3 predictions
    """
    try:
        clf = _load_classifier()
        result = clf(clause_text, CLAUSE_TYPES, multi_label=False)
        return {
            "clause_type": result["labels"][0],
            "confidence":  round(result["scores"][0], 3),
            "top_3": {
                label: round(score, 3)
                for label, score in zip(result["labels"][:3], result["scores"][:3])
            },
        }
    except Exception as e:
        return {"clause_type": "general clause", "confidence": 0.0, "top_3": {}, "error": str(e)}


def extract_clauses(document_text: str) -> list[dict]:
    """
    Extract and classify all clauses from a document.

    Returns
    -------
    list[dict]
        text        : original clause text
        clause_type : predicted type
        confidence  : model confidence
        top_3       : top 3 predictions with scores
    """
    chunks = chunk_by_clause(document_text)
    results = []

    for chunk in chunks:
        classification = classify_clause(chunk)
        results.append({
            "text":        chunk,
            "clause_type": classification["clause_type"],
            "confidence":  classification["confidence"],
            "top_3":       classification.get("top_3", {}),
        })

    return results


def group_by_type(document_text: str) -> dict[str, list[str]]:
    """
    Return a dict mapping clause type → list of matching clause texts.
    Useful for quickly finding all liability or termination clauses.
    """
    clauses = extract_clauses(document_text)
    grouped: dict[str, list[str]] = {}
    for c in clauses:
        ctype = c["clause_type"]
        grouped.setdefault(ctype, []).append(c["text"])
    return grouped