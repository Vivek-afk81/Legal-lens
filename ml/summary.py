from transformers import pipeline

_summarizer = None

ASPECTS = [
    ("parties",      "Who are the parties involved in this agreement?"),
    ("duration",     "What is the duration or term of this agreement?"),
    ("termination",  "How and when can this agreement be terminated?"),
    ("liability",    "What are the liability limitations or exclusions?"),
    ("disputes",     "How are disputes or conflicts resolved?"),
]


def _load_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
        )
    return _summarizer


def summarize(text: str, mode: str = "brief") -> str | dict:
    """
    Summarise a legal document.

    Args:
        text : document string
        mode : "brief"    → single plain-English summary (str)
               "detailed" → aspect-by-aspect dict

    Returns:
        str  (brief mode)
        dict (detailed mode) with keys: parties, duration, termination,
                                         liability, disputes
    """
    model = _load_summarizer()

    if mode == "brief":
        prompt = (
            "Summarise the following legal agreement in simple, plain English "
            "so a non-lawyer can understand it:\n\n"
            f"{text}\n\nSummary:"
        )
        out = model(prompt, max_new_tokens=200, do_sample=False)
        return out[0]["generated_text"].strip()

    if mode == "detailed":
        summaries = {}
        for key, question in ASPECTS:
            prompt = (
                f"Based on this legal document, answer: {question}\n\n"
                f"Document:\n{text}\n\nAnswer:"
            )
            out = model(prompt, max_new_tokens=100, do_sample=False)
            summaries[key] = out[0]["generated_text"].strip()
        return summaries

    raise ValueError(f"Unknown mode '{mode}'. Use 'brief' or 'detailed'.")


def summarize_clause(clause_text: str) -> str:
    """Explain a single clause in plain English."""
    model = _load_summarizer()
    prompt = f"Explain this legal clause in simple language:\n{clause_text}\n\nSimple explanation:"
    out = model(prompt, max_new_tokens=100, do_sample=False)
    return out[0]["generated_text"].strip()