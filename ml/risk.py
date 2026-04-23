from transformers import pipeline

_classifier = None

RISK_LABELS = [
    "high risk liability clause",
    "high risk indemnification clause",
    "medium risk termination clause",
    "medium risk penalty clause",
    "low risk standard clause",
]

_LABEL_TO_LEVEL = {
    "high risk liability clause":         ("High",   "Liability"),
    "high risk indemnification clause":   ("High",   "Indemnification"),
    "medium risk termination clause":     ("Medium", "Termination"),
    "medium risk penalty clause":         ("Medium", "Penalty"),
    "low risk standard clause":           ("Low",    "Standard"),
}


def _load_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _classifier


# ── ML-based detection ────────────────────────────────────────────────────────

def detect_risk_ml(clause: str):
    """
    Zero-shot classification → (risk_level, risk_type, confidence).
    Falls back to keyword matching on error.
    """
    try:
        clf = _load_classifier()
        out = clf(clause, RISK_LABELS, multi_label=False)
        top_label  = out["labels"][0]
        confidence = out["scores"][0]
        level, rtype = _LABEL_TO_LEVEL.get(top_label, ("Low", "Unknown"))
        return level, rtype, round(confidence, 3)
    except Exception:
        return detect_risk_keyword(clause)


# ── Keyword fallback ──────────────────────────────────────────────────────────

def detect_risk_keyword(clause: str):
    c = clause.lower()
    if any(w in c for w in ("not liable", "no liability", "consequential", "indemnif")):
        return "High", "Liability", 1.0
    if any(w in c for w in ("terminate", "termination")):
        return "Medium", "Termination", 1.0
    if any(w in c for w in ("penalty", "breach", "damages")):
        return "High", "Penalty", 1.0
    return "Low", "Standard", 1.0


# ── Public API ────────────────────────────────────────────────────────────────

def detect_risk(clause: str):
    """Simple wrapper kept for backward compatibility (returns risk level str)."""
    level, _, _ = detect_risk_keyword(clause)
    return level


def analyze_risks(chunks: list[str], use_ml: bool = True) -> list[dict]:
    """
    Analyse risk for every chunk.

    Returns list of dicts:
        clause     : original text
        risk       : "High" | "Medium" | "Low"
        risk_type  : e.g. "Liability", "Termination"
        confidence : float  (1.0 for keyword path)
        method     : "ml" | "keyword"
    """
    results = []
    for chunk in chunks:
        if use_ml:
            level, rtype, conf = detect_risk_ml(chunk)
            method = "ml"
        else:
            level, rtype, conf = detect_risk_keyword(chunk)
            method = "keyword"

        results.append({
            "clause":     chunk,
            "risk":       level,
            "risk_type":  rtype,
            "confidence": conf,
            "method":     method,
        })
    return results