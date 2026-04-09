# ml/risk.py

def detect_risk(clause):
    clause = clause.lower()

    if "terminate" in clause:
        return "Medium"
    elif "not liable" in clause or "no liability" in clause:
        return "High"
    elif "penalty" in clause:
        return "High"
    else:
        return "Low"


def analyze_risks(chunks):
    results = []

    for c in chunks:
        risk = detect_risk(c)
        results.append({
            "clause": c,
            "risk": risk
        })

    return results