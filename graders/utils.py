def safe_score(score: float) -> float:
    """
    Ensures score is strictly between 0 and 1 exclusive.
    Required by OpenEnv hackathon validator — 0.0 and 1.0 both fail.

    Bounds: 0.05 (floor) and 0.95 (ceiling) — not 0.0 or 1.0.
    - Perfect match → 0.95 (not 1.0)
    - No match      → 0.05 (not 0.0)
    """
    score = float(score)
    if score <= 0.0:
        return 0.05
    if score >= 1.0:
        return 0.95
    return round(min(0.95, max(0.05, score)), 4)
