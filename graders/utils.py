def safe_score(score: float) -> float:
    """
    Ensures score is strictly between 0 and 1 exclusive.
    Required by OpenEnv hackathon validator — 0.0 and 1.0 both fail.
    """
    score = float(score)
    if score <= 0.0:
        return 0.001
    if score >= 1.0:
        return 0.999
    return round(score, 4)
