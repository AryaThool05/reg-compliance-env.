def test_no_score_is_exactly_zero_or_one():
    """This test mirrors the exact Phase 2 validator check."""
    from graders.grader_easy import grade_easy
    from graders.grader_medium import grade_medium
    from graders.grader_hard import grade_hard
    from models import Action
    
    all_scores = []
    
    # Best-case inputs
    best_action_easy = Action(violation_ids=["ART6-1"], severity="high", explanation="test", fix_suggestion="fix")
    best_action_medium = Action(violation_ids=["PURPOSE LIMITATION", "MINIMIS", "LAWFUL BASIS CONSENT", "TRANSPARENT", "RETENTION"], severity="high", explanation="test", fix_suggestion="test"*20)
    best_action_hard = Action(violation_ids=["V1", "V2"], severity="high", explanation="test", fix_suggestion="This is a very long fix suggestion that should definitely be over 50 characters to trigger the maximum score.")
    
    # Worst-case inputs
    worst_action = Action(violation_ids=[], severity="none", explanation="test", fix_suggestion=None)
    
    # Partial inputs
    partial_action = Action(violation_ids=["V1"], severity="low", explanation="test", fix_suggestion="short fix")
    
    # Ground truths
    gt_easy = {}
    gt_medium = {}
    gt_hard = {"fixed_violations": ["V1"], "new_violations": ["V2"]}
    
    # Collect scores
    for action in [best_action_easy, worst_action, partial_action]:
        all_scores.append(grade_easy(action, gt_easy).score)
    for action in [best_action_medium, worst_action, partial_action]:
        all_scores.append(grade_medium(action, gt_medium).score)
    for action in [best_action_hard, worst_action, partial_action]:
        all_scores.append(grade_hard(action, gt_hard).score)
        
    for score in all_scores:
        assert score > 0.0, f"Score {score} is exactly 0.0 — will fail Phase 2"
        assert score < 1.0, f"Score {score} is exactly 1.0 — will fail Phase 2"
        assert 0.0 < score < 1.0, f"Score {score} out of strict range (0,1)"
