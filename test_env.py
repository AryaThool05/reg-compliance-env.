"""Quick integration test for RegComplianceEnv."""
import asyncio
from environment import RegComplianceEnv
from models import Action


async def test_all():
    env = RegComplianceEnv()

    # --- Easy ---
    r = await env.reset("easy")
    print(f"[easy] RESET OK  | article_ref={r.observation.article_ref}")
    a = Action(violation_ids=["ART6-CONSENT"], severity="high", explanation="No consent obtained")
    s = await env.step(a)
    print(f"[easy] STEP  OK  | reward={s.reward}, done={s.done}")
    st = await env.state()
    print(f"[easy] STATE OK  | step_count={st['step_count']}, done={st['done']}")

    # --- Medium ---
    r = await env.reset("medium")
    print(f"\n[med]  RESET OK  | article_ref={r.observation.article_ref}")
    a = Action(
        violation_ids=["ART5-RETENTION", "ART6-LAWFUL-BASIS", "ART13-TRANSPARENCY"],
        severity="high",
        explanation="Multiple violations found",
    )
    s = await env.step(a)
    print(f"[med]  STEP  OK  | reward={s.reward}, done={s.done}")

    # --- Hard ---
    r = await env.reset("hard")
    print(f"\n[hard] RESET OK  | article_ref={r.observation.article_ref}")
    a = Action(
        violation_ids=["ART6-LAWFUL-BASIS"],
        severity="high",
        explanation="Lawful basis was fixed in v2",
        fix_suggestion="The policy should explicitly state the specific lawful basis under Article 6 for each processing activity to achieve full compliance.",
    )
    s = await env.step(a)
    print(f"[hard] STEP  OK  | reward={s.reward}, done={s.done}")

    # --- Cleanup ---
    await env.close()
    print("\nCLOSE OK — all tasks passed!")


asyncio.run(test_all())
