"""Promotion gate: replays a candidate rule against observed evidence.

The gate computes every number itself from stored episodes. It never accepts a score
from the proposer. Thresholds are pinned here, not in anything the watcher can write.
"""
from __future__ import annotations

from harness.rules import RuleError, apply_rules, fires, validate

# Pinned promotion policy (set before looking at held-out results).
MIN_FIXES = 1          # candidate must turn at least one observed miss into a correct call
MAX_REGRESSIONS = 0    # and must not fire on any claim the payer actually paid


def evaluate(candidate: dict, active_rules: list[dict], evidence: list[dict]) -> dict:
    """evidence: [{'claim_id', 'features', 'observed': {'outcome', 'carcs'}}], one per claim
    (latest observation), drawn only from the training stream."""
    try:
        validate(candidate)
    except RuleError as e:
        return {"verdict": "rejected", "stage": "structure", "reason": str(e),
                "fixes": [], "regressions": [], "redundant": []}

    fixes, regressions, redundant = [], [], []
    for ev in evidence:
        if not fires(candidate, ev["features"]):
            continue
        obs = ev["observed"]
        if obs["outcome"] == "paid":
            regressions.append(ev["claim_id"])
            continue
        current = apply_rules(active_rules, ev["features"])
        right_code = candidate["predict"]["carc"] in obs["carcs"]
        if current and current["carc"] in obs["carcs"]:
            redundant.append(ev["claim_id"])
        elif right_code:
            fixes.append(ev["claim_id"])
        else:
            regressions.append(ev["claim_id"])  # fires on a denial but predicts the wrong reason

    if len(regressions) > MAX_REGRESSIONS:
        verdict, reason = "rejected", (
            f"would wrongly deny {len(regressions)} claim(s) the payer paid or denied for a different reason: "
            + ", ".join(regressions))
    elif len(fixes) < MIN_FIXES:
        verdict, reason = "rejected", "fixes no observed miss" + (" (already covered)" if redundant else "")
    else:
        verdict, reason = "promote", f"fixes {len(fixes)} observed miss(es), 0 regressions"
    return {"verdict": verdict, "stage": "replay", "reason": reason, "fixes": fixes,
            "regressions": regressions, "redundant": redundant,
            "thresholds": {"min_fixes": MIN_FIXES, "max_regressions": MAX_REGRESSIONS},
            "evidence_size": len(evidence)}
