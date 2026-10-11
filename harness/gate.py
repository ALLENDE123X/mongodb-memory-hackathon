"""Promotion gate: replays a candidate rule against observed evidence.

The gate computes every number itself from stored episodes. It never accepts a score
from the proposer. Thresholds are pinned here, not in anything the watcher can write.

Denial rule:    fixes   = denied claims (right CARC) no active denial rule already covers
                breaks  = any paid claim it fires on, or a denial it would explain with the wrong CARC
Exception rule: fixes   = paid claims the scrubber wrongly held that no active exception covers
                breaks  = any denied claim it fires on (it would let a real denial through)
"""
from __future__ import annotations

from harness.rules import RuleError, apply_rules, fires, outcome_of, validate

# Pinned promotion policy (set before looking at held-out results).
MIN_FIXES = 1          # candidate must correct at least one observed case
MAX_REGRESSIONS = 0    # and must not contradict any observed outcome


def evaluate(candidate: dict, active_rules: list[dict], evidence: list[dict]) -> dict:
    """evidence: [{'claim_id', 'features', 'observed': {'outcome', 'carcs'}, 'predicted'?}],
    one per claim (latest observation), drawn only from the training stream."""
    try:
        validate(candidate)
    except RuleError as e:
        return {"verdict": "rejected", "stage": "structure", "reason": str(e),
                "fixes": [], "regressions": [], "redundant": []}

    kind = outcome_of(candidate)
    fixes, regressions, redundant = [], [], []
    for ev in evidence:
        if not fires(candidate, ev["features"]):
            continue
        obs = ev["observed"]
        if kind == "denied":
            if obs["outcome"] == "paid":
                regressions.append(ev["claim_id"])
                continue
            current = apply_rules(active_rules, ev["features"], "denied")
            if current and current["carc"] in obs["carcs"]:
                redundant.append(ev["claim_id"])
            elif candidate["predict"]["carc"] in obs["carcs"]:
                fixes.append(ev["claim_id"])
            else:
                regressions.append(ev["claim_id"])  # right claim, wrong reason
        else:
            if obs["outcome"] == "denied":
                regressions.append(ev["claim_id"])
                continue
            held = (ev.get("predicted") or {}).get("predicted_outcome") == "denied"
            if apply_rules(active_rules, ev["features"], "paid") or not held:
                redundant.append(ev["claim_id"])
            else:
                fixes.append(ev["claim_id"])

    noun = "wrongly deny" if kind == "denied" else "wave through"
    if len(regressions) > MAX_REGRESSIONS:
        verdict, reason = "rejected", (
            f"would {noun} {len(regressions)} claim(s) whose real outcome contradicts it: " + ", ".join(regressions))
    elif len(fixes) < MIN_FIXES:
        verdict, reason = "rejected", "corrects no observed case" + (" (already covered)" if redundant else "")
    else:
        verdict, reason = "promote", f"corrects {len(fixes)} observed case(s), 0 regressions"
    return {"verdict": verdict, "kind": kind, "stage": "replay", "reason": reason, "fixes": fixes,
            "regressions": regressions, "redundant": redundant,
            "thresholds": {"min_fixes": MIN_FIXES, "max_regressions": MAX_REGRESSIONS},
            "evidence_size": len(evidence)}
