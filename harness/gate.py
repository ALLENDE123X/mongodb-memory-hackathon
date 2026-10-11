"""Promotion gate: replays a candidate rule against observed evidence.

The gate computes every number itself from stored episodes. It never accepts a score
from the proposer. Thresholds are pinned here, not in anything the watcher can write.

Denial rule:    fixes   = denied claims (right CARC) no active denial rule already covers
                breaks  = any paid claim it fires on, or a denial it would explain with the wrong CARC
Exception rule: fixes   = paid claims the scrubber wrongly held that no active exception covers
                breaks  = any denied claim it fires on (it would let a real denial through)
Replacement:    consolidation swaps active rules for one more general rule; it must keep every
                claim they explained and contradict nothing (see `evaluate_replacement`)
"""
from __future__ import annotations

import json

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


def _canon(rule: dict) -> tuple:
    """Order-insensitive identity of a rule's scope + conditions + prediction."""
    return (rule.get("payer"), sorted(json.dumps(c, sort_keys=True) for c in rule.get("when", [])),
            json.dumps(rule.get("predict", {}), sort_keys=True))


def _explained(rules: list[dict], ev: dict, kind: str) -> bool:
    """Does this ruleset get this claim right for its kind? Denial: the winning rule names an
    observed CARC. Exception: some exception rule covers the paid claim."""
    hit = apply_rules(rules, ev["features"], kind)
    if kind == "denied":
        return ev["observed"]["outcome"] == "denied" and bool(hit) and hit["carc"] in ev["observed"]["carcs"]
    return ev["observed"]["outcome"] == "paid" and hit is not None


def evaluate_replacement(new_rule: dict, replaced_rules: list[dict], active_rules: list[dict],
                         evidence: list[dict]) -> dict:
    """Consolidation gate: may `new_rule` take the place of `replaced_rules` in the active set?

    Promote only if (a) every claim the active set explained before is still explained after the
    swap (so nothing the replaced rules fixed is lost), (b) the new rule contradicts zero observed
    outcomes, and (c) it replaces >= 1 active rule of its own kind and is not condition-identical
    to any active rule. Equal observed coverage is fine: a broader rule that fires on the same
    training claims is exactly what consolidation is for. Losing coverage never is.
    """
    base = {"stage": "replacement", "replaces": [r.get("rule_id") for r in replaced_rules],
            "fixes": [], "regressions": [], "lost": [], "preserved": [],
            "thresholds": {"max_regressions": MAX_REGRESSIONS, "max_lost": 0, "min_fixes": MIN_FIXES},
            "evidence_size": len(evidence)}
    try:
        validate(new_rule)
    except RuleError as e:
        return {**base, "verdict": "rejected", "stage": "structure", "reason": str(e)}
    kind = outcome_of(new_rule)
    active_ids = {r["rule_id"] for r in active_rules}
    replaced_ids = {r.get("rule_id") for r in replaced_rules}
    reason = None
    if not replaced_rules:
        reason = "replaces no rule"
    elif not replaced_ids <= active_ids:
        reason = "replaces a rule that is not active: " + ", ".join(sorted(map(str, replaced_ids - active_ids)))
    elif any(outcome_of(r) != kind for r in replaced_rules):
        reason = "replaced rules must predict the same outcome kind as the replacement"
    elif any(_canon(new_rule) == _canon(r) for r in active_rules):
        reason = "condition-identical to an active rule"
    if reason:
        return {**base, "verdict": "rejected", "kind": kind, "reason": reason}

    after = [r for r in active_rules if r["rule_id"] not in replaced_ids] + [new_rule]
    fixes, regressions, lost, preserved = [], [], [], []
    for ev in evidence:
        obs = ev["observed"]
        if fires(new_rule, ev["features"]):
            if kind == "denied" and (obs["outcome"] == "paid" or new_rule["predict"]["carc"] not in obs["carcs"]):
                regressions.append(ev["claim_id"])
            elif kind == "paid" and obs["outcome"] == "denied":
                regressions.append(ev["claim_id"])
            else:
                fixes.append(ev["claim_id"])
        if _explained(active_rules, ev, kind):
            if not _explained(after, ev, kind):
                lost.append(ev["claim_id"])
            elif any(fires(r, ev["features"]) for r in replaced_rules):
                preserved.append(ev["claim_id"])

    if len(regressions) > MAX_REGRESSIONS:
        verdict, reason = "rejected", (f"replacement contradicts {len(regressions)} observed outcome(s): "
                                       + ", ".join(regressions))
    elif lost:
        verdict, reason = "rejected", (f"replacement loses {len(lost)} claim(s) the replaced rules explained: "
                                       + ", ".join(lost))
    elif len(fixes) < MIN_FIXES:
        verdict, reason = "rejected", "replacement explains no observed claim"
    else:
        verdict, reason = "promote", (f"replaces {len(replaced_rules)} rule(s), keeps all {len(preserved)} "
                                      f"claim(s) they explained, 0 regressions")
    return {**base, "verdict": verdict, "kind": kind, "reason": reason, "fixes": fixes,
            "regressions": regressions, "lost": lost, "preserved": preserved}
