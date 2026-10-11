"""Promotion gate: replays a candidate rule against observed evidence.

The gate computes every number itself from stored episodes. It never accepts a score
from the proposer. Thresholds are pinned here, not in anything the watcher can write.

Denial rule:    fixes   = denied claims (right CARC) no active denial rule already covers
                breaks  = any paid claim it fires on, or a denial it would explain with the wrong CARC
Exception rule: fixes   = paid claims the scrubber wrongly held that no active exception covers
                breaks  = any denied claim it fires on (it would let a real denial through)
Replacement:    consolidation swaps active rules for one more general rule; it must keep every
                claim they explained and contradict nothing (see `evaluate_replacement`)

Min evidence:   a denial rule is only as trustworthy as the paid counterexamples it was replayed
                against, so it needs MIN_PAYER_PAID_SEEN distinct paid claims from its payer
                (from any payer for a "*" rule). Counted by claim_id, so re-scrubbing the same
                claims every iteration cannot inflate it.
"""
from __future__ import annotations

import json

from harness.rules import RuleError, apply_rules, fires, outcome_of, validate

# Pinned promotion policy (set before looking at held-out results).
MIN_FIXES = 1          # candidate must correct at least one observed case
MAX_REGRESSIONS = 0    # and must not contradict any observed outcome
MIN_PAYER_PAID_SEEN = 2  # denial rule needs this many distinct paid claims from its payer replayed
INSUFFICIENT = "insufficient evidence"  # reason prefix; a revision cannot fix it, only more claims can


def paid_seen(payer: str, evidence: list[dict]) -> int:
    """Distinct paid claim_ids from `payer` ("*" = any payer) in the replayed evidence."""
    return len({ev["claim_id"] for ev in evidence if ev["observed"]["outcome"] == "paid"
                and payer in ("*", ev.get("payer") or (ev.get("features") or {}).get("payer"))})


def _insufficient(payer: str, seen: int) -> str:
    return (f"{INSUFFICIENT}: only {seen} distinct paid {payer} claim(s) replayed, "
            f"MIN_PAYER_PAID_SEEN={MIN_PAYER_PAID_SEEN} (pinned in harness/gate.py); retry when more arrive")


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
    elif kind == "denied" and (seen := paid_seen(candidate["payer"], evidence)) < MIN_PAYER_PAID_SEEN:
        verdict, reason = "rejected", _insufficient(candidate["payer"], seen)
    else:
        verdict, reason = "promote", f"corrects {len(fixes)} observed case(s), 0 regressions"
    return {"verdict": verdict, "kind": kind, "stage": "replay", "reason": reason, "fixes": fixes,
            "regressions": regressions, "redundant": redundant,
            "thresholds": {"min_fixes": MIN_FIXES, "max_regressions": MAX_REGRESSIONS,
                           "min_payer_paid_seen": MIN_PAYER_PAID_SEEN},
            "evidence_size": len(evidence)}


LIST_OPS = {"in", "not_in", "contains_any", "contains_none", "startswith_any", "none_startswith"}


def _key(x) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def _norm_cond(cond: dict):
    """Canonical form of one condition: line groups sorted and deduped, list values sorted and
    deduped, and a one-element `in` written as `eq`. Two conditions that always agree on every
    claim because of ordering or duplication normalise to the same value."""
    for group in ("any_line", "all_lines"):
        if group in cond:
            return {group: sorted({_key(_norm_cond(c)) for c in cond[group]})}
    op, value = cond.get("op"), cond.get("value")
    if op in LIST_OPS and isinstance(value, (list, tuple)):
        value = [json.loads(k) for k in sorted({_key(v) for v in value})]
        if op == "in" and len(value) == 1:
            op, value = "eq", value[0]
    return {"field": cond.get("field"), "op": op, "value": value}


def _repeats(conds: list) -> bool:
    """True if a rule's `when` (or one of its line groups) lists the same condition twice."""
    keys = [_key(_norm_cond(c)) for c in conds]
    if len(keys) != len(set(keys)):
        return True
    return any(_repeats(c.get("any_line") or c.get("all_lines")) for c in conds
               if c.get("any_line") or c.get("all_lines"))


def _canon(rule: dict) -> tuple:
    """Identity of a rule's scope + conditions + prediction, insensitive to ordering anywhere."""
    return (rule.get("payer"), sorted({_key(_norm_cond(c)) for c in rule.get("when") or []}),
            _key(rule.get("predict") or {}))


def _correct(rule: dict, ev: dict, kind: str) -> bool:
    """Does this single rule fire on the claim and agree with the observed outcome (and CARC)?"""
    if not fires(rule, ev["features"]):
        return False
    if kind == "denied":
        return ev["observed"]["outcome"] == "denied" and rule["predict"].get("carc") in ev["observed"]["carcs"]
    return ev["observed"]["outcome"] == "paid"


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

    Promote only if
      (a) the new rule itself correctly fires on every claim any replaced rule correctly fired on
          (no narrowing), and nothing the whole active set explained is lost after the swap,
      (b) the new rule contradicts zero observed outcomes, and
      (c) it replaces >= 1 active rule of its own kind and payer (never a "*" rule), and is not
          condition-identical to any active rule nor repeats a condition.
    Equal observed coverage is fine: a broader rule that fires on the same training claims is
    exactly what consolidation is for. Losing coverage never is.
    """
    replaced_rules = list({r.get("rule_id"): r for r in replaced_rules}.values())  # dedupe, keep order
    base = {"stage": "replacement", "replaces": [r.get("rule_id") for r in replaced_rules],
            "fixes": [], "regressions": [], "lost": [], "narrowed": [], "preserved": [],
            "thresholds": {"max_regressions": MAX_REGRESSIONS, "max_lost": 0, "min_fixes": MIN_FIXES,
                           "min_payer_paid_seen": MIN_PAYER_PAID_SEEN},
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
    elif any(r.get("payer") == "*" for r in replaced_rules):
        reason = "wildcard (\"*\") rules are not consolidated"
    elif any(r.get("payer") != new_rule["payer"] for r in replaced_rules):
        reason = f"replaced rules must all belong to payer {new_rule['payer']!r} (no cross-payer replacement)"
    elif any(outcome_of(r) != kind for r in replaced_rules):
        reason = "replaced rules must predict the same outcome kind as the replacement"
    elif _repeats(new_rule["when"]):
        reason = "rule repeats a condition"
    elif any(_canon(new_rule) == _canon(r) for r in active_rules):
        reason = "condition-identical to an active rule"
    if reason:
        return {**base, "verdict": "rejected", "kind": kind, "reason": reason}

    after = [r for r in active_rules if r["rule_id"] not in replaced_ids] + [new_rule]
    fixes, regressions, lost, narrowed, preserved = [], [], [], [], []
    for ev in evidence:
        obs = ev["observed"]
        if fires(new_rule, ev["features"]):
            if kind == "denied" and (obs["outcome"] == "paid" or new_rule["predict"]["carc"] not in obs["carcs"]):
                regressions.append(ev["claim_id"])
            elif kind == "paid" and obs["outcome"] == "denied":
                regressions.append(ev["claim_id"])
            else:
                fixes.append(ev["claim_id"])
        if any(_correct(r, ev, kind) for r in replaced_rules):
            (preserved if _correct(new_rule, ev, kind) else narrowed).append(ev["claim_id"])
        if _explained(active_rules, ev, kind) and not _explained(after, ev, kind):
            lost.append(ev["claim_id"])

    if len(regressions) > MAX_REGRESSIONS:
        verdict, reason = "rejected", (f"replacement contradicts {len(regressions)} observed outcome(s): "
                                       + ", ".join(regressions))
    elif narrowed:
        verdict, reason = "rejected", (f"replacement no longer fires on {len(narrowed)} claim(s) the replaced "
                                       "rules explained: " + ", ".join(narrowed))
    elif lost:
        verdict, reason = "rejected", (f"replacement loses {len(lost)} claim(s) the active rules explained: "
                                       + ", ".join(lost))
    elif len(fixes) < MIN_FIXES:
        verdict, reason = "rejected", "replacement explains no observed claim"
    elif kind == "denied" and (seen := paid_seen(new_rule["payer"], evidence)) < MIN_PAYER_PAID_SEEN:
        verdict, reason = "rejected", _insufficient(new_rule["payer"], seen)
    else:
        verdict, reason = "promote", (f"replaces {len(replaced_rules)} rule(s), fires correctly on all "
                                      f"{len(preserved)} claim(s) they explained, 0 regressions")
    return {**base, "verdict": verdict, "kind": kind, "reason": reason, "fixes": fixes,
            "regressions": regressions, "lost": lost, "narrowed": narrowed, "preserved": preserved}
