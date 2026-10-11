"""Watcher: turns the scrubber's misses into candidate payer rules.

It reads observed denials (episodic memory), the active rules, and every rule the gate
previously rejected with the reason (its own lessons). It never sees the simulator's rule
table or held-out claims. It proposes; only the gate can promote.
"""
from __future__ import annotations

import json
import os

from agents.llm import complete_json
from harness.payer_sim import CARC_TEXT

SYSTEM = """You improve a medical claim scrubber by writing executable payer rules.
Every training claim was really submitted and the payer's remittance is attached. You get two
kinds of cases for one payer:

1. "uncovered_denials": the payer DENIED these and no learned rule explains them yet (the
   scrubber may have guessed some of them right, but a guess is not knowledge). From the
   denial codes (CARC/RARC) and the claim details, infer the payer's hidden policy and write a
   DENIAL rule:   "predict": {"outcome": "denied", "carc": "<code>"}
2. "false_holds": the payer PAID these, but the scrubber wrongly predicted a denial (see
   scrubber_said). Infer what this payer does NOT enforce and write an EXCEPTION rule:
   "predict": {"outcome": "paid"}   e.g. "this payer does not enforce its usual edit on this service class".

Write ONE rule per variant of each distinct cause, in this JSON rule language:
{"rule_id": "", "payer": "<PAYER>" or "*",
 "description": "<one sentence policy>",
 "when": [<condition>, ...],            # ALL must hold
 "predict": {...}}

Claim-level condition: {"field": F, "op": OP, "value": V}
  F in: payer, pos, claim_filing_code, days_to_submit, prior_auth, dx_codes, in_network,
        line_count, total_charge, cpts, days_since_last_preventive, context.<key>
Line-level condition (true if ANY single line matches all): {"any_line": [{"field": F, "op": OP, "value": V}, ...]}
  line F in: cpt, modifiers, units, charge, dx_pointer_count, dx_codes (diagnoses this line points to)
OP: eq, ne, in, not_in, gt, gte, lt, lte, missing, present, contains, not_contains,
    contains_any, contains_none, startswith_any, none_startswith, between ([lo, hi] numeric)
Diagnosis codes have no decimal point (E559, not E55.9). Dates are not allowed as values.

A replay gate checks every rule against every claim seen so far. A denial rule is promoted
only if it explains at least one uncovered denial with the right CARC and fires on ZERO paid
claims. An exception rule is promoted only if it fixes at least one false hold and fires on
ZERO denied claims.

Write the MOST GENERAL rule that contradicts no observed claim. Rules are scored on claims
you have never seen, and a rule that names one example code will miss its siblings. So:
- Prefer code ranges (`between`, `startswith_any`) and code families over single codes: if two
  denied codes are 99213 and 99214, write the E/M family range 99202-99215, not either code.
- Describe the CLASS of procedure, not the example you were shown, rather than naming the
  second code of one observed pair. Express a code pair as two any_line conditions (or
  claim-level cpts with contains / contains_any); use between on line cpt, not on cpts.
- Drop every condition the evidence does not force. Keep the payer; add a narrower condition
  (a specific code, a modifier, a place of service) only when a paid counterexample in
  "paid_claims" (or a denied one in "denied_claims", for exceptions) would otherwise match.
- For numeric fields use the thresholds in "threshold_evidence": a cutoff must sit above the
  largest value on a paid claim and at or below the smallest value among denials WITH THE SAME
  CARC ("min_on_denied_by_carc"); denials with other CARCs have other causes, so ignore them.
- Cover every variant of the same cause (for example every place of service or modifier
  pattern that triggers the same denial) with one rule per variant: "when" is AND-only, so
  variants of one cause are separate rules sharing the description stem and the CARC.
Do not memorise individual claims. Learn from previously rejected rules and their reasons; do
not repeat them. When told a rule was too broad, narrow it only as far as the named claims
require, keeping everything else general.

Respond with JSON only: {"rules": [<rule>, ...], "notes": "<short reasoning>"}"""


def claim_brief(ev: dict) -> dict:
    f = ev["features"]
    return {"claim_id": ev["claim_id"], "payer": f["payer"], "pos": f["pos"], "prior_auth": f["prior_auth"],
            "days_to_submit": f["days_to_submit"], "in_network": f["in_network"], "dx_codes": f["dx_codes"],
            "days_since_last_preventive": f["days_since_last_preventive"],
            "lines": [{k: ln[k] for k in ("cpt", "modifiers", "units", "charge", "dx_codes")} for ln in f["lines"]],
            "context": {k: v for k, v in f["context"].items() if k != "submission_date"},
            "observed": {"outcome": ev["observed"]["outcome"],
                         "carcs": [f"{c}: {CARC_TEXT.get(c, '')}" for c in ev["observed"]["carcs"]],
                         "rarcs": ev["observed"].get("rarcs", [])},
            "scrubber_said": ev.get("predicted")}


THRESHOLD_FIELDS = ("days_to_submit", "days_since_last_preventive", "context.visit_count_ytd")


def _numeric(ev: dict, field: str):
    f = ev["features"]
    v = f["context"].get(field.split(".", 1)[1]) if field.startswith("context.") else f.get(field)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def threshold_evidence(paid: list[dict], denied: list[dict]) -> dict:
    """Per numeric field seen in evidence: the max on a paid claim, and the min on a denied
    claim per CARC.

    A cutoff rule for one CARC is only safe above the paid max and at or below the smallest
    value among denials with that same CARC. Denials with other CARCs have other causes, so
    pooling them would pull the bound below the paid max and make every cutoff look impossible.
    """
    out = {}
    for field in THRESHOLD_FIELDS:
        paid_vals = [v for v in (_numeric(e, field) for e in paid) if v is not None]
        by_carc: dict[str, float] = {}
        for e in denied:
            v = _numeric(e, field)
            if v is None:
                continue
            for carc in e["observed"]["carcs"]:
                by_carc[carc] = min(v, by_carc.get(carc, v))
        if paid_vals or by_carc:
            out[field] = {"max_on_paid_claim": max(paid_vals, default=None),
                          "min_on_denied_by_carc": dict(sorted(by_carc.items()))}
    return out


def propose(uncovered: list[dict], false_holds: list[dict], *, paid_examples: list[dict],
            denied_examples: list[dict], active_rules: list[dict], lessons: list[dict],
            model: str | None = None, feedback: str | None = None) -> tuple[list[dict], dict]:
    model = model or os.environ.get("WATCHER_MODEL", "")
    payload = {
        "uncovered_denials": [claim_brief(m) for m in uncovered],
        "false_holds": [claim_brief(m) for m in false_holds],
        "paid_claims": [claim_brief(p) for p in paid_examples],
        "denied_claims": [claim_brief(d) for d in denied_examples],
        "threshold_evidence": threshold_evidence(paid_examples, denied_examples),
        "active_rules": [{k: r[k] for k in ("payer", "description", "when", "predict")} for r in active_rules],
        "previously_rejected_rules": [{"description": r["description"], "when": r["when"], "predict": r["predict"],
                                       "rejected_because": r.get("rejection_reason")} for r in lessons],
    }
    user = json.dumps(payload, indent=1, default=str)
    if feedback:
        user += f"\n\nYour previous proposal was rejected by the replay gate: {feedback}\nRevise it."
    out, meta = complete_json(model, SYSTEM, user, max_tokens=4000)
    return out.get("rules", []) or [], {**meta, "notes": out.get("notes")}
