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
You are given claims the scrubber got WRONG: the payer denied them but the scrubber predicted
"paid" (or the wrong reason). From the denial codes and the claim details, infer the payer's
hidden policy and write ONE precise rule per distinct cause, in this JSON rule language:

{"rule_id": "", "payer": "<PAYER>" or "*",
 "description": "<one sentence policy, e.g. 'NORTHSTAR requires prior auth for MRI/CT 70551, 72148, 74177'>",
 "when": [<condition>, ...],            # ALL must hold
 "predict": {"outcome": "denied", "carc": "<code>"}}

Claim-level condition: {"field": F, "op": OP, "value": V}
  F in: payer, pos, claim_filing_code, days_to_submit, prior_auth, dx_codes, in_network,
        line_count, total_charge, cpts, days_since_last_preventive, context.<key>
Line-level condition (true if ANY single line matches all): {"any_line": [{"field": F, "op": OP, "value": V}, ...]}
  line F in: cpt, modifiers, units, charge, dx_pointer_count, dx_codes (diagnoses this line points to)
OP: eq, ne, in, not_in, gt, gte, lt, lte, missing, present, contains, not_contains,
    contains_any, contains_none, startswith_any, none_startswith, between ([lo, hi] numeric)
Diagnosis codes have no decimal point (E559, not E55.9). Dates are not allowed as values.

A rule is promoted only if, replayed on every claim seen so far, it fixes at least one miss
and fires on ZERO claims the payer paid. So make rules as narrow as the evidence requires:
use the payer, the specific codes, and the distinguishing condition (missing auth, modifier,
threshold, place of service). Do not memorise individual claims.
Use the paid claims you are shown as counter-examples: your rule must not match them.
Learn from previously rejected rules and their reasons; do not repeat their mistakes.

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


def propose(misses: list[dict], *, paid_examples: list[dict], active_rules: list[dict],
            lessons: list[dict], model: str | None = None, feedback: str | None = None) -> tuple[list[dict], dict]:
    model = model or os.environ.get("WATCHER_MODEL", "")
    payload = {
        "missed_denials": [claim_brief(m) for m in misses],
        "paid_claims_same_payer": [claim_brief(p) for p in paid_examples],
        "active_rules": [{k: r[k] for k in ("payer", "description", "when", "predict")} for r in active_rules],
        "previously_rejected_rules": [{"description": r["description"], "when": r["when"],
                                       "rejected_because": r.get("rejection_reason")} for r in lessons],
    }
    user = json.dumps(payload, indent=1, default=str)
    if feedback:
        user += f"\n\nYour previous proposal was rejected by the replay gate: {feedback}\nRevise it."
    out, meta = complete_json(model, SYSTEM, user, max_tokens=3000)
    return out.get("rules", []) or [], {**meta, "notes": out.get("notes")}
