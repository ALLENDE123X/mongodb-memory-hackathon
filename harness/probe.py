"""Stage probe: push a deliberately over-broad rule through the real gate.

    python -m harness.probe --run-id run2

The candidate is recorded in Atlas exactly like a watcher proposal (rule, evaluation,
audit), so the rejection is real, not a slide.
"""
from __future__ import annotations

import argparse
import json

from harness.gate import evaluate
from harness.loop import latest_train_evidence
from memory.atlas_schema import get_db
from memory.store import Store

OVERBROAD = {
    "payer": "NORTHSTAR",
    "description": "NORTHSTAR denies all advanced imaging (MRI/CT 70551, 72148, 74177)",
    "when": [{"any_line": [{"field": "cpt", "op": "in", "value": ["70551", "72148", "74177"]}]}],
    "predict": {"outcome": "denied", "carc": "197"},
}


def run_probe(store: Store, run_id: str) -> dict:
    """Replay OVERBROAD through the real gate and record it. Writes exactly one rule doc
    (status rejected), one evaluation doc and one audit doc; the head never moves."""
    head, active = store.active()
    evidence = latest_train_evidence(store, run_id)
    rule = {**OVERBROAD, "rule_id": f"PROBE-{store.db.rules.count_documents({}) + 1:03d}"}
    verdict = evaluate(rule, active, evidence)
    ev_id = store.insert_evaluation({"run_id": run_id, "rule_id": rule["rule_id"], "probe": True,
                                     "baseline_version": head["version"], **verdict})
    store.insert_rule(rule, status="candidate", iteration=-1, evidence=[])
    if verdict["verdict"] == "promote":  # should never happen; refuse to promote a probe anyway
        store.set_rule_status(rule["rule_id"], "parked", rejection_reason="probe rules are never promoted")
    else:
        store.set_rule_status(rule["rule_id"], "rejected", rejection_reason=verdict["reason"],
                              regressions=verdict["regressions"])
    store.audit("rejection", run_id=run_id, rule_id=rule["rule_id"], reason=verdict["reason"],
                description=rule["description"], probe=True, evaluation_id=ev_id)
    return {"rule": rule, "verdict": verdict, "head": head, "evaluation_id": ev_id}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="run2")
    args = ap.parse_args()
    store = Store(get_db())
    out = run_probe(store, args.run_id)
    rule, verdict, head = out["rule"], out["verdict"], out["head"]
    print(f"proposed : {rule['description']}")
    print(f"replayed : {verdict['evidence_size']} observed claims, ruleset v{head['version']}")
    print(f"verdict  : {verdict['verdict'].upper()}  {verdict['reason']}")
    print(f"active ruleset still v{store.active()[0]['version']}")
    print(json.dumps({k: verdict[k] for k in ("fixes", "regressions", "redundant")}))


if __name__ == "__main__":
    main()
