"""Export a run from Atlas into demo/run.json for the demo page.

    python -m demo.export --run-id run1

Read-only against a db handle. Fields written by later tickets (consolidation, semantic_kb,
taxonomy, raindrop) are read defensively: absent means the page hides or says "not enabled".
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Memory type -> Atlas collections that hold it (the page shows "0 / not enabled" when empty).
MEMORY_TYPES = {
    "semantic": ("semantic_kb",),
    "episodic": ("episodes",),
    "procedural": ("rules", "rulesets"),
    "taxonomic": ("taxonomy",),
}
LEDGER_KINDS = ("proposal", "promotion", "rejection", "consolidation", "probe")
_KNOWN = {"_id", "kind", "created_at", "run_id", "iteration", "payer", "rule_id", "description", "rule_kind",
          "reason", "fixes", "regressions", "version", "probe", "attempt", "n_rules", "notes", "model"}


def label_for(version, rule_count) -> str:
    """'no learned rules' only when the ruleset really is empty (a resumed run may start with rules)."""
    return "no learned rules" if not rule_count else f"ruleset v{version}"


def _lst(x) -> list:
    """Stored fixes/regressions can be None on partial documents; treat as empty."""
    return x if isinstance(x, list) else []


def _claim_ids(episode_ids) -> list[str]:
    return [e.rsplit(":", 1)[-1] for e in (episode_ids or [])]


def _rule_kind(rule: dict) -> str:
    return "exception" if (rule.get("predict") or {}).get("outcome") == "paid" else "denial"


def _in_run(doc: dict, run_id: str) -> bool:
    return doc.get("run_id") in (None, run_id)


def export(db, run_id: str) -> dict:
    run = db.runs.find_one({"_id": run_id}) or {}
    history = run.get("history", [])
    curve = [{"iteration": h["iteration"], "ruleset_version": h["metrics"]["ruleset_version"],
              "rule_count": h["metrics"]["rule_count"],
              "label": label_for(h["metrics"]["ruleset_version"], h["metrics"]["rule_count"]),
              "heldout": h["metrics"]["heldout"], "train": h["metrics"]["train"]} for h in history]
    evals = [e for e in db.evaluations.find({}).sort("created_at", 1) if _in_run(e, run_id)]
    eval_by_rule = {e["rule_id"]: e for e in evals}

    rules = []
    for r in db.rules.find({}).sort("created_at", 1):
        ev = eval_by_rule.get(r["_id"], {})
        rules.append({"rule_id": r["_id"], "payer": r["payer"], "description": r.get("description"),
                      "status": r["status"], "iteration": r.get("proposed_in_iteration"),
                      "version": r.get("promoted_version"), "kind": _rule_kind(r),
                      "carc": (r.get("predict") or {}).get("carc"),  # exception rules have no carc
                      "when": r["when"], "evidence": r.get("evidence_episode_ids", []),
                      "learned_from": _claim_ids(r.get("evidence_episode_ids")),
                      "fixes": _lst(ev.get("fixes")), "regressions": _lst(r.get("regressions")) or _lst(ev.get("regressions")),
                      "rejection_reason": r.get("rejection_reason"),
                      "superseded_by": r.get("superseded_by"),
                      "probe": bool(ev.get("probe")) or r.get("proposed_in_iteration") == -1})

    gate = [{"rule_id": e["rule_id"], "iteration": e.get("iteration"), "verdict": e.get("verdict"),
             "kind": e.get("kind"), "reason": e.get("reason"), "stage": e.get("stage"),
             "fixes": len(_lst(e.get("fixes"))), "regressions": len(_lst(e.get("regressions"))),
             "redundant": len(_lst(e.get("redundant"))), "evidence_size": e.get("evidence_size"),
             "baseline_version": e.get("baseline_version"), "probe": bool(e.get("probe"))} for e in evals]

    rule_by_id = {r["rule_id"]: r for r in rules}
    ledger = []
    for a in db.audits.find({}).sort("created_at", 1):
        if not _in_run(a, run_id):
            continue
        kind = "probe" if a.get("probe") else a["kind"]
        rid = a.get("rule_id")
        rule = rule_by_id.get(rid, {})
        ev = eval_by_rule.get(rid, {})
        ledger.append({"kind": kind, "created_at": a.get("created_at"), "iteration": a.get("iteration"),
                       "payer": a.get("payer") or rule.get("payer"), "rule_id": rid,
                       "description": a.get("description") or rule.get("description"),
                       "rule_kind": a.get("rule_kind") or rule.get("kind"),
                       "reason": a.get("reason") or a.get("notes"),
                       "fixes": len(_lst(a.get("fixes")) or _lst(ev.get("fixes"))),
                       "regressions": len(_lst(ev.get("regressions"))), "version": a.get("version"),
                       "extra": {k: v for k, v in a.items() if k not in _KNOWN}})

    learning = []
    for h in history:
        if "learning" not in h:
            continue
        s = h["learning"]
        proposed, promoted = s.get("proposed", 0), s.get("promoted", 0)
        learning.append({"iteration": h["iteration"],
                         **{k: s.get(k) for k in ("uncovered_denials", "false_holds", "misses", "proposed",
                                                  "promoted", "rejected", "revised")},
                         "n": proposed, "acceptance_rate": round(promoted / proposed, 3) if proposed else None,
                         "n_train": (h["metrics"].get("train") or {}).get("n")})

    counts = {c: db[c].count_documents({}) for c in
              ("episodes", "rules", "rulesets", "evaluations", "audits", "semantic_kb", "taxonomy")}
    memory_counts = {t: [{"collection": c, "count": counts.get(c, db[c].count_documents({}))} for c in cols]
                     for t, cols in MEMORY_TYPES.items()}
    out = {"run_id": run_id, "scrubber_model": history[0].get("scrubber_model") if history else None,
           "watcher_model": history[0].get("watcher_model") if history else None,
           "baseline": curve[0] if curve else None, "curve": curve, "rules": rules, "learning": learning,
           "rsi": learning, "gate": gate, "ledger": ledger, "memory_counts": memory_counts, "counts": counts}
    tel = run.get("telemetry") or {}  # JOB-14 telemetry: dashboard link and Signals sent
    out["raindrop"] = {"dashboard_url": os.environ.get("RAINDROP_DASHBOARD_URL"),
                       "events_sent": tel.get("events_sent", {}), "total": tel.get("total", 0)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="run1")
    ap.add_argument("--out", default="demo/run.json")
    args = ap.parse_args()
    from memory.atlas_schema import get_db  # imported here so tests never need a connection string
    data = export(get_db(), args.run_id)
    Path(args.out).write_text(json.dumps(data, indent=2, default=str))
    print(f"wrote {args.out}: {len(data['curve'])} iterations, {len(data['rules'])} rules")


if __name__ == "__main__":
    main()
