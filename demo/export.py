"""Export a run from Atlas into demo/run.json for the demo page.

    python -m demo.export --run-id run1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory.atlas_schema import get_db


def export(db, run_id: str) -> dict:
    run = db.runs.find_one({"_id": run_id}) or {}
    history = run.get("history", [])
    curve = [{"iteration": h["iteration"], "ruleset_version": h["metrics"]["ruleset_version"],
              "rule_count": h["metrics"]["rule_count"], "heldout": h["metrics"]["heldout"],
              "train": h["metrics"]["train"]} for h in history]
    rules = []
    for r in db.rules.find({}).sort("created_at", 1):
        rules.append({"rule_id": r["_id"], "payer": r["payer"], "description": r.get("description"),
                      "status": r["status"], "iteration": r.get("proposed_in_iteration"),
                      "version": r.get("promoted_version"), "carc": r["predict"]["carc"],
                      "when": r["when"], "evidence": r.get("evidence_episode_ids", []),
                      "rejection_reason": r.get("rejection_reason")})
    learning = [{"iteration": h["iteration"], **{k: v for k, v in h.get("learning", {}).items()
                                                 if k in ("misses", "proposed", "promoted", "rejected", "revised")}}
                for h in history if "learning" in h]
    return {"run_id": run_id, "scrubber_model": history[0].get("scrubber_model") if history else None,
            "watcher_model": history[0].get("watcher_model") if history else None,
            "curve": curve, "rules": rules, "learning": learning,
            "counts": {c: db[c].count_documents({}) for c in
                       ("episodes", "rules", "rulesets", "evaluations", "audits")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="run1")
    ap.add_argument("--out", default="demo/run.json")
    args = ap.parse_args()
    data = export(get_db(), args.run_id)
    Path(args.out).write_text(json.dumps(data, indent=2, default=str))
    print(f"wrote {args.out}: {len(data['curve'])} iterations, {len(data['rules'])} rules")


if __name__ == "__main__":
    main()
