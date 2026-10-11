"""The recursive loop: scrub -> payer responds -> remember -> propose -> gate -> promote -> repeat.

    python -m harness.loop --run-id demo --iterations 3 --reset
    python -m harness.loop --run-id demo --iterations 3          # resume after a crash/restart

Iteration k scrubs every claim with ruleset v(k). Training claims are submitted to the
simulated payer and their remittances become learning evidence. Held-out claims are scored
the same way but their outcomes are never shown to the watcher or used for retrieval.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from agents.scrubber import scrub
from agents.watcher import propose
from eval.metrics import score
from eval.synthetic_claims import load_claims, public_view
from harness.features import features
from harness.gate import evaluate
from harness.payer_sim import adjudicate, outcome_summary
from harness.rules import apply_rules
from memory.store import ConflictError, Store

MAX_REVISIONS = 1


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# -- episodic memory helpers --------------------------------------------------------------

def latest_train_evidence(store: Store, run_id: str) -> list[dict]:
    """Latest observation per training claim: what the gate replays and the watcher reads."""
    latest: dict[str, dict] = {}
    for ep in store.episodes({"run_id": run_id, "split": "train"}):
        latest[ep["claim_id"]] = ep
    return list(latest.values())


def similar_past(view: dict, evidence: list[dict], k: int = 3) -> list[dict]:
    feats = features(view)
    cpts = set(feats["cpts"])
    scored = []
    for ep in evidence:
        if ep["claim_id"] == view["id"]:
            continue
        overlap = len(cpts & set(ep["cpts"]))
        same_payer = ep["payer"] == feats["payer"]
        if overlap or same_payer:
            scored.append((overlap * 2 + same_payer, ep))
    scored.sort(key=lambda t: -t[0])
    return [ep for _, ep in scored[:k]]


# -- one pass over the claims ------------------------------------------------------------

def scrub_pass(store: Store, run_id: str, iteration: int) -> dict:
    head, rules = store.active()
    memory = latest_train_evidence(store, run_id)
    rows = defaultdict(list)
    todo = []
    for split in ("train", "heldout"):
        for rec in load_claims(split):
            ep_id = f"{run_id}:{iteration}:{rec['id']}"
            existing = store.db.episodes.find_one({"_id": ep_id})
            if existing:  # idempotent resume: never re-submit a claim
                rows[split].append(existing)
            else:
                todo.append((split, rec, ep_id))

    def work(item):
        split, rec, ep_id = item
        view = public_view(rec)
        decision = scrub(view, rules=rules, similar=similar_past(view, memory))
        remit = adjudicate(rec)  # the payer, standing in for a clearinghouse round-trip
        observed = outcome_summary(remit)
        feats = features(view)
        return {"_id": ep_id, "run_id": run_id, "iteration": iteration, "split": split,
                "claim_id": rec["id"], "payer": rec["payer"], "cpts": feats["cpts"],
                "ruleset_version": head["version"], "predicted": decision, "observed": observed,
                "correct": decision["predicted_outcome"] == observed["outcome"]
                and (observed["outcome"] == "paid" or decision.get("carc") in observed["carcs"]),
                "features": feats, "remittance": remit, "visible_to_learning": split == "train"}

    with ThreadPoolExecutor(max_workers=int(os.environ.get("SCRUB_WORKERS", "8"))) as pool:
        for ep in pool.map(work, todo):
            store.append_episode(ep)
            rows[ep["split"]].append(ep)
    metrics = {split: score(eps) for split, eps in rows.items()}
    metrics["ruleset_version"] = head["version"]
    metrics["rule_count"] = len(rules)
    return metrics


# -- learning -----------------------------------------------------------------------------

def _uncovered_denial(active: list[dict], ep: dict) -> bool:
    """Denied, and no promoted denial rule explains it (an LLM guess is not knowledge)."""
    if ep["observed"]["outcome"] != "denied":
        return False
    hit = apply_rules(active, ep["features"], "denied")
    return not (hit and hit["carc"] in ep["observed"]["carcs"])


def _false_hold(active: list[dict], ep: dict) -> bool:
    """Paid, but the scrubber held it, and no exception rule covers it yet."""
    return (ep["observed"]["outcome"] == "paid" and ep["predicted"]["predicted_outcome"] == "denied"
            and not apply_rules(active, ep["features"], "paid"))


def learn(store: Store, run_id: str, iteration: int) -> dict:
    evidence = latest_train_evidence(store, run_id)
    _, active = store.active()
    uncovered = [e for e in evidence if _uncovered_denial(active, e)]
    holds = [e for e in evidence if _false_hold(active, e)]
    by_payer: dict[str, dict[str, list]] = defaultdict(lambda: {"uncovered": [], "holds": []})
    for e in uncovered:
        by_payer[e["payer"]]["uncovered"].append(e)
    for e in holds:
        by_payer[e["payer"]]["holds"].append(e)
    stats = {"uncovered_denials": len(uncovered), "false_holds": len(holds), "proposed": 0, "promoted": 0,
             "rejected": 0, "revised": 0, "promoted_rules": [], "rejected_rules": []}
    seq = store.db.rules.count_documents({}) + 1

    for payer, groups in sorted(by_payer.items()):
        paid_examples = [e for e in evidence if e["payer"] == payer and e["observed"]["outcome"] == "paid"]
        denied_examples = [e for e in evidence if e["payer"] == payer and e["observed"]["outcome"] == "denied"]
        feedback = None
        for attempt in range(MAX_REVISIONS + 1):
            head, active = store.active()
            lessons = store.rejected_rules()
            try:
                candidates, meta = propose(groups["uncovered"], groups["holds"], paid_examples=paid_examples,
                                           denied_examples=denied_examples, active_rules=active,
                                           lessons=lessons, feedback=feedback)
            except Exception as e:  # noqa: BLE001
                log(f"  watcher failed for {payer}: {e}")
                break
            store.audit("proposal", run_id=run_id, iteration=iteration, payer=payer, attempt=attempt,
                        n_rules=len(candidates), notes=meta.get("notes"), model=meta["model"])
            rejected_feedback = []
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                cand = {**cand, "rule_id": f"R-{seq:03d}", "payer": cand.get("payer") or payer}
                seq += 1
                stats["proposed"] += 1
                stats["revised"] += attempt > 0
                head, active = store.active()
                verdict = evaluate(cand, active, evidence)
                ev_id = store.insert_evaluation({"run_id": run_id, "iteration": iteration, "rule_id": cand["rule_id"],
                                                 "baseline_version": head["version"], **verdict})
                evid = [f"{run_id}:{iteration}:{c}" for c in verdict["fixes"]]
                try:
                    store.insert_rule(cand, status="candidate", iteration=iteration, evidence=evid)
                except Exception:  # duplicate id after resume
                    continue
                kind = "exception" if (cand.get("predict") or {}).get("outcome") == "paid" else "denial"
                if verdict["verdict"] == "promote":
                    try:
                        v = store.promote([cand["rule_id"]], expected_version=head["version"],
                                          evaluation_id=ev_id, note=cand.get("description", ""))
                    except ConflictError as e:
                        store.set_rule_status(cand["rule_id"], "parked", rejection_reason=str(e))
                        continue
                    store.set_rule_status(cand["rule_id"], "promoted", promoted_version=v)
                    store.audit("promotion", run_id=run_id, iteration=iteration, rule_id=cand["rule_id"],
                                version=v, fixes=verdict["fixes"], description=cand.get("description"), rule_kind=kind)
                    stats["promoted"] += 1
                    stats["promoted_rules"].append({"rule_id": cand["rule_id"], "version": v, "kind": kind,
                                                    "description": cand.get("description"), "fixes": verdict["fixes"]})
                    log(f"  PROMOTED v{v} {cand['rule_id']} [{kind}]: {cand.get('description')} (fixes {verdict['fixes']})")
                else:
                    store.set_rule_status(cand["rule_id"], "rejected", rejection_reason=verdict["reason"],
                                          regressions=verdict.get("regressions", []))
                    store.audit("rejection", run_id=run_id, iteration=iteration, rule_id=cand["rule_id"],
                                reason=verdict["reason"], description=cand.get("description"), rule_kind=kind)
                    stats["rejected"] += 1
                    stats["rejected_rules"].append({"rule_id": cand["rule_id"], "kind": kind,
                                                    "description": cand.get("description"), "reason": verdict["reason"]})
                    rejected_feedback.append(f"{cand.get('description')!r}: {verdict['reason']}")
                    log(f"  rejected {cand['rule_id']} [{kind}]: {cand.get('description')} -> {verdict['reason']}")
            _, active = store.active()
            groups = {"uncovered": [e for e in groups["uncovered"] if _uncovered_denial(active, e)],
                      "holds": [e for e in groups["holds"] if _false_hold(active, e)]}
            if not rejected_feedback or not (groups["uncovered"] or groups["holds"]):
                break
            feedback = "; ".join(rejected_feedback)
    return stats


# -- driver -------------------------------------------------------------------------------

def run(store: Store, run_id: str, iterations: int) -> list[dict]:
    store.bootstrap()
    state = store.run_state(run_id) or {}
    history = state.get("history", [])
    start = len(history)
    if start:
        log(f"resuming {run_id} at iteration {start} (ruleset v{store.active()[0]['version']} loaded from the database)")
    for it in range(start, iterations + 1):
        log(f"iteration {it}: scrubbing with ruleset v{store.active()[0]['version']}")
        metrics = scrub_pass(store, run_id, it)
        log(f"  heldout balanced acc {metrics['heldout']['balanced_accuracy']}  "
            f"train {metrics['train']['balanced_accuracy']}  rules {metrics['rule_count']}")
        entry = {"iteration": it, "metrics": metrics,
                 "scrubber_model": os.environ.get("SCRUBBER_MODEL"), "watcher_model": os.environ.get("WATCHER_MODEL")}
        if it < iterations:
            log(f"iteration {it}: watcher learning from training misses")
            entry["learning"] = learn(store, run_id, it)
        history.append(entry)
        store.checkpoint(run_id, history=history, phase="done" if it == iterations else "running")
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="demo")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--reset", action="store_true", help="drop all harness collections first")
    ap.add_argument("--mock", action="store_true", help="use an in-memory database instead of Atlas")
    args = ap.parse_args()
    if args.mock:
        import mongomock
        db = mongomock.MongoClient().db
    else:
        from memory.atlas_schema import get_db
        db = get_db()
    store = Store(db)
    if args.reset:
        for c in ("episodes", "rules", "rulesets", "heads", "evaluations", "runs", "audits"):
            db[c].drop()
    history = run(store, args.run_id, args.iterations)
    print("\niteration  heldout_bal_acc  heldout_acc_w_reason  unnecessary_holds  rules")
    for h in history:
        m = h["metrics"]["heldout"]
        print(f"{h['iteration']:>9}  {m['balanced_accuracy']!s:>15}  {m['accuracy_with_reason']!s:>20}  "
              f"{m['unnecessary_holds']:>17}  {h['metrics']['rule_count']:>5}")


if __name__ == "__main__":
    main()
