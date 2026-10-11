"""Stage demo: five beats that tell the story in three minutes.

    python -m harness.demo --mock                  # fully offline, in-memory database (run from the repo root)
    python -m harness.demo --run-id run2           # against Atlas (read-only except beat 3)
    python -m harness.demo --run-id run2 --beat 3  # any beat alone

Beats: 1 before (a held-out claim the baseline missed), 2 learned (the promoted rule, the
training claims that taught it, its gate stats), 3 refuses (an over-broad rule is rejected),
4 remembers (a fresh connection loads the ruleset from the database and scrubs by rule),
5 score (the held-out curve).

Everything is read from the store. The only write is beat 3's probe, which records exactly
one rule doc (status rejected), one evaluation doc and one audit doc and never moves the
active ruleset or head. `--mock` seeds an in-memory mongomock database by running the real
loop with a fake model, so nothing touches the network or Atlas.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from unittest import mock

import agents.scrubber as scrubber_mod
import agents.watcher as watcher_mod
from eval.synthetic_claims import load_claims, public_view
from harness.features import features
from harness.payer_sim import CARC_TEXT, adjudicate, outcome_summary
from harness.probe import run_probe
from harness.rules import apply_rules
from memory.store import Store

BEATS = {1: "BEFORE", 2: "LEARNED", 3: "REFUSES", 4: "REMEMBERS", 5: "SCORE"}
WRITE_COLLECTIONS = ("rules", "evaluations", "audits")
MOCK_BANNER = "[MOCK: stand-in watcher seeded from the hand-written oracle rules, not a learned result]"
_state = {"mock": False}


# -- offline seeding ----------------------------------------------------------------------

def _fake_scrub_llm(model, system, user, **kw):
    """The baseline: a model that waves everything through."""
    return {"predicted_outcome": "paid", "carc": None, "confidence": 0.5, "reasons": []}, \
        {"model": "fake", "latency_s": 0, "cached": False}


MOCK_DESCRIPTIONS = {
    "PR-01": "NORTHSTAR requires prior authorization for advanced imaging (70551, 72148, 74177)",
    "PR-02-NS": "NORTHSTAR denies claims filed more than 180 days after the date of service",
    "PR-02-BV": "BAYVIEW denies claims filed more than 90 days after the date of service",
    "PR-02-CA": "CASCADE denies claims filed more than a year after the date of service",
    "PR-03a": "CASCADE requires modifier 25 on an E/M billed with an injection or vaccine administration",
    "PR-03b": "CASCADE requires modifier 25 on an E/M billed with a procedure",
    "PR-04": "NORTHSTAR does not accept modifier 59 on NCCI edit pairs",
    "PR-05": "BAYVIEW bundles venipuncture 36415 into lab panels",
    "PR-06": "NORTHSTAR allows one preventive visit per rolling 365 days",
    "PR-07": "NORTHSTAR caps physical therapy at 20 visits per plan year",
    "PR-08": "CASCADE covers vitamin D testing (82306) only for listed diagnoses",
    "PR-09a": "BAYVIEW telehealth from home must use POS 10, not 02",
    "PR-09b": "BAYVIEW telehealth from home requires modifier 95",
    "PR-10": "BAYVIEW denies non-emergency out-of-network services",
    "PR-11": "Every service line needs a valid diagnosis pointer",
    "PR-12": "Every service line needs a charge greater than zero",
}


def _fake_watch_llm(model, system, user, **kw):
    """A stand-in watcher that proposes sound rules plus one over-broad rule the gate must reject."""
    from tests.test_rule_language import ORACLE  # hand-written rules, used only to seed --mock
    payload = json.loads(user.split("\n\nYour previous")[0])
    payers = {m["payer"] for m in payload["uncovered_denials"] + payload["false_holds"]}
    rules = [dict(r, rule_id="", description=MOCK_DESCRIPTIONS.get(r["rule_id"], r["description"]))
             for r in ORACLE if r["payer"] in payers or r["payer"] == "*"]
    rules.append({"rule_id": "", "payer": sorted(payers)[0], "description": "deny everything",
                  "when": [{"field": "line_count", "op": "gte", "value": 1}],
                  "predict": {"outcome": "denied", "carc": "16"}})
    return {"rules": rules, "notes": "mock watcher"}, {"model": "fake", "latency_s": 0, "cached": False}


def seed_mock(run_id: str = "mock"):
    """Run the real loop once against mongomock with the fake model. Returns the client."""
    import mongomock

    from harness import loop
    _state["mock"] = True
    client = mongomock.MongoClient()
    with mock.patch.object(scrubber_mod, "complete_json", _fake_scrub_llm), \
            mock.patch.object(watcher_mod, "complete_json", _fake_watch_llm), \
            contextlib.redirect_stdout(io.StringIO()):
        loop.run(Store(client.db), run_id, iterations=1)
    return client


# -- printing helpers ---------------------------------------------------------------------

def heading(n: int) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  BEAT {n} of {len(BEATS)}: {BEATS[n]}\n{bar}")
    if _state["mock"] and n in (2, 5):
        print(MOCK_BANNER)


def _compact(obj):
    """Shorten long value lists so a rule fits on a slide."""
    if isinstance(obj, list):
        return obj if len(obj) <= 6 else [*obj[:3], f"... {len(obj)} items"]
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items()}
    return obj


def _carc(code: str) -> str:
    return f"CARC {code} ({CARC_TEXT.get(code, 'see X12 code list')})"


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


# -- the beats ----------------------------------------------------------------------------

def _history(store: Store, run_id: str) -> list[dict]:
    state = store.run_state(run_id)
    if not state or not state.get("history"):
        raise SystemExit(f"no run {run_id!r} in the database; run harness.loop first or use --mock")
    return state["history"]


def beat1(store: Store, run_id: str, claim_id: str | None = None) -> dict:
    """A held-out claim the baseline missed and a promoted rule now covers."""
    heading(1)
    history = _history(store, run_id)
    _, active = store.active()
    base_it = history[0]["iteration"]
    missed = []
    for ep in store.episodes({"run_id": run_id, "iteration": base_it, "split": "heldout"}):
        obs = ep["observed"]
        hit = apply_rules(active, ep["features"], "denied")
        if (obs["outcome"] == "denied" and ep["predicted"]["predicted_outcome"] != "denied"
                and hit and hit["carc"] in obs["carcs"]):
            missed.append(ep)
    missed.sort(key=lambda e: e["claim_id"])
    ep = next((e for e in missed if e["claim_id"] == claim_id), None) if claim_id else (missed[0] if missed else None)
    if ep is None:
        raise SystemExit("no held-out claim was missed by the baseline and is now covered by a rule")
    f, obs, pred = ep["features"], ep["observed"], ep["predicted"]
    print(f"Held-out claim {ep['claim_id']}  payer {ep['payer']}  CPT {', '.join(ep['cpts'])}  "
          f"charge ${f['total_charge']:,.2f}")
    print("This claim was never shown to the watcher. Its outcome was not learning evidence.\n")
    print(f"  baseline (ruleset v{ep['ruleset_version']}) predicted : {pred['predicted_outcome'].upper()}  "
          f"(source: {pred.get('source')})")
    print(f"  the payer actually returned            : {obs['outcome'].upper()}  "
          + "; ".join(_carc(c) for c in obs["carcs"]) + (f"  RARC {', '.join(obs['rarcs'])}" if obs.get("rarcs") else ""))
    print("  => the baseline would have submitted a claim that was always going to bounce.")
    return {"claim_id": ep["claim_id"], "payer": ep["payer"], "baseline": pred["predicted_outcome"],
            "observed": obs["outcome"], "carcs": obs["carcs"], "episode": ep}


def beat2(store: Store, run_id: str, ep: dict) -> dict:
    """The promoted rule that covers the claim, who taught it, and what the gate computed."""
    heading(2)
    head, active = store.active()
    hit = apply_rules(active, ep["features"], "denied")
    rule = store.db.rules.find_one({"_id": hit["rule_id"]})
    ev = store.db.evaluations.find_one({"rule_id": rule["_id"]})
    print(f"Promoted rule {rule['_id']} (ruleset v{rule.get('promoted_version')}, written by the watcher in iteration "
          f"{rule.get('proposed_in_iteration')})")
    print(f"  policy   : {rule['description']}")
    print(f"  predicts : {rule['predict']['outcome']} with {_carc(rule['predict']['carc'])}")
    print(f"  when     : {json.dumps(_shorten(_compact(rule['when'])), separators=(',', ':'))}")
    taught = [store.db.episodes.find_one({"_id": i}) for i in rule.get("evidence_episode_ids", [])]
    taught = [t for t in taught if t]
    print(f"\nTraining claims that taught it ({len(taught)}):")
    for t in taught:
        print(f"  - {t['claim_id']}  {t['payer']}  CPT {', '.join(t['cpts'])}  -> "
              f"{t['observed']['outcome']} {', '.join(t['observed']['carcs'])}")
    held_out_clean = all(t["split"] == "train" for t in taught) and ep["claim_id"] not in {t["claim_id"] for t in taught}
    print(f"\nGate stats (computed by the harness, never reported by a model):")
    print(f"  replayed {ev['evidence_size']} observed training claims | fixes {len(ev['fixes'])} | "
          f"regressions {len(ev['regressions'])} | already covered {len(ev.get('redundant') or [])}")
    if ev.get("replaces"):
        print(f"  consolidation: replaced {', '.join(ev['replaces'])}, kept all "
              f"{len(ev.get('preserved') or [])} claim(s) they explained, lost {len(ev.get('lost') or [])}")
    print(f"  pinned thresholds {ev['thresholds']} | verdict {ev['verdict'].upper()}: {ev['reason']}")
    print(f"  held-out claim {ep['claim_id']} absent from the evidence: {held_out_clean}")
    return {"rule_id": rule["_id"], "taught": [t["claim_id"] for t in taught], "evaluation": ev,
            "held_out_clean": held_out_clean, "head_version": head["version"]}


def _shorten(x, keep: int = 6):
    """Long code lists (e.g. a consolidated imaging family) print as the first few plus a count."""
    if isinstance(x, list):
        if len(x) > keep and all(not isinstance(v, (dict, list)) for v in x):
            return x[:keep] + [f"... +{len(x) - keep} more"]
        return [_shorten(v, keep) for v in x]
    if isinstance(x, dict):
        return {k: _shorten(v, keep) for k, v in x.items()}
    return x


def beat3(store: Store, run_id: str) -> dict:
    """Push an over-broad rule through the real gate."""
    heading(3)
    before_head = store.db.heads.find_one({"_id": "main"})
    before = {c: store.db[c].count_documents({}) for c in WRITE_COLLECTIONS}
    out = run_probe(store, run_id)
    rule, verdict = out["rule"], out["verdict"]
    after_head = store.db.heads.find_one({"_id": "main"})
    wrote = {c: store.db[c].count_documents({}) - before[c] for c in WRITE_COLLECTIONS}
    status = store.db.rules.find_one({"_id": rule["rule_id"]})["status"]
    print(f"Someone proposes: {rule['description']}")
    print(f"  replayed against {verdict['evidence_size']} observed training claims "
          f"on ruleset v{before_head['version']}")
    print(f"  verdict : {verdict['verdict'].upper()}  {verdict['reason']}")
    unchanged = (before_head["version"], before_head["revision"]) == (after_head["version"], after_head["revision"])
    print(f"\nWhat this beat wrote: {wrote['rules']} rule doc (status {status}), {wrote['evaluations']} evaluation doc, "
          f"{wrote['audits']} audit doc. Nothing else.")
    print(f"Active ruleset/head unchanged: v{after_head['version']} (revision {after_head['revision']}) = {unchanged}")
    return {"rule_id": rule["rule_id"], "verdict": verdict["verdict"], "status": status, "wrote": wrote,
            "head_unchanged": unchanged, "regressions": verdict["regressions"]}


def beat4(reconnect, expect_version: int, shown: dict) -> dict:
    """A brand-new connection: no caches, no process state. Rules come from the database."""
    heading(4)
    fresh = Store(reconnect())
    head, rules = fresh.active()
    print(f"New process, new connection. Loaded ruleset v{head['version']} from the database "
          f"({len(rules)} rules). Same version as before the restart: {head['version'] == expect_version}")
    rec = _fresh_claim(rules, shown)
    view = public_view(rec)
    feats = features(view)
    assert apply_rules(rules, feats), "chosen claim must be covered by a rule (rule path only)"

    def no_model(*a, **k):
        raise AssertionError("beat 4 must not call a model")
    with mock.patch.object(scrubber_mod, "complete_json", no_model):
        decision = scrubber_mod.scrub(view, rules=rules, similar=[])
    payer = outcome_summary(adjudicate(rec))
    print(f"\nScrubbing {view['id']} ({view['payer']}, CPT {', '.join(feats['cpts'])}) with no model call:")
    print(f"  scrubber : {decision['action'].upper()}  predicts {decision['predicted_outcome']} "
          f"{_carc(decision['carc'])}  via {decision['source']}")
    print(f"  reason   : {decision['reasons'][0]}")
    print(f"  payer    : {payer['outcome']} {', '.join(payer['carcs'])}  (matches: "
          f"{decision['carc'] in payer['carcs']})")
    return {"version": head["version"], "rules": len(rules), "claim_id": view["id"], "source": decision["source"],
            "model": decision["model"], "carc": decision["carc"], "payer_carcs": payer["carcs"]}


def _fresh_claim(rules: list[dict], shown: dict) -> dict:
    """A different held-out claim (never learning evidence) that a learned denial rule covers,
    preferring another payer than beat 1 so the audience sees the rules are not one trick."""
    pool = [r for r in load_claims("heldout")
            if r["id"] != shown["claim_id"] and apply_rules(rules, features(public_view(r)))]
    if not pool:
        raise SystemExit("no other held-out claim is covered by a learned rule")
    from harness.payer_sim import adjudicate, outcome_summary

    def agrees(r: dict) -> bool:  # the rule's call matches what the payer actually returned
        hit = apply_rules(rules, features(public_view(r)))
        return hit["carc"] in outcome_summary(adjudicate(r))["carcs"]
    pool.sort(key=lambda r: (not agrees(r), r["payer"] == shown["payer"], r["id"]))
    return pool[0]


def beat5(store: Store, run_id: str) -> dict:
    """The held-out curve, straight from the run history."""
    heading(5)
    history = _history(store, run_id)
    print("Held-out claims only. The scrubber is judged on outcomes it never learned from.\n")
    print(f"{'iter':>4} {'ruleset':>8} {'rules':>6} {'balanced acc':>13} {'acc w/ reason':>14} "
          f"{'denials caught':>15} {'unnecessary holds':>18}")
    rows = []
    for h in history:
        m, hm = h["metrics"], h["metrics"]["heldout"]
        caught = f"{hm['denials_caught_right_reason']}/{hm['n_denied']}"
        print(f"{h['iteration']:>4} {'v' + str(m['ruleset_version']):>8} {m['rule_count']:>6} "
              f"{_pct(hm['balanced_accuracy']):>13} {_pct(hm['accuracy_with_reason']):>14} "
              f"{caught:>15} {str(hm['unnecessary_holds']) + '/' + str(hm['n_paid']):>18}")
        rows.append({"iteration": h["iteration"], "balanced_accuracy": hm["balanced_accuracy"],
                     "unnecessary_holds": hm["unnecessary_holds"], "rules": m["rule_count"]})
    n = history[-1]["metrics"]["heldout"]["n"]
    print(f"\nn = {n} held-out claims, so one claim moves accuracy by about {100 / n:.1f} points. "
          f"Balanced accuracy and unnecessary holds stop an always-deny scrubber from looking good.")
    return {"rows": rows, "n": n}


# -- driver -------------------------------------------------------------------------------

def run_beats(reconnect, run_id: str, beats: list[int], claim_id: str | None = None) -> dict:
    store = Store(reconnect())
    out: dict = {}
    # beats 1, 2 and 4 hang off one claim; pick it quietly when beat 1 is not being shown
    if 1 in beats:
        b1 = out[1] = beat1(store, run_id, claim_id)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            b1 = beat1(store, run_id, claim_id)
    if 2 in beats:
        out[2] = beat2(store, run_id, b1["episode"])
    if 3 in beats:
        out[3] = beat3(store, run_id)
    if 4 in beats:
        out[4] = beat4(reconnect, store.active()[0]["version"], b1)
    if 5 in beats:
        out[5] = beat5(store, run_id)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-id", default=None, help="run to present (default: run2, or 'mock' with --mock)")
    ap.add_argument("--beat", type=int, choices=sorted(BEATS), action="append", help="run only this beat (repeatable)")
    ap.add_argument("--claim", default=None, help="held-out claim id to feature in beats 1 and 2")
    ap.add_argument("--mock", action="store_true", help="offline: seed an in-memory database with a fake-model loop")
    args = ap.parse_args(argv)
    run_id = args.run_id or ("mock" if args.mock else "run2")
    _state["mock"] = args.mock
    if args.mock:
        import mongomock
        client = seed_mock(run_id)
        reconnect = lambda: mongomock.MongoClient(_store=client._store).db  # noqa: E731  a new client, same data
    else:
        from memory.atlas_schema import get_db

        def reconnect():
            get_db.cache_clear()  # a genuinely new MongoClient, as after a restart
            return get_db()
    print("Recursive Revenue-Cycle Harness: stage demo"
          + ("  [MOCK: in-memory database, fake model, offline]" if args.mock else f"  [run {run_id}]"))
    run_beats(reconnect, run_id, args.beat or sorted(BEATS), args.claim)


if __name__ == "__main__":
    main()
