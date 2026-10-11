"""JOB-1: threshold evidence, concrete revision feedback, and policy-style scrubber notes."""
import json

import mongomock

import agents.scrubber as scrubber_mod
import agents.watcher as watcher_mod
from agents.scrubber import _payer_notes
from harness import loop
from memory.store import Store


def _ev(claim_id, outcome, **feats):
    base = {"payer": "P", "pos": "11", "prior_auth": None, "days_to_submit": None, "in_network": True,
            "dx_codes": [], "days_since_last_preventive": None, "lines": [], "context": {}}
    ctx = feats.pop("context", {})
    return {"claim_id": claim_id, "features": {**base, **feats, "context": ctx},
            "observed": {"outcome": outcome, "carcs": [], "rarcs": []}}


def test_threshold_evidence_max_paid_min_denied():
    paid = [_ev("p1", "paid", days_to_submit=20, context={"visit_count_ytd": 3}),
            _ev("p2", "paid", days_to_submit=45)]
    denied = [_ev("d1", "denied", days_to_submit=90, days_since_last_preventive=200,
                  context={"visit_count_ytd": 13}),
              _ev("d2", "denied", days_to_submit=120)]
    out = watcher_mod.threshold_evidence(paid, denied)
    assert out["days_to_submit"] == {"max_on_paid_claim": 45, "min_on_denied_claim": 90}
    assert out["context.visit_count_ytd"] == {"max_on_paid_claim": 3, "min_on_denied_claim": 13}
    assert out["days_since_last_preventive"] == {"max_on_paid_claim": None, "min_on_denied_claim": 200}


def test_watcher_payload_contains_threshold_summary(monkeypatch):
    seen = {}

    def fake(model, system, user, **kw):
        seen["system"], seen["user"] = system, user
        return {"rules": [], "notes": ""}, {"model": "fake", "latency_s": 0, "cached": False}

    monkeypatch.setattr(watcher_mod, "complete_json", fake)
    watcher_mod.propose([], [], paid_examples=[_ev("p1", "paid", days_to_submit=30)],
                        denied_examples=[_ev("d1", "denied", days_to_submit=95)],
                        active_rules=[], lessons=[])
    payload = json.loads(seen["user"])
    assert payload["threshold_evidence"]["days_to_submit"] == {"max_on_paid_claim": 30, "min_on_denied_claim": 95}
    assert "as narrow as" not in seen["system"]
    assert "MOST GENERAL" in seen["system"]


def test_revision_feedback_includes_regressing_claim_briefs(monkeypatch):
    monkeypatch.setattr(scrubber_mod, "complete_json", lambda *a, **k: (
        {"predicted_outcome": "paid", "carc": None, "confidence": 0.5, "reasons": []},
        {"model": "fake", "latency_s": 0, "cached": False}))
    feedbacks = []

    def fake_watch(model, system, user, **kw):
        feedbacks.append(user.split("\n\nYour previous", 1)[1] if "\n\nYour previous" in user else None)
        payload = json.loads(user.split("\n\nYour previous")[0])
        payer = sorted({m["payer"] for m in payload["uncovered_denials"] + payload["false_holds"]})[0]
        rule = {"rule_id": "", "payer": payer, "description": "deny everything",
                "when": [{"field": "line_count", "op": "gte", "value": 1}],
                "predict": {"outcome": "denied", "carc": "16"}}
        return {"rules": [rule], "notes": ""}, {"model": "fake", "latency_s": 0, "cached": False}

    monkeypatch.setattr(watcher_mod, "complete_json", fake_watch)
    client = mongomock.MongoClient()
    store = Store(client.db)
    store.bootstrap()
    loop.scrub_pass(store, "t", 0)
    loop.learn(store, "t", 0)
    assert feedbacks[0] is None and any(feedbacks[1:]), "a rejected rule must trigger a revision with feedback"
    revision = next(f for f in feedbacks if f)
    evidence = loop.latest_train_evidence(store, "t")
    paid_ids = {e["claim_id"] for e in evidence if e["observed"]["outcome"] == "paid"}
    # the regressing claims appear as full briefs (payer, lines, observed), not bare ids
    assert any(f'"claim_id": "{c}"' in revision for c in paid_ids)
    assert '"lines"' in revision and '"observed"' in revision and "wrongly matched" in revision


def test_payer_notes_frame_rules_as_examples_not_limits():
    rules = [{"payer": "P", "description": "P bundles 36415 with 80053", "predict": {"outcome": "denied", "carc": "97"}}]
    notes = _payer_notes("P", rules)
    assert notes.startswith("Learned payer policy (examples, not exhaustive")
    assert "P bundles 36415 with 80053 (denies with CARC 97)" in notes
    assert _payer_notes("Q", rules) == "No payer-specific rules learned yet."
