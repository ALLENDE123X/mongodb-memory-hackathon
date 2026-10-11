"""End-to-end loop with a fake model: checks plumbing, gating, promotion and resume."""
import mongomock

import agents.scrubber as scrubber_mod
import agents.watcher as watcher_mod
from harness import loop
from memory.store import Store
from tests.test_rule_language import ORACLE


def fake_scrub_llm(model, system, user, **kw):
    return {"predicted_outcome": "paid", "carc": None, "confidence": 0.5, "reasons": []}, \
        {"model": "fake", "latency_s": 0, "cached": False}


def fake_watch_llm(model, system, user, **kw):
    import json
    payload = json.loads(user.split("\n\nYour previous")[0])
    payers = {m["payer"] for m in payload["uncovered_denials"] + payload["false_holds"]}
    rules = [dict(r, rule_id="") for r in ORACLE if r["payer"] in payers or r["payer"] == "*"]
    # plus one over-broad rule the gate must reject
    rules.append({"rule_id": "", "payer": sorted(payers)[0], "description": "deny everything",
                  "when": [{"field": "line_count", "op": "gte", "value": 1}],
                  "predict": {"outcome": "denied", "carc": "16"}})
    return {"rules": rules, "notes": "fake"}, {"model": "fake", "latency_s": 0, "cached": False}


def test_loop_learns_rejects_and_resumes(monkeypatch):
    monkeypatch.setattr(scrubber_mod, "complete_json", fake_scrub_llm)
    monkeypatch.setattr(watcher_mod, "complete_json", fake_watch_llm)
    client = mongomock.MongoClient()
    history = loop.run(Store(client.db), "t", iterations=1)
    first, second = history[0]["metrics"]["heldout"], history[1]["metrics"]["heldout"]
    assert first["balanced_accuracy"] == 0.5          # always-paid baseline
    assert second["balanced_accuracy"] > 0.9          # learned rules transfer to held-out
    assert second["unnecessary_holds"] == 0
    learning = history[0]["learning"]
    assert learning["promoted"] > 0 and learning["rejected"] > 0
    assert any("deny everything" == r["description"] for r in learning["rejected_rules"])
    # held-out outcomes never become learning evidence
    assert all(e["split"] == "train" for e in loop.latest_train_evidence(Store(client.db), "t"))
    # restart: a new process resumes from the database without re-submitting
    n = client.db.episodes.count_documents({})
    again = loop.run(Store(client.db), "t", iterations=1)
    assert len(again) == 2 and client.db.episodes.count_documents({}) == n
