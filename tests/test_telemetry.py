"""Raindrop telemetry: a no-op without a key, correct calls with a fake SDK, no answer key in payloads."""
import json

import mongomock
import pytest

import agents.scrubber as scrubber_mod
import agents.watcher as watcher_mod
from agents import telemetry
from eval.synthetic_claims import load_claims
from harness import loop
from memory.store import Store
from tests.test_loop import fake_scrub_llm, fake_watch_llm


class FakeSDK:
    def __init__(self):
        self.calls = []

    def _rec(name):
        def f(self, **kw):
            self.calls.append((name, kw))
        return f

    track_ai = _rec("track_ai")
    track_signal = _rec("track_signal")
    track = _rec("track")
    flush = _rec("flush")
    shutdown = _rec("shutdown")


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    for k in ("RAINDROP_WRITE_KEY", "RAINDROP_SAMPLE", "RAINDROP_DISABLE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(scrubber_mod, "complete_json", fake_scrub_llm)
    monkeypatch.setattr(watcher_mod, "complete_json", fake_watch_llm)
    telemetry.set_sdk(None)
    yield
    telemetry.set_sdk(None)


def _run(sdk=None):
    if sdk:
        telemetry.set_sdk(sdk)
    return loop.run(Store(mongomock.MongoClient().db), "t", iterations=1, slices=1)


def test_no_key_is_a_noop(monkeypatch):
    monkeypatch.setenv("RAINDROP_WRITE_KEY", "rd_should_never_be_used_in_tests")
    assert not telemetry.enabled()  # pytest guard: even a real key in the env sends nothing
    history = _run()
    assert history[1]["metrics"]["heldout"]["balanced_accuracy"] > 0.9
    assert telemetry.summary()["total"] == 0


def test_fake_sdk_gets_scrub_events_and_signals():
    sdk = FakeSDK()
    _run(sdk)
    ai = [kw for n, kw in sdk.calls if n == "track_ai" and kw["event"] == "claim_scrub"]
    sig = {kw["event_id"]: kw for n, kw in sdk.calls if n == "track_signal" and kw["name"] == "payer_outcome"}
    n_claims = len(load_claims("train")) + len(load_claims("heldout"))
    assert len(ai) == 2 * n_claims == len(sig)  # iterations 0 and 1
    first = ai[0]
    assert first["event_id"].startswith("t:0:") and first["convo_id"] == "t:0"
    assert first["user_id"] in {"NORTHSTAR", "BAYVIEW", "CASCADE"}
    assert first["model"] == "fake"
    assert set(first["properties"]) == {"ruleset_version", "source", "split", "iteration"}
    for e in ai:
        s = sig[e["event_id"]]
        assert s["signal_type"] == "feedback" and s["sentiment"] in ("POSITIVE", "NEGATIVE")
    # always-paid baseline: every denial is a missed_denial (NEGATIVE), every paid claim is right
    base = [sig[e["event_id"]] for e in ai if e["convo_id"] == "t:0"]
    for s in base:
        p = s["properties"]
        assert (p["outcome"], p["error_type"], s["sentiment"]) in {
            ("paid", "none", "POSITIVE"), ("denied", "missed_denial", "NEGATIVE")}
    assert any(s["comment"].startswith("denied CO-") for s in base)
    assert [n for n, _ in sdk.calls][-1] == "flush"


def test_fake_sdk_gets_proposals_and_gate_verdicts():
    sdk = FakeSDK()
    _run(sdk)
    props = {kw["event_id"]: kw for n, kw in sdk.calls if n == "track_ai" and kw["event"] == "rule_proposal"}
    verdicts = {kw["event_id"]: kw for n, kw in sdk.calls if n == "track_signal" and kw["name"] == "gate_verdict"}
    assert props and set(props) == set(verdicts)
    assert {v["sentiment"] for v in verdicts.values()} == {"POSITIVE", "NEGATIVE"}
    rejected = [k for k, p in props.items() if "deny everything" in p["output"]]
    assert rejected and all(verdicts[k]["sentiment"] == "NEGATIVE" for k in rejected)
    promos = [kw for n, kw in sdk.calls if n == "track" and kw["event"] == "ruleset_promotion"]
    assert promos and all(isinstance(p["properties"]["ruleset_version"], int) for p in promos)
    assert telemetry.summary()["events_sent"]["ruleset_promotion"] == len(promos)


def test_no_hidden_fields_in_any_payload():
    sdk = FakeSDK()
    _run(sdk)
    blob = json.dumps(sdk.calls, default=str)
    hidden = {"ground_truth", "violated_rules", "category"}
    for rec in load_claims("train") + load_claims("heldout"):
        hidden |= set(rec["ground_truth"]["violated_rules"])
        assert rec["claim"]["subscriber"]["memberId"] not in blob
        assert rec["ground_truth"]["reason"] not in blob
    for h in hidden:
        assert h not in blob, h
    for n, kw in sdk.calls:
        if n == "track_ai":
            assert "split" not in json.loads(kw["input"]) and "split" not in json.loads(kw["output"])


def test_clean_redacts_answer_key_ids_and_keys():
    out = telemetry.clean({"ground_truth": 1, "x": ["see PR-03a and PR-02-NS", {"split": "heldout", "ok": 1}]})
    assert out == {"x": ["see [rule] and [rule]", {"ok": 1}]}


def test_sampling_and_disable(monkeypatch):
    monkeypatch.setenv("RAINDROP_DISABLE", "claim_scrub,payer_outcome")
    sdk = FakeSDK()
    _run(sdk)
    assert not [1 for n, kw in sdk.calls if kw.get("event") == "claim_scrub" or kw.get("name") == "payer_outcome"]
    assert any(kw.get("event") == "rule_proposal" for _, kw in sdk.calls)
    monkeypatch.setenv("RAINDROP_DISABLE", "")
    monkeypatch.setenv("RAINDROP_SAMPLE", "0")
    sdk2 = FakeSDK()
    _run(sdk2)
    assert not [n for n, _ in sdk2.calls if n in ("track_ai", "track_signal", "track")]
    monkeypatch.setenv("RAINDROP_SAMPLE", "0.5")
    sdk3 = FakeSDK()
    _run(sdk3)
    ai = {kw["event_id"] for n, kw in sdk3.calls if n == "track_ai" and kw["event"] == "claim_scrub"}
    sig = {kw["event_id"] for n, kw in sdk3.calls if n == "track_signal" and kw["name"] == "payer_outcome"}
    assert ai == sig and 0 < len(ai) < 2 * (len(load_claims("train")) + len(load_claims("heldout")))


def test_sdk_failure_never_breaks_the_loop():
    class Boom(FakeSDK):
        def track_ai(self, **kw):
            raise RuntimeError("network down")
    history = _run(Boom())
    assert history[1]["metrics"]["heldout"]["balanced_accuracy"] > 0.9


def test_export_has_raindrop_block(monkeypatch):
    from demo.export import export
    monkeypatch.setenv("RAINDROP_DASHBOARD_URL", "https://app.raindrop.ai/example")
    db = mongomock.MongoClient().db
    telemetry.set_sdk(FakeSDK())
    loop.run(Store(db), "t", iterations=1, slices=1)
    block = export(db, "t")["raindrop"]
    assert block["dashboard_url"] == "https://app.raindrop.ai/example"
    assert block["total"] == sum(block["events_sent"].values()) > 0
    telemetry.set_sdk(None)
    assert export(mongomock.MongoClient().db, "none")["raindrop"] == {
        "dashboard_url": "https://app.raindrop.ai/example", "events_sent": {}, "total": 0}


def test_malformed_inputs_never_raise():
    telemetry.set_sdk(FakeSDK())
    ep = {"_id": "t:0:X", "run_id": "t", "iteration": 0, "split": "train", "payer": "NORTHSTAR",
          "ruleset_version": 0, "correct": False,
          "predicted": {"predicted_outcome": "paid"}, "observed": {"outcome": "denied", "carcs": ["197"]},
          "features": {"pos": "11", "dx_codes": [], "lines": [{"cpt": "74177"}]}}  # line missing "modifiers"
    telemetry.scrub_event(ep)
    telemetry.scrub_event({})
    telemetry.rule_proposal("t", 0, "NORTHSTAR", {}, [], None)       # no rule_id
    telemetry.gate_verdict("t", "R-1", None, True)                    # verdict not a dict
    telemetry.ruleset_promotion("t", 0, "NORTHSTAR", "R-1", object())  # unserialisable version is fine too


def test_signal_requires_its_parent_event(monkeypatch):
    monkeypatch.setenv("RAINDROP_DISABLE", "claim_scrub,rule_proposal")
    sdk = FakeSDK()
    _run(sdk)
    assert not [1 for n, _ in sdk.calls if n == "track_signal"]  # no orphan Signals
    assert [1 for n, kw in sdk.calls if n == "track" and kw["event"] == "ruleset_promotion"]


def test_run_doc_telemetry_only_when_enabled():
    db = mongomock.MongoClient().db
    loop.run(Store(db), "t", iterations=1, slices=1)
    assert "telemetry" not in db.runs.find_one({"_id": "t"})
    db2 = mongomock.MongoClient().db
    telemetry.set_sdk(FakeSDK())
    loop.run(Store(db2), "t", iterations=1, slices=1)
    assert db2.runs.find_one({"_id": "t"})["telemetry"]["total"] > 0
