"""Streaming training slices, the min-evidence gate check, and the LLM concurrency guard."""
import sys
import types

import mongomock
import pytest

import agents.llm as llm
import agents.scrubber as scrubber_mod
import agents.watcher as watcher_mod
from eval.synthetic_claims import load_claims
from harness import loop
from harness.gate import MIN_PAYER_PAID_SEEN, evaluate, evaluate_replacement
from memory.store import Store
from tests.test_loop import fake_scrub_llm, fake_watch_llm
from tests.test_rules_gate_store import AUTH_RULE, EVIDENCE


def test_slices_are_stratified_and_every_payer_has_a_paid_claim_in_slice_0():
    slices = loop.train_slices(3)
    train = {r["id"]: r for r in load_claims("train")}
    assert sorted(c for s in slices for c in s) == sorted(train)  # a partition
    for payer in {r["payer"] for r in train.values()}:
        assert any(train[c]["payer"] == payer and train[c]["ground_truth"]["outcome"] == "paid" for c in slices[0])
    bayview_paid = [i for i, s in enumerate(slices) for c in s
                    if train[c]["payer"] == "BAYVIEW" and train[c]["ground_truth"]["outcome"] == "paid"]
    assert bayview_paid == [0, 1]
    assert loop.revealed_train_ids(9, 3) == set(train)  # past the last slice: everything


def test_streaming_loop_grows_evidence_and_scores_heldout_every_iteration(monkeypatch):
    monkeypatch.setattr(scrubber_mod, "complete_json", fake_scrub_llm)
    monkeypatch.setattr(watcher_mod, "complete_json", fake_watch_llm)
    client = mongomock.MongoClient()
    history = loop.run(Store(client.db), "s", iterations=3)
    sizes = [len(s) for s in loop.train_slices(3)]
    assert [h["n"] for h in history] == [sizes[0], sizes[0] + sizes[1], 24, 24]
    assert [h["metrics"]["train"]["n"] for h in history] == [h["n"] for h in history]
    n_heldout = len(load_claims("heldout"))
    assert all(h["metrics"]["heldout"]["n"] == n_heldout for h in history)
    reasons = [r["reason"] for r in history[0]["learning"]["rejected_rules"]]
    assert any("MIN_PAYER_PAID_SEEN" in r and "BAYVIEW" in r for r in reasons)  # 1 paid BAYVIEW seen
    assert history[-1]["metrics"]["heldout"]["balanced_accuracy"] > 0.9
    # idempotent resume with the stored slice count, even if the caller passes another
    n = client.db.episodes.count_documents({})
    again = loop.run(Store(client.db), "s", iterations=3, slices=1)
    assert len(again) == 4 and client.db.episodes.count_documents({}) == n
    assert client.db.runs.find_one({"_id": "s"})["slices"] == 3


def test_min_payer_paid_seen_counts_distinct_claims():
    one_paid = [EVIDENCE[0], EVIDENCE[1]]  # one paid NORTHSTAR claim
    v = evaluate(AUTH_RULE, [], one_paid + [dict(EVIDENCE[1])] * 3)  # re-scrubs of B must not count
    assert MIN_PAYER_PAID_SEEN == 2
    assert v["verdict"] == "rejected" and "MIN_PAYER_PAID_SEEN=2" in v["reason"]
    assert "pinned in harness/gate.py" in v["reason"]
    assert evaluate(AUTH_RULE, [], EVIDENCE)["verdict"] == "promote"  # B and D: two distinct


def test_min_payer_paid_seen_applies_to_denial_replacements():
    wider = {**AUTH_RULE, "rule_id": "R-NS-5", "when": [
        {"any_line": [{"field": "cpt", "op": "in", "value": ["70551", "70553", "72148", "74177"]}]},
        {"field": "prior_auth", "op": "missing"}]}
    thin = evaluate_replacement(wider, [AUTH_RULE], [AUTH_RULE], [EVIDENCE[0], EVIDENCE[1]])
    assert thin["verdict"] == "rejected" and thin["reason"].startswith("insufficient evidence")
    assert evaluate_replacement(wider, [AUTH_RULE], [AUTH_RULE], EVIDENCE)["verdict"] == "promote"


class FakeRateLimit(Exception):
    pass


def test_rate_limits_back_off_exponentially_then_surface(monkeypatch):
    calls, sleeps = [], []

    def completion(**kw):
        calls.append(kw)
        raise FakeRateLimit("429 rate limit exceeded")

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    monkeypatch.setenv("LLM_CASSETTE", "off")
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        llm.complete_json("fake/model", "s", "u")
    assert len(calls) == llm.MAX_TRIES == 3
    assert sleeps == [5.0, 10.0]  # exponential, nothing after the last try
    assert llm._slots._value == llm._slots._initial_value  # every slot released
    assert llm._rate_limited(Exception("529 Overloaded")) and not llm._rate_limited(ValueError("bad json"))
