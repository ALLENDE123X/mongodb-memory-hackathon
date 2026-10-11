"""JOB-2: rule consolidation. Gated replacement, superseded rulesets, loop step with zero misses."""
import json

import mongomock
import pytest

import agents.scrubber as scrubber_mod
import agents.watcher as watcher_mod
from harness import loop
from harness.gate import evaluate_replacement
from memory.store import ConflictError, Store
from tests.test_loop import fake_scrub_llm
from tests.test_rule_language import ORACLE


def lines(*cpts):
    return [{"cpt": c, "modifiers": [], "units": 1, "charge": 25.0, "dx_pointer_count": 1} for c in cpts]


def ev(claim_id, outcome, cpts, carc=None):
    return {"claim_id": claim_id, "features": {"payer": "BAYVIEW", "context": {}, "lines": lines(*cpts)},
            "observed": {"outcome": outcome, "carcs": [carc] if carc else []}}


def rule(rid, when, carc="97", payer="BAYVIEW"):
    return {"rule_id": rid, "payer": payer, "description": rid, "when": when,
            "predict": {"outcome": "denied", "carc": carc}}


VENIPUNCTURE = {"any_line": [{"field": "cpt", "op": "eq", "value": "36415"}]}
N1 = rule("N1", [VENIPUNCTURE, {"any_line": [{"field": "cpt", "op": "eq", "value": "80053"}]}])
N2 = rule("N2", [VENIPUNCTURE, {"any_line": [{"field": "cpt", "op": "eq", "value": "80048"}]}])
LAB_RANGE = rule("NEW", [VENIPUNCTURE, {"any_line": [{"field": "cpt", "op": "between", "value": [80047, 89398]}]}])
EVIDENCE = [ev("D1", "denied", ["36415", "80053"], "97"), ev("D2", "denied", ["36415", "80048"], "97"),
            ev("P1", "paid", ["36415", "99213"]), ev("P2", "paid", ["80053"]),
            ev("D3", "denied", ["36415", "85025"], "16")]  # same pattern, different cause


def no_d3(evidence=EVIDENCE):
    return [e for e in evidence if e["claim_id"] != "D3"]


def test_equivalent_coverage_but_broader_merge_is_promoted():
    v = evaluate_replacement(LAB_RANGE, [N1, N2], [N1, N2], no_d3())
    assert v["verdict"] == "promote", v["reason"]
    assert sorted(v["fixes"]) == ["D1", "D2"] and sorted(v["preserved"]) == ["D1", "D2"]
    assert v["regressions"] == [] and v["lost"] == []


def test_generalizing_a_single_rule_with_equal_coverage_is_promoted():
    v = evaluate_replacement(LAB_RANGE, [N1], [N1, N2], no_d3())
    assert v["verdict"] == "promote", v["reason"]


def test_coverage_losing_replacement_is_rejected():
    narrow = rule("NEW", [VENIPUNCTURE, {"any_line": [{"field": "cpt", "op": "between", "value": [80050, 80060]}]}])
    v = evaluate_replacement(narrow, [N1, N2], [N1, N2], no_d3())
    assert v["verdict"] == "rejected" and v["lost"] == ["D2"]


def test_narrowing_is_rejected_even_when_another_remaining_rule_still_explains_it():
    other = rule("O", [{"any_line": [{"field": "cpt", "op": "in", "value": ["80048"]}]}, VENIPUNCTURE])
    narrow = rule("NEW", [VENIPUNCTURE, {"any_line": [{"field": "cpt", "op": "in", "value": ["80053", "80054"]}]}])
    v = evaluate_replacement(narrow, [N1, N2], [N1, N2, other], no_d3())
    assert v["verdict"] == "rejected" and v["narrowed"] == ["D2"] and v["lost"] == []


def test_duplicate_replaces_ids_are_deduped():
    v = evaluate_replacement(LAB_RANGE, [N1, N1, N2], [N1, N2], no_d3())
    assert v["verdict"] == "promote" and v["replaces"] == ["N1", "N2"]


def test_replacement_firing_on_a_paid_claim_is_rejected():
    v = evaluate_replacement(rule("NEW", [VENIPUNCTURE]), [N1, N2], [N1, N2], no_d3())
    assert v["verdict"] == "rejected" and v["regressions"] == ["P1"]


def test_replacement_firing_on_a_denial_with_another_carc_is_rejected():
    v = evaluate_replacement(LAB_RANGE, [N1, N2], [N1, N2], EVIDENCE)
    assert v["verdict"] == "rejected" and v["regressions"] == ["D3"]


LINE_GROUP = rule("LG", [{"any_line": [{"field": "units", "op": "gte", "value": 1},
                                       {"field": "cpt", "op": "in", "value": ["80053", "80048"]}]}])


@pytest.mark.parametrize("new,replaced,why", [
    ({**N1, "rule_id": "NEW", "when": list(reversed(N1["when"]))}, [N1], "condition-identical"),
    (LAB_RANGE, [], "replaces no rule"),
    (LAB_RANGE, [rule("GONE", [VENIPUNCTURE])], "not active"),
    ({**LAB_RANGE, "predict": {"outcome": "paid"}}, [N1], "same outcome kind"),
    ({**LAB_RANGE, "payer": "*"}, [N1, N2], "cross-payer"),
    (LAB_RANGE, [{**N1, "payer": "NORTHSTAR"}], "cross-payer"),
    ({**LAB_RANGE, "payer": "*"}, [{**N1, "payer": "*"}], "wildcard"),
    # identity survives reordering inside line groups and list values, and `in [x]` == `eq x`
    (rule("NEW", [{"any_line": [{"field": "cpt", "op": "in", "value": ["80053"]}]},
                  {"any_line": [{"field": "cpt", "op": "eq", "value": "36415"}]}]), [N2], "condition-identical"),
    (rule("NEW", [{"any_line": [{"field": "cpt", "op": "in", "value": ["80048", "80053"]},
                                {"field": "units", "op": "gte", "value": 1}]}]), [N2], "condition-identical"),
    (rule("NEW", [{"any_line": [{"field": "units", "op": "gte", "value": 1},
                                {"field": "cpt", "op": "in", "value": ["80053", "80048", "80053"]}]}]), [N2],
     "condition-identical"),
    ({**LAB_RANGE, "when": LAB_RANGE["when"] + [VENIPUNCTURE]}, [N1, N2], "repeats a condition"),
    (rule("NEW", [{"any_line": [{"field": "cpt", "op": "eq", "value": "36415"},
                                {"field": "cpt", "op": "in", "value": ["36415"]}]}]), [N1], "repeats a condition"),
])
def test_replacement_structural_rejections(new, replaced, why):
    active = [N1, N2, LINE_GROUP] + [r for r in replaced if r["rule_id"] in ("N1", "N2") and r not in (N1, N2)]
    v = evaluate_replacement(new, replaced, active, no_d3())
    assert v["verdict"] == "rejected" and why in v["reason"]


def test_store_replace_rules_versions_supersedes_and_cas_guards():
    s = Store(mongomock.MongoClient().db)
    s.bootstrap()
    for r in (N1, N2, LAB_RANGE):
        s.insert_rule(r, status="candidate", iteration=0, evidence=[])
    assert s.promote(["N1", "N2"], expected_version=0, evaluation_id="e1", note="narrow") == 1
    v = s.replace_rules("NEW", ["N1", "N2"], expected_version=1, evaluation_id="e2", note="merge",
                        audit={"run_id": "t", "iteration": 0})
    assert v == 2
    head, active = s.active()
    assert head["version"] == 2 and [r["_id"] for r in active] == ["NEW"]
    assert s.db.rulesets.find_one({"_id": "rs-1"})["rule_ids"] == ["N1", "N2"]  # history is immutable
    rs2 = s.db.rulesets.find_one({"_id": "rs-2"})
    assert rs2["parent"] == "rs-1" and rs2["removed"] == ["N1", "N2"] and rs2["added"] == ["NEW"]
    for rid in ("N1", "N2"):
        doc = s.db.rules.find_one({"_id": rid})
        assert doc["status"] == "superseded" and doc["superseded_by"] == "NEW"
    audit = s.db.audits.find_one({"kind": "consolidation"})
    assert audit["replaces"] == ["N1", "N2"] and audit["version"] == 2 and audit["run_id"] == "t"
    with pytest.raises(ConflictError):  # stale head
        s.replace_rules("NEW", ["N1"], expected_version=1, evaluation_id="e3", note="stale")
    with pytest.raises(ConflictError):  # already superseded
        s.replace_rules("NEW", ["N1"], expected_version=2, evaluation_id="e3", note="gone")
    assert s.active()[0]["version"] == 2


def test_loop_consolidates_even_with_zero_misses(monkeypatch):
    calls = {"consolidate": 0}

    def fake(model, system, user, **kw):
        meta = {"model": "fake", "latency_s": 0, "cached": False}
        payload = json.loads(user.split("\n\nYour previous")[0])
        if system != watcher_mod.CONSOLIDATE_SYSTEM:
            payers = {m["payer"] for m in payload["uncovered_denials"] + payload["false_holds"]}
            return {"rules": [dict(r, rule_id="") for r in ORACLE if r["payer"] in payers or r["payer"] == "*"]}, meta
        if payload["payer"] != "BAYVIEW":
            return {"replacements": []}, meta
        calls["consolidate"] += 1
        if calls["consolidate"] < 2:  # only act in the second learning iteration
            return {"replacements": []}, meta
        target = next(r["rule_id"] for r in payload["active_rules"] if r["description"] == "PR-02-BV")
        late = {"field": "days_to_submit", "op": "gte", "value": 91}
        return {"replacements": [
            {"op": "replace", "replaces": [target], "rule": {"payer": "BAYVIEW", "description": "any delay",
             "when": [{"field": "days_to_submit", "op": "gt", "value": 0}], "predict": {"outcome": "denied", "carc": "29"}}},
            {"op": "replace", "replaces": [target], "rule": {"payer": "BAYVIEW", "description": "timely filing 90d",
             "when": [late], "predict": {"outcome": "denied", "carc": "29"}}},
        ]}, meta

    monkeypatch.setattr(scrubber_mod, "complete_json", fake_scrub_llm)
    monkeypatch.setattr(watcher_mod, "complete_json", fake)
    client = mongomock.MongoClient()
    history = loop.run(Store(client.db), "c", iterations=2, slices=1)
    second = history[1]
    assert second["learning"]["uncovered_denials"] == 0 and second["learning"]["false_holds"] == 0
    cons = second["consolidation"]
    assert "BAYVIEW" in cons["payers"] and cons["promoted"] == 1 and cons["rejected"] == 1
    assert [r["verdict"] for r in cons["replacements"]] == ["rejected", "promote"]
    old = client.db.rules.find_one({"description": "PR-02-BV", "status": "superseded"})
    assert old and old["superseded_by"] == cons["replacements"][1]["rule_id"]
    assert history[2]["metrics"]["rule_count"] == history[1]["metrics"]["rule_count"]
    assert history[2]["metrics"]["heldout"]["balanced_accuracy"] >= history[1]["metrics"]["heldout"]["balanced_accuracy"]


def test_consolidate_prompt_does_not_leak_answer_key():
    for needle in ("80048", "80053", "36415", "96372", "17000", "11102", "82306", "70551", "72148", "PAYER_RULES"):
        assert needle not in watcher_mod.CONSOLIDATE_SYSTEM, needle


def _ruleset_store():
    s = Store(mongomock.MongoClient().db)
    s.bootstrap()
    for r in (N1, N2, LAB_RANGE):
        s.insert_rule(r, status="candidate", iteration=0, evidence=[])
    s.promote(["N1", "N2"], expected_version=0, evaluation_id="e1", note="narrow")
    return s


def test_existing_ruleset_version_is_a_conflict_not_a_crash():
    s = _ruleset_store()
    s.db.rulesets.insert_one({"_id": "rs-2", "version": 2, "rule_ids": [], "parent": "rs-1"})  # left by a lost CAS
    with pytest.raises(ConflictError):
        s.promote(["NEW"], expected_version=1, evaluation_id="e", note="x")
    with pytest.raises(ConflictError):
        s.replace_rules("NEW", ["N1", "N2"], expected_version=1, evaluation_id="e", note="x")
    assert s.active()[0]["version"] == 1


def test_consolidation_rejects_replacing_another_payers_rule(monkeypatch):
    s = _ruleset_store()
    ns = rule("NS1", [{"field": "days_to_submit", "op": "gt", "value": 180}], carc="29", payer="NORTHSTAR")
    s.insert_rule(ns, status="candidate", iteration=0, evidence=[])
    s.promote(["NS1"], expected_version=1, evaluation_id="e2", note="ns")

    def fake(payer, mine, **kw):
        reps = [] if payer != "BAYVIEW" else [{"op": "replace", "replaces": ["NS1"], "rule": {
            "payer": "BAYVIEW", "description": "steal", "when": ns["when"], "predict": ns["predict"]}}]
        return reps, {"model": "fake"}

    monkeypatch.setattr(loop, "consolidate", fake)
    stats = loop.consolidate_rules(s, "r", 0)
    assert stats["promoted"] == 0 and stats["rejected"] == 1
    assert "not active BAYVIEW rules: NS1" in stats["replacements"][0]["reason"]
    assert s.db.rules.find_one({"_id": "NS1"})["status"] == "candidate"


def test_malformed_stored_rules_never_break_prompts_or_export(monkeypatch):
    from demo.export import export
    seen = []
    monkeypatch.setattr(watcher_mod, "complete_json",
                        lambda *a, **k: (seen.append(a) or {"rules": [], "replacements": []}, {"model": "fake"}))
    bad = {"rule_id": "BAD", "replaces": ["N1"], "rejection_reason": "x"}  # no description/when/predict
    watcher_mod.propose([], [], paid_examples=[], denied_examples=[], active_rules=[], lessons=[bad])
    watcher_mod.consolidate("BAYVIEW", [N1], paid_examples=[], denied_examples=[], lessons=[bad])
    assert len(seen) == 2
    db = mongomock.MongoClient().db
    db.rules.insert_one({"_id": "BAD", "created_at": 0})
    assert export(db, "none")["rules"][0]["when"] == []
