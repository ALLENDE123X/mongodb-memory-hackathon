import mongomock
import pytest

from harness.gate import evaluate
from harness.rules import RuleError, apply_rules, fires, validate
from memory.store import ConflictError, Store

AUTH_RULE = {
    "rule_id": "R-NS-1", "payer": "NORTHSTAR",
    "description": "NORTHSTAR requires prior auth for advanced imaging",
    "when": [{"any_line": [{"field": "cpt", "op": "in", "value": ["70551", "72148", "74177"]}]},
             {"field": "prior_auth", "op": "missing"}],
    "predict": {"outcome": "denied", "carc": "197"},
}
ALL_IMAGING = {
    "rule_id": "R-NS-2", "payer": "NORTHSTAR", "description": "deny all imaging",
    "when": [{"any_line": [{"field": "cpt", "op": "in", "value": ["70551", "72148", "74177"]}]}],
    "predict": {"outcome": "denied", "carc": "197"},
}


def feats(payer, cpt, auth=None, **ctx):
    return {"payer": payer, "prior_auth": auth, "context": ctx,
            "lines": [{"cpt": cpt, "modifiers": [], "units": 1, "charge": 900.0, "dx_pointer_count": 1}]}


EVIDENCE = [
    {"claim_id": "A", "features": feats("NORTHSTAR", "72148"), "observed": {"outcome": "denied", "carcs": ["197"]}},
    {"claim_id": "B", "features": feats("NORTHSTAR", "72148", auth="PA123"), "observed": {"outcome": "paid", "carcs": []}},
    {"claim_id": "C", "features": feats("CASCADE", "72148"), "observed": {"outcome": "paid", "carcs": []}},
    {"claim_id": "D", "features": feats("NORTHSTAR", "99213"), "observed": {"outcome": "paid", "carcs": []}},
]


def test_rule_fires_only_for_its_payer_and_conditions():
    assert fires(AUTH_RULE, EVIDENCE[0]["features"])
    assert not fires(AUTH_RULE, EVIDENCE[1]["features"])
    assert not fires(AUTH_RULE, EVIDENCE[2]["features"])
    assert apply_rules([AUTH_RULE], EVIDENCE[0]["features"])["carc"] == "197"


def test_rules_cannot_reference_identifiers():
    bad = {**AUTH_RULE, "when": [{"field": "member_id", "op": "eq", "value": "SYN-1"}]}
    with pytest.raises(RuleError):
        validate(bad)


def test_gate_promotes_precise_rule_and_rejects_overbroad_one():
    good = evaluate(AUTH_RULE, [], EVIDENCE)
    assert good["verdict"] == "promote" and good["fixes"] == ["A"]
    bad = evaluate(ALL_IMAGING, [], EVIDENCE)
    assert bad["verdict"] == "rejected" and bad["regressions"] == ["B"]


def test_gate_rejects_redundant_rule():
    again = {**AUTH_RULE, "rule_id": "R-NS-3"}
    assert evaluate(again, [AUTH_RULE], EVIDENCE)["verdict"] == "rejected"


def test_store_promotion_is_versioned_and_cas_guarded():
    s = Store(mongomock.MongoClient().db)
    s.bootstrap()
    s.bootstrap()  # idempotent
    head, rules = s.active()
    assert head["version"] == 0 and rules == []
    s.insert_rule(AUTH_RULE, status="candidate", iteration=1, evidence=["ep-1"])
    ev = s.insert_evaluation({"rule_id": AUTH_RULE["rule_id"], "verdict": "promote"})
    assert s.promote([AUTH_RULE["rule_id"]], expected_version=0, evaluation_id=ev, note="t") == 1
    head, rules = s.active()
    assert head["version"] == 1 and [r["_id"] for r in rules] == ["R-NS-1"]
    with pytest.raises(ConflictError):
        s.promote([], expected_version=0, evaluation_id=ev, note="stale")


def test_restart_recovers_rules_from_db_alone():
    client = mongomock.MongoClient()
    s = Store(client.db)
    s.bootstrap()
    s.insert_rule(AUTH_RULE, status="promoted", iteration=1, evidence=[])
    s.promote(["R-NS-1"], expected_version=0, evaluation_id="e", note="t")
    fresh = Store(client.db)  # new process, same database
    head, rules = fresh.active()
    assert head["version"] == 1 and rules[0]["predict"]["carc"] == "197"


CASCADE_IMAGING_OK = {
    "rule_id": "R-CA-1", "payer": "CASCADE", "description": "CASCADE does not require auth for imaging",
    "when": [{"any_line": [{"field": "cpt", "op": "in", "value": ["70551", "72148", "74177"]}]}],
    "predict": {"outcome": "paid"},
}


def test_exception_rule_promoted_only_for_false_holds_and_never_over_a_denial():
    held = dict(EVIDENCE[2], predicted={"predicted_outcome": "denied"})
    ok = evaluate(CASCADE_IMAGING_OK, [], [EVIDENCE[0], EVIDENCE[1], held])
    assert ok["verdict"] == "promote" and ok["fixes"] == ["C"]
    everything_ns = {**CASCADE_IMAGING_OK, "rule_id": "R-NS-9", "payer": "NORTHSTAR",
                     "description": "NORTHSTAR pays all imaging"}
    bad = evaluate(everything_ns, [], [EVIDENCE[0], dict(EVIDENCE[1], predicted={"predicted_outcome": "denied"})])
    assert bad["verdict"] == "rejected" and bad["regressions"] == ["A"]
    assert apply_rules([CASCADE_IMAGING_OK], EVIDENCE[2]["features"], "paid")["rule_id"] == "R-CA-1"
    assert apply_rules([CASCADE_IMAGING_OK], EVIDENCE[2]["features"]) is None
