"""The watcher can only learn what the rule language can express. This test writes the
hidden payer rules in that language by hand and checks they reproduce every adjudication.
It does NOT feed the watcher: it proves the ceiling is reachable."""
from eval.synthetic_claims import load_claims
from harness.features import features
from harness.payer_sim import VITAMIN_D_COVERED_ICD10
from harness.rules import apply_rules, validate

EM = [str(c) for c in range(99202, 99216)]
PROC_OR_ADMIN = {"any_line": [{"field": "cpt", "op": "in",
                               "value": ["96372", "96373", "96374", "G0008", "G0009", "G0010",
                                         "90460", "90461", "90471", "90472", "90473", "90474"]}]}


def R(rid, payer, when, carc):
    return {"rule_id": rid, "payer": payer, "description": rid, "when": when,
            "predict": {"outcome": "denied", "carc": carc}}


ORACLE = [
    R("PR-01", "NORTHSTAR", [{"any_line": [{"field": "cpt", "op": "in", "value": ["70551", "72148", "74177"]}]},
                             {"field": "prior_auth", "op": "missing"}], "197"),
    R("PR-02-NS", "NORTHSTAR", [{"field": "days_to_submit", "op": "gt", "value": 180}], "29"),
    R("PR-02-BV", "BAYVIEW", [{"field": "days_to_submit", "op": "gt", "value": 90}], "29"),
    R("PR-02-CA", "CASCADE", [{"field": "days_to_submit", "op": "gt", "value": 365}], "29"),
    R("PR-03a", "CASCADE", [{"any_line": [{"field": "cpt", "op": "in", "value": EM},
                                          {"field": "modifiers", "op": "not_contains", "value": "25"}]},
                            PROC_OR_ADMIN], "4"),
    R("PR-03b", "CASCADE", [{"any_line": [{"field": "cpt", "op": "in", "value": EM},
                                          {"field": "modifiers", "op": "not_contains", "value": "25"}]},
                            {"any_line": [{"field": "cpt", "op": "between", "value": [10004, 69990]}]}], "4"),
    R("PR-04", "NORTHSTAR", [{"any_line": [{"field": "cpt", "op": "in", "value": ["97164", "11102", "17000"]},
                                           {"field": "modifiers", "op": "contains", "value": "59"}]}], "4"),
    R("PR-05", "BAYVIEW", [{"any_line": [{"field": "cpt", "op": "eq", "value": "36415"}]},
                           {"any_line": [{"field": "cpt", "op": "between", "value": [80047, 89398]}]}], "97"),
    R("PR-06", "NORTHSTAR", [{"any_line": [{"field": "cpt", "op": "between", "value": [99381, 99397]}]},
                             {"field": "days_since_last_preventive", "op": "lt", "value": 365}], "119"),
    R("PR-07", "NORTHSTAR", [{"any_line": [{"field": "cpt", "op": "in",
                                            "value": ["97110", "97112", "97140", "97530", "97161", "97162", "97163", "97164"]}]},
                             {"field": "context.visit_count_ytd", "op": "gte", "value": 20}], "119"),
    R("PR-08", "CASCADE", [{"any_line": [{"field": "cpt", "op": "eq", "value": "82306"},
                                         {"field": "dx_codes", "op": "contains_none",
                                          "value": sorted(VITAMIN_D_COVERED_ICD10)}]}], "167"),
    R("PR-09a", "BAYVIEW", [{"field": "pos", "op": "eq", "value": "02"},
                            {"any_line": [{"field": "cpt", "op": "in", "value": EM}]}], "5"),
    R("PR-09b", "BAYVIEW", [{"field": "pos", "op": "eq", "value": "10"},
                            {"any_line": [{"field": "cpt", "op": "in", "value": EM},
                                          {"field": "modifiers", "op": "not_contains", "value": "95"}]}], "4"),
    R("PR-10", "BAYVIEW", [{"field": "in_network", "op": "eq", "value": False},
                           {"field": "pos", "op": "ne", "value": "23"}], "242"),
    R("PR-11", "*", [{"any_line": [{"field": "dx_pointer_count", "op": "eq", "value": 0}]}], "16"),
    R("PR-12", "*", [{"any_line": [{"field": "charge", "op": "lte", "value": 0}]}], "16"),
]


def test_oracle_rules_are_valid():
    for r in ORACLE:
        validate(r)


def test_rule_language_can_express_every_hidden_rule():
    wrong = []
    for rec in load_claims():
        gt = rec["ground_truth"]
        hit = apply_rules(ORACLE, features(rec))
        if gt["outcome"] == "paid":
            if hit:
                wrong.append((rec["id"], "fired on paid claim", hit["rule_id"]))
        elif not hit:
            wrong.append((rec["id"], "missed denial", gt["carc"]))
        elif hit["carc"] != gt["carc"] and not (gt["carc"] in ("4", "5") and hit["carc"] in ("4", "5")):
            wrong.append((rec["id"], f"carc {hit['carc']} != {gt['carc']}", hit["rule_id"]))
    assert not wrong, wrong
