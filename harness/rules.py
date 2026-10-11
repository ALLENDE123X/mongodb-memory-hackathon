"""Executable payer rules.

The watcher proposes rules as small JSON condition trees instead of free text, so the
gate can replay them deterministically against known outcomes without an LLM.

A rule:
    {
      "rule_id": "R-NORTHSTAR-0001",
      "payer": "NORTHSTAR",                     # or "*" for every payer
      "description": "NORTHSTAR requires prior auth for advanced imaging",
      "when": [ <condition>, ... ],             # all must hold
      "predict": {"outcome": "denied", "carc": "197"}
    }

Conditions read from the flattened claim features (see `harness.features`):
    {"field": "days_to_submit", "op": "gt", "value": 180}
    {"field": "prior_auth", "op": "missing"}
    {"field": "dx_codes", "op": "none_startswith", "value": ["E55", "N18"]}
    {"any_line": [{"field": "cpt", "op": "in", "value": ["72148"]},
                  {"field": "modifiers", "op": "contains_none", "value": ["25"]}]}
"""
from __future__ import annotations

from typing import Any

# Fields a rule may reference. Identifiers (member ids, NPIs, control numbers, exact
# dates) are deliberately absent so a rule cannot memorise individual claims.
CLAIM_FIELDS = {
    "payer", "pos", "claim_filing_code", "days_to_submit", "prior_auth", "dx_codes",
    "in_network", "line_count", "total_charge", "cpts", "days_since_last_preventive",
}
CONTEXT_PREFIX = "context."
LINE_FIELDS = {"cpt", "modifiers", "units", "charge", "dx_pointer_count", "dx_codes"}

OPS = {
    "eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "missing", "present",
    "contains", "not_contains", "contains_any", "contains_none",
    "startswith_any", "none_startswith", "between",
}


class RuleError(ValueError):
    pass


def _get(obj: dict, field: str) -> Any:
    if field.startswith(CONTEXT_PREFIX):
        return (obj.get("context") or {}).get(field[len(CONTEXT_PREFIX):])
    return obj.get(field)


def _as_list(v: Any) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def _apply(op: str, actual: Any, expected: Any) -> bool:
    if op == "missing":
        return actual in (None, "", [], {})
    if op == "present":
        return actual not in (None, "", [], {})
    if actual is None:
        return False
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "in":
        return actual in _as_list(expected)
    if op == "not_in":
        return actual not in _as_list(expected)
    if op == "between":  # inclusive numeric range, e.g. CPT 80047-89398
        try:
            return float(expected[0]) <= float(actual) <= float(expected[1])
        except (TypeError, ValueError, IndexError):
            return False
    if op in ("gt", "gte", "lt", "lte"):
        try:
            a, e = float(actual), float(expected)
        except (TypeError, ValueError):
            return False
        return {"gt": a > e, "gte": a >= e, "lt": a < e, "lte": a <= e}[op]
    items = [str(x) for x in _as_list(actual)]
    exp = [str(x) for x in _as_list(expected)]
    if op == "contains":
        return exp[0] in items
    if op == "not_contains":
        return exp[0] not in items
    if op == "contains_any":
        return any(e in items for e in exp)
    if op == "contains_none":
        return not any(e in items for e in exp)
    if op == "startswith_any":
        return any(i.startswith(e) for i in items for e in exp)
    if op == "none_startswith":
        return not any(i.startswith(e) for i in items for e in exp)
    raise RuleError(f"unknown op {op!r}")


def _cond(cond: dict, feats: dict) -> bool:
    if "any_line" in cond:
        return any(all(_cond(c, line) for c in cond["any_line"]) for line in feats.get("lines", []))
    if "all_lines" in cond:
        return all(all(_cond(c, line) for c in cond["all_lines"]) for line in feats.get("lines", []))
    return _apply(cond["op"], _get(feats, cond["field"]), cond.get("value"))


def fires(rule: dict, feats: dict) -> bool:
    if rule.get("payer") not in ("*", feats.get("payer")):
        return False
    return all(_cond(c, feats) for c in rule["when"])


def validate(rule: dict) -> None:
    """Structural guard: raises RuleError if the rule is malformed or out of bounds."""
    for key in ("rule_id", "payer", "description", "when", "predict"):
        if key not in rule:
            raise RuleError(f"missing {key}")
    if not rule["when"]:
        raise RuleError("rule has no conditions")
    if rule["predict"].get("outcome") != "denied" or not rule["predict"].get("carc"):
        raise RuleError("predict must be {'outcome': 'denied', 'carc': <code>}")

    def check(cond: dict, line_level: bool) -> None:
        if "any_line" in cond or "all_lines" in cond:
            if line_level:
                raise RuleError("nested line conditions are not allowed")
            for c in cond.get("any_line") or cond.get("all_lines"):
                check(c, True)
            return
        field, op = cond.get("field"), cond.get("op")
        allowed = LINE_FIELDS if line_level else CLAIM_FIELDS
        if not (field in allowed or (not line_level and str(field).startswith(CONTEXT_PREFIX))):
            raise RuleError(f"field {field!r} not allowed{' on a line' if line_level else ''}")
        if op not in OPS:
            raise RuleError(f"unknown op {op!r}")

    for c in rule["when"]:
        check(c, False)


def specificity(rule: dict) -> int:
    def count(c: dict) -> int:
        inner = c.get("any_line") or c.get("all_lines")
        return sum(count(x) for x in inner) if inner else 1
    return sum(count(c) for c in rule["when"]) + (0 if rule["payer"] == "*" else 1)


def apply_rules(rules: list[dict], feats: dict) -> dict | None:
    """Most specific firing rule wins. Returns {'rule_id', 'carc'} or None."""
    hits = [r for r in rules if fires(r, feats)]
    if not hits:
        return None
    best = max(hits, key=specificity)
    return {"rule_id": best["rule_id"], "carc": best["predict"]["carc"], "description": best["description"]}
