"""Scoring that an always-deny or always-pay scrubber cannot game."""
from __future__ import annotations


def score(rows: list[dict]) -> dict:
    """rows: [{'predicted': {'predicted_outcome', 'carc'}, 'observed': {'outcome', 'carcs'}}]"""
    paid = [r for r in rows if r["observed"]["outcome"] == "paid"]
    denied = [r for r in rows if r["observed"]["outcome"] == "denied"]
    paid_ok = sum(r["predicted"]["predicted_outcome"] == "paid" for r in paid)
    denied_caught = [r for r in denied if r["predicted"]["predicted_outcome"] == "denied"]
    denied_exact = sum(r["predicted"].get("carc") in r["observed"]["carcs"] for r in denied_caught)

    def rate(a: int, b: int) -> float | None:
        return round(a / b, 3) if b else None

    specificity = rate(paid_ok, len(paid))        # paid claims correctly let through
    sensitivity = rate(len(denied_caught), len(denied))  # denials caught before submission
    balanced = round((specificity + sensitivity) / 2, 3) if paid and denied else None
    correct = paid_ok + denied_exact
    return {
        "n": len(rows), "n_paid": len(paid), "n_denied": len(denied),
        "balanced_accuracy": balanced,
        "accuracy_with_reason": rate(correct, len(rows)),   # paid right, or denial caught with the right CARC
        "denials_caught": len(denied_caught), "denials_caught_right_reason": denied_exact,
        "unnecessary_holds": len(paid) - paid_ok, "unnecessary_hold_rate": rate(len(paid) - paid_ok, len(paid)),
        "missed_denials": len(denied) - len(denied_caught),
    }
