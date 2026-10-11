"""Synthetic claim dataset for the revenue-cycle harness.

All data is fictional. `ground_truth` and `split` are the answer key and must never be
shown to the scrubber; use `public_view()` to build scrubber inputs.

Validate with:  .venv/bin/python -m eval.synthetic_claims
"""
from __future__ import annotations

import copy
import json
import re
import sys
from collections import Counter
from datetime import date
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).with_name("synthetic_claims.json")

PAYERS = ("NORTHSTAR", "CASCADE", "BAYVIEW")
CATEGORIES = ("imaging", "e_and_m", "lab", "therapy", "preventive", "procedure", "telehealth")
OUTCOMES = ("paid", "denied")
SPLITS = ("train", "heldout")
RULE_IDS = tuple(f"PR-{i:02d}" for i in range(1, 13))
HIDDEN_KEYS = ("ground_truth", "split")

REQUIRED = (
    "id", "payer", "claim_type", "category", "patient", "provider", "date_of_service",
    "submission_date", "place_of_service", "diagnoses", "lines", "prior_auth_number",
    "context", "ground_truth", "split",
)


@lru_cache(maxsize=1)
def _all_claims() -> tuple[dict, ...]:
    with DATA_PATH.open() as f:
        return tuple(json.load(f))


def load_claims(split: str | None = None) -> list[dict]:
    """Return deep copies of all claims, optionally filtered to 'train' or 'heldout'."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS} or None, got {split!r}")
    return [copy.deepcopy(c) for c in _all_claims() if split is None or c["split"] == split]


def get_claim(claim_id: str) -> dict:
    for c in _all_claims():
        if c["id"] == claim_id:
            return copy.deepcopy(c)
    raise KeyError(claim_id)


def public_view(claim: dict) -> dict:
    """The claim as the scrubber sees it: no ground_truth, no split."""
    return {k: copy.deepcopy(v) for k, v in claim.items() if k not in HIDDEN_KEYS}


# ---------------------------------------------------------------- validation

def _validate(claims: list[dict]) -> list[str]:
    errs: list[str] = []
    if len(claims) != 40:
        errs.append(f"expected 40 claims, got {len(claims)}")
    ids = [c.get("id") for c in claims]
    if len(set(ids)) != len(ids):
        errs.append(f"duplicate ids: {[i for i, n in Counter(ids).items() if n > 1]}")

    for c in claims:
        cid = c.get("id", "?")
        missing = [k for k in REQUIRED if k not in c]
        if missing:
            errs.append(f"{cid}: missing {missing}")
            continue
        if not re.fullmatch(r"CLM-\d{3}", cid):
            errs.append(f"{cid}: bad id format")
        if c["payer"] not in PAYERS:
            errs.append(f"{cid}: bad payer {c['payer']}")
        if c["category"] not in CATEGORIES:
            errs.append(f"{cid}: bad category {c['category']}")
        if c["claim_type"] != "professional":
            errs.append(f"{cid}: claim_type must be professional")
        if c["split"] not in SPLITS:
            errs.append(f"{cid}: bad split {c['split']}")
        p, pr = c["patient"], c["provider"]
        if not (isinstance(p.get("age"), int) and p.get("sex") in ("F", "M")
                and re.fullmatch(r"SYN-\d{6}", p.get("member_id", ""))):
            errs.append(f"{cid}: bad patient block")
        if not (re.fullmatch(r"99\d{8}", pr.get("npi", "")) and isinstance(pr.get("in_network"), bool)
                and pr.get("specialty")):
            errs.append(f"{cid}: bad provider block")
        try:
            dos = date.fromisoformat(c["date_of_service"])
            sub = date.fromisoformat(c["submission_date"])
            if dos.year != 2026 or sub.year != 2026 or sub < dos:
                errs.append(f"{cid}: dates must be in 2026 with submission >= DOS")
        except ValueError:
            errs.append(f"{cid}: unparseable dates")
        if not c["diagnoses"]:
            errs.append(f"{cid}: no diagnoses")
        if not c["lines"]:
            errs.append(f"{cid}: no lines")
        for i, ln in enumerate(c["lines"], 1):
            if ln.get("line") != i:
                errs.append(f"{cid}: line numbering")
            if any(not (1 <= d <= len(c["diagnoses"])) for d in ln.get("dx_pointers", [])):
                errs.append(f"{cid}: line {i} dx pointer out of range")
            if not isinstance(ln.get("charge_cents"), int) or not isinstance(ln.get("units"), int):
                errs.append(f"{cid}: line {i} charge/units must be int")
        gt = c["ground_truth"]
        if gt.get("outcome") not in OUTCOMES:
            errs.append(f"{cid}: bad outcome")
        elif gt["outcome"] == "denied":
            if not gt.get("carc"):
                errs.append(f"{cid}: denied without carc")
            if not gt.get("violated_rules"):
                errs.append(f"{cid}: denied without violated_rules")
        else:
            if gt.get("carc") is not None:
                errs.append(f"{cid}: paid claim must have carc null")
            if gt.get("violated_rules"):
                errs.append(f"{cid}: paid claim lists violated rules")
        for r in gt.get("violated_rules", []):
            if r not in RULE_IDS:
                errs.append(f"{cid}: unknown rule {r}")
        if not gt.get("reason"):
            errs.append(f"{cid}: empty reason")
        if any(k in public_view(c) for k in HIDDEN_KEYS):
            errs.append(f"{cid}: public_view leaks hidden keys")

    splits = Counter(c.get("split") for c in claims)
    if splits["train"] != 24 or splits["heldout"] != 16:
        errs.append(f"split counts {dict(splits)} != train 24 / heldout 16")
    cats = {c.get("category") for c in claims}
    if set(CATEGORIES) - cats:
        errs.append(f"categories not represented: {set(CATEGORIES) - cats}")
    for r in RULE_IDS:
        in_splits = {c["split"] for c in claims if r in c["ground_truth"].get("violated_rules", [])}
        if in_splits != set(SPLITS):
            errs.append(f"{r} must appear in both splits, found in {sorted(in_splits)}")
    return errs


def _summary(claims: list[dict]) -> None:
    print("\nOutcome x payer x split")
    print(f"{'payer':<10} {'split':<8} {'paid':>5} {'denied':>7} {'total':>6}")
    for payer in PAYERS:
        for split in SPLITS:
            sub = [c for c in claims if c["payer"] == payer and c["split"] == split]
            n_paid = sum(c["ground_truth"]["outcome"] == "paid" for c in sub)
            print(f"{payer:<10} {split:<8} {n_paid:>5} {len(sub) - n_paid:>7} {len(sub):>6}")
    for split in SPLITS:
        sub = [c for c in claims if c["split"] == split]
        n_paid = sum(c["ground_truth"]["outcome"] == "paid" for c in sub)
        print(f"{'ALL':<10} {split:<8} {n_paid:>5} {len(sub) - n_paid:>7} {len(sub):>6}")
    n_paid = sum(c["ground_truth"]["outcome"] == "paid" for c in claims)
    print(f"{'ALL':<10} {'all':<8} {n_paid:>5} {len(claims) - n_paid:>7} {len(claims):>6}")

    print("\nRule coverage")
    print(f"{'rule':<6} {'carc':<6} {'train':>6} {'heldout':>8}  claims")
    for r in RULE_IDS:
        hits = [c for c in claims if r in c["ground_truth"]["violated_rules"]]
        carcs = "/".join(sorted({c["ground_truth"]["carc"] for c in hits}))
        tr = sum(c["split"] == "train" for c in hits)
        print(f"{r:<6} {carcs:<6} {tr:>6} {len(hits) - tr:>8}  {', '.join(c['id'] for c in hits)}")

    print("\nCategory counts:", dict(Counter(c["category"] for c in claims)))


def main() -> int:
    claims = load_claims()
    errs = _validate(claims)
    _summary(claims)
    if errs:
        print(f"\nFAIL: {len(errs)} problem(s)")
        for e in errs:
            print("  -", e)
        return 1
    print("\nOK: dataset valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
