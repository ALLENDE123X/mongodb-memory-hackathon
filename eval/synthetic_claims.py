"""Synthetic claim dataset for the revenue-cycle harness.

Each record is {id, payer, category, split, ground_truth, claim, context}:
  claim   - an 837P professional claim request body using the Stedi / Change Healthcare JSON
            field names (POST /change/medicalnetwork/professionalclaims/v3/submission)
  context - eligibility / history facts a billing team has that are not on the 837
            (submission_date, provider_in_network, visit_count_ytd, last_preventive_dos, ...)

All data is fictional. `ground_truth`, `split` and `category` are answer-key / eval metadata and
must never be shown to the scrubber; use `public_view()` to build scrubber inputs.

Validate with:  .venv/bin/python -m eval.synthetic_claims
"""
from __future__ import annotations

import copy
import json
import re
import sys
from collections import Counter
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).with_name("synthetic_claims.json")

PAYERS = ("NORTHSTAR", "CASCADE", "BAYVIEW")
CATEGORIES = ("imaging", "e_and_m", "lab", "therapy", "preventive", "procedure", "telehealth")
OUTCOMES = ("paid", "denied")
SPLITS = ("train", "heldout")
RULE_IDS = tuple(f"PR-{i:02d}" for i in range(1, 13))
HIDDEN_KEYS = ("ground_truth", "split", "category")

REQUIRED = ("id", "payer", "category", "split", "ground_truth", "claim", "context")
PUBLIC_KEYS = ("id", "payer", "claim", "context")

# Payer routing ids (fictional) and the real X12 claim filing indicator for each plan type:
# CI = Commercial Insurance Co., 16 = HMO Medicare Risk (Medicare Advantage), MC = Medicaid.
PAYER_IDS = {"NORTHSTAR": "SYNNS", "CASCADE": "SYNCA", "BAYVIEW": "SYNBV"}
CLAIM_FILING_CODES = {"NORTHSTAR": "CI", "CASCADE": "16", "BAYVIEW": "MC"}

CLAIM_INFO_REQUIRED = (
    "claimFilingCode", "patientControlNumber", "claimChargeAmount", "placeOfServiceCode",
    "claimFrequencyCode", "signatureIndicator", "planParticipationCode",
    "benefitsAssignmentCertificationIndicator", "releaseInformationCode",
    "healthCareCodeInformation", "serviceLines",
)
PROF_SERVICE_REQUIRED = (
    "procedureIdentifier", "procedureCode", "lineItemChargeAmount", "measurementUnit",
    "serviceUnitCount", "compositeDiagnosisCodePointers",
)
CONTEXT_REQUIRED = ("submission_date", "provider_in_network", "eligibility_active", "plan_type")
_AMOUNT = re.compile(r"\d+\.\d{2}")
_YMD = re.compile(r"\d{8}")


@lru_cache(maxsize=1)
def _all_claims() -> tuple[dict, ...]:
    with DATA_PATH.open() as f:
        return tuple(json.load(f))


def load_claims(split: str | None = None) -> list[dict]:
    """Return deep copies of all claim records, optionally filtered to 'train' or 'heldout'."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS} or None, got {split!r}")
    return [copy.deepcopy(c) for c in _all_claims() if split is None or c["split"] == split]


def get_claim(claim_id: str) -> dict:
    for c in _all_claims():
        if c["id"] == claim_id:
            return copy.deepcopy(c)
    raise KeyError(claim_id)


def public_view(claim: dict) -> dict:
    """The record as the scrubber sees it: id, payer, the 837P `claim` body and `context`.

    No ground_truth, no split, no category."""
    return {k: copy.deepcopy(claim[k]) for k in PUBLIC_KEYS}


def _ymd_to_date(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:]))


# ---------------------------------------------------------------- validation

def _validate_837(cid: str, payer: str, clm: dict, ctx: dict) -> list[str]:
    errs: list[str] = []
    for k in ("controlNumber", "tradingPartnerServiceId", "tradingPartnerName", "submitter",
              "receiver", "subscriber", "billing", "claimInformation"):
        if k not in clm:
            errs.append(f"{cid}: claim missing {k}")
    if errs:
        return errs
    if clm["tradingPartnerServiceId"] != PAYER_IDS[payer] or clm["tradingPartnerName"] != payer \
            or clm["receiver"].get("organizationName") != payer:
        errs.append(f"{cid}: payer routing (tradingPartner*/receiver) does not match {payer}")
    sub = clm["subscriber"]
    if not (re.fullmatch(r"SYN-\d{6}", sub.get("memberId", "")) and sub.get("gender") in ("F", "M")
            and _YMD.fullmatch(sub.get("dateOfBirth", "")) and sub.get("lastName")):
        errs.append(f"{cid}: bad subscriber block")
    bil = clm["billing"]
    if not (re.fullmatch(r"99\d{8}", bil.get("npi", "")) and bil.get("organizationName")
            and re.fullmatch(r"[0-9A-Z]{10}", bil.get("taxonomyCode", ""))):
        errs.append(f"{cid}: bad billing provider block")

    ci = clm["claimInformation"]
    missing = [k for k in CLAIM_INFO_REQUIRED if k not in ci]
    if missing:
        return errs + [f"{cid}: claimInformation missing {missing}"]
    if ci["patientControlNumber"] != cid:
        errs.append(f"{cid}: patientControlNumber != id")
    if ci["claimFilingCode"] != CLAIM_FILING_CODES[payer]:
        errs.append(f"{cid}: claimFilingCode {ci['claimFilingCode']} wrong for {payer}")
    if ci["claimFrequencyCode"] != "1":
        errs.append(f"{cid}: claimFrequencyCode must be 1 (original)")
    if not re.fullmatch(r"\d{2}", ci["placeOfServiceCode"]):
        errs.append(f"{cid}: bad placeOfServiceCode")

    dx = ci["healthCareCodeInformation"]
    if not dx:
        errs.append(f"{cid}: no diagnoses")
    for k, d in enumerate(dx):
        want = "ABK" if k == 0 else "ABF"
        if d.get("diagnosisTypeCode") != want:
            errs.append(f"{cid}: diagnosis {k + 1} type must be {want}")
        if not re.fullmatch(r"[A-Z]\d[0-9A-Z]{1,5}", d.get("diagnosisCode", "")):
            errs.append(f"{cid}: diagnosis {k + 1} must be ICD-10-CM without a decimal point")

    lines = ci["serviceLines"]
    if not lines:
        errs.append(f"{cid}: no service lines")
    total = Decimal("0")
    sub_date = date.fromisoformat(ctx["submission_date"])
    for i, sl in enumerate(lines, 1):
        ps = sl.get("professionalService", {})
        miss = [k for k in PROF_SERVICE_REQUIRED if k not in ps]
        if miss:
            errs.append(f"{cid}: line {i} missing {miss}")
            continue
        if sl.get("providerControlNumber") != f"{cid}-{i}":
            errs.append(f"{cid}: line {i} providerControlNumber")
        if not _YMD.fullmatch(sl.get("serviceDate", "")):
            errs.append(f"{cid}: line {i} serviceDate must be YYYYMMDD")
        else:
            dos = _ymd_to_date(sl["serviceDate"])
            if dos.year != 2026 or sub_date.year != 2026 or sub_date < dos:
                errs.append(f"{cid}: line {i} dates must be in 2026 with submission >= DOS")
        if ps["procedureIdentifier"] != "HC" or ps["measurementUnit"] != "UN":
            errs.append(f"{cid}: line {i} must use HC procedure codes and UN units")
        if not _AMOUNT.fullmatch(ps["lineItemChargeAmount"]):
            errs.append(f"{cid}: line {i} charge must be a 2-decimal string")
        else:
            total += Decimal(ps["lineItemChargeAmount"])
        if not ps["serviceUnitCount"].isdigit():
            errs.append(f"{cid}: line {i} serviceUnitCount must be an integer string")
        ptrs = ps["compositeDiagnosisCodePointers"].get("diagnosisCodePointers")
        if not isinstance(ptrs, list) or any(not (str(p).isdigit() and 1 <= int(p) <= len(dx)) for p in ptrs):
            errs.append(f"{cid}: line {i} dx pointer out of range")
        if len(ptrs or []) > 4:
            errs.append(f"{cid}: line {i} has more than 4 dx pointers")
    if not _AMOUNT.fullmatch(ci["claimChargeAmount"]) or Decimal(ci["claimChargeAmount"]) != total:
        errs.append(f"{cid}: claimChargeAmount {ci['claimChargeAmount']} != sum of lines {total}")
    pa = ci.get("claimSupplementalInformation", {}).get("priorAuthorizationNumber")
    if pa is not None and not pa:
        errs.append(f"{cid}: empty priorAuthorizationNumber")
    return errs


def _validate(claims: list[dict]) -> list[str]:
    errs: list[str] = []
    if len(claims) != 46:
        errs.append(f"expected 46 claims, got {len(claims)}")
    ids = [c.get("id") for c in claims]
    if len(set(ids)) != len(ids):
        errs.append(f"duplicate ids: {[i for i, n in Counter(ids).items() if n > 1]}")

    for c in claims:
        cid = c.get("id", "?")
        missing = [k for k in REQUIRED if k not in c]
        if missing:
            errs.append(f"{cid}: missing {missing}")
            continue
        if set(c) != set(REQUIRED):
            errs.append(f"{cid}: unexpected top-level keys {sorted(set(c) - set(REQUIRED))}")
        if not re.fullmatch(r"CLM-\d{3}", cid):
            errs.append(f"{cid}: bad id format")
        if c["payer"] not in PAYERS:
            errs.append(f"{cid}: bad payer {c['payer']}")
            continue
        if c["category"] not in CATEGORIES:
            errs.append(f"{cid}: bad category {c['category']}")
        if c["split"] not in SPLITS:
            errs.append(f"{cid}: bad split {c['split']}")
        ctx = c["context"]
        cmiss = [k for k in CONTEXT_REQUIRED if k not in ctx]
        if cmiss:
            errs.append(f"{cid}: context missing {cmiss}")
            continue
        if not isinstance(ctx["provider_in_network"], bool) or not isinstance(ctx["eligibility_active"], bool):
            errs.append(f"{cid}: context booleans")
        try:
            date.fromisoformat(ctx["submission_date"])
        except ValueError:
            errs.append(f"{cid}: unparseable submission_date")
            continue
        errs.extend(_validate_837(cid, c["payer"], c["claim"], ctx))

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
        pv = public_view(c)
        if set(pv) != set(PUBLIC_KEYS) or "ground_truth" in json.dumps(pv):
            errs.append(f"{cid}: public_view leaks hidden keys")

    splits = Counter(c.get("split") for c in claims)
    if splits["train"] != 24 or splits["heldout"] != 22:
        errs.append(f"split counts {dict(splits)} != train 24 / heldout 22")
    held = Counter(c["ground_truth"].get("outcome") for c in claims if c.get("split") == "heldout")
    if held["paid"] != held["denied"]:
        errs.append(f"heldout must be balanced paid/denied, got {dict(held)}")
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

    print("\nPaid traps by split (paid claims; each looks risky but is clean for its payer)")
    for split in SPLITS:
        traps = [c["id"] for c in claims if c["split"] == split and c["ground_truth"]["outcome"] == "paid"]
        print(f"{split:<8} {len(traps):>3}  {', '.join(traps)}")

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
