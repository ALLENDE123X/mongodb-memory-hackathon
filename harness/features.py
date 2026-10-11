"""Flatten an 837P claim record into the fields rules and the scrubber reason over."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from harness.payer_sim import PAYER_BY_ID


def _ymd(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def features(record: dict) -> dict:
    claim, ctx = record["claim"], record.get("context") or {}
    ci = claim["claimInformation"]
    dx = [d.get("diagnosisCode", "") for d in ci.get("healthCareCodeInformation", [])]
    lines = []
    for sl in ci.get("serviceLines", []):
        ps = sl.get("professionalService", {})
        ptrs = (ps.get("compositeDiagnosisCodePointers") or {}).get("diagnosisCodePointers") or []
        valid = [int(p) for p in ptrs if str(p).isdigit() and 1 <= int(p) <= len(dx)]
        lines.append({
            "cpt": ps.get("procedureCode", ""),
            "modifiers": list(ps.get("procedureModifiers") or []),
            "units": float(ps.get("serviceUnitCount", "1")),
            "charge": float(Decimal(ps.get("lineItemChargeAmount", "0"))),
            "dx_pointer_count": len(valid),
            "dx_codes": [dx[p - 1] for p in valid],
            "service_date": sl.get("serviceDate"),
        })
    first_dos = min(_ymd(ln["service_date"]) for ln in lines) if lines else None
    submitted = date.fromisoformat(ctx["submission_date"]) if ctx.get("submission_date") else None
    last_prev = date.fromisoformat(ctx["last_preventive_dos"]) if ctx.get("last_preventive_dos") else None
    return {
        "payer": PAYER_BY_ID.get(claim.get("tradingPartnerServiceId"), record.get("payer")),
        "pos": ci.get("placeOfServiceCode"),
        "claim_filing_code": ci.get("claimFilingCode"),
        "prior_auth": (ci.get("claimSupplementalInformation") or {}).get("priorAuthorizationNumber"),
        "dx_codes": dx,
        "in_network": ctx.get("provider_in_network"),
        "days_to_submit": (submitted - first_dos).days if submitted and first_dos else None,
        "days_since_last_preventive": (first_dos - last_prev).days if last_prev and first_dos else None,
        "line_count": len(lines),
        "total_charge": float(Decimal(ci.get("claimChargeAmount", "0"))),
        "cpts": [ln["cpt"] for ln in lines],
        "lines": lines,
        "context": ctx,
    }
