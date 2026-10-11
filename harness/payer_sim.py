"""Deterministic payer adjudication simulator.

`adjudicate(record)` takes a stored claim record ({id, payer, claim, context, ...}), reads only
the 837P `claim` body (Stedi / Change Healthcare JSON field names) and the `context` block
(the payer's own eligibility / accumulator / network data), applies each payer's adjudication
policy, and returns an 835 ERA in the shape of Stedi's 835 report JSON
(GET /2024-04-01/reports/{transactionId}/835): transactions[].detailInfo[].paymentInfo[] with
claimPaymentInfo, serviceLines[].servicePaymentInformation, serviceLines[].serviceAdjustments
and serviceLines[].healthCareCheckRemarkCodes.

`outcome_summary(remit)` collapses an 835 into {"outcome", "carcs", "rarcs", "adjustments"}.

Remittance realism: adjustment groups follow convention (PR for benefit-maximum and
coverage-terminated denials, CO for the rest), NORTHSTAR and CASCADE lines carry patient cost
share (PR-2 coinsurance / PR-3 copay) so CLP05 is non-zero, and the header fields real payers
always send (payer address, technical contact, facility code) are present. The group never
decides denied vs paid: a line is denied when it paid 0 and carries a CARC outside the cost
share / contractual set {1, 2, 3, 45}.

The simulator is self-contained: it never reads the answer key (ground_truth / split /
category) or any rules document. Same input -> same output, no randomness, no clock.
"""
from __future__ import annotations

import uuid
import zlib
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

# ----------------------------------------------------------------------------- payers

PAYER_BY_ID = {"SYNNS": "NORTHSTAR", "SYNCA": "CASCADE", "SYNBV": "BAYVIEW"}

PAYER_PROFILE = {
    # allowed amount as a fraction of billed charge (before a small per-code offset)
    # copay_em: fixed copay on an E/M visit; coinsurance: share of allowed on everything else.
    # BAYVIEW (Medicaid-style) has no cost share. Addresses / phones are fictional (555-01xx).
    "NORTHSTAR": {"allowed_pct": Decimal("0.80"), "oon_allowed_pct": Decimal("0.70"),
                  "tax_id": "990000001", "copay_em": Decimal("25.00"), "coinsurance": Decimal("0.20"),
                  "address": {"address1": "100 NORTHSTAR PLAZA", "city": "MINNEAPOLIS",
                              "state": "MN", "postalCode": "55401"}, "phone": "8005550101"},
    "CASCADE": {"allowed_pct": Decimal("0.74"), "tax_id": "990000002",
                "copay_em": Decimal("20.00"), "coinsurance": Decimal("0.20"),
                "address": {"address1": "200 CASCADE WAY", "city": "SEATTLE",
                            "state": "WA", "postalCode": "98101"}, "phone": "8005550102"},
    "BAYVIEW": {"allowed_pct": Decimal("0.70"), "tax_id": "990000003",
                "address": {"address1": "300 BAYVIEW BLVD", "city": "OAKLAND",
                            "state": "CA", "postalCode": "94607"}, "phone": "8005550103"},
}

# CARCs that are cost share or a contractual write-down, never a denial reason by themselves.
NON_DENIAL_CARCS = frozenset({"1", "2", "3", "45"})

# Timely filing. CASCADE follows Medicare's 12-month / 1-calendar-year rule (42 CFR 424.44);
# NORTHSTAR and BAYVIEW use contract windows in days.
FILING_LIMIT_DAYS = {"NORTHSTAR": 180, "BAYVIEW": 90}

# ----------------------------------------------------------------------------- code sets

ADVANCED_IMAGING_AUTH = {"70551", "72148", "74177"}  # NORTHSTAR prior-auth list

PT_CODES = {"97110", "97112", "97140", "97530", "97161", "97162", "97163", "97164"}

PREVENTIVE_EM = {f"{c}" for c in range(99381, 99388)} | {f"{c}" for c in range(99391, 99398)}

LAB_PANEL_OR_TEST = lambda code: code.isdigit() and 80047 <= int(code) <= 89398  # noqa: E731

VENIPUNCTURE = "36415"
VITAMIN_D_25OH = "82306"

# NCCI practitioner PTP edits used by the dataset: (column 1, column 2) -> modifier indicator.
# 97164 pairs: NCCI Policy Manual for Medicare Services (2026) Ch. XI §P.3 (therapy services in
# column 1, PT re-evaluation 97164 in column 2, bypass with 59/XU).
# 17000/11102: NCCI Policy Manual (2026) Ch. III §I.14 (biopsy 11102 paired with destruction
# 17000, bypass with 59/XE/XS). Column order of this pair is not asserted; the check is
# order-agnostic for it.
NCCI_PTP = {
    ("97110", "97164"): "1",
    ("97112", "97164"): "1",
    ("97140", "97164"): "1",
    ("97530", "97164"): "1",
    ("17000", "11102"): "1",
}
NCCI_ORDER_UNVERIFIED = {("17000", "11102")}
X_MODIFIERS = {"XE", "XS", "XP", "XU"}

# Medicare Advantage-style procedure / administration codes that make a same-day E/M need -25.
ADMIN_CODES = {"96372", "96373", "96374", "G0008", "G0009", "G0010",
               "90460", "90461", "90471", "90472", "90473", "90474"}

# CASCADE vitamin D (82306) medical-necessity list: ICD-10-CM codes that support medical
# necessity in CGS Administrators Billing and Coding article A56798 v29 (LCD L33996
# "Vitamin D Assay Testing", revision effective 08/06/2026), ranges expanded (183 codes).
VITAMIN_D_COVERED_ICD10 = frozenset({
    "E200", "E20810", "E20811", "E20812", "E20818", "E20819", "E2089", "E209", "E210", "E211",
    "E212", "E213", "E41", "E43", "E550", "E559", "E673", "E678", "E68", "E8331", "E8332",
    "E8339", "E8351", "E8352", "E840", "E8411", "E8419", "E848", "E892", "K5000", "K50011",
    "K50012", "K50013", "K50014", "K50018", "K50111", "K50112", "K50113", "K50114", "K50118",
    "K5080", "K50811", "K50812", "K50813", "K50814", "K50818", "K5090", "K50911", "K50912",
    "K50913", "K50914", "K50918", "K5100", "K51011", "K51012", "K51013", "K51014", "K51018",
    "K5120", "K51211", "K51212", "K51213", "K51214", "K51218", "K5130", "K51311", "K51312",
    "K51313", "K51314", "K51318", "K5140", "K51411", "K51412", "K51413", "K51414", "K51418",
    "K5150", "K51511", "K51512", "K51513", "K51514", "K51518", "K520", "K702", "K7030", "K7031",
    "K741", "K742", "K743", "K744", "K7469", "K769", "K831", "K900", "K901", "K902", "K903",
    "K9041", "K9049", "K9089", "K909", "K912", "M8000XA", "M80011A", "M80012A", "M80021A",
    "M80022A", "M80031A", "M80032A", "M80041A", "M80042A", "M80051A", "M80052A", "M80061A",
    "M80062A", "M80071A", "M80072A", "M8008XA", "M800B1A", "M800B1D", "M800B1G", "M800B1K",
    "M800B1P", "M800B1S", "M800B2A", "M800B2D", "M800B2G", "M800B2K", "M800B2P", "M800B2S",
    "M810", "M816", "M818", "M830", "M831", "M832", "M833", "M834", "M835", "M838", "M8580",
    "M85811", "M85812", "M85821", "M85822", "M85831", "M85832", "M85841", "M85842", "M85851",
    "M85852", "M85861", "M85862", "M85871", "M85872", "M8588", "M8589", "M899", "M949", "N1830",
    "N1831", "N1832", "N184", "N185", "N186", "N2581", "Q782", "R82994", "Z6830", "Z6831",
    "Z6832", "Z6833", "Z6834", "Z6835", "Z6836", "Z6837", "Z6838", "Z6839", "Z6841", "Z6842",
    "Z6843", "Z6844", "Z6845",
})

# X12 CARC / RARC descriptions (abbreviated from the X12 code lists) for the codes emitted here.
CARC_TEXT = {
    "4": "The procedure code is inconsistent with the modifier used.",
    "5": "The procedure code/type of bill is inconsistent with the place of service.",
    "16": "Claim/service lacks information or has submission/billing error(s).",
    "27": "Expenses incurred after coverage terminated.",
    "29": "The time limit for filing has expired.",
    "45": "Charge exceeds fee schedule/maximum allowable or contracted/legislated fee arrangement.",
    "97": "The benefit for this service is included in the payment/allowance for another "
          "service/procedure that has already been adjudicated.",
    "119": "Benefit maximum for this time period or occurrence has been reached.",
    "167": "This (these) diagnosis(es) is (are) not covered.",
    "197": "Precertification/authorization/notification/pre-treatment absent.",
    "242": "Services not provided by network/primary care providers.",
}
GROUP_TEXT = {"CO": "Contractual Obligations", "PR": "Patient Responsibility",
              "OA": "Other Adjustments"}

# Denials the patient owes (PR); every other denial is the provider's to fix (CO). 119 is a
# benefit maximum, which is a plan-design limit at the commercial payer only; 27 is coverage
# terminated at any payer.
PR_DENIAL_CARCS = {"27": None, "119": {"NORTHSTAR"}}  # carc -> payers (None = every payer)

CENT = Decimal("0.01")


# ----------------------------------------------------------------------------- helpers

def _money(d: Decimal) -> str:
    return str(d.quantize(CENT, rounding=ROUND_HALF_UP))


def _denial_group(payer: str, carc: str) -> str:
    payers = PR_DENIAL_CARCS.get(carc, ())
    return "PR" if payers is None or payer in payers else "CO"


def _cost_share(payer: str, ln: "_Line", allowed: Decimal) -> list[tuple[str, Decimal]]:
    """Deterministic patient cost share on a paid line: [(carc, amount)], paid = allowed - sum.

    E/M visits get a fixed copay (PR-3), preventive visits none (ACA), everything else
    coinsurance on the allowed amount (PR-2). CASCADE waives cost share on lab tests (MA
    Part B). The copay is capped at half the allowed amount so a paid line never rounds to 0.
    """
    profile = PAYER_PROFILE[payer]
    if "coinsurance" not in profile or ln.code in PREVENTIVE_EM:
        return []
    if _is_em(ln.code):
        copay = min(profile["copay_em"], (allowed / 2).quantize(CENT, rounding=ROUND_HALF_UP))
        return [("3", copay)]
    if payer == "CASCADE" and LAB_PANEL_OR_TEST(ln.code):
        return []
    return [("2", (allowed * profile["coinsurance"]).quantize(CENT, rounding=ROUND_HALF_UP))]


def _ymd(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _add_one_year(d: date) -> date:
    try:
        return d.replace(year=d.year + 1)
    except ValueError:  # Feb 29
        return d.replace(year=d.year + 1, day=28)


def _is_em(code: str) -> bool:
    return code.isdigit() and (99202 <= int(code) <= 99215 or 99281 <= int(code) <= 99285)


def _is_surgical_procedure(code: str) -> bool:
    return code.isdigit() and 10004 <= int(code) <= 69990


def _is_telehealth(pos: str, mods: set[str], ctx: dict) -> bool:
    return pos in ("02", "10") or bool(mods & {"95", "GT", "93"}) \
        or "audio" in str(ctx.get("visit_mode", "")).lower()


class _Line:
    __slots__ = ("idx", "sl", "code", "mods", "units", "charge", "ptrs", "denial")

    def __init__(self, idx: int, sl: dict):
        ps = sl.get("professionalService", {})
        self.idx = idx
        self.sl = sl
        self.code = ps.get("procedureCode", "")
        self.mods = set(ps.get("procedureModifiers", []) or [])
        self.units = ps.get("serviceUnitCount", "1")
        self.charge = Decimal(ps.get("lineItemChargeAmount", "0"))
        self.ptrs = (ps.get("compositeDiagnosisCodePointers") or {}).get("diagnosisCodePointers") or []
        self.denial: tuple[str, list[str]] | None = None  # (carc, rarcs)

    def deny(self, carc: str, rarcs: list[str]) -> None:
        if self.denial is None:  # first failing edit wins
            self.denial = (carc, rarcs)


# ----------------------------------------------------------------------------- edits

def _claim_level_denial(payer: str, ci: dict, lines: list[_Line], ctx: dict) -> tuple[str, list[str]] | None:
    first_dos = min(_ymd(ln.sl["serviceDate"]) for ln in lines)
    received = date.fromisoformat(ctx["submission_date"])
    if payer == "CASCADE":
        late = received > _add_one_year(first_dos)
    else:
        late = (received - first_dos).days > FILING_LIMIT_DAYS[payer]
    if late:
        return "29", []
    if payer == "BAYVIEW" and not ctx.get("provider_in_network", True) \
            and ci.get("placeOfServiceCode") != "23":
        return "242", []
    return None


def _line_edits(payer: str, ci: dict, lines: list[_Line], ctx: dict) -> None:
    dx = [d.get("diagnosisCode", "") for d in ci.get("healthCareCodeInformation", [])]
    pos = ci.get("placeOfServiceCode", "")
    codes = {ln.code for ln in lines}
    prior_auth = (ci.get("claimSupplementalInformation") or {}).get("priorAuthorizationNumber")
    dos = min(_ymd(ln.sl["serviceDate"]) for ln in lines)

    for ln in lines:
        # Front-end completeness edits (all payers): CARC 16 always carries a RARC.
        if not ln.ptrs or any(not str(p).isdigit() or not 1 <= int(p) <= len(dx) for p in ln.ptrs):
            ln.deny("16", ["M76", "MA130"])
        if ln.charge <= 0:
            ln.deny("16", ["M79", "MA130"])

        if payer == "NORTHSTAR":
            if ln.code in ADVANCED_IMAGING_AUTH and not prior_auth:
                ln.deny("197", ["M62"])
            if ln.code in PREVENTIVE_EM and ctx.get("last_preventive_dos"):
                last = date.fromisoformat(ctx["last_preventive_dos"])
                if (dos - last).days < 365:
                    ln.deny("119", ["N130"])
            if ln.code in PT_CODES and int(ctx.get("visit_count_ytd", 0)) >= 20:
                ln.deny("119", ["N362"])

        if payer == "CASCADE":
            if _is_em(ln.code) and "25" not in ln.mods and any(
                    c in ADMIN_CODES or _is_surgical_procedure(c) for c in codes - {ln.code}):
                ln.deny("4", ["N822"])
            if ln.code == VITAMIN_D_25OH:
                pointed = {dx[int(p) - 1] for p in ln.ptrs if str(p).isdigit() and 1 <= int(p) <= len(dx)}
                if not pointed & VITAMIN_D_COVERED_ICD10:
                    ln.deny("167", ["N115"])

        if payer == "BAYVIEW":
            if ln.code == VENIPUNCTURE and any(LAB_PANEL_OR_TEST(c) for c in codes):
                ln.deny("97", ["M15"])
            if _is_em(ln.code) and _is_telehealth(pos, ln.mods, ctx):
                if pos != "10":
                    ln.deny("5", ["M77"])
                elif "95" not in ln.mods:
                    ln.deny("4", ["N823"] if ln.mods & {"GT", "93"} else ["N822"])

    # NCCI PTP edits: column-2 line needs a PTP-associated modifier. NORTHSTAR accepts only
    # the X{EPSU} subset; CASCADE and BAYVIEW also accept 59.
    by_code: dict[str, list[_Line]] = {}
    for ln in lines:
        by_code.setdefault(ln.code, []).append(ln)
    for (c1, c2), ccmi in NCCI_PTP.items():
        if c1 not in by_code or c2 not in by_code:
            continue
        candidates = by_code[c2] + (by_code[c1] if (c1, c2) in NCCI_ORDER_UNVERIFIED else [])
        if ccmi == "1" and any(ln.mods & X_MODIFIERS for ln in candidates):
            continue
        with59 = [ln for ln in candidates if "59" in ln.mods]
        if ccmi == "1" and with59:
            if payer == "NORTHSTAR":
                for ln in with59:
                    ln.deny("4", ["N823"])
            continue
        for ln in by_code[c2]:  # unmodified pair: column 2 is bundled
            ln.deny("97", ["M15"])


# ----------------------------------------------------------------------------- public API

def adjudicate(record: dict) -> dict:
    """Adjudicate one claim record and return an 835 ERA (Stedi 835 report JSON shape)."""
    clm = record["claim"]
    ctx = record.get("context", {})
    ci = clm["claimInformation"]
    payer = PAYER_BY_ID.get(clm.get("tradingPartnerServiceId", ""), record.get("payer"))
    profile = PAYER_PROFILE[payer]
    lines = [_Line(i, sl) for i, sl in enumerate(ci["serviceLines"], 1)]

    if not ctx.get("eligibility_active", True):
        for ln in lines:
            ln.deny("27", [])  # expenses incurred after coverage terminated
    claim_denial = _claim_level_denial(payer, ci, lines, ctx)
    if claim_denial:
        for ln in lines:
            ln.deny(*claim_denial)
    else:
        _line_edits(payer, ci, lines, ctx)

    pct = profile["allowed_pct"]
    if payer == "NORTHSTAR" and not ctx.get("provider_in_network", True):
        pct = profile["oon_allowed_pct"]

    out_lines = []
    total_paid = Decimal("0")
    total_pr = Decimal("0")
    for ln in lines:
        ps = ln.sl.get("professionalService", {})
        adjustments, remarks = [], []
        if ln.denial:
            carc, rarcs = ln.denial
            paid = Decimal("0")
            allowed = Decimal("0")
            group = _denial_group(payer, carc)
            adjustments.append(_adj(group, carc, ln.charge))
            if group == "PR":
                total_pr += ln.charge
            remarks = rarcs
        else:
            offset = Decimal(zlib.crc32(ln.code.encode()) % 5) / 100  # 0.00-0.04, deterministic
            allowed = (ln.charge * (pct + offset)).quantize(CENT, rounding=ROUND_HALF_UP)
            if ln.charge - allowed > 0:
                adjustments.append(_adj("CO", "45", ln.charge - allowed))
            share = _cost_share(payer, ln, allowed)
            for carc, amount in share:
                adjustments.append(_adj("PR", carc, amount))
            paid = allowed - sum((a for _, a in share), Decimal("0"))
            total_pr += allowed - paid
        total_paid += paid
        spi = {
            "adjudicatedProcedureCode": ln.code,
            "lineItemChargeAmount": _money(ln.charge),
            "lineItemProviderPaymentAmount": _money(paid),
            "productOrServiceIDQualifier": "HC",
            "productOrServiceIDQualifierValue": "Health Care Financing Administration Common "
                                                "Procedural Coding System (HCPCS) Codes",
            "unitsOfServicePaidCount": "0" if ln.denial else ln.units,
        }
        if ps.get("procedureModifiers"):
            spi["adjudicatedProcedureModifierCodes"] = list(ps["procedureModifiers"])
        line_out = {
            "lineItemControlNumber": ln.sl.get("providerControlNumber", str(ln.idx)),
            "serviceDate": ln.sl["serviceDate"],
            "servicePaymentInformation": spi,
            "serviceAdjustments": adjustments,
            "serviceSupplementalAmounts": {"allowedActual": _money(allowed)},
        }
        if remarks:
            line_out["healthCareCheckRemarkCodes"] = [
                {"codeListQualifierCode": "HE", "codeListQualifierCodeValue": "Claim Payment Remark Codes",
                 "remarkCode": r} for r in remarks]
        out_lines.append(line_out)

    total_charge = Decimal(ci["claimChargeAmount"])
    all_denied = all(ln.denial for ln in lines)
    received = date.fromisoformat(ctx["submission_date"])
    produced = received + timedelta(days=14)
    pcn = ci["patientControlNumber"]
    icn = f"{zlib.crc32((payer + pcn).encode()):010d}{pcn[-3:]}"
    sub = clm.get("subscriber", {})

    payment_info = {
        "claimPaymentInfo": {
            "patientControlNumber": pcn,
            "claimStatusCode": "4" if all_denied else "1",  # 4 = denied, 1 = processed as primary
            "totalClaimChargeAmount": _money(total_charge),
            "claimPaymentAmount": _money(total_paid),
            "patientResponsibilityAmount": _money(total_pr),  # CLP05 = sum of PR adjustments
            "facilityTypeCode": ci.get("placeOfServiceCode", ""),  # CLP08
            "claimFilingIndicatorCode": ci.get("claimFilingCode", ""),
            "payerClaimControlNumber": icn,
            "claimFrequencyCode": ci.get("claimFrequencyCode", "1"),
        },
        "patientName": {"firstName": sub.get("firstName", ""), "lastName": sub.get("lastName", ""),
                        "memberId": sub.get("memberId", "")},
        "claimReceivedDate": received.strftime("%Y%m%d"),
        "serviceLines": out_lines,
    }
    billing = clm.get("billing", {})
    return {
        "meta": {"applicationMode": "simulation", "senderId": clm.get("tradingPartnerServiceId", payer),
                 "traceId": str(uuid.uuid5(uuid.NAMESPACE_URL, f"payer-sim/{payer}/{pcn}"))},
        "transactions": [{
            "controlNumber": clm.get("controlNumber", ""),
            "productionDate": produced.strftime("%Y%m%d"),
            "financialInformation": {
                "transactionHandlingCode": "I" if total_paid > 0 else "H",
                "totalActualProviderPaymentAmount": _money(total_paid),
                "creditOrDebitFlagCode": "C",
                "paymentMethodCode": "ACH" if total_paid > 0 else "NON",  # NON: nothing to pay
                **({"paymentFormatCode": "CCP"} if total_paid > 0 else {}),
                "checkIssueOrEFTEffectiveDate": produced.strftime("%Y%m%d"),
                "payerIdentifier": "1" + profile["tax_id"],
            },
            "paymentAndRemitReassociationDetails": {
                "traceTypeCode": "1",
                "checkOrEFTTraceNumber": icn,
                "originatingCompanyIdentifier": "1" + profile["tax_id"],
            },
            "payer": {
                "name": payer, "payerIdentificationNumber": clm.get("tradingPartnerServiceId", ""),
                "address": dict(profile["address"]),
                "technicalContactInformation": [{"contactName": "ERA SUPPORT",
                                                 "contactMethods": [{"phone": profile["phone"]}]}],
            },
            # Stedi has no separate qualifier field: the `npi` key is the XX-qualified payee id.
            "payee": {"name": billing.get("organizationName", ""), "npi": billing.get("npi", ""),
                      "payeeIdentification": billing.get("npi", "")},
            "detailInfo": [{"assignedNumber": "1", "paymentInfo": [payment_info]}],
        }],
    }


def _adj(group: str, carc: str, amount: Decimal) -> dict:
    return {
        "claimAdjustmentGroupCode": group,
        "claimAdjustmentGroupCodeValue": GROUP_TEXT[group],
        "adjustmentReasonCode1": carc,
        "adjustmentReason1": CARC_TEXT.get(carc, ""),
        "adjustmentAmount1": _money(amount),
    }


def outcome_summary(remit: dict) -> dict:
    """{"outcome": "paid"|"denied", "carcs": [...], "rarcs": [...], "adjustments": [...]} from an 835.

    A line is denied when it paid 0 and carries a CARC outside {1, 2, 3, 45} (cost share and
    contractual write-downs); a claim is denied when any line is denied. The adjustment group
    never decides that: a PR-119 benefit-maximum line is a denial, a paid line with CO-45 and
    PR-2 is paid. `carcs` lists CARCs outside that set (any group) and `rarcs` their remarks,
    in line order, de-duplicated. `adjustments` is the same list with the group attached
    ("PR-119", "CO-197") as extra context; CARCs stay the learning signal."""
    carcs: list[str] = []
    rarcs: list[str] = []
    adjustments: list[str] = []
    denied = False
    for tx in remit.get("transactions", []):
        for detail in tx.get("detailInfo", []):
            for pi in detail.get("paymentInfo", []):
                for sl in pi.get("serviceLines", []):
                    flagged = [a for a in sl.get("serviceAdjustments", [])
                               if a.get("adjustmentReasonCode1") not in NON_DENIAL_CARCS]
                    if not flagged:
                        continue
                    paid = Decimal(sl["servicePaymentInformation"]["lineItemProviderPaymentAmount"])
                    denied = denied or paid == 0
                    for a in flagged:
                        carc = a["adjustmentReasonCode1"]
                        tag = f"{a['claimAdjustmentGroupCode']}-{carc}"
                        carcs += [carc] if carc not in carcs else []
                        adjustments += [tag] if tag not in adjustments else []
                    rarcs += [r["remarkCode"] for r in sl.get("healthCareCheckRemarkCodes", [])
                              if r["remarkCode"] not in rarcs]
    return {"outcome": "denied" if denied else "paid", "carcs": carcs, "rarcs": rarcs,
            "adjustments": adjustments}
