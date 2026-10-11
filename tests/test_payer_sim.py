"""Payer simulator agrees with the answer key, and the 837P / 835 shapes balance."""
from decimal import Decimal

import pytest

from eval.synthetic_claims import _validate, load_claims, public_view
from harness.features import features
from harness.payer_sim import adjudicate, outcome_summary

CLAIMS = load_claims()
IDS = [c["id"] for c in CLAIMS]


def _sim_input(record: dict) -> dict:
    # The simulator gets the stored record minus the answer key.
    return {k: v for k, v in record.items() if k not in ("ground_truth", "split", "category")}


def _payment_info(remit: dict) -> dict:
    return remit["transactions"][0]["detailInfo"][0]["paymentInfo"][0]


@pytest.mark.parametrize("record", CLAIMS, ids=IDS)
def test_outcome_matches_ground_truth(record):
    gt = record["ground_truth"]
    summary = outcome_summary(adjudicate(_sim_input(record)))
    assert summary["outcome"] == gt["outcome"], summary
    if gt["outcome"] == "denied":
        if "PR-09" in gt["violated_rules"]:
            assert summary["carcs"] in (["4"], ["5"]) and summary["carcs"] == [gt["carc"]], summary
        else:
            assert summary["carcs"] == [gt["carc"]], summary
    else:
        assert summary["carcs"] == [] and summary["rarcs"] == []


@pytest.mark.parametrize("record", CLAIMS, ids=IDS)
def test_claim_charge_amount_balances(record):
    ci = record["claim"]["claimInformation"]
    lines = sum(Decimal(sl["professionalService"]["lineItemChargeAmount"]) for sl in ci["serviceLines"])
    assert Decimal(ci["claimChargeAmount"]) == lines


def _pr(sl: dict) -> Decimal:
    return sum((Decimal(a["adjustmentAmount1"]) for a in sl["serviceAdjustments"]
                if a["claimAdjustmentGroupCode"] == "PR"), Decimal("0"))


@pytest.mark.parametrize("record", CLAIMS, ids=IDS)
def test_remit_balances(record):
    remit = adjudicate(_sim_input(record))
    pi = _payment_info(remit)
    cpi = pi["claimPaymentInfo"]
    assert cpi["patientControlNumber"] == record["id"]
    assert Decimal(cpi["totalClaimChargeAmount"]) == Decimal(record["claim"]["claimInformation"]["claimChargeAmount"])
    paid = pr_total = Decimal("0")
    for sl in pi["serviceLines"]:
        spi = sl["servicePaymentInformation"]
        charge, pay = Decimal(spi["lineItemChargeAmount"]), Decimal(spi["lineItemProviderPaymentAmount"])
        adj = sum(Decimal(a["adjustmentAmount1"]) for a in sl["serviceAdjustments"])
        assert charge == pay + adj  # every line balances: charge = payment + adjustments
        by_carc = {a["adjustmentReasonCode1"]: a for a in sl["serviceAdjustments"]}
        if pay == 0:  # denied line: one adjustment for the full charge
            assert len(sl["serviceAdjustments"]) == 1 and adj == charge
        else:  # paid line: allowed 70-85% of charge; CO-45 = charge - allowed; PR-2/PR-3 on allowed
            allowed = Decimal(sl["serviceSupplementalAmounts"]["allowedActual"])
            assert Decimal("0.70") <= allowed / charge <= Decimal("0.85")
            assert allowed == pay + _pr(sl)
            assert Decimal(by_carc["45"]["adjustmentAmount1"]) == charge - allowed
            assert by_carc["45"]["claimAdjustmentGroupCode"] == "CO"
            assert set(by_carc) <= {"45", "2", "3"}
            assert all(a["claimAdjustmentGroupCode"] == "PR" for c, a in by_carc.items() if c != "45")
        paid += pay
        pr_total += _pr(sl)
    assert Decimal(cpi["claimPaymentAmount"]) == paid
    assert Decimal(cpi["patientResponsibilityAmount"]) == pr_total  # CLP05 = sum of PR
    assert Decimal(remit["transactions"][0]["financialInformation"]["totalActualProviderPaymentAmount"]) == paid
    all_denied = all(Decimal(sl["servicePaymentInformation"]["lineItemProviderPaymentAmount"]) == 0
                     for sl in pi["serviceLines"])
    assert cpi["claimStatusCode"] == ("4" if all_denied else "1")


def _groups_by_carc(record: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for sl in _payment_info(adjudicate(_sim_input(record)))["serviceLines"]:
        for a in sl["serviceAdjustments"]:
            out.setdefault(a["adjustmentReasonCode1"], set()).add(a["claimAdjustmentGroupCode"])
    return out


def test_denial_groups_follow_convention():
    seen: dict[tuple[str, str], set[str]] = {}
    for record in CLAIMS:
        for carc, groups in _groups_by_carc(record).items():
            seen.setdefault((record["payer"], carc), set()).update(groups)
    for (payer, carc), groups in seen.items():
        if carc in ("2", "3"):
            assert groups == {"PR"} and payer != "BAYVIEW"
        elif carc == "27" or (carc == "119" and payer == "NORTHSTAR"):
            assert groups == {"PR"}, (payer, carc)
        else:
            assert groups == {"CO"}, (payer, carc)
    assert ("NORTHSTAR", "119") in seen and ("NORTHSTAR", "2") in seen


def test_northstar_119_denial_is_pr_group_but_still_denied():
    record = next(c for c in CLAIMS if c["ground_truth"]["outcome"] == "denied"
                  and c["ground_truth"]["carc"] == "119")
    summary = outcome_summary(adjudicate(_sim_input(record)))
    assert summary["outcome"] == "denied" and summary["carcs"] == ["119"]
    assert summary["adjustments"] == ["PR-119"]
    # a denied PR line is owed by the patient: it counts in CLP05
    cpi = _payment_info(adjudicate(_sim_input(record)))["claimPaymentInfo"]
    assert Decimal(cpi["patientResponsibilityAmount"]) > 0


def test_paid_claim_with_pr_reads_as_paid():
    record = next(c for c in CLAIMS if c["ground_truth"]["outcome"] == "paid" and c["payer"] == "NORTHSTAR")
    remit = adjudicate(_sim_input(record))
    sls = _payment_info(remit)["serviceLines"]
    assert any(_pr(sl) > 0 for sl in sls)
    codes = {a["adjustmentReasonCode1"] for sl in sls for a in sl["serviceAdjustments"]}
    assert "45" in codes and codes & {"2", "3"}
    assert outcome_summary(remit) == {"outcome": "paid", "carcs": [], "rarcs": [], "adjustments": []}


def test_cost_share_by_payer():
    for record in CLAIMS:
        pi = _payment_info(adjudicate(_sim_input(record)))
        pr = Decimal(pi["claimPaymentInfo"]["patientResponsibilityAmount"])
        if record["payer"] == "BAYVIEW":
            assert pr == 0, record["id"]
    em = next(c for c in CLAIMS if c["payer"] == "NORTHSTAR" and c["ground_truth"]["outcome"] == "paid"
              and any(sl["professionalService"]["procedureCode"] in {"99213", "99214"}
                      for sl in c["claim"]["claimInformation"]["serviceLines"]))
    for sl in _payment_info(adjudicate(_sim_input(em)))["serviceLines"]:
        if sl["servicePaymentInformation"]["adjudicatedProcedureCode"] in {"99213", "99214"}:
            prs = [a for a in sl["serviceAdjustments"] if a["claimAdjustmentGroupCode"] == "PR"]
            assert [(a["adjustmentReasonCode1"], a["adjustmentAmount1"]) for a in prs] == [("3", "25.00")]


def test_outcome_summary_group_never_decides():
    def remit(paid, *adjs):
        return {"transactions": [{"detailInfo": [{"paymentInfo": [{"serviceLines": [{
            "servicePaymentInformation": {"lineItemProviderPaymentAmount": paid},
            "serviceAdjustments": [{"claimAdjustmentGroupCode": g, "adjustmentReasonCode1": c} for g, c in adjs],
        }]}]}]}]}
    assert outcome_summary(remit("80.00", ("CO", "45"), ("PR", "2")))["outcome"] == "paid"
    assert outcome_summary(remit("0.00", ("PR", "119")))["outcome"] == "denied"
    assert outcome_summary(remit("0.00", ("CO", "119")))["carcs"] == ["119"]
    assert outcome_summary(remit("0.00", ("PR", "1")))["outcome"] == "paid"  # cost share only


def test_always_present_header_fields():
    for record in CLAIMS:
        remit = adjudicate(_sim_input(record))
        tx = remit["transactions"][0]
        pi = _payment_info(remit)
        assert pi["claimPaymentInfo"]["facilityTypeCode"] == record["claim"]["claimInformation"]["placeOfServiceCode"]
        assert {"address1", "city", "state", "postalCode"} <= set(tx["payer"]["address"])
        assert tx["payer"]["technicalContactInformation"][0]["contactMethods"][0]["phone"]
        assert tx["payee"]["payeeIdentification"] == tx["payee"]["npi"] != ""
        fi = tx["financialInformation"]
        assert (fi["paymentMethodCode"] == "ACH") == (Decimal(fi["totalActualProviderPaymentAmount"]) > 0)


def test_carc_16_always_has_a_remark_code():
    for record in CLAIMS:
        for sl in _payment_info(adjudicate(_sim_input(record)))["serviceLines"]:
            if any(a["adjustmentReasonCode1"] == "16" for a in sl["serviceAdjustments"]):
                assert sl.get("healthCareCheckRemarkCodes"), record["id"]


def test_deterministic():
    for record in CLAIMS:
        assert adjudicate(_sim_input(record)) == adjudicate(_sim_input(record))


def test_simulator_ignores_answer_key():
    record = next(c for c in CLAIMS if c["ground_truth"]["outcome"] == "denied")
    flipped = dict(record, ground_truth={"outcome": "paid", "carc": None, "violated_rules": [], "reason": "x"})
    assert adjudicate(flipped) == adjudicate(_sim_input(record))
    assert adjudicate(public_view(record)) == adjudicate(_sim_input(record))


def test_dataset_validates():
    assert _validate(load_claims()) == []


def test_validator_catches_unbalanced_charge():
    bad = load_claims()
    bad[0]["claim"]["claimInformation"]["claimChargeAmount"] = "1.00"
    assert any("claimChargeAmount" in e for e in _validate(bad))


def test_public_view_shape():
    for record in CLAIMS:
        pv = public_view(record)
        assert set(pv) == {"id", "payer", "claim", "context"}


def test_coverage_terminated_27_is_pr_at_every_payer():
    for payer in ("NORTHSTAR", "CASCADE", "BAYVIEW"):
        record = dict(_sim_input(next(c for c in CLAIMS if c["payer"] == payer)))
        record["context"] = dict(record["context"], eligibility_active=False)
        summary = outcome_summary(adjudicate(record))
        assert summary["outcome"] == "denied" and summary["carcs"] == ["27"], payer
        assert summary["adjustments"] == ["PR-27"], payer


def test_claim_brief_shows_group_and_carc():
    from agents.watcher import claim_brief
    record = next(c for c in CLAIMS if c["ground_truth"]["outcome"] == "denied"
                  and c["ground_truth"]["carc"] == "119")
    observed = outcome_summary(adjudicate(_sim_input(record)))
    brief = claim_brief({"claim_id": record["id"], "features": features(public_view(record)),
                         "observed": observed})
    assert brief["observed"]["adjustments"] == ["PR-119"]
    assert brief["observed"]["carcs"][0].startswith("119:")
