"""Payer simulator agrees with the answer key, and the 837P / 835 shapes balance."""
from decimal import Decimal

import pytest

from eval.synthetic_claims import _validate, load_claims, public_view
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


@pytest.mark.parametrize("record", CLAIMS, ids=IDS)
def test_remit_balances(record):
    remit = adjudicate(_sim_input(record))
    pi = _payment_info(remit)
    cpi = pi["claimPaymentInfo"]
    assert cpi["patientControlNumber"] == record["id"]
    assert Decimal(cpi["totalClaimChargeAmount"]) == Decimal(record["claim"]["claimInformation"]["claimChargeAmount"])
    paid = Decimal("0")
    for sl in pi["serviceLines"]:
        spi = sl["servicePaymentInformation"]
        charge, pay = Decimal(spi["lineItemChargeAmount"]), Decimal(spi["lineItemProviderPaymentAmount"])
        adj = sum(Decimal(a["adjustmentAmount1"]) for a in sl["serviceAdjustments"])
        assert charge == pay + adj  # every line balances: charge = payment + adjustments
        assert all(a["claimAdjustmentGroupCode"] == "CO" for a in sl["serviceAdjustments"])
        if pay == 0:  # denied line: one CO adjustment for the full charge
            assert len(sl["serviceAdjustments"]) == 1 and adj == charge
        else:  # paid line: 70-85% allowed, CO-45 for the rest
            assert Decimal("0.70") <= pay / charge <= Decimal("0.85")
            assert [a["adjustmentReasonCode1"] for a in sl["serviceAdjustments"]] == ["45"]
        paid += pay
    assert Decimal(cpi["claimPaymentAmount"]) == paid
    assert Decimal(remit["transactions"][0]["financialInformation"]["totalActualProviderPaymentAmount"]) == paid
    all_denied = all(Decimal(sl["servicePaymentInformation"]["lineItemProviderPaymentAmount"]) == 0
                     for sl in pi["serviceLines"])
    assert cpi["claimStatusCode"] == ("4" if all_denied else "1")


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
