"""The stage demo must run end to end offline: mongomock + fake model, no network, no Atlas."""
import socket

import mongomock
import pytest

from harness import demo


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("demo tried to open a network connection")
    monkeypatch.setattr(socket.socket, "connect", boom)


@pytest.fixture(scope="module")
def seeded():
    return demo.seed_mock("t")


def reconnect_for(client):
    return lambda: mongomock.MongoClient(_store=client._store).db


def test_mock_prints_all_five_beats(capsys):
    demo.main(["--mock"])
    out = capsys.readouterr().out
    for n, name in demo.BEATS.items():
        assert f"BEAT {n} of 5: {name}" in out
    assert "—" not in out  # no em dashes in printed prose
    assert "1 rule doc (status rejected), 1 evaluation doc, 1 audit doc" in out


def test_beats_tell_a_true_story(seeded, capsys):
    out = demo.run_beats(reconnect_for(seeded), "t", [1, 2, 3, 4, 5])
    b1, b2, b3, b4, b5 = (out[i] for i in range(1, 6))
    # 1: held-out claim the baseline waved through and the payer denied
    assert b1["baseline"] == "paid" and b1["observed"] == "denied" and b1["carcs"]
    assert b1["episode"]["split"] == "heldout"
    # 2: the covering rule was promoted from training evidence only, with gate stats
    assert b2["taught"] and b1["claim_id"] not in b2["taught"] and b2["held_out_clean"]
    assert b2["evaluation"]["verdict"] == "promote" and not b2["evaluation"]["regressions"]
    # 3: probe writes exactly one rule (rejected), one evaluation, one audit; head unchanged
    assert b3["verdict"] == "rejected" and b3["status"] == "rejected" and b3["head_unchanged"]
    assert b3["wrote"] == {"rules": 1, "evaluations": 1, "audits": 1}
    # 4: fresh connection, rule path, no model
    assert b4["source"].startswith("rule:") and b4["model"] is None
    assert b4["carc"] in b4["payer_carcs"] and b4["claim_id"] != b1["claim_id"]
    # 5: curve improves on held-out
    assert b5["rows"][-1]["balanced_accuracy"] > b5["rows"][0]["balanced_accuracy"]
    capsys.readouterr()


@pytest.mark.parametrize("beat", [1, 2, 3, 4, 5])
def test_each_beat_runs_alone(beat, capsys):
    demo.main(["--mock", "--beat", str(beat)])
    out = capsys.readouterr().out
    assert f"BEAT {beat} of 5" in out
    assert out.count("BEAT ") == 1
