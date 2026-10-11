"""The answer key never reaches a model prompt, and unrevealed training slices stay invisible.

Every scrubber, watcher-propose and watcher-consolidate prompt of a fake-model mongomock run is captured and scanned for the
hidden fields (`ground_truth`, `violated_rules`, `split`, `category`), any hidden rule id
(PR-01..PR-12), and held-out claim ids. `test_scanner_is_live` proves the scan would fail if
`public_view` ever leaked `ground_truth`. The fake watcher replays the hidden ORACLE rules, whose
descriptions are their PR ids, so those are renamed before they enter memory: a real watcher only
ever sees what is in its prompt.
"""
import json
import re

import mongomock
import pytest

import agents.scrubber as scrubber_mod
import agents.watcher as watcher_mod
from eval.synthetic_claims import load_claims
from harness import loop
from memory.store import Store
from tests.test_loop import fake_scrub_llm, fake_watch_llm

HIDDEN = ("ground_truth", "violated_rules", '"split"', '"category"')
RULE_ID = re.compile(r"\bPR-(0[1-9]|1[0-2])\b")
HELDOUT = {r["id"] for r in load_claims("heldout")}
CLAIM_ID = re.compile(r"CLM-\d{3}")
SUBJECT = re.compile(r'"patientControlNumber": "(CLM-\d{3})"')


def renamed_watcher(model, system, user, **kw):
    out, meta = fake_watch_llm(model, system, user, **kw)
    if "rules" not in out:  # consolidation reply
        return out, meta
    rules = [{**r, "description": f"learned {r['payer']} rule {i}"} for i, r in enumerate(out["rules"])]
    return {**out, "rules": rules}, meta


@pytest.fixture
def prompts(monkeypatch):
    seen: list[tuple[str, str]] = []

    def capture(who, fake):
        def call(model, system, user, **kw):
            kind = "consolidate" if system == watcher_mod.CONSOLIDATE_SYSTEM else who
            seen.append((kind, system + "\n" + user))
            return fake(model, system, user, **kw)
        return call

    monkeypatch.setattr(scrubber_mod, "complete_json", capture("scrubber", fake_scrub_llm))
    monkeypatch.setattr(watcher_mod, "complete_json", capture("watcher", renamed_watcher))
    return seen


def leaks(who: str, text: str) -> list[str]:
    found = [k for k in HIDDEN if k in text] + RULE_ID.findall(text)
    ids = CLAIM_ID.findall(text)
    heldout = set(ids) & HELDOUT
    # only a held-out claim's own scrubber prompt may name a held-out id, and only its own:
    # the claim under review is the last id in the prompt (its 837 patientControlNumber)
    own = SUBJECT.findall(text)
    allowed = {own[-1]} if who == "scrubber" and own and own[-1] in HELDOUT else set()
    if heldout - allowed:
        found.append(f"held-out ids {sorted(heldout - allowed)}")
    return found


def test_no_prompt_carries_the_answer_key(prompts):
    loop.run(Store(mongomock.MongoClient().db), "leak", iterations=3)
    assert {w for w, _ in prompts} == {"scrubber", "watcher", "consolidate"}
    assert any(set(CLAIM_ID.findall(t)) & HELDOUT for w, t in prompts if w == "scrubber")  # held-out was scored
    bad = [(w, leaks(w, t)) for w, t in prompts if leaks(w, t)]
    assert not bad, bad[:3]


def test_scanner_is_live(prompts, monkeypatch):
    """If public_view leaked the answer key, the scan above would catch it."""
    def leaky_view(rec):  # a regression that smuggles the answer key into a field the prompt shows
        return {"id": rec["id"], "payer": rec["payer"], "claim": rec["claim"],
                "context": {**rec["context"], "ground_truth": rec["ground_truth"]}}

    monkeypatch.setattr(loop, "public_view", leaky_view)
    store = Store(mongomock.MongoClient().db)
    store.bootstrap()
    loop.scrub_pass(store, "live", 0)
    assert any("ground_truth" in leaks(w, t) for w, t in prompts)
    assert any(RULE_ID.search(t) for _, t in prompts)


def test_unrevealed_slice_is_invisible_until_revealed(prompts):
    client = mongomock.MongoClient()
    store = Store(client.db)
    store.bootstrap()
    later = set(loop.train_slices(3)[2])

    def everywhere() -> str:
        dump = {c: list(client.db[c].find({})) for c in client.db.list_collection_names()}
        return json.dumps(dump, default=str) + "\n".join(t for _, t in prompts)

    loop.scrub_pass(store, "s", 0)
    loop.learn(store, "s", 0)
    assert any(w == "watcher" for w, _ in prompts)
    assert not later & set(CLAIM_ID.findall(everywhere())), "slice-2 claim visible during iteration 0"
    loop.scrub_pass(store, "s", 2)  # revealed now: the same scan finds them, so the check is live
    assert later <= set(CLAIM_ID.findall(everywhere()))
