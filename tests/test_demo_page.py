"""Demo exporter and static page: export keys against a mongomock run, defensive fields, page hygiene."""
import json
import re
from pathlib import Path

import mongomock

import agents.scrubber as scrubber_mod
import agents.watcher as watcher_mod
from demo.export import export, label_for
from harness import loop
from memory.store import Store
from tests.test_loop import fake_scrub_llm, fake_watch_llm

DEMO = Path(__file__).resolve().parent.parent / "demo"
KEYS = {"run_id", "baseline", "curve", "rules", "learning", "rsi", "gate", "ledger", "memory_counts", "counts"}


def seeded_db(monkeypatch):
    monkeypatch.setattr(scrubber_mod, "complete_json", fake_scrub_llm)
    monkeypatch.setattr(watcher_mod, "complete_json", fake_watch_llm)
    db = mongomock.MongoClient().db
    store = Store(db)
    loop.run(store, "t", iterations=1)
    return db, store


def test_export_has_all_keys_and_consistent_stats(monkeypatch):
    db, _ = seeded_db(monkeypatch)
    out = export(db, "t")
    assert KEYS <= out.keys()
    assert {"dashboard_url", "events_sent", "total"} <= out["raindrop"].keys()  # JOB-14 block, kept as is
    assert out["baseline"]["label"] == "no learned rules" and out["baseline"]["rule_count"] == 0
    assert out["curve"][1]["label"] == f"ruleset v{out['curve'][1]['ruleset_version']}"
    r = out["rsi"][0]
    assert {"uncovered_denials", "false_holds", "proposed", "promoted", "rejected", "revised"} <= r.keys()
    assert r["acceptance_rate"] == round(r["promoted"] / r["proposed"], 3) and r["n"] == r["proposed"]
    assert out["gate"] and all({"fixes", "regressions", "evidence_size"} <= g.keys() for g in out["gate"])
    rejected = [g for g in out["gate"] if g["verdict"] != "promote"]
    assert rejected and any(g["regressions"] for g in rejected)
    kinds = {e["kind"] for e in out["ledger"]}
    assert {"proposal", "promotion", "rejection"} <= kinds
    # absent collections show up as zero, never missing
    assert out["memory_counts"]["semantic"] == [{"collection": "semantic_kb", "count": 0}]
    assert out["memory_counts"]["taxonomic"] == [{"collection": "taxonomy", "count": 0}]
    assert out["memory_counts"]["episodic"][0]["count"] == db.episodes.count_documents({})
    json.dumps(out, default=str)


def test_export_handles_exception_superseded_consolidation_and_probe(monkeypatch):
    db, store = seeded_db(monkeypatch)
    db.rules.insert_one({"_id": "R-900", "payer": "NORTHSTAR", "status": "promoted", "promoted_version": 99,
                         "proposed_in_iteration": 0, "description": "pays mod 59 with XU", "when": [],
                         "predict": {"outcome": "paid"}, "evidence_episode_ids": ["t:0:CLM-044"]})
    first = db.rules.find_one({"status": "promoted", "_id": {"$ne": "R-900"}})["_id"]
    db.rules.update_one({"_id": first}, {"$set": {"status": "superseded", "superseded_by": "R-900"}})
    store.audit("consolidation", run_id="t", iteration=0, rule_id="R-900", superseded=[first], description="merged")
    store.audit("rejection", run_id="t", rule_id="PROBE-1", reason="would wrongly deny", probe=True)
    db.semantic_kb.insert_one({"_id": "x"})
    out = export(db, "t")
    exc = next(r for r in out["rules"] if r["rule_id"] == "R-900")
    assert exc["kind"] == "exception" and exc["carc"] is None and exc["learned_from"] == ["CLM-044"]
    sup = next(r for r in out["rules"] if r["rule_id"] == first)
    assert sup["status"] == "superseded" and sup["superseded_by"] == "R-900"
    by_kind = {e["kind"]: e for e in out["ledger"]}
    assert by_kind["consolidation"]["extra"]["superseded"] == [first]
    assert "probe" in by_kind
    assert out["memory_counts"]["semantic"][0]["count"] == 1


def test_baseline_label_only_when_rule_count_is_zero():
    assert label_for(0, 0) == "no learned rules"
    assert label_for(4, 4) == "ruleset v4"   # a resumed run may start with rules


def test_committed_run_json_still_parses_with_original_keys():
    d = json.loads((DEMO / "run.json").read_text())
    assert d["curve"] and d["rules"]


def test_html_references_only_relative_resources_and_has_required_copy():
    html = (DEMO / "index.html").read_text()
    assert not re.search(r"<script[^>]+src=", html) and "<link" not in html and "@import" not in html
    assert not re.search(r"(src|href)\s*=\s*[\"']?(https?:)?//", html)
    assert "fetch(src" in html and '"run.json"' in html and 'get("data")' in html
    assert "All claims synthetic. Payer simulator stands in for real 835 remittances." in html
    assert "0 / not enabled" in html and "prefers-color-scheme: dark" in html
    assert "—" not in html and "–" not in html  # no em or en dashes in page copy


def test_export_tolerates_none_fixes_and_regressions(monkeypatch):
    db, store = seeded_db(monkeypatch)
    db.evaluations.update_many({}, {"$set": {"fixes": None, "regressions": None, "redundant": None}})
    db.rules.update_many({}, {"$set": {"regressions": None}})
    store.audit("promotion", run_id="t", rule_id="nope", fixes=None)
    out = export(db, "t")
    assert all(g["fixes"] == 0 and g["regressions"] == 0 for g in out["gate"])
    assert all(r["fixes"] == [] and r["regressions"] == [] for r in out["rules"])
    assert next(e for e in out["ledger"] if e["rule_id"] == "nope")["fixes"] == 0


def test_sample_is_flagged_and_page_shows_banner_and_rsi_na():
    sample = json.loads((DEMO / "sample_run.json").read_text())
    assert sample["synthetic_sample"] is True and sample["_note"] and sample["run_id"] == "sample"
    html = (DEMO / "index.html").read_text()
    assert "SAMPLE DATA, NOT A RESULT" in html and 'd.run_id === "sample" || !!d.synthetic_sample' in html
    assert '"SAMPLE "' in html                                   # title prefix
    assert "l.uncovered_denials ?? l.misses" in html and '?? "n/a"' in html
    assert "Rules learned in round N are first scored at iteration N+1." in html
    assert "Open the Raindrop dashboard" in html
