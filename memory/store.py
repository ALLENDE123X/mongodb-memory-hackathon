"""Atlas-authoritative store for the harness.

Collections
  episodes     episodic: one document per scrub attempt (decision + observed remittance)
  rules        procedural: every rule ever proposed, with status and provenance
  rulesets     procedural: immutable, versioned snapshots of the promoted rule set
  heads        pointer to the active ruleset; promotion is a compare-and-swap on it
  evaluations  gate verdicts, computed by the harness (never asserted by a model)
  runs         loop checkpoints so a restarted process resumes where it stopped
  audits       append-only log of proposals, rejections, promotions, consolidations, recoveries
  semantic_kb  semantic: past denials + payer notes with Voyage embeddings
  taxonomy     taxonomic: payer -> claim category -> denial category (CARC group)

Promotion write order: rules + evaluation are durable first, then the head moves only if
it still points at the version the candidate was evaluated against.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

HEAD_ID = "main"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConflictError(RuntimeError):
    """The head moved since the candidate was evaluated."""


class Store:
    def __init__(self, db):
        self.db = db

    # -- setup -----------------------------------------------------------------
    def ensure_indexes(self) -> None:
        d = self.db
        d.episodes.create_index([("run_id", ASCENDING), ("iteration", ASCENDING)])
        d.episodes.create_index([("claim_id", ASCENDING), ("created_at", DESCENDING)])
        d.episodes.create_index([("payer", ASCENDING), ("observed.outcome", ASCENDING)])
        d.rules.create_index([("payer", ASCENDING), ("status", ASCENDING)])
        d.rulesets.create_index([("version", DESCENDING)], unique=True)
        d.evaluations.create_index([("rule_id", ASCENDING)])
        d.audits.create_index([("created_at", ASCENDING)])
        d.taxonomy.create_index([("kind", ASCENDING), ("key", ASCENDING)], unique=True)

    def bootstrap(self) -> None:
        """Create ruleset v0 (empty) and the head if this is a fresh database."""
        self.ensure_indexes()
        try:
            self.db.rulesets.insert_one(
                {"_id": "rs-0", "version": 0, "parent": None, "rule_ids": [], "created_at": _now(),
                 "evaluation_id": None, "note": "baseline: no learned rules"})
        except DuplicateKeyError:
            pass
        self.db.heads.update_one(
            {"_id": HEAD_ID},
            {"$setOnInsert": {"ruleset_id": "rs-0", "version": 0, "revision": 0, "updated_at": _now()}},
            upsert=True)

    # -- episodic --------------------------------------------------------------
    def append_episode(self, ep: dict) -> str:
        ep = {"_id": ep.get("_id") or f"ep-{uuid.uuid4().hex[:12]}", "created_at": _now(), **ep}
        try:
            self.db.episodes.insert_one(ep)
        except DuplicateKeyError:
            pass  # idempotent replays after a restart
        return ep["_id"]

    def episodes(self, query: dict | None = None) -> list[dict]:
        return list(self.db.episodes.find(query or {}).sort("created_at", ASCENDING))

    # -- procedural ------------------------------------------------------------
    def active(self) -> tuple[dict, list[dict]]:
        head = self.db.heads.find_one({"_id": HEAD_ID})
        rs = self.db.rulesets.find_one({"_id": head["ruleset_id"]})
        rules = list(self.db.rules.find({"_id": {"$in": rs["rule_ids"]}}))
        return head, rules

    def insert_rule(self, rule: dict, *, status: str, iteration: int, evidence: list[str]) -> None:
        self.db.rules.insert_one({
            "_id": rule["rule_id"], **rule, "status": status, "proposed_in_iteration": iteration,
            "evidence_episode_ids": evidence, "created_at": _now()})

    def set_rule_status(self, rule_id: str, status: str, **extra) -> None:
        self.db.rules.update_one({"_id": rule_id}, {"$set": {"status": status, "updated_at": _now(), **extra}})

    def rejected_rules(self, payer: str | None = None) -> list[dict]:
        q = {"status": {"$in": ["rejected", "parked"]}}
        if payer:
            q["payer"] = payer
        return list(self.db.rules.find(q).sort("created_at", ASCENDING))

    def insert_evaluation(self, ev: dict) -> str:
        ev = {"_id": f"eval-{uuid.uuid4().hex[:12]}", "created_at": _now(), **ev}
        self.db.evaluations.insert_one(ev)
        return ev["_id"]

    def promote(self, rule_ids: list[str], *, expected_version: int, evaluation_id: str, note: str) -> int:
        """Append a new immutable ruleset and CAS the head onto it."""
        head = self.db.heads.find_one({"_id": HEAD_ID})
        if head["version"] != expected_version:
            raise ConflictError(f"head is v{head['version']}, candidate evaluated against v{expected_version}")
        parent = self.db.rulesets.find_one({"_id": head["ruleset_id"]})
        new_version = expected_version + 1
        new_id = f"rs-{new_version}"
        self.db.rulesets.insert_one({
            "_id": new_id, "version": new_version, "parent": parent["_id"],
            "rule_ids": parent["rule_ids"] + rule_ids, "added": rule_ids,
            "evaluation_id": evaluation_id, "note": note, "created_at": _now()})
        moved = self.db.heads.find_one_and_update(
            {"_id": HEAD_ID, "version": expected_version},
            {"$set": {"ruleset_id": new_id, "version": new_version, "updated_at": _now()},
             "$inc": {"revision": 1}},
            return_document=ReturnDocument.AFTER)
        if moved is None:
            raise ConflictError("head moved during promotion")
        return new_version

    def replace_rules(self, new_rule_id: str, replaced_ids: list[str], *, expected_version: int,
                      evaluation_id: str, note: str, audit: dict | None = None) -> int:
        """Consolidation: append a ruleset = parent - replaced + new, CAS the head onto it, then
        mark the replaced rules superseded. Old rulesets stay intact, so history is replayable."""
        head = self.db.heads.find_one({"_id": HEAD_ID})
        if head["version"] != expected_version:
            raise ConflictError(f"head is v{head['version']}, replacement evaluated against v{expected_version}")
        parent = self.db.rulesets.find_one({"_id": head["ruleset_id"]})
        missing = [r for r in replaced_ids if r not in parent["rule_ids"]]
        if missing or not replaced_ids:
            raise ConflictError(f"rules not in active ruleset {parent['_id']}: {missing or 'none named'}")
        new_version = expected_version + 1
        new_id = f"rs-{new_version}"
        self.db.rulesets.insert_one({
            "_id": new_id, "version": new_version, "parent": parent["_id"],
            "rule_ids": [r for r in parent["rule_ids"] if r not in replaced_ids] + [new_rule_id],
            "added": [new_rule_id], "removed": list(replaced_ids),
            "evaluation_id": evaluation_id, "note": note, "created_at": _now()})
        moved = self.db.heads.find_one_and_update(
            {"_id": HEAD_ID, "version": expected_version},
            {"$set": {"ruleset_id": new_id, "version": new_version, "updated_at": _now()},
             "$inc": {"revision": 1}},
            return_document=ReturnDocument.AFTER)
        if moved is None:
            raise ConflictError("head moved during replacement")
        self.db.rules.update_many({"_id": {"$in": list(replaced_ids)}},
                                  {"$set": {"status": "superseded", "superseded_by": new_rule_id,
                                            "superseded_in_version": new_version, "updated_at": _now()}})
        self.audit("consolidation", rule_id=new_rule_id, replaces=list(replaced_ids), version=new_version,
                   evaluation_id=evaluation_id, note=note, **(audit or {}))
        return new_version

    # -- runs + audits ---------------------------------------------------------
    def checkpoint(self, run_id: str, **state) -> None:
        self.db.runs.update_one({"_id": run_id}, {"$set": {**state, "updated_at": _now()},
                                                  "$setOnInsert": {"created_at": _now()}}, upsert=True)

    def run_state(self, run_id: str) -> dict | None:
        return self.db.runs.find_one({"_id": run_id})

    def audit(self, kind: str, **payload) -> None:
        self.db.audits.insert_one({"kind": kind, "created_at": _now(), **payload})
