"""Atlas connection + collection handles for the four memory types.

traces       - episodic: every scrub decision + reasoning, with the real adjudication outcome
policies     - procedural: versioned policy snapshots (rules + tool whitelist) per iteration
rules        - procedural: per-payer rules the watcher derived, with provenance (which denials taught it)
semantic_kb  - semantic: payer policy text / coding guidance, Voyage embeddings
taxonomy     - taxonomic: payer -> claim category -> denial category (CARC groups)
"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

load_dotenv(".env.local")
load_dotenv()

COLLECTIONS = ("traces", "policies", "rules", "semantic_kb", "taxonomy")


@lru_cache
def get_db():
    client = MongoClient(os.environ["MONGODB_URI"], appname="recursive-rcm-harness")
    return client[os.environ.get("MONGODB_DB", "recursive_harness")]


def ensure_indexes():
    db = get_db()
    db.traces.create_index([("iteration", ASCENDING), ("claim_id", ASCENDING)])
    db.traces.create_index([("created_at", DESCENDING)])
    db.policies.create_index([("version", DESCENDING)], unique=True)
    db.rules.create_index([("payer", ASCENDING), ("status", ASCENDING), ("introduced_in", ASCENDING)])
    db.taxonomy.create_index([("kind", ASCENDING), ("name", ASCENDING)], unique=True)


if __name__ == "__main__":
    db = get_db()
    print("ping:", db.client.admin.command("ping"))
    ensure_indexes()
    print("collections:", db.list_collection_names())
