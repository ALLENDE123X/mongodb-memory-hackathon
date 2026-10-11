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
from pymongo import MongoClient

load_dotenv(".env.local")
load_dotenv()

COLLECTIONS = ("traces", "policies", "rules", "semantic_kb", "taxonomy")


@lru_cache
def get_db():
    client = MongoClient(os.environ["MONGODB_URI"], appname="recursive-rcm-harness")
    return client[os.environ.get("MONGODB_DB", "recursive_harness")]


if __name__ == "__main__":
    from memory.store import Store

    db = get_db()
    print("ping:", db.client.admin.command("ping"))
    Store(db).bootstrap()
    print("collections:", sorted(db.list_collection_names()))
