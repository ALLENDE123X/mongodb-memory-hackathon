"""Atlas connection + the collections behind the four memory types (see memory/store.py).

episodes     - episodic: every scrub decision + reasoning, with the real adjudication outcome
rules        - procedural: every rule ever proposed, with status and provenance
rulesets     - procedural: immutable, versioned snapshots of the promoted rule set
heads        - procedural: pointer to the active ruleset (compare-and-swap on promotion)
evaluations  - gate verdicts computed by the harness
runs         - loop checkpoints so a restarted process resumes
audits       - append-only log of proposals, rejections, promotions
semantic_kb  - semantic: Voyage embeddings of training outcomes (Atlas Vector Search)
taxonomy     - taxonomic: payer -> claim category -> denial category (CARC groups)
"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(".env.local")
load_dotenv()

COLLECTIONS = ("episodes", "rules", "rulesets", "heads", "evaluations", "runs", "audits", "semantic_kb", "taxonomy")


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
