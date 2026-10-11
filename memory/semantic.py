"""Semantic memory: Voyage embeddings of training outcomes, retrieved with Atlas Vector Search.

One `semantic_kb` doc per distinct *training* claim (`_id` = "<run_id>:<claim_id>"), holding a short
text summary of the claim and its latest observed outcome plus the embedding. A new observation of
the same claim updates its doc instead of adding another, and the doc is only re-embedded when the
summary text changed. Input is the training-evidence list the loop already builds
(`harness.loop.latest_train_evidence`), so any slicing applied to that list applies here too:
held-out episodes are never embedded, and hits for claims outside the list are dropped.

The summary has no claim ids, no ground_truth, no split, no category. Retrieval returns the
evidence episodes themselves, so the scrubber sees the same shape as with the lexical fallback.

Fallback: without VOYAGE_API_KEY, or on a mongomock database, `semantic_memory()` returns None and
the loop keeps using the lexical `similar_past`. Atlas index creation is `ensure_vector_index`,
run once by an operator (`python -m memory.semantic --ensure-index`), never implicitly.
"""
from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone
from typing import Protocol

from harness.features import features
from harness.payer_sim import CARC_TEXT

COLLECTION = "semantic_kb"
DEFAULT_MODEL = "voyage-3.5"
DEFAULT_DIM = 1024
EMBED_BATCH = 128


def model_name() -> str:
    return os.environ.get("VOYAGE_MODEL", DEFAULT_MODEL)


def index_name() -> str:
    return os.environ.get("VECTOR_INDEX", "semantic_kb_vector")


# -- text summaries ----------------------------------------------------------------------

def _claim_text(feats: dict) -> str:
    lines = "; ".join(" ".join([ln["cpt"], *[f"mod {m}" for m in ln["modifiers"]]]) + f" x{ln['units']:g}"
                      for ln in feats["lines"])
    return (f"Payer {feats['payer']}. Place of service {feats['pos']}. Lines: {lines}. "
            f"Diagnoses: {', '.join(feats['dx_codes'])}.")


def query_text(view: dict) -> str:
    """Public view of a claim about to be scrubbed: no outcome yet."""
    return _claim_text(features(view))


def episode_text(ep: dict) -> str:
    obs = ep["observed"]
    if obs["outcome"] == "paid":
        outcome = "Outcome: paid."
    else:
        carcs = "; ".join(f"CARC {c}: {CARC_TEXT.get(c, '')}".strip() for c in obs["carcs"])
        rarcs = f" RARC {', '.join(obs['rarcs'])}." if obs.get("rarcs") else ""
        outcome = f"Outcome: denied. {carcs}.{rarcs}"
    return f"{_claim_text(ep['features'])} {outcome}"


# -- embedders ---------------------------------------------------------------------------

class Embedder(Protocol):
    def embed(self, texts: list[str], input_type: str) -> list[list[float]]: ...


class VoyageEmbedder:
    def __init__(self, model: str | None = None):
        import voyageai  # lazy: only needed when a key is configured
        self.client = voyageai.Client()  # reads VOYAGE_API_KEY
        self.model = model or model_name()

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            out += self._embed_batch(texts[i:i + EMBED_BATCH], input_type)
        return out

    def _embed_batch(self, batch: list[str], input_type: str, attempts: int = 3) -> list[list[float]]:
        """Retry rate limits with exponential backoff (1s, 2s); other errors propagate."""
        from voyageai.error import RateLimitError
        for n in range(attempts):
            try:
                return self.client.embed(batch, model=self.model, input_type=input_type).embeddings
            except RateLimitError:
                if n == attempts - 1:
                    raise
                time.sleep(2 ** n)
        raise AssertionError("unreachable")


def _is_mongomock(db) -> bool:
    return type(db.client).__module__.startswith("mongomock")


def semantic_memory(db, embedder: Embedder | None = None) -> "SemanticMemory | None":
    """The vector-backed memory, or None when the loop should use the lexical fallback."""
    if embedder is None:
        if not os.environ.get("VOYAGE_API_KEY") or _is_mongomock(db):
            return None
        embedder = VoyageEmbedder()
    return SemanticMemory(db, embedder)


# -- the memory --------------------------------------------------------------------------

class SemanticMemory:
    def __init__(self, db, embedder: Embedder):
        self.col = db[COLLECTION]
        self.embedder = embedder
        self.model = getattr(embedder, "model", model_name())

    def sync(self, run_id: str, evidence: list[dict]) -> int:
        """Upsert one vector per distinct training claim. Returns how many were (re)embedded."""
        latest = {ep["claim_id"]: ep for ep in evidence
                  if ep.get("split") == "train" and ep.get("visible_to_learning", True)}
        have = {d["claim_id"]: d["text_hash"] for d in self.col.find({"run_id": run_id}, {"claim_id": 1, "text_hash": 1})}
        todo = []
        for cid, ep in sorted(latest.items()):
            text = episode_text(ep)
            h = hashlib.sha256(f"{self.model}|{text}".encode()).hexdigest()
            if have.get(cid) != h:
                todo.append((cid, ep, text, h))
        if not todo:
            return 0
        vectors = self.embedder.embed([t[2] for t in todo], "document")
        now = datetime.now(timezone.utc)
        for (cid, ep, text, h), vec in zip(todo, vectors):
            self.col.update_one(
                {"_id": f"{run_id}:{cid}"},
                {"$set": {"run_id": run_id, "claim_id": cid, "payer": ep["payer"], "episode_id": ep["_id"],
                          "split": "train", "text": text, "text_hash": h, "model": self.model,
                          "embedding": list(vec), "updated_at": now}},
                upsert=True)
        return len(todo)

    def _vector_search(self, qvec: list[float], run_id: str, limit: int) -> list[dict]:
        return list(self.col.aggregate([
            {"$vectorSearch": {"index": index_name(), "path": "embedding", "queryVector": qvec,
                               "numCandidates": max(50, limit * 10), "limit": limit,
                               "filter": {"run_id": run_id}}},
            {"$project": {"claim_id": 1, "score": {"$meta": "vectorSearchScore"}}}]))

    def similar(self, view: dict, evidence: list[dict], run_id: str, k: int = 3) -> list[dict]:
        """Top-k most similar past training outcomes, excluding the claim itself and any claim
        that is not in the given evidence list."""
        by_claim = {ep["claim_id"]: ep for ep in evidence
                    if ep.get("split") == "train" and ep.get("visible_to_learning", True)}
        qvec = self.embedder.embed([query_text(view)], "query")[0]
        out = []
        for hit in self._vector_search(qvec, run_id, limit=k * 4 + 1):
            ep = by_claim.get(hit["claim_id"])
            if ep is not None and hit["claim_id"] != view["id"]:
                out.append(ep)
            if len(out) == k:
                break
        return out


# -- Atlas Vector Search index ----------------------------------------------------------

def vector_index_definition() -> dict:
    dim = int(os.environ.get("VOYAGE_DIM", DEFAULT_DIM))
    return {"name": index_name(), "type": "vectorSearch",
            "definition": {"fields": [
                {"type": "vector", "path": "embedding", "numDimensions": dim, "similarity": "cosine"},
                {"type": "filter", "path": "run_id"}]}}


def ensure_vector_index(db) -> bool:
    """Create the Atlas Vector Search index on semantic_kb if missing. Returns True if created.
    Atlas only: operators run this once; tests never call it against a real cluster."""
    from pymongo.operations import SearchIndexModel
    spec = vector_index_definition()
    if COLLECTION not in db.list_collection_names():  # create_search_index needs the collection to exist
        db.create_collection(COLLECTION)
    col = db[COLLECTION]
    if any(ix.get("name") == spec["name"] for ix in col.list_search_indexes()):
        return False
    col.create_search_index(SearchIndexModel(definition=spec["definition"], name=spec["name"], type="vectorSearch"))
    return True


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--print-index", action="store_true", help="print the index definition and exit")
    ap.add_argument("--ensure-index", action="store_true", help="create the index on the configured Atlas DB")
    args = ap.parse_args()
    if args.ensure_index:
        from memory.atlas_schema import get_db
        print("created" if ensure_vector_index(get_db()) else "already exists")
        print("Atlas builds the index asynchronously: wait for status READY before running the loop.")
    else:
        print(json.dumps(vector_index_definition(), indent=2))
