"""Semantic + taxonomic memory: no network, mongomock, and a fake embedder."""
import json
import math

import mongomock

import agents.scrubber as scrubber_mod
import agents.watcher as watcher_mod
from eval.synthetic_claims import load_claims
from harness import loop
from memory import semantic, taxonomy
from memory.store import Store
from tests.test_loop import fake_scrub_llm


class FakeEmbedder:
    """Deterministic bag-of-tokens hashing into 32 dims."""
    model = "fake-embed"

    def __init__(self):
        self.calls = []

    def embed(self, texts, input_type):
        self.calls.append((input_type, len(texts)))
        out = []
        for t in texts:
            v = [0.0] * 32
            for tok in t.lower().replace(".", " ").replace(",", " ").split():
                v[hash(tok) % 32] += 1.0
            out.append(v)
        return out


class LocalSemantic(semantic.SemanticMemory):
    """mongomock has no $vectorSearch: rank the collection's docs by cosine instead."""

    def _vector_search(self, qvec, run_id, limit):
        def cos(a, b):
            return sum(x * y for x, y in zip(a, b)) / ((math.hypot(*a) * math.hypot(*b)) or 1)
        docs = [d for d in self.col.find({"run_id": run_id})]
        docs.sort(key=lambda d: -cos(qvec, d["embedding"]))
        return docs[:limit]


def _scrubbed(monkeypatch, iterations=(0,)):
    monkeypatch.setattr(scrubber_mod, "complete_json", fake_scrub_llm)
    store = Store(mongomock.MongoClient().db)
    store.bootstrap()
    for it in iterations:
        loop.scrub_pass(store, "t", it, slices=1)
    return store


def test_category_and_carc_groups():
    assert taxonomy.cpt_category("99213") == "evaluation_management"
    assert taxonomy.cpt_category("74177") == "imaging"
    assert taxonomy.cpt_category("82306") == "lab"
    assert taxonomy.cpt_category("97110") == "therapy"
    assert taxonomy.cpt_category("27447") == "surgery"
    assert taxonomy.cpt_category("G0008") == "other"
    assert [taxonomy.carc_group(c) for c in ("197", "29", "4", "5", "97", "119", "167", "242", "16", "99")] == [
        "authorization", "timely_filing", "coding_modifier", "coding_modifier", "bundling", "benefit_limit",
        "coverage", "coverage", "missing_info", "other"]


def test_taxonomy_counts_distinct_claims_and_train_only(monkeypatch):
    store = _scrubbed(monkeypatch, iterations=(0, 1))  # same claims scrubbed twice
    evidence = loop.latest_train_evidence(store, "t")
    counts = taxonomy.taxonomy_counts(evidence)
    expected = sum(len({taxonomy.cpt_category(c) for c in e["cpts"]}) * len({taxonomy.carc_group(c) for c in e["observed"]["carcs"]})
                   for e in evidence if e["observed"]["outcome"] == "denied")
    assert expected > 0 and sum(d["count"] for d in counts.values()) == expected  # no double counting
    train_ids = {c["id"] for c in load_claims("train")}
    assert all(set(d["claim_ids"]) <= train_ids for d in counts.values())
    # held-out episodes handed in by mistake are ignored
    heldout = store.episodes({"run_id": "t", "split": "heldout"})
    assert taxonomy.taxonomy_counts(evidence + heldout) == counts
    # synced docs mirror the counts and are rebuilt, not accumulated
    stored = {d["key"]: d["count"] for d in store.db.taxonomy.find({"kind": taxonomy.KIND})}
    assert stored == {k: d["count"] for k, d in counts.items()}
    taxonomy.sync_taxonomy(store.db, "t", evidence)
    assert {d["key"]: d["count"] for d in store.db.taxonomy.find({})} == stored
    payer = next(k.split("|")[0] for k in counts)
    line = taxonomy.taxonomy_summary(evidence, payer)
    assert line.startswith(f"{payer}: ") and " -> " in line and " x" in line
    assert "no denials" in taxonomy.taxonomy_summary(evidence, "NOBODY")


def test_fallback_without_key_or_on_mongomock(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    assert semantic.semantic_memory(mongomock.MongoClient().db) is None
    monkeypatch.setenv("VOYAGE_API_KEY", "not-a-real-key")  # key set, but the DB is mongomock
    assert semantic.semantic_memory(mongomock.MongoClient().db) is None
    store = _scrubbed(monkeypatch)
    assert store.db.semantic_kb.count_documents({}) == 0  # lexical path wrote no vectors
    view = {"id": "x", "payer": "P", "claim": load_claims("train")[0]["claim"], "context": {}}
    ev = [{"claim_id": "a", "payer": "P", "cpts": ["99213"]}]
    assert loop.retrieve_similar(view, ev, None, "t") == loop.similar_past(view, ev)


def test_semantic_sync_embeds_only_training_one_vector_per_claim(monkeypatch):
    store = _scrubbed(monkeypatch, iterations=(0,))
    emb = FakeEmbedder()
    sem = LocalSemantic(store.db, emb)
    evidence = loop.latest_train_evidence(store, "t")
    heldout = store.episodes({"run_id": "t", "split": "heldout"})
    assert heldout
    n = sem.sync("t", evidence + heldout)  # held-out handed in by mistake
    docs = list(store.db.semantic_kb.find({}))
    assert n == len(evidence) == len(docs)
    train_ids = {c["id"] for c in load_claims("train")}
    assert {d["claim_id"] for d in docs} == train_ids and all(d["split"] == "train" for d in docs)
    for d in docs:  # nothing identifying or hidden in the embedded text
        assert d["claim_id"] not in d["text"] and "ground_truth" not in d["text"]
    # same observation again: nothing re-embedded; new observation: same doc updated, not duplicated
    assert sem.sync("t", evidence) == 0
    changed = dict(evidence[0], observed={"outcome": "denied", "carcs": ["197"], "rarcs": ["N54"]})
    assert sem.sync("t", [changed] + evidence[1:]) == 1
    assert store.db.semantic_kb.count_documents({}) == len(evidence)
    assert "CARC 197" in store.db.semantic_kb.find_one({"claim_id": changed["claim_id"]})["text"]


def test_similar_excludes_self_and_claims_outside_evidence(monkeypatch):
    store = _scrubbed(monkeypatch)
    sem = LocalSemantic(store.db, FakeEmbedder())
    evidence = loop.latest_train_evidence(store, "t")
    sem.sync("t", evidence)
    rec = load_claims("train")[0]
    from eval.synthetic_claims import public_view
    hits = sem.similar(public_view(rec), evidence, "t", k=3)
    assert len(hits) == 3 and rec["id"] not in {h["claim_id"] for h in hits}
    sliced = evidence[:5]  # a narrower evidence slice (ticket 3) narrows retrieval too
    assert {h["claim_id"] for h in sem.similar(public_view(rec), sliced, "t", k=3)} <= {e["claim_id"] for e in sliced}


def test_loop_uses_semantic_memory_and_gives_watcher_the_taxonomy(monkeypatch):
    monkeypatch.setattr(scrubber_mod, "complete_json", fake_scrub_llm)
    seen = []

    def watch(model, system, user, **kw):
        seen.append(json.loads(user.split("\n\nYour previous")[0]))
        return {"rules": [], "notes": "none"}, {"model": "fake", "latency_s": 0, "cached": False}

    monkeypatch.setattr(watcher_mod, "complete_json", watch)
    emb = FakeEmbedder()
    monkeypatch.setattr(loop, "semantic_memory", lambda db: LocalSemantic(db, emb))
    store = Store(mongomock.MongoClient().db)
    loop.run(store, "t", iterations=1, slices=1)
    train_ids = {c["id"] for c in load_claims("train")}
    assert {d["claim_id"] for d in store.db.semantic_kb.find({})} == train_ids
    assert any(c[0] == "query" for c in emb.calls) and any(c[0] == "document" for c in emb.calls)
    assert seen and all("payer_denial_taxonomy" in p for p in seen)
    assert " -> " in seen[0]["payer_denial_taxonomy"]


class _FakeCol:
    def __init__(self):
        self.created = []

    def list_search_indexes(self):
        return [{"name": m.document["name"]} for m in self.created]

    def create_search_index(self, model):
        self.created.append(model)


class _FakeDb:
    def __init__(self, with_collection=False):
        self.col = _FakeCol()
        self.made = ["semantic_kb"] if with_collection else []

    def list_collection_names(self):
        return list(self.made)

    def create_collection(self, name):
        self.made.append(name)

    def __getitem__(self, name):
        return self.col


def test_ensure_vector_index_is_idempotent_and_matches_definition(monkeypatch):
    monkeypatch.setenv("VOYAGE_DIM", "1024")
    db = _FakeDb()
    assert semantic.ensure_vector_index(db) is True and semantic.ensure_vector_index(db) is False
    assert db.made == ["semantic_kb"]  # collection created before the index
    doc = db.col.created[0].document
    assert doc["type"] == "vectorSearch" and doc["name"] == "semantic_kb_vector"
    assert doc["definition"]["fields"] == [
        {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
        {"type": "filter", "path": "run_id"}]


def test_short_vector_results_are_topped_up_lexically(monkeypatch):
    store = _scrubbed(monkeypatch)
    sem = LocalSemantic(store.db, FakeEmbedder())  # nothing synced: the vector path returns []
    evidence = loop.latest_train_evidence(store, "t")
    from eval.synthetic_claims import public_view
    view = public_view(load_claims("train")[0])
    notes = set()
    got = loop.retrieve_similar(view, evidence, sem, "t", notes)
    assert got == loop.similar_past(view, evidence) and got and "short" in notes
    # partially filled: vector hit kept first, lexical fills the rest, no duplicates, never the claim itself
    sem.sync("t", evidence[:1])
    got = loop.retrieve_similar(view, evidence, sem, "t")
    ids = [e["claim_id"] for e in got]
    assert len(ids) == len(set(ids)) and view["id"] not in ids and len(ids) == 3


def test_similar_ignores_non_train_evidence(monkeypatch):
    store = _scrubbed(monkeypatch)
    sem = LocalSemantic(store.db, FakeEmbedder())
    evidence = loop.latest_train_evidence(store, "t")
    sem.sync("t", evidence)
    heldout = store.episodes({"run_id": "t", "split": "heldout"})
    from eval.synthetic_claims import public_view
    hits = sem.similar(public_view(load_claims("train")[0]), heldout, "t")
    assert hits == []


def test_memory_failures_never_crash_scrub_pass(monkeypatch):
    class Boom:
        model = "boom"

        def embed(self, texts, input_type):
            raise RuntimeError("voyage down")

    monkeypatch.setattr(scrubber_mod, "complete_json", fake_scrub_llm)
    monkeypatch.setattr(loop, "semantic_memory", lambda db: LocalSemantic(db, Boom()))
    store = Store(mongomock.MongoClient().db)
    store.bootstrap()
    loop.scrub_pass(store, "t", 0, slices=1)
    loop.scrub_pass(store, "t", 1, slices=1)  # second pass retrieves with an embedder that raises: lexical fallback
    assert store.db.semantic_kb.count_documents({}) == 0

    def no_construct(db):
        raise RuntimeError("cannot build embedder")

    monkeypatch.setattr(loop, "semantic_memory", no_construct)
    monkeypatch.setattr(loop, "sync_taxonomy", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tax")))
    assert loop.scrub_pass(store, "t", 2, slices=1)["train"]


def test_voyage_embedder_retries_rate_limits(monkeypatch):
    from voyageai.error import RateLimitError
    monkeypatch.setattr(semantic.time, "sleep", lambda s: None)
    emb = object.__new__(semantic.VoyageEmbedder)
    emb.model = "m"

    class Client:
        def __init__(self, fails):
            self.fails, self.n = fails, 0

        def embed(self, batch, model, input_type):
            self.n += 1
            if self.n <= self.fails:
                raise RateLimitError("slow down")
            return type("R", (), {"embeddings": [[1.0] for _ in batch]})()

    emb.client = Client(2)
    assert emb.embed(["a", "b"], "document") == [[1.0], [1.0]] and emb.client.n == 3
    emb.client = Client(3)
    try:
        emb.embed(["a"], "document")
        raise AssertionError("should have raised after 3 attempts")
    except RateLimitError:
        assert emb.client.n == 3
