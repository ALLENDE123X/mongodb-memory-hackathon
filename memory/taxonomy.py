"""Taxonomic memory: payer -> claim category -> denial category, with counts.

Docs in `taxonomy`: {kind: "payer_category_denial", key: "<payer>|<category>|<carc-group>", count, ...}.
Category comes from the CPT family of the billed lines, never from the hidden `category` field.
CARC group comes from the remittance's denial codes.

Counts are always derived from the training evidence list the loop hands in (latest observation
per distinct training claim), so re-scrubbing the same claims every iteration cannot inflate them
and held-out claims can never contribute. Syncing is a full, idempotent rebuild of this run's docs.
A claim with several CPT families or several CARC groups counts once under each combination.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone

KIND = "payer_category_denial"

# (low, high, category); first match wins, anything else is "other".
CPT_FAMILIES = (
    (99202, 99499, "evaluation_management"),
    (70010, 79999, "imaging"),
    (80047, 89398, "lab"),
    (97010, 97799, "therapy"),
    (10004, 69990, "surgery"),
)

CARC_GROUPS = {
    "197": "authorization", "29": "timely_filing", "4": "coding_modifier", "5": "coding_modifier",
    "97": "bundling", "119": "benefit_limit", "167": "coverage", "242": "coverage", "16": "missing_info",
}


def cpt_category(cpt: str) -> str:
    if cpt.isdigit():
        n = int(cpt)
        for lo, hi, name in CPT_FAMILIES:
            if lo <= n <= hi:
                return name
    return "other"


def carc_group(carc: str) -> str:
    return CARC_GROUPS.get(str(carc), "other")


def _keys(ep: dict) -> list[str]:
    obs = ep["observed"]
    if obs["outcome"] != "denied":
        return []
    cats = sorted({cpt_category(c) for c in ep["cpts"]})
    groups = sorted({carc_group(c) for c in obs["carcs"]})
    return [f"{ep['payer']}|{cat}|{grp}" for cat in cats for grp in groups]


def taxonomy_counts(evidence: list[dict]) -> dict[str, dict]:
    """{key: {"count": n, "claim_ids": [...]}} over distinct claims (latest observation each)."""
    latest = {ep["claim_id"]: ep for ep in evidence
              if ep.get("split") == "train" and ep.get("visible_to_learning", True)}
    claims: dict[str, list[str]] = defaultdict(list)
    for cid, ep in sorted(latest.items()):
        for key in _keys(ep):
            claims[key].append(cid)
    return {k: {"count": len(v), "claim_ids": v} for k, v in sorted(claims.items())}


def taxonomy_summary(evidence: list[dict], payer: str, limit: int = 8) -> str:
    """Compact watcher-facing line, e.g. 'BAYVIEW: lab -> bundling x2; imaging -> authorization x1'."""
    rows = Counter()
    for key, doc in taxonomy_counts(evidence).items():
        p, cat, grp = key.split("|")
        if p == payer:
            rows[(cat, grp)] = doc["count"]
    if not rows:
        return f"{payer}: no denials observed yet"
    top = sorted(rows.items(), key=lambda t: (-t[1], t[0]))[:limit]
    return f"{payer}: " + "; ".join(f"{cat} -> {grp} x{n}" for (cat, grp), n in top)


def sync_taxonomy(db, run_id: str, evidence: list[dict]) -> int:
    """Rebuild this run's taxonomy docs from the evidence. Returns the number of docs."""
    counts = taxonomy_counts(evidence)
    now = datetime.now(timezone.utc)
    for key, doc in counts.items():
        payer, cat, grp = key.split("|")
        db.taxonomy.update_one(
            {"kind": KIND, "key": key},
            {"$set": {"payer": payer, "category": cat, "carc_group": grp, "count": doc["count"],
                      "claim_ids": doc["claim_ids"], "run_id": run_id, "updated_at": now}},
            upsert=True)
    db.taxonomy.delete_many({"kind": KIND, "run_id": run_id, "key": {"$nin": list(counts)}})
    return len(counts)
