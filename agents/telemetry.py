"""Raindrop telemetry: third-party observability for the scrubber and the watcher.

Every call here is fire-and-forget and write-only. Nothing is ever read back from Raindrop, so
nothing it knows can reach the watcher or the scrubber (held-out isolation).

It is a no-op unless RAINDROP_WRITE_KEY is set, and always a no-op under pytest unless a test
injects a fake SDK with `set_sdk`. A telemetry failure never breaks the loop.

Env:
  RAINDROP_WRITE_KEY       enables sending
  RAINDROP_SAMPLE          fraction of events to send, 0..1 (default 1.0). Sampling is keyed on the
                           event id, so a Signal is sent exactly when its event was.
  RAINDROP_DISABLE         comma list of event kinds to skip: claim_scrub, payer_outcome,
                           rule_proposal, gate_verdict, ruleset_promotion
  RAINDROP_DASHBOARD_URL   linked from the demo export (not used for sending)

Payloads are built from episodes and rules only (never from claim records), and every string is
scrubbed of the hidden answer key's rule ids as a second line of defence.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
from functools import wraps

from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv()

KINDS = ("claim_scrub", "payer_outcome", "rule_proposal", "gate_verdict", "ruleset_promotion")
_HIDDEN_KEYS = {"ground_truth", "split", "category", "violated_rules"}
_ANSWER_KEY_ID = re.compile(r"\bPR-\d+\w*(?:-\w+)*", re.I)

_lock = threading.Lock()
_sdk = None          # the raindrop module, or a test fake
_tried = False
_counts: dict[str, int] = {}


# -- lifecycle ----------------------------------------------------------------------------

def set_sdk(sdk) -> None:
    """Inject an SDK object (tests). Pass None to reset to the environment-driven default."""
    global _sdk, _tried
    with _lock:
        _sdk, _tried = sdk, sdk is not None
        _counts.clear()


def _client():
    global _sdk, _tried
    if _sdk is not None:
        return _sdk
    if _tried or "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
        return None
    with _lock:
        _tried = True
        key = os.environ.get("RAINDROP_WRITE_KEY")
        if not key:
            return None
        try:
            import raindrop.analytics as raindrop
            raindrop.init(key, tracing_enabled=False, auto_instrument=False, app_git=False, local_workshop_url=None)
            _sdk = raindrop
        except Exception:  # noqa: BLE001
            _sdk = None
    return _sdk


def enabled() -> bool:
    return _client() is not None


def flush() -> None:
    sdk = _client()
    if sdk:
        _safe(sdk.flush)


def shutdown() -> None:
    sdk = _client()
    if sdk:
        _safe(sdk.shutdown)


def summary(prior: dict | None = None) -> dict:
    """Events sent by kind (this process plus `prior` from a resumed run). Stored with the run
    for the demo export."""
    sent = {k: _counts.get(k, 0) + (prior or {}).get(k, 0) for k in {*_counts, *(prior or {})}}
    return {"sample": _sample_rate(), "events_sent": sent, "total": sum(sent.values())}


# -- gating and sanitising ------------------------------------------------------------------

def _safe(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except Exception:  # noqa: BLE001
        return None


def _sample_rate() -> float:
    try:
        return min(1.0, max(0.0, float(os.environ.get("RAINDROP_SAMPLE", "1.0"))))
    except ValueError:
        return 1.0


def _on(kind: str, event_id: str) -> bool:
    if kind in {k.strip() for k in os.environ.get("RAINDROP_DISABLE", "").split(",")}:
        return False
    rate = _sample_rate()
    if rate >= 1.0:
        return True
    bucket = int(hashlib.sha256(event_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < rate


def clean(obj):
    """Drop hidden-answer-key fields and redact answer-key rule ids, recursively."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if k not in _HIDDEN_KEYS}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, str):
        return _ANSWER_KEY_ID.sub("[rule]", obj)
    return obj


def _dump(obj) -> str:
    return json.dumps(clean(obj), sort_keys=True, default=str)


def _never_raises(fn):
    """Telemetry must never break the loop: payload building is guarded as well as the SDK call."""
    @wraps(fn)
    def wrapper(*args, **kw):
        try:
            return fn(*args, **kw)
        except Exception:  # noqa: BLE001
            return None
    return wrapper


def _send(kind: str, key: str, call: str, /, parent: str | None = None, **kw) -> None:
    """parent: the event kind a Signal hangs off; the Signal is sent only if that was sent too."""
    sdk = _client()
    if not sdk or not _on(kind, key) or (parent and not _on(parent, key)):
        return
    try:
        getattr(sdk, call)(**kw)
    except Exception:  # noqa: BLE001
        return
    _counts[kind] = _counts.get(kind, 0) + 1


# -- scrub events -----------------------------------------------------------------------------

def claim_summary(ep: dict) -> dict:
    """Compact synthetic claim summary from an episode: no member ids, no answer key."""
    f = ep["features"]
    return {"payer": ep["payer"], "pos": f.get("pos"), "dx": f.get("dx_codes"),
            "lines": [{"cpt": ln["cpt"], "modifiers": ln["modifiers"]} for ln in f.get("lines", [])]}


def error_type(ep: dict) -> str:
    pred, obs = ep["predicted"], ep["observed"]
    if obs["outcome"] == "denied" and pred["predicted_outcome"] != "denied":
        return "missed_denial"
    if obs["outcome"] == "paid" and pred["predicted_outcome"] == "denied":
        return "unnecessary_hold"
    if ep.get("correct"):
        return "none"
    return "wrong_reason"


@_never_raises
def scrub_event(ep: dict) -> None:
    """One claim_scrub event plus its payer_outcome Signal, from a finished episode."""
    if not enabled():
        return
    pred, obs, eid = ep["predicted"], ep["observed"], ep["_id"]
    decision = {k: pred.get(k) for k in ("predicted_outcome", "carc", "confidence", "reasons", "source", "action")}
    _send("claim_scrub", eid, "track_ai", user_id=ep["payer"], event="claim_scrub", event_id=eid,
          model=pred.get("model") or "rule", input=_dump(claim_summary(ep)), output=_dump(decision),
          convo_id=f"{ep['run_id']}:{ep['iteration']}",
          properties={"ruleset_version": ep["ruleset_version"], "source": str(pred.get("source")),
                      "split": ep["split"], "iteration": ep["iteration"]})
    carcs = list(obs.get("carcs") or [])
    etype = error_type(ep)
    comment = f"denied CO-{', CO-'.join(carcs)}" if obs["outcome"] == "denied" and carcs else obs["outcome"]
    _send("payer_outcome", eid, "track_signal", parent="claim_scrub", event_id=eid, name="payer_outcome", signal_type="feedback",
          sentiment="POSITIVE" if ep.get("correct") else "NEGATIVE", comment=comment,
          properties={"outcome": obs["outcome"], "carcs": ",".join(carcs), "error_type": etype})


# -- watcher proposals and gate verdicts ----------------------------------------------------------

def _rule_event_id(run_id: str, rule_id: str) -> str:
    return f"{run_id}:{rule_id}"


@_never_raises
def rule_proposal(run_id: str, iteration: int, payer: str, rule: dict, evidence_ids: list[str], model: str | None) -> None:
    if not enabled():
        return
    eid = _rule_event_id(run_id, rule["rule_id"])
    _send("rule_proposal", eid, "track_ai", user_id=payer, event="rule_proposal", event_id=eid,
          model=model or "watcher", input=_dump({"payer": payer, "evidence": evidence_ids}), output=_dump(rule),
          convo_id=f"{run_id}:{iteration}", properties={"iteration": iteration})


@_never_raises
def gate_verdict(run_id: str, rule_id: str, verdict: dict, promoted: bool) -> None:
    if not enabled():
        return
    eid = _rule_event_id(run_id, rule_id)
    _send("gate_verdict", eid, "track_signal", parent="rule_proposal", event_id=eid, name="gate_verdict", signal_type="feedback",
          sentiment="POSITIVE" if promoted else "NEGATIVE", comment=clean(str(verdict.get("reason") or verdict.get("verdict"))),
          properties={"verdict": verdict.get("verdict"), "fixes": len(verdict.get("fixes") or []),
                      "regressions": len(verdict.get("regressions") or [])})


@_never_raises
def ruleset_promotion(run_id: str, iteration: int, payer: str, rule_id: str, version: int) -> None:
    if not enabled():
        return
    eid = f"{run_id}:v{version}"
    _send("ruleset_promotion", eid, "track", user_id=payer, event="ruleset_promotion", event_id=eid,
          properties={"ruleset_version": version, "rule_id": rule_id, "iteration": iteration})
