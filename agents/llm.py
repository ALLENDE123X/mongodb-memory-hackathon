"""Model-agnostic JSON completion via LiteLLM, with record/replay.

Models are named LiteLLM-style in env vars, e.g.
  SCRUBBER_MODEL=openai/<model-id>        WATCHER_MODEL=gemini/<model-id>
  SCRUBBER_MODEL=anthropic/<model-id>     (needs ANTHROPIC_API_KEY)
  SCRUBBER_MODEL=synthetic/<hf-model-id>  (synthetic.new, OpenAI-compatible)

LLM_CASSETTE=record|replay|off (default record): every response is appended to
data/cassette.jsonl keyed by a hash of (model, messages), so a known-good run can be
replayed byte-for-byte on stage without network.

LLM_CONCURRENCY (default 6) bounds in-flight provider calls across all threads, so the scrub
pool and watcher cannot stampede a rate limit; rate-limit errors back off exponentially.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv()

CASSETTE = Path(os.environ.get("LLM_CASSETTE_PATH", "data/cassette.jsonl"))
SYNTHETIC_BASE = "https://api.synthetic.new/v1"

MAX_TRIES = 3

_cache: dict[str, str] | None = None
_lock = threading.Lock()
_slots = threading.BoundedSemaphore(int(os.environ.get("LLM_CONCURRENCY", "6")))


def _rate_limited(e: Exception) -> bool:
    """Rate limits and provider overload (Anthropic 529, ServiceUnavailable) both mean: wait longer."""
    text = f"{type(e).__name__} {e}".lower()
    return any(k in text for k in ("ratelimit", "rate limit", "429", "529", "overloaded", "serviceunavailable"))


def _backoff(attempt: int, e: Exception) -> float:
    """Seconds to wait before the next try: exponential on rate limits, linear otherwise."""
    return 5.0 * 2 ** attempt if _rate_limited(e) else 2.0 * (attempt + 1)


def _key(model: str, messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps([model, messages], sort_keys=True).encode()).hexdigest()


def _load() -> dict[str, str]:
    global _cache
    if _cache is None:
        _cache = {}
        if CASSETTE.exists():
            for line in CASSETTE.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    _cache[row["key"]] = row["content"]
    return _cache


def _route(model: str) -> dict:
    if model.startswith("synthetic/"):
        return {"model": "openai/" + model.split("/", 1)[1], "api_base": SYNTHETIC_BASE,
                "api_key": os.environ.get("SYNTHETIC_API_KEY")}
    return {"model": model}


def _parse(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def complete_json(model: str, system: str, user: str, *, max_tokens: int = 2000) -> tuple[dict, dict]:
    """Returns (parsed_json, meta). meta has model, latency_s, cached, raw."""
    if not model:
        raise RuntimeError("model not configured: set SCRUBBER_MODEL / WATCHER_MODEL in .env.local")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    mode = os.environ.get("LLM_CASSETTE", "record")
    key = _key(model, messages)
    if mode in ("record", "replay") and key in _load():
        raw = _load()[key]
        return _parse(raw), {"model": model, "latency_s": 0.0, "cached": True, "raw": raw}
    if mode == "replay":
        raise RuntimeError(f"cassette miss in replay mode for {model}")

    import litellm  # imported lazily so tests don't pay for it

    # Some models reject temperature=0 or JSON mode; drop what a provider doesn't support.
    # Run-to-run determinism comes from the cassette, not from sampling settings.
    litellm.drop_params = True
    last_err = None
    for attempt in range(MAX_TRIES):
        t0 = time.time()
        try:
            with _slots:  # held only for the call itself, never while backing off
                resp = litellm.completion(messages=messages, temperature=0, max_tokens=max_tokens,
                                          response_format={"type": "json_object"}, **_route(model))
            raw = resp.choices[0].message.content or ""
            parsed = _parse(raw)
            if mode == "record":
                with _lock:
                    CASSETTE.parent.mkdir(parents=True, exist_ok=True)
                    with CASSETTE.open("a") as f:
                        f.write(json.dumps({"key": key, "model": model, "content": raw}) + "\n")
                    _load()[key] = raw
            return parsed, {"model": model, "latency_s": round(time.time() - t0, 2), "cached": False, "raw": raw}
        except Exception as e:  # noqa: BLE001 - surface after retries
            last_err = e
            if attempt + 1 < MAX_TRIES:
                time.sleep(_backoff(attempt, e))
    raise RuntimeError(f"{model} failed after {MAX_TRIES} attempts: {last_err}") from last_err
