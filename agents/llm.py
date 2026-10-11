"""Model-agnostic JSON completion via LiteLLM, with record/replay.

Models are named LiteLLM-style in env vars, e.g.
  SCRUBBER_MODEL=openai/<model-id>        WATCHER_MODEL=gemini/<model-id>
  SCRUBBER_MODEL=anthropic/<model-id>     (needs ANTHROPIC_API_KEY)
  SCRUBBER_MODEL=synthetic/<hf-model-id>  (Bruce's synthetic.new, OpenAI-compatible)

LLM_CASSETTE=record|replay|off (default record): every response is appended to
data/cassette.jsonl keyed by a hash of (model, messages), so a known-good run can be
replayed byte-for-byte on stage without network.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv()

CASSETTE = Path(os.environ.get("LLM_CASSETTE_PATH", "data/cassette.jsonl"))
SYNTHETIC_BASE = "https://api.synthetic.new/v1"

_cache: dict[str, str] | None = None


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

    last_err = None
    for attempt in range(3):
        t0 = time.time()
        try:
            resp = litellm.completion(messages=messages, temperature=0, max_tokens=max_tokens,
                                      response_format={"type": "json_object"}, **_route(model))
            raw = resp.choices[0].message.content or ""
            parsed = _parse(raw)
            if mode == "record":
                CASSETTE.parent.mkdir(parents=True, exist_ok=True)
                with CASSETTE.open("a") as f:
                    f.write(json.dumps({"key": key, "model": model, "content": raw}) + "\n")
                _load()[key] = raw
            return parsed, {"model": model, "latency_s": round(time.time() - t0, 2), "cached": False, "raw": raw}
        except Exception as e:  # noqa: BLE001 - surface after retries
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{model} failed after 3 attempts: {last_err}")
