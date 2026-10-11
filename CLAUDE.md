# Recursive Revenue-Cycle Harness: Agent Operating Guide

Hackathon build (SF Tech Week, 2026-10-11, due 5:00 PM PT). A claim-scrubber agent and a watcher agent that learns hidden payer rules from denials and evolves the scrubber's rules from its own failure traces. Billing/coverage only, never clinical judgment; synthetic data only, no PHI. All memory lives in MongoDB Atlas. Judges need it to run inside the provided Atlas sandbox.

## HARD STOPS (non-negotiable for every agent)
1. **Never merge your own PR.** Implementation agents open a PR against `main` and stop. Only the orchestrator merges.
2. **`git add` explicit paths only.** Never `git add -A` / `git add .`. Never commit `.env*` (except `.env.example`), `data/cassette.jsonl`, `data/*.log`, or debugging artifacts.
3. **Never write to the live Atlas database.** `MONGODB_URI` in `.env.local` points at the shared demo cluster that a real run may be using. All tests use `mongomock`. Do not run `python -m harness.loop`, `harness.probe`, or `demo.export` against Atlas, and do not drop or modify any collection there. If a ticket needs a live check, stop and say so in the PR body; the orchestrator runs it.
4. **Never call paid model APIs from tests.** Tests monkeypatch `complete_json` (see `tests/test_loop.py`). CI has no API keys.
5. **Hidden answer key stays hidden.** `eval/PAYER_RULES.md` and each claim's `ground_truth` / `split` must never reach the scrubber or watcher prompts or retrieval. The scrubber only sees `public_view(record)`. Held-out outcomes must never become learning evidence or retrieval memory.
6. **The gate computes its own numbers.** Never let a model-reported score decide a promotion. Pinned thresholds live in `harness/gate.py`.
7. **No `git stash`.** Agents run in parallel worktrees and the stash is shared.

## Conventions
- Python 3.12. Create a venv in your worktree: `uv venv -p 3.12 && uv pip install -r requirements.txt`. Run `python -m pytest -q` before opening a PR; CI runs the same.
- One ticket, one branch, one PR. PR title starts with the ticket id (`JOB-<n>: ...`).
- Keep the existing style: module docstring explaining the why, small functions, no new frameworks.
- Secrets only in `.env.local` (gitignored). `.env.example` is the documented shape.

## Map
- `eval/` synthetic 837P claims (`synthetic_claims.json`), loader/validator, metrics. `PAYER_RULES.md` = hidden answer key.
- `harness/payer_sim.py` deterministic payer returning 835-style remittances. `harness/features.py` flattens a claim. `harness/rules.py` executable rule language. `harness/gate.py` replay gate. `harness/loop.py` the recursive loop. `harness/probe.py` stage probe.
- `agents/` LiteLLM JSON client with record/replay cassette, scrubber, watcher.
- `memory/store.py` Atlas collections: episodes, rules, rulesets, heads (CAS), evaluations, runs, audits, semantic_kb, taxonomy.
- `demo/export.py` exports a run from Atlas to `demo/run.json`.
