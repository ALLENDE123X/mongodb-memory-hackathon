# Recursive Revenue-Cycle Harness: Agent Operating Guide

Hackathon build (SF Tech Week, 2026-10-11, due 5:00 PM PT). A claim-scrubber agent and a watcher agent that learns hidden payer rules from denials and evolves the scrubber's rules from its own failure traces. Billing/coverage only, never clinical judgment; synthetic data only, no PHI. All memory lives in MongoDB Atlas (sandbox cluster). Judges need it to run inside the provided Atlas sandbox.

## Rules
- Secrets only in `.env.local` (gitignored). `.env.example` is the documented shape.
- `git add` explicit paths only. Never `git add -A` / `git add .`.
- `eval/PAYER_RULES.md` and each claim's `ground_truth` are hidden: the scrubber only ever sees `public_view(claim)`. The watcher sees outcomes (paid/denied + CARC) via traces, never PAYER_RULES.md.
- Python 3.12 venv at `.venv` (`uv pip install -r requirements.txt`).
