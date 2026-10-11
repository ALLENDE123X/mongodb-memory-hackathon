# Recursive Revenue-Cycle Harness

A claim-scrubbing agent that improves itself without weight updates. Before a healthcare claim goes out, the **scrubber** predicts whether the payer will pay or deny it and why. A second **watcher** agent mines the scrubber's misses against real adjudication outcomes (denials + CARC codes) and rewrites the scrubber's payer-specific rules, retrieval filters, and tool access. Every payer has its own hidden rulebook; the harness learns it from denials and keeps it in MongoDB Atlas.

All four memory types live in Atlas:
- **Semantic** - payer policy text and coding guidance (Voyage AI embeddings)
- **Episodic** - every scrub decision, reasoning, and the real outcome
- **Procedural** - the evolving per-payer ruleset the watcher derives
- **Taxonomic** - payer -> claim category -> denial category (CARC groups)

Billing and coverage only, never clinical judgment. All data is synthetic, no PHI.

Built at the SF Tech Week Harness Engineering hackathon (Shack15, Oct 11 2026) by Bruce Beckley, Arjun Prasad, Pranav Lende.

## Layout
- `agents/` - `scrubber.py` (scrubs a claim, writes a trace), `watcher.py` (proposes rule updates from traces)
- `memory/` - Atlas schema + unified retrieval across memory types
- `harness/` - the loop: scrub batch -> traces -> watcher -> rule update -> repeat
- `eval/` - 40 synthetic claims with ground-truth adjudication + metrics
- `demo/` - accuracy curve + rule-evolution timeline

## Setup
```
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env.local   # fill in values
python -m memory.atlas_schema
python -m eval.synthetic_claims
```

## New work and prior art
Every line of code in this repository was written during the hackathon (after the 10:30 AM PT start on Oct 11, 2026); see the commit history. The gated-learning design (candidate rule, replay against known outcomes, conditional promotion, restart from durable state) is adapted from Bruce Beckley's pre-event MemGit architecture notes. No MemGit code is used here.

All claims, members, providers, and payers are synthetic. `eval/PAYER_RULES.md` is the simulator's hidden answer key: the scrubber and watcher never read it.
