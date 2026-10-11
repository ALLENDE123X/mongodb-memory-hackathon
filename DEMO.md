# Demo runbook

Three minutes on stage, one minute on video, then Q&A. Two people: the **Narrator** talks, the **Driver** types. Fill in names here: Narrator TBD, Driver TBD.

Every beat is one command and prints big headings in plain English. The Driver never edits anything live.

## Before you walk on
1. Terminal at the repo root, font large, dark background, venv active. Run `python -m harness.demo --run-id RUN --beat 5` once so imports are warm and you know the run id is right (RUN = final run id, TBD). Beat 5 only reads.
2. Confirm Atlas answers: `python -m memory.atlas_schema` prints `ping` and the collection list (it is idempotent: it ensures indexes and ruleset v0 exist and does not touch existing documents). No Wi-Fi, no ping: use the fallback below now, not mid-talk.
3. Have the demo page (issue #5, `python -m http.server -d demo`) open in a second tab as the closing visual, if it has merged.
4. Know your numbers: final held-out balanced accuracy TBD, baseline TBD, rules learned TBD, rules rejected TBD. They come from beat 5, so read them off the screen rather than from memory.

## The three minutes
Run the whole thing with `python -m harness.demo --run-id RUN`, or one beat at a time with `--beat N`. One beat at a time is better on stage: it gives the Narrator control of the pacing.

| Time | Beat | Driver runs | Narrator says |
|---|---|---|---|
| 0:00 | Setup | nothing | "Every payer has a hidden rulebook. A claim scrubber that has never seen it waves bad claims through. Ours learns the rulebook from denials, keeps it in MongoDB Atlas, and only accepts a new rule after it proves itself against real outcomes." |
| 0:20 | 1 Before | `python -m harness.demo --run-id RUN --beat 1` | "Here is a held-out claim. The watcher never saw it. The baseline said paid. The payer denied it with the code on screen. This is a real miss, with the payer's own reason." |
| 0:50 | 2 Learned | `... --beat 2` | "After learning, a rule now covers it. Here are the training claims that taught it, and here is what the gate computed: how many real misses it fixes, how many real outcomes it contradicts. The gate computes those numbers itself. No model gets to grade its own homework. And note the line at the bottom: this held-out claim was not in the evidence." |
| 1:30 | 3 Refuses | `... --beat 3` | "Now we try to cheat. Someone proposes 'this payer denies all advanced imaging'. Sounds plausible, and it would catch more denials. The gate replays it against real outcomes, finds a claim that was paid, and rejects it. Look at the last two lines: the probe wrote exactly one rejected rule, one evaluation and one audit record, and the active ruleset did not move." |
| 2:00 | 4 Remembers | `... --beat 4` | "Memory is the database, not the process. New connection, no caches. It loads the active ruleset from Atlas, same version, and scrubs a claim it has never held in memory. No model call, just the rule path, and it matches what the payer returns." |
| 2:30 | 5 Score | `... --beat 5` | "Held-out claims only. Baseline to final, balanced accuracy so denying everything cannot win, and unnecessary holds so over-flagging shows up. There are 22 of them, so one claim is about 4.5 points. Read it as a trend." |
| 2:50 | Close | demo page, if merged | "Learned from outcomes, refused a bad rule, remembered across a restart. Everything lives in MongoDB Atlas." |

Choosing the claim: beat 1 picks the lowest-numbered held-out claim the baseline missed that a promoted rule now covers. Pass `--claim CLM-0xx` to feature a different one (used by beats 1 and 2; beat 4 automatically picks a different claim, preferring another payer).

## What beat 3 writes (say this if asked)
Beat 3 is the only step that writes. Against Atlas it adds exactly one rule document (status `rejected`), one evaluation document and one audit document, and it leaves the active ruleset and head unchanged. The beat prints these counts and the head version before and after, measured from the database, not asserted.

Rehearse beat 3 with `--mock`, and run it against Atlas at most once or twice. Every live run adds another probe rule (`PROBE-nnn`) to the `rules` collection and shows up in the demo export, so a dozen rehearsals make the ledger noisy.

## Fallback if Wi-Fi dies
`python -m harness.demo --mock` runs all five beats with no network and no Atlas. It seeds an in-memory database by running the real loop with a fake model standing in for the scrubber and the watcher, so the gate, the store, the probe and the rule path are all real.

What changes, and what to say: the rules in mock mode come from a stand-in watcher and the baseline is a scrubber that always says paid, so the numbers on screen are mock numbers and are not our results. Say so plainly: "This is the offline rehearsal. Same code path, stand-in model. The recorded numbers are on the results slide." Do not read mock accuracy as the headline.

`LLM_CASSETTE=replay` makes a model-backed run reproduce byte for byte from `data/cassette.jsonl` with no model calls. It helps when the model API is down but Atlas is up (for example, re-running `harness.loop` for a fresh run). It does not help when Atlas is unreachable, because the stage script reads from the database. Wi-Fi down means `--mock`.

## One-minute video shot list
Screen recording at 1080p with a voiceover, terminal text at least 28 pt.

| Seconds | Shot | Voiceover |
|---|---|---|
| 0-8 | Title card: "Recursive Revenue-Cycle Harness", then the architecture diagram from the README | "Every payer has a hidden rulebook. Our scrubber learns it from denials, and proves each rule before it uses it." |
| 8-20 | Beat 1 output, zoom on `baseline predicted: PAID` and the payer's CARC line | "A held-out claim. The baseline would have submitted it. The payer denied it." |
| 20-32 | Beat 2 output, highlight the training claims and the gate stats line | "The watcher learned a rule from training claims. The gate replayed it against real outcomes: one fix, zero regressions, computed by the harness." |
| 32-42 | Beat 3 output, highlight `REJECTED` and `Active ruleset/head unchanged` | "An over-broad rule is rejected, and the active ruleset does not move." |
| 42-50 | Beat 4 output, highlight `Loaded ruleset v... from the database` and `no model call` | "A fresh process loads the ruleset from MongoDB Atlas and scrubs by rule, no model call." |
| 50-60 | Beat 5 table, then the Atlas collections or the demo page | "Held-out balanced accuracy goes from TBD to TBD. Everything is synthetic, and everything lives in Atlas." |

## Q&A crib
**Why not fine-tune?** Rules are inspectable, versioned and revertible. A payer changes a policy and we add or retire one rule; fine-tuning bakes it into weights and we cannot show a judge, an auditor or a payer why a claim was held. There is also no gradient to compute here: the signal is a denial code, not a labeled dataset.

**Why MongoDB?** The loop needs different shapes in one place: episodes with nested remittances, rule documents with conditions, immutable ruleset versions, and a head pointer we update with a compare-and-swap so two writers cannot clobber each other. Restart recovery is just reading the database. Embeddings for semantic memory (issue #4) sit next to the rest.

**"You wrote the hidden rules, so of course it can learn them."** True that we wrote the simulator, and the README says so. The check is transfer: the watcher sees 24 training claims and is scored on 22 held-out claims it never saw, including paid trap claims that look risky but are clean for that payer (for example, imaging without auth at a payer that does not require it). A rule that memorizes a training claim, or one that over-flags, loses points there. Several rules also mirror real public CMS policy, listed in the README honesty section.

**"Couldn't it just deny everything?"** That is why we report balanced accuracy and unnecessary holds, not plain accuracy. Denying everything gets every denial and fails every paid claim. The gate also rejects any rule that fires on a claim that was paid. Beat 3 shows that live.

**Is it safe? Who decides what gets promoted?** The gate does, from replayed outcomes. A model-reported score never decides anything. The rule is pinned in code: at least one real fix and zero contradicted outcomes. Promotion is an append-only ruleset version plus a compare-and-swap on the head, so every version stays auditable and the head can point back at an earlier one, and the probe shows a bad rule being refused.

**Does it leak the answer key?** The scrubber sees only the claim and its context. The watcher sees training outcomes, never the simulator's rule table or held-out outcomes. Held-out episodes are flagged not visible to learning and never reach the gate's evidence. Beat 2 prints that the featured held-out claim is absent from the evidence.

**What about humans?** That is the v2 plan in the README roadmap (issues #7 to #11). A review queue lets a biller edit, override or hold a flagged claim. The semantics matter: if a human overrides a hold and the payer pays the claim, that is strong evidence the rule was wrong, and after two such confirmed false holds on distinct claims the harness automatically parks the rule through a new ruleset version. Held claims nobody resubmits are excluded as censored, because we never learn what would have happened. None of that is built yet.

**How big is the improvement, really?** Read it off beat 5: baseline TBD, final TBD, on 22 held-out claims. One claim is about 4.5 points, so we call it a trend. We would need more claims and more payers before claiming a benchmark.

**What is real and what is simulated?** Claims, members and payers are synthetic and the payer is a simulator that returns 835-style remittances. The agent loop, the gate, the store, the Atlas persistence and the restart behavior are real code running against a real database.
