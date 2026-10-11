# Payer Rules (HIDDEN GROUND TRUTH)

**Do not load this file into the scrubber or watcher prompts.** It is the answer key for
`eval/synthetic_claims.json`. The watcher must rediscover these rules from denials
(outcome + CARC) stored in memory.

All payers, members, providers and claims are fictional. NPIs are fake (prefix `99`), member IDs are `SYN-xxxxxx`.
Scope: billing, coverage and coding rules only. Nothing here is clinical guidance.

## Payers

| Payer | Model | Notes |
|---|---|---|
| `NORTHSTAR` | Commercial PPO | Pays out-of-network at OON benefit level. Strict auth and benefit limits. |
| `CASCADE` | Medicare Advantage-style | Medicare-like coding rules, generous filing window, no imaging auth. |
| `BAYVIEW` | Medicaid managed care-style | Short filing window, strict network, bundles venipuncture. |

## Adjudication convention

A claim is `denied` if any line is denied; `ground_truth.carc` is the CARC on the denied line(s).
Each denied claim violates exactly one rule, except CLM-031, which has both generic defects (PR-11 + PR-12).

## Rules

| ID | Payer(s) | Rule | CARC | Why a generic reviewer misses it |
|---|---|---|---|---|
| PR-01 | NORTHSTAR | Prior authorization required for advanced imaging CPT 70551 (MRI brain), 72148 (MRI lumbar), 74177 (CT abd/pelvis w/ contrast). No `prior_auth_number` = deny. CASCADE and BAYVIEW do not require auth for these. | 197 | Auth lists are payer-specific; a generic reviewer either never flags imaging auth or flags it everywhere (over-flagging CASCADE). |
| PR-02 | All (payer-specific windows) | Timely filing measured from DOS to submission date: BAYVIEW 90 days, NORTHSTAR 180 days, CASCADE 365 days. | 29 | Generic default assumption is "about a year", so 105-200 day lags look fine. |
| PR-03 | CASCADE | Modifier 25 is required on an E/M billed the same day as ANY procedure, injection or administration code, including 96372 and vaccine admin G0008/90471. NORTHSTAR and BAYVIEW pay E/M + vaccine admin without 25. | 4 | Many reviewers only expect 25 with minor surgical procedures, not with injections or vaccine admin. |
| PR-04 | NORTHSTAR | Modifier 59 is not accepted to bypass NCCI pair edits; the column-2 code must carry XE/XS/XP/XU (e.g. 97140 with 97530; 11102 with 17110). | 4 | Modifier 59 is the textbook "distinct procedure" fix, so a claim with 59 looks correct. |
| PR-05 | BAYVIEW | Venipuncture 36415 is bundled into lab panels/tests (80053, 80061, 85025, etc.); separately billed 36415 is denied. NORTHSTAR and CASCADE pay 36415 separately. | 97 | 36415 is routinely payable under Medicare and commercial plans. |
| PR-06 | NORTHSTAR | Preventive visits 99395/99396: one per rolling 365 days (not calendar year). Compare DOS to `context.last_preventive_dos`. | 119 | Generic assumption is "one per calendar year", so a visit in a new calendar year looks fine. |
| PR-07 | NORTHSTAR | Physical therapy (97110/97112/97140/97530) limited to 20 visits per plan year. `context.visit_count_ytd` is visits already used; deny when it is >= 20. CASCADE has no hard cap (KX supports continuation). | 119 | Visit caps vary by plan; nothing on the claim says where the cap is. |
| PR-08 | CASCADE | 25-OH vitamin D (82306) covered only with a supporting diagnosis family: E55.x (vitamin D deficiency), M80.x/M81.x (osteoporosis), N18.x (CKD), K90.x (malabsorption), E21.x (hyperparathyroidism). Fatigue, obesity, screening are not covered. | 167 | Fatigue/obesity look like a reasonable indication; other payers pay them. |
| PR-09 | BAYVIEW | Telehealth E/M must be billed POS 10 (not 02) AND modifier 95 (GT not accepted). Wrong POS -> CARC 5; missing/incorrect modifier -> CARC 4. CASCADE pays POS 02 without 95. | 5 / 4 | POS 02 and GT are both legitimate telehealth codes elsewhere. |
| PR-10 | BAYVIEW | Non-emergency services from out-of-network providers are not covered, even with a referral on file. Emergency services (POS 23) are covered OON. NORTHSTAR pays OON at OON benefit. | 242 | A generic reviewer cannot tell which plan types have OON benefits; the referral looks like it should cure it. |
| PR-11 | All (generic) | Every service line must have at least one valid diagnosis pointer. | 16 | Should be caught by any competent reviewer. |
| PR-12 | All (generic) | Every line must have a charge greater than $0.00. | 16 | Should be caught by any competent reviewer. |

## Coverage in the dataset

Each rule is violated in at least one `train` claim and one `heldout` claim. PR-02 is violated
twice in train (BAYVIEW and NORTHSTAR) and once in heldout (BAYVIEW).

Paid "trap" claims (look risky, are clean under that payer):
- CASCADE advanced imaging without auth (PR-01 contrast), train + heldout
- CASCADE claim submitted 262 days after DOS (PR-02 contrast)
- CASCADE 99213-25 + 11102 (PR-03 done right); BAYVIEW E/M + vaccine admin without 25 (PR-03 contrast)
- NORTHSTAR 97530 + 97140-XU (PR-04 done right)
- CASCADE 80053 + 36415 (PR-05 contrast); BAYVIEW lab with no 36415
- NORTHSTAR preventive more than 365 days after the last one (PR-06)
- CASCADE PT at visit 27 with KX (PR-07 contrast)
- NORTHSTAR 82306 with fatigue (PR-08 contrast)
- CASCADE telehealth POS 02 without 95 (PR-09 contrast)
- NORTHSTAR out-of-network office visit; BAYVIEW out-of-network emergency visit POS 23 (PR-10 contrast)
- NORTHSTAR imaging with an auth number on file (PR-01 done right)
