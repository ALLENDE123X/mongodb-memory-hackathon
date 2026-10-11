# Payer Rules (HIDDEN GROUND TRUTH)

**Do not load this file into the scrubber or watcher prompts.** It is the answer key for
`eval/synthetic_claims.json`. The watcher must rediscover these rules from denials
(outcome + CARC) stored in memory.

All payers, members, providers and claims are fictional. NPIs are fake (prefix `99`), member IDs are `SYN-xxxxxx`.
Scope: billing, coverage and coding rules only. Nothing here is clinical guidance.

## Record shape

Each record in `synthetic_claims.json` is `{id, payer, category, split, ground_truth, claim, context}`.
`claim` is an 837P professional claim request body using the Stedi / Change Healthcare JSON field
names (`POST /change/medicalnetwork/professionalclaims/v3/submission`,
https://www.stedi.com/docs/api-reference/post-healthcare-claims): `claimInformation.serviceLines[].professionalService`
with `procedureCode` / `procedureModifiers` / `lineItemChargeAmount` / `serviceUnitCount` /
`compositeDiagnosisCodePointers.diagnosisCodePointers`, `healthCareCodeInformation` (ABK then ABF, no
decimal point), `claimSupplementalInformation.priorAuthorizationNumber`, etc. `context` holds facts a
billing team has that are not on the 837 (`submission_date`, `provider_in_network`, `visit_count_ytd`,
`last_preventive_dos`, ...). The scrubber sees `public_view()` = `{id, payer, claim, context}`.

`harness/payer_sim.py` implements these rules in code (`adjudicate(record)` returns an 835 ERA in the
Stedi 835 report JSON shape; `outcome_summary(remit)` gives outcome + CARCs + RARCs). It never reads this
file or `ground_truth`. `tests/test_payer_sim.py` checks it reproduces every `ground_truth`.

## Payers

| Payer | Model | Payer ID (837 `tradingPartnerServiceId`) | `claimFilingCode` | Notes |
|---|---|---|---|---|
| `NORTHSTAR` | Commercial PPO | `SYNNS` | `CI` (Commercial Insurance Co.) | Pays out-of-network at OON benefit level. Strict auth and benefit limits. |
| `CASCADE` | Medicare Advantage-style | `SYNCA` | `16` (HMO Medicare Risk) | Medicare-like coding rules, Medicare filing window, no imaging auth. |
| `BAYVIEW` | Medicaid managed care-style | `SYNBV` | `MC` (Medicaid) | Short filing window, strict network, bundles venipuncture. |

Claim filing indicator codes are the real X12 values from Stedi's claims code lists
(https://www.stedi.com/docs/healthcare/claims-code-lists#claim-filing-indicator-codes). Payer IDs are fictional.

## Adjudication convention

A claim is `denied` if any line is denied; `ground_truth.carc` is the CARC on the denied line(s).
In the simulator's 835, a denied line pays $0.00 with one CO adjustment for the full line charge; a paid
line pays a deterministic 70-84% of charge with a CO-45 adjustment for the rest. Claim-level edits
(timely filing PR-02, BAYVIEW out-of-network PR-10) deny every line; the others deny only the offending
line, so a "denied" claim can still be partially paid (`claimStatusCode` 1) unless every line is denied
(`claimStatusCode` 4). Every CARC / RARC emitted was checked against the X12 code lists
(https://x12.org/codes/claim-adjustment-reason-codes, https://x12.org/codes/remittance-advice-remark-codes,
all active, no stop date, checked 2026-10-11).
Each denied claim violates exactly one rule, except CLM-031, which has both generic defects (PR-11 + PR-12).

## Rules

Grounding: **CMS** = the rule mirrors real public CMS policy (source below); **contractual (fictional)** =
a made-up per-payer contract term.

| ID | Payer(s) | Rule | CARC / RARC | Grounding | Why a generic reviewer misses it |
|---|---|---|---|---|---|
| PR-01 | NORTHSTAR | Prior authorization required for advanced imaging CPT 70551 (MRI brain), 72148 (MRI lumbar), 74177 (CT abd/pelvis w/ contrast). No `claimSupplementalInformation.priorAuthorizationNumber` = deny. CASCADE and BAYVIEW do not require auth for these. | 197 / M62 | contractual (fictional) | Auth lists are payer-specific; a generic reviewer either never flags imaging auth or flags it everywhere (over-flagging CASCADE). |
| PR-02 | All (payer-specific windows) | Timely filing measured from DOS to `context.submission_date`. CASCADE: Medicare's rule, 12 months / 1 calendar year after the date of service (deny if received after DOS + 1 year). NORTHSTAR: 180 days (contract). BAYVIEW: 90 days (contract). Denies every line. | 29 / - | CASCADE window: **CMS** [1]. NORTHSTAR / BAYVIEW windows: contractual (fictional) | Generic default assumption is "about a year", so 105-200 day lags look fine. |
| PR-03 | CASCADE | Modifier 25 is required on an E/M billed the same day as ANY procedure (CPT 10004-69990), injection or administration code, including 96372 and vaccine admin G0008/90471. NORTHSTAR and BAYVIEW pay E/M + vaccine admin without 25. | 4 / N822 | contractual (fictional) | Many reviewers only expect 25 with minor surgical procedures, not with injections or vaccine admin. |
| PR-04 | NORTHSTAR | NCCI PTP edits with modifier indicator 1: NORTHSTAR does not accept modifier 59 to bypass them; the column-2 code must carry XE/XS/XP/XU. Pairs used: 97110 + 97164 (PT re-evaluation, needs XU) and 17000 + 11102 (biopsy of a separate lesion, needs XS). CASCADE and BAYVIEW accept 59 (as Medicare does). | 4 / N823 | Edit pairs and indicator: **CMS** [2][3]. 59-not-accepted: contractual (fictional) | Modifier 59 is the textbook "distinct procedure" fix, so a claim with 59 looks correct. |
| PR-05 | BAYVIEW | Venipuncture 36415 is bundled into lab panels/tests (CPT 80047-89398, e.g. 80053, 80061, 85025); separately billed 36415 is denied. NORTHSTAR and CASCADE pay 36415 separately. | 97 / M15 | contractual (fictional); contrast is real: NCCI does not bundle 36415 into lab panels and Original Medicare pays it under the Clinical Laboratory Fee Schedule [4] | 36415 is routinely payable under Medicare and commercial plans. |
| PR-06 | NORTHSTAR | Preventive visits 99381-99397: one per rolling 365 days (not calendar year). Deny when DOS - `context.last_preventive_dos` < 365 days. | 119 / N130 | contractual (fictional) | Generic assumption is "one per calendar year", so a visit in a new calendar year looks fine. |
| PR-07 | NORTHSTAR | Physical therapy (97110/97112/97140/97530, evals 97161-97164) limited to 20 visits per plan year. `context.visit_count_ytd` is visits already used; deny PT lines when it is >= 20. CASCADE has no hard cap (KX supports continuation). | 119 / N362 | contractual (fictional) | Visit caps vary by plan; nothing on the claim says where the cap is. |
| PR-08 | CASCADE | 25-OH vitamin D (82306) is covered only when a diagnosis pointed to by the 82306 line is on the CGS "Vitamin D Assay Testing" covered ICD-10 list (183 codes: e.g. E55.0/E55.9 deficiency, E20.x/E21.0-E21.3 parathyroid, K50/K51/K90.x malabsorption and IBD, M80.x/M81.x osteoporosis, M83.x osteomalacia, N18.30-N18.6 CKD, Z68.30-Z68.45 BMI >= 30). Fatigue (R53.83), obesity coded without a BMI Z68 code (E66.9) and screening are not covered. NORTHSTAR and BAYVIEW have no diagnosis restriction. | 167 / N115 | **CMS** [5] | Fatigue/obesity look like a reasonable indication; other payers pay them. |
| PR-09 | BAYVIEW | Telehealth E/M (patient at home) must be billed POS 10 (not 02) AND modifier 95 (GT / 93 not accepted). Wrong POS -> CARC 5; missing/incorrect modifier -> CARC 4. CASCADE pays POS 02 without 95. | 5 / M77; 4 / N823 (GT) or N822 (none) | POS 02 vs 10 definitions: **CMS** [6]. Modifier 95 requirement and CASCADE leniency: contractual (fictional) | POS 02 and GT are both legitimate telehealth codes elsewhere. |
| PR-10 | BAYVIEW | Non-emergency services from out-of-network providers (`context.provider_in_network` false) are not covered, even with a referral on file. Emergency services (POS 23) are covered OON. NORTHSTAR pays OON at OON benefit (lower allowed amount). Denies every line. | 242 / - | contractual (fictional) | A generic reviewer cannot tell which plan types have OON benefits; the referral looks like it should cure it. |
| PR-11 | All (generic) | Every service line must have at least one valid `diagnosisCodePointers` entry (1..number of diagnoses). | 16 / M76 + MA130 | X12 837P / CARC 16 usage (needs a RARC) | Should be caught by any competent reviewer. |
| PR-12 | All (generic) | Every line must have `lineItemChargeAmount` greater than 0.00. | 16 / M79 + MA130 | X12 837P / CARC 16 usage | Should be caught by any competent reviewer. |

Simulator baseline (not one of the 12 hidden rules, not exercised by any claim): an NCCI pair from the
table above billed with no PTP-associated modifier on either line has its column-2 line denied CO-97 / M15
by every payer; `eligibility_active: false` would deny CO-27.

### Sources (all fetched 2026-10-11)

1. **Timely filing (PR-02, CASCADE).** Medicare Claims Processing Manual, Pub. 100-04, Chapter 1, §70
   "Time Limitations for Filing Part A and Part B Claims" (Rev. 2140) and §70.4 (Rev. 12909): claims must be
   filed "no later than 12 months, or 1 calendar year, after the date the services were furnished"
   (42 CFR 424.44). https://www.cms.gov/regulations-and-guidance/guidance/manuals/downloads/clm104c01.pdf
   Change: CASCADE's window was "365 days"; it is now DOS + 1 calendar year (no claim outcome changed;
   CLM-014 at 262 days is still paid).
2. **NCCI PTP, therapy pair (PR-04).** NCCI Policy Manual for Medicare Services, 2026, Chapter XI §P.3
   (revision date 1/1/2026): "The NCCI program contains PTP edits with Column One codes of the physical
   medicine and rehabilitation therapy services and Column Two codes of the physical therapy ... re-evaluation
   CPT codes of 97164 ...", bypassable "with modifier 59 or XU appended to CPT code 97164", i.e. a
   modifier-indicator-1 edit. Chapter I §E defines CCMI 0/1/9.
   https://www.cms.gov/files/document/11-chapter11a-ncci-medicare-policy-manual-2026-final.pdf
   https://www.cms.gov/files/document/01-chapter1-ncci-medicare-policy-manual-2026-final.pdf
   Change: the old pair 97530 + 97140 could not be confirmed (see "Not verified"), so CLM-020 / CLM-026
   now bill 97110-GP x2 + 97164 (CLM-020: 97164-XU,GP paid; CLM-026: 97164-59,GP denied CARC 4).
3. **NCCI PTP, procedure pair (PR-04).** NCCI Policy Manual for Medicare Services, 2026, Chapter III §I.14:
   "The NCCI PTP edits pairing CPT codes 11102, 11104, and 11106 ... each with CPT codes 17000 and 17004 ...
   Use of modifier 59 or XE or XS with these NCCI PTP edits is only appropriate if the 2 procedures ... are
   performed on separate lesions" (modifier indicator 1); §I.2: biopsy codes 11102-11107 are separately
   reportable with lesion removal only for a different lesion (so 11102 is the code that carries the modifier).
   https://www.cms.gov/files/document/03-chapter3-ncci-medicare-policy-manual-2026-final.pdf
   Change: the old pair 17110 + 11102 is not named in the manual, so CLM-032 / CLM-043 now bill
   17000 (dx L57.0 actinic keratosis) + 11102-59 (dx D48.5). Outcomes unchanged (CLM-032 NORTHSTAR denied
   CARC 4; CLM-043 CASCADE paid).
   File version checked: the current practitioner PTP edit files on
   https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-procedure-procedure-ptp-edits
   are **Practitioner PTP Edits v323r0, effective Oct. 1, 2026 (posted Sept. 2, 2026)**. The ungated
   "Practitioner Services PTP Quarterly Additions, Deletions, and Revisions, effective October 1, 2026" zip
   was downloaded and contains no changes to 97110/97164, 17000/11102 or 36415 pairs.
4. **36415 (PR-05 contrast).** The NCCI Policy Manual 2026 chapters on pathology/lab (Ch. X) and medicine
   (Ch. XI) contain no 36415 bundling with lab panels; Medicare pays 36415 as a specimen collection fee on
   the Clinical Laboratory Fee Schedule (https://www.cms.gov/medicare/payment/fee-schedules/clinical-laboratory-fee-schedule-clfs).
   BAYVIEW's bundling is a fictional contract term (many Medicaid / commercial plans do bundle it).
5. **Vitamin D (PR-08).** CGS Administrators (MAC, Part A/B) LCD **L33996 "Vitamin D Assay Testing"**
   (v33, effective 08/06/2026) and its Billing and Coding article **A56798 v29** (revision effective
   08/06/2026, "Annual review, no changes"), group 1 "ICD-10-CM Codes that Support Medical Necessity"
   (183 codes). The simulator embeds that exact list.
   https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdid=33996&ver=33
   https://www.cms.gov/medicare-coverage-database/view/article.aspx?articleid=56798&ver=29
   Change: the old family list (E55.x, M80/M81.x, N18.x, K90.x, E21.x) is replaced by the article list.
   Differences: N18.1/N18.2/N18.9 and E21.4/E21.5 are not covered; parathyroid E20.x, IBD K50/K51,
   cirrhosis, osteomalacia M83.x, BMI Z68.30-Z68.45 and others are. No claim outcome changed.
6. **Telehealth POS (PR-09).** CMS Place of Service Code Set
   (https://www.cms.gov/medicare/coding-billing/place-of-service-codes/code-sets): POS 02 "Telehealth Provided
   Other than in Patient's Home" (patient is not located in their home); POS 10 "Telehealth Provided in
   Patient's Home" (effective January 1, 2022, available to Medicare April 1, 2022). Both BAYVIEW telehealth
   claims have `context.patient_location` = home, so POS 02 is a real coding error there (CLM-012 CARC 5).
   CASCADE paying POS 02 for a patient at home (CLM-004) is fictional contract leniency.

### Not verified

- **Full NCCI PTP file.** The v323r0 practitioner PTP edit files (and the CMS Coverage API's
  LCD/article data endpoints) are only served after accepting the AMA CPT license click-through. That
  acceptance was not given, so column-1/column-2 order and modifier indicators were taken from the NCCI
  Policy Manual text, not from the edit file. The column order of 17000/11102 is not asserted; the
  simulator treats that pair order-agnostically. 97530/97140 (the old therapy pair) may well be a real
  indicator-1 edit, but it could not be confirmed from an ungated primary source, so it was swapped.
- **Modifier 95 / GT under 2026 Medicare rules.** Only the POS definitions were verified on cms.gov.
  Whether Original Medicare in 2026 still wants POS 02/10 vs. the in-person POS + modifier 95 was not
  verified; the BAYVIEW modifier-95 rule stays a fictional Medicaid plan term.
- **CMS 36415 payment amount** (CLFS) was not checked; only the "separately payable, not NCCI-bundled" point is used.
- **Claim filing code for Medicare Advantage.** `16` (HMO Medicare Risk) is the conventional value for MA
  plans; Stedi's code list lists it, but no CMS document was checked for it.

## Coverage in the dataset

Each rule is violated in at least one `train` claim and one `heldout` claim. PR-02 is violated
twice in train (BAYVIEW and NORTHSTAR) and once in heldout (BAYVIEW).

Paid "trap" claims (look risky, are clean under that payer):
- CASCADE advanced imaging without auth (PR-01 contrast), train + heldout
- CASCADE claim submitted 262 days after DOS (PR-02 contrast; inside Medicare's 1-calendar-year window)
- CASCADE 99213-25 + 11102 (PR-03 done right); BAYVIEW E/M + vaccine admin without 25 (PR-03 contrast)
- NORTHSTAR 97110 + 97164-XU (PR-04 done right)
- CASCADE 80053 + 36415 (PR-05 contrast); BAYVIEW lab with no 36415
- NORTHSTAR preventive more than 365 days after the last one (PR-06)
- CASCADE PT at visit 27 with KX (PR-07 contrast)
- NORTHSTAR 82306 with fatigue (PR-08 contrast)
- CASCADE telehealth POS 02 without 95, patient at home (PR-09 contrast; CASCADE contract leniency)
- NORTHSTAR out-of-network office visit; BAYVIEW out-of-network emergency visit POS 23 (PR-10 contrast)
- NORTHSTAR imaging with an auth number on file (PR-01 done right)
- CLM-041: NORTHSTAR office visit submitted 150 days after DOS (PR-02 contrast; inside the 180-day window), heldout
- CLM-042: NORTHSTAR 99213 + flu vaccine + 90471 without modifier 25 (PR-03 contrast), heldout
- CLM-043: CASCADE 17000 + 11102-59 (PR-04 contrast; only NORTHSTAR rejects 59), heldout
- CLM-044: CASCADE PT visit 24 with GP/KX (PR-07 contrast; no CASCADE cap), heldout
- CLM-045: BAYVIEW MRI lumbar 72148 with no prior auth (PR-01 contrast), heldout
- CLM-046: BAYVIEW preventive 99395 331 days after the last one (PR-06 contrast; rolling-365 limit is NORTHSTAR-only), heldout

Split balance: train 24 (11 paid / 13 denied), heldout 22 (11 paid / 11 denied), 46 claims total.
