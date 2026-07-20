# ROUND-3 MEASUREMENT INSTRUMENTS — BUILD STATUS (20 July 2026)

Build-order step 1 of ROUND3_REVIEW.md (D instruments). Everything here
was built WITHOUT touching bluebird_controller_dqn.py (bcd is owned by
the parallel round-3 mechanism build; this module set only reads it).

Code: `round3_instruments.py` (D1 rebuild), `round3_m6_telemetry.py`
(M6 scaffold, data side), `test_c_instruments_lock.py` (C-instruments
lock), plus additions to `micro_battery_common.py` (D2 config echo,
checkpoint fix, dual-mode hooks, null-banded bar printing),
`micro_battery_m4.py` (D3 bar redesign), `micro_battery_m2xl.py`
(locked state_read extraction), `credit_diagnostics.py` (config echo).

Status table, measured null bands and the measured budget table are in
section 7 (all runs completed 20 July; nothing below is TBD).

## 1. D1 — ordering instrument rebuilt (`round3_instruments.py`)

Replaces the instrument the 20-July audit voided (T5: null band
[-0.62, +0.85] swallows the 0.4 bar, effective n ~ 1; T6: +0.18 basis
ceiling).

- Probe sets from N >= 8 INDEPENDENT scenario seeds per test family
  (pre-registered: states_per_seed = 3, n_seeds = 8, 24 states/family).
  Families: `conflict` (M1/M3 geometry: 2-starter NOOP -> LoS) and
  `holdfire` (M2 geometry: NOOP-clean level-separated proximate pair,
  the round-1 finder criteria verbatim). Per-seed statistics
  (`rho_perseed_mean` ± sd over seeds) are the independent-n readout.
- Every instrument ships its 30-random-init untrained null band
  (mean/sd/p2.5/p97.5; torch seeds 7000-7029, T5 discipline) BESIDE
  every reading; `bar(..., null_band=...)` prints "INSIDE NULL BAND, no
  verdict" whenever a reading cannot beat 30 untrained nets. Null bands
  are NOOP-basis ONLY (review's null-basis rule).
- Dual-basis GT lifecycle: NOOP-basis GT built once per probe set,
  rescored per checkpoint at full cadence (net forwards only, ~1 s);
  frozen-policy-basis tables ONLY at the pre-registered checkpoint set
  {init, quarter, half, final} (`frozen_checkpoint_episodes()`), each
  interpreted via the T6-style basis-agreement Spearman printed beside
  it, NEVER via a null band of its own.
- rho at c in {0, c_train, 1}; rho_mid DROPPED. Extra locked scalars:
  NOOP-argmax fraction, NOOP margin mean/p95 (the C2 tolerance
  derivation's noise input), NOOP-level median deviation.
- Fingerprint + HARD-FAIL staleness: every table/band JSON carries
  {scenario seeds, probe steps, strata, h per seed, env config, pricing
  echo (D2 folded in), env_termination_version} + a sha256; readers
  raise SystemExit on any mismatch. The A1 recompute trigger is
  mechanical: the fingerprint reads
  `getattr(bcd, "ENV_TERMINATION_VERSION", "pre-A1")` — the A1 build
  MUST define/bump that constant in bcd, which auto-invalidates every
  pre-A1 table. Frozen-basis rebuilds additionally re-harvest the
  probe states and hard-fail on steps/strata drift.

Reproduce:
```
.venv/bin/python round3_instruments.py --scan
.venv/bin/python round3_instruments.py --build
.venv/bin/python round3_instruments.py --nulls
.venv/bin/python round3_instruments.py --frozen --ckpt <pt> # 4 ckpts only
.venv/bin/python round3_instruments.py --read --ckpt <pt>
.venv/bin/python round3_instruments.py --budget
```

## 2. D2 — bookkeeping

- `micro_battery_common.bcd_config_echo(args)`: every cf / pricing /
  nstep / floor / W_rec flag in one dict — pricing + reward constants
  read LIVE from bcd, run flags from args, and the round-3 flags that
  do not exist yet (W_REC, NOOP tie-break tolerance names) probed by
  getattr and echoed as "absent" until the bcd build lands.
  `write_summary` injects it into EVERY battery summary JSON;
  `credit_diagnostics.py` injects it into its diagnostics JSONs;
  `run_training_arm` stores it in every checkpoint's `extra`.
- M4 no-checkpoint gap FIXED: `run_training_arm` now saves checkpoints
  on the rho cadence even when `rho_fn is None` (the T8 forensics
  deviation: rounds 1 AND 2 of M4 left no checkpoint).
- The GT-relevant echo slice is folded into the D1 fingerprint
  (`gt_relevant_echo()`), so config drift and GT staleness are ONE
  mechanism, as the review requires.
- NOT DONE HERE (bcd-owned, flagged): persisting target_net in
  checkpoints and the D4 CF_K_FLOOR CLI override live in
  bluebird_controller_dqn.py / its save_checkpoint — out of this
  work-package's file domain. The bcd owner must pick these up.

## 3. D3 — M4 bar redesign (`micro_battery_m4.py`)

- B1-HARD: ANY through-flight command in the greedy eval episode =
  hard fail (count == 0). The round-1 bar passed first-command at
  chance ~0.65 while the policy gave the through-flight 10/60 commands.
- B1-FI: first-intervention test over >= 10 probe episodes on
  independent qualifying seeds (`--fi_episodes`, default 10; pool =
  19-Jul shortlist + seeds 100-259, finder re-verified). Pass = first
  issued command targets a crossing member in EVERY episode; chance
  floor 0.667^10 ~ 0.018 printed with the bar.
- B2 (commands/ep <= 2) UNCHANGED. B3 (probe-argmax through-flight
  count == 0) UNCHANGED — it is one of the C-locked reads.
- DOCUMENTED OPERATIONALIZATION (flagged, not silent): on non-primary
  probe seeds "acceptable" = CROSSING MEMBERSHIP from the finder — the
  full A15 oracle costs ~10 min/seed and the review's 0.667 chance
  arithmetic is exactly crossing-vs-through membership; the primary
  seed is additionally reported against its oracle acceptable set.

## 4. C-instruments lock (`test_c_instruments_lock.py`)

Regression test asserting `probe_argmax`, `noop_argmax_fraction`,
`rho_eval`, `rho_eval_subset`, the M2-XL read (extracted to
`micro_battery_m2xl.state_read`) plus `noop_level_hits` and
`round3_instruments.ordering_read` are BIT-IDENTICAL under any
tolerance value:

1. tolerance sweep {0, 0.1, 0.7, 5, 1e9} over every plausible
   agent-level and bcd-module-level tolerance name — outputs compared
   with exact equality;
2. selection-path poisoning: `select_candidate` / `generate_action`
   replaced with raising stubs — instruments must still run (proves
   they read `candidate_q` directly and can never inherit the
   tolerance or the count bonus);
3. `probe_argmax` pinned to the direct candidate_q Hurwicz argmax with
   the exact-tie-to-NOOP rule.

RESULT: ALL PASS (run 20 July). NOTE: pytest is not installed in
.venv — the file is pytest-compatible but was run as
`.venv/bin/python test_c_instruments_lock.py`.

Dual-mode (tolerance-off) reporting hooks for the command-RATE bars
(which cannot be de-biased from logs): `micro_battery_common.
detect_noop_tolerance` + `greedy_eval_dual_mode` +
`dual_mode_cmd_rate_bar` — one extra tolerance-off greedy episode per
read; the bar is SCORED on the tolerance-off number, the tolerance-on
number is reported as the policy-level rate. Degrades to single-mode
with an explicit "tolerance absent in bcd" note today.

## 5. M6 scaffold (`round3_m6_telemetry.py`) — data side only

Parser/reporter for the per-arc telemetry the re-review specifies:
per-arc [armed?, causal-test GT, resolved?, time-in-band, W_rec paid,
flag events], per-boundary [armed?, live-clearance?, m per window,
Q_target pre/post-floor, band re-entry within nstep], per-episode
counters [armed_win_count, w_rec_paid_total, tolerance_invoked,
bonus_override, in_band_tolerance_flips (hard bar == 0)].

- Flag-coverage instrument: fraction of ENDOGENOUS band entries
  (causal test as ground truth) whose causing command was sign-flagged,
  split inside/outside the 15 nm / 20 FL ramp; pre-registered
  prediction (near-zero outside, partial inside) printed with the
  numbers; FAIL line when unflagged outside-ramp endogenous onsets
  carry material W_rec mass (draft materiality threshold 10% of total
  W_rec mass — DRAFT, flagged for review ratification).
- Laundering exposure metric: joint rate of floor-active AND
  band-re-entry-within-nstep, plus floor-active fraction on
  live-clearance boundaries (the two alarm series).
- Boundary fire-rate counter printed so "per-pair resolutions nonzero,
  global collapses zero" is visible in one place.
- GRACEFUL DEGRADATION verified: run against an existing battery JSONL
  it reports every field "ABSENT (bcd not yet emitting X)"; run
  against a synthetic contract-conforming JSONL every aggregate
  computes (both runs on 20 July, see status table).
- FLAGGED: the re-review specifies field CONTENT but not literal key
  names. The names in `round3_m6_telemetry.FIELD_MAP` are therefore
  the proposed binding contract for the bcd build; if bcd ships
  different names, update FIELD_MAP (single indirection point).

## 6. Deviations and flags (prominent)

1. M6 JSONL field NAMES are this package's proposal (content per
   re-review); bcd build must adopt them or FIELD_MAP must be updated.
2. M4 B1-FI uses crossing-membership on non-primary seeds (oracle
   acceptable-set only on the primary seed) — documented above.
3. target_net checkpoint persistence + D4 cf_k floor override are
   bcd-side and remain with the bcd owner (out of file domain).
4. M6 W_rec materiality threshold (10%) is a DRAFT constant.
5. pytest missing from .venv; lock test run standalone.
6. Conflict family needed a WIDER scan than pre-drafted: only 7/8
   qualifying seeds in 100-399 (2-starter NOOP-LoS rate ~2.3%); the
   8th (seed 446) came from an extension to 400-799. The hard-fail
   FLAG fired as designed; bank_conflict.json now holds 8.
7. The canonical tables are built at the battery default pricing
   gate/0.2. Arms that must match 12d pricing (smooth/0.5) need a
   REBUILD under that pricing — mixing hard-fails by fingerprint (the
   staleness selftest below proves it fires).
8. The demonstration frozen-basis/read checkpoint (round-1 m2 ep800)
   was TRAINED smooth/0.5 and is read here under gate/0.2 —
   load_agent's loud re-pricing note fired; intentional for the
   machinery demonstration, and the basis-agreement numbers below are
   gate/0.2 numbers, not comparable to T6's smooth/0.5 +0.180.

## 7. Status table + measured numbers (all runs 20 July 2026)

| Deliverable | Built | Run now | Result |
|---|---|---|---|
| D1 seed banks (N>=8/family) | YES | YES | conflict [134,144,152,168,210,350,376,446]; holdfire [119,167,176,193,242,259,289,313] |
| D1 NOOP-basis GT tables (24 states, 264 branches each) | YES | YES | conflict 41 s, 16 hot; holdfire 46 s, 16 hot |
| D1 null bands (30 inits, seeds 7000-7029) | YES | YES | tables below |
| D1 frozen-basis tables (ep800 demo ckpt) | YES | YES | conflict 157 s, agreement +0.497 (sd 0.524); holdfire 179 s, agreement +0.426 (sd 0.479) |
| D1 staleness hard-fail | YES | YES (selftest) | SystemExit fires on pricing mismatch (gate/0.2 table vs smooth/0.5 live) |
| D2 config echo in summaries + checkpoints | YES | YES (M5 2-ep run) | 34-key echo in summary JSON and in ep2 checkpoint; W_REC / tolerance echo "absent" as designed |
| D2 M4 checkpoint fix (rho_fn=None) | YES | YES | "checkpoint saved (no rho_fn)" + ep .pt on disk in both M5 and M4 smokes |
| D3 M4 bars (hard-fail + first-intervention) | YES | YES (3-ep smoke) | B1-H: 8 through-flight cmds -> MISS; B1-FI: 6/10 first interventions -> MISS; B2: 11 cmds -> MISS; B3: 20/20 through argmax -> MISS. Smoke-scale MISSes expected; harness proof complete. (One dup seed in the FI list during the smoke — dedupe fixed immediately after.) |
| C-instruments lock test | YES | YES | ALL PASS: bit-identical across tolerances {0, 0.1, 0.7, 5, 1e9}; instruments run with select_candidate/generate_action poisoned |
| C dual-mode reporting hooks | YES | degrades | "tolerance absent in bcd — single mode" until bcd ships it |
| M6 scaffold | YES | YES | all fields "ABSENT" vs real battery JSONL (graceful degradation verified); full aggregation verified on synthetic contract JSONL incl. flag-coverage FAIL path and in-band-flip hard bar |

### Measured null bands (30 untrained inits, mean rho over states; the
### pre-registered 0.4 rho bar is OUTSIDE every band — the T5 defect
### (bar swallowed by the null, band ~[-0.6, +0.85]) is repaired)

conflict family, full candidate set (24 states / 8 seeds):
  c0.0 rho  mean +0.024 sd 0.138 band [-0.174, +0.265]
  c0.5 rho  mean +0.021 sd 0.143 band [-0.183, +0.296]
  c1.0 rho  mean +0.014 sd 0.153 band [-0.182, +0.301]
  c0.5 noop_argmax_frac band [0.000, 0.909]; noop_margin_mean band
  [-0.212, +0.797]; noop_margin_p95 band [-0.035, +1.037]
conflict family, lateral (M3) subset:
  c0.5 rho  mean +0.010 sd 0.180 band [-0.295, +0.349]
holdfire family, full candidate set:
  c0.0 rho  mean +0.081 sd 0.208 band [-0.407, +0.362]
  c0.5 rho  mean +0.077 sd 0.199 band [-0.363, +0.374]
  c1.0 rho  mean +0.072 sd 0.191 band [-0.295, +0.370]
  c0.5 noop_argmax_frac band [0.000, 0.970]; noop_margin_mean band
  [-0.215, +0.777]; noop_margin_p95 band [-0.078, +0.971]
  c0.5 noop_level_med_dev band [+0.070, +0.757]

### Demonstration read (round-1 m2 ep800, re-priced gate/0.2) — the
### instrument reproduces the audit's diagnosis with the null printed
### beside every number:

- rho INSIDE the null band at every c on both families and both bases
  (no ordering signal in the round-1 net — matches T5/T6);
- NOOP margin mean +4.7..+5.6 vs null band upper ~0.8 (the compulsion
  signature, now null-referenced);
- holdfire NOOP-level median dev +26.8..+33.3 vs null [+0.03, +1.09]
  (the Q(NOOP) doom, now null-referenced).

### Measured budget table (pre-registered unit: ~240 s / 20-state
### dual-basis table from t6_run.log; this build came in UNDER budget)

| phase (per family, 24 states, 264 branches) | measured wall |
|---|---|
| NOOP-basis GT (once per probe set)          | 41-46 s |
| null bands, 30 inits, 3 c-values            | 2-3 s |
| per-checkpoint NOOP rescore (net fwd only)  | ~1 s |
| frozen-basis table (per ckpt; 4 ckpts/arm)  | 157-179 s |
| => dual-basis unit (NOOP + frozen)          | ~200-225 s / 24 states |
| full-run frozen budget per arm per family   | ~4 x 170 s ~ 11 min |
| seed scan (302 seeds, both families)        | ~195 s (+20 s extension) |

### Reproduce commands

```
.venv/bin/python round3_instruments.py --scan
.venv/bin/python round3_instruments.py --build
.venv/bin/python round3_instruments.py --nulls
.venv/bin/python round3_instruments.py --frozen --ckpt checkpoints/micro_battery/m2/m2_seed167_20260719_205235_ep800.pt --ckpt_tag m2r1_ep800
.venv/bin/python round3_instruments.py --read   --ckpt checkpoints/micro_battery/m2/m2_seed167_20260719_205235_ep800.pt --ckpt_tag m2r1_ep800
.venv/bin/python round3_instruments.py --budget
.venv/bin/python test_c_instruments_lock.py
.venv/bin/python round3_m6_telemetry.py --jsonl <battery jsonl>
.venv/bin/python micro_battery_m4.py --gates --train --episodes 3 --scenario_seed 134   # D3 smoke
.venv/bin/python micro_battery_m5.py --gates --train --episodes 2 --scenario_seed 109 --rho_every 25  # D2 ckpt-fix proof
```

Artifacts: checkpoints/round3_instruments/{bank_*,gt_noop_*,nulls_*,
gt_frozen_*}.json (all fingerprinted; readers hard-fail on staleness).
