# RUN 13 — FIX C, FINAL PRE-REGISTERED DESIGN (v2)

Status: POST-AUDIT REWRITE, 19 July 2026. Supersedes the audited v1
draft. Incorporates audit amendments A1-A15 (RUN13_FIX_C_AUDIT.md) and
JK rulings D1-D6. Run 13 is STAGE 1 ONLY (ruling D3c): counterfactual
replay. The search rebuild is a separate future project; the v1 stage-2
section is deleted, its salvage notes live in the audit file.

## 0. Evidence base (corrected per D4)

- Fair-basis 12c vs 12d (frozen shared states): rho_taken 0.110 vs
  0.093 — the full 12d credit package produced NO ordering improvement.
- Discriminator: rho_taken weakly positive, rho_untaken null → untaken
  candidates never receive targets; the coverage hole binds.
- TRUE-conflict re-measurement (stratum 0, <10 nm, 7 frozen states,
  19 July): 12d-final prices NOOP −30.5 vs GT −42.9 (moderate ~12
  optimism), crowns NOOP 0%. The earlier "−7.6 vs −32" figure was a
  SAFE-stratum artifact (label inversion, fixed). Implication: the
  compulsion pathology lives in SAFE states (inventing interventions
  for traffic that needs none) — M2/M2-XL target exactly this.
- Knee rerun under the 12d-final policy (19 July): median action-vs-NOOP
  gap crosses zero at H≈24-30 but reaches decision-relevant size only
  at H=36 (median 0.97 vs fee 0.1; 5.0 at H=48); delay-not-avoid still
  26:1; delay-miscredited-as-avoidance 74% @H=24 → 48% @H=36; personal
  knees: 63% ≤36, 85% ≤48.

## 1. Mechanism — counterfactual replay (CF-replay)

At selected live-episode states s, for each of K selected untaken
candidates a′: clone env at s, apply a′, then run the FROZEN CURRENT
POLICY (ruling D1) for H_cf steps, pricing every step with the live CBP
reward layer (A3: absolute-potential pricing REJECTED — Bellman-
inconsistent cross-stream targets). Store CF transition
(s, a′, G_cf, s_H, cf=True). Target:

    y = G_cf + gamma^H_cf * Q_target(s_H, a_cont(s_H))

where a_cont(s_H) is the continuation policy's actual action at s_H
(A1 Branch A bootstrap — no NOOP-degenerate tail).

### Continuation (D1 = frozen-policy, Branch A of A1)

- Continuation policy = the training net frozen at branch time, greedy
  (epsilon 0), with the LIVE episode's last_issued dict deep-copied at
  branch time into BOTH the continuation policy and the CF candidate
  mask (A13.1) + selftest: CF suffix step-0 masked action set == live
  masked set at s.
- Mandatory NOOP probes: a′ = NOOP is always in the CF set at selected
  states (A1: the selection rule never picks NOOP on its own).
- Staleness machinery (A1 Branch A): every CF transition stamped with
  generation episode; age cap A_max (pre-registered below); recompute
  path via retained in-RAM env clones. MEASURED 19 July: 0.27 MB/clone
  → 1200 clones ≈ 0.31 GB; retention is not a constraint up to ~10k.
- rho_untaken instrument basis MATCHES training basis by construction
  (D1/C2 GT uses frozen-policy continuation) — no co-primary GT column
  needed (that was Branch B machinery).

### Horizon (D5 + A10)

- Live n-step: n = 36 (D5 ruling; knee evidence above).
- H_cf = 36, matched to nstep (A4 depth-symmetry: no gamma^12 vs
  gamma^36 tail mismatch between live and CF targets).
- CPA-matched exclusion rule (A10): any state feeding a rho bar must
  satisfy H*6s >= that state's flagged-conflict CPA; far-CPA states
  (knee tail, ~15%) are excluded from headline rho strata (not from
  training). Logged per state.

### Selection rule (A5-corrected strata)

1. States: min vertically-proximate pair separation < 10 nm first
   (stratum 0); 10-30 nm at p = 0.3; PLUS a state-level random floor
   p = 0.05 across all strata (safe-state CF probes are what M2 needs).
2. Candidates at a state: mandatory NOOP probe; then (a) interval-
   overlap-with-argmax ambiguity; (b) lowest CF-side count cells;
   (c) candidate-level random floor p = 0.1.
3. Exclusions (A13.2): reissue-masked candidates; BEFORE_ENTRY-target
   candidates (physical no-ops — store the analytic target
   G_cf(NOOP) − CMD_COST from one shared NOOP rollout instead).
4. CF selection uses a SEPARATE CF-side count table and a dedicated RNG
   stream; live exploration tables and live RNG are never touched from
   CF code (A12.3).

### Budget (D2 = cap rises; coverage-first per A2)

- Pre-registered coverage target: >= 20% of stratum-0 (s, a′) pairs per
  buffer lifetime receive a CF target (exact K per state solved from
  the realized episode mix; K floor 3 incl. NOOP probe).
- Cost model (A2-corrected): one state costs K*(1+H_cf) priced steps +
  K+1 deepcopy-equivalents (deepcopy charged at ~1.5 step-equivalents;
  StepBudget extended to charge clones, A2.2).
- Budget is BANKED ACROSS EPISODES (A2.3), not per-episode — at micro
  scale a per-episode cap makes CF inert and M1's own gate unreachable.
- Realized per-episode CF-target coverage logged at BOTH scales;
  pre-registered abort/flag if the full run's realized coverage rate
  falls >10x below the micro rate that certified the mechanism (A2.4).
- Accepted consequence (JK, D2): wall-clock rises; train smaller/slower
  on this machine; more compute possible for later runs.

### Replay integration (A14 sampler; A12 quarantine)

- CF pool sampled share-proportionally with a pre-registered boost
  (terminal-boost pattern); the 1:3 cf:real ratio is a never-expected-
  to-bind upper bound; realized cf batch fraction + per-item cf replay
  counts logged per checkpoint; median-sampled-cf-age tripwire.
- Age cap A_max = 400 episodes; recompute-on-expiry for up to 2000
  retained clones (0.54 GB), else drop.
- Tracker quarantine (A12.1, D6 signed off): cf flag plumbed through
  train_step; bootstrap_hits, _tripwire_devs, bsupport_* skip cf rows;
  parallel cf_* series logged; the tripwire revival trigger AND its
  ep500 baseline read the LIVE-ONLY series.
- NOOP-head calibration check (A12.2): net Q(s, NOOP) vs scripted-NOOP
  GT on the frozen shared set at every milestone — CF-target-validity
  covariate (the −30.5-vs−42.9 instrument).

### Pricing instrumentation (A3)

- Per-CF-rollout event fraction logged (floor 20% flagged).
- Per-state spread of G_cf across the K candidates (bootstrap excluded)
  logged — direct measure of ordering content in returns.
- Pre-registered interpretation rule: rising micro rho with near-zero
  full-scale G_cf spread = gate passed in a regime the full run does
  not inhabit → treat as NOT certified.
- Sparse-event micro variant (decisive event beyond H_cf) in battery
  (M1-far below).

## 2. Build order

1. M0 harness + CF branch machinery in bluebird_controller_dqn.py
   (--cf_replay, default OFF; the four 12d silent-killer surfaces are
   build requirements: last_issued snapshot timing, DeliveryClock
   observe discipline, matched-depth H_cf = nstep, terminal-pool
   handling for CF windows).
2. M0 passes (gating) → RA-CF gate (A3): RA1/RA4/RA5/RA6 + NOOP-zero
   re-run through the production CF branch path.
3. Micro battery M1-M6 (below) with --cf_replay on where specified.
4. All gates green → run-13 launch decision returns to JK with the
   battery table.

## 3. Micro battery (repaired; gates run 13)

- M0 CF-identity (A4, NEW, gates everything): teacher-forced replay of
  a fixed-seed episode's RECORDED action sequence through the CF
  pricing path at every taken (s,a); bitwise asserts: G_cf == windower
  n-step return; synthesized mask/next-action fields == stored live
  8-tuple; branch clock advanced once per step. Terminal windows:
  separate collapsed-semantics assert. NOT re-derived from live policy
  (count bonus + adaptive c make trajectories legitimately diverge).
- M1 level-allocation: clean >= 0.80 AND full-candidate rho > 0.4 with
  --cf_replay on (12d-era null: 0.069). GT horizon obeys the
  CPA-matched rule (A10): micro --h raised so H*6s covers the
  scenario's CPA.
- M1-far (A3 sparse-event variant): decisive event at step ~20 > the
  OLD H=12 draft — with H_cf = 36 the event is in-window; the variant
  instead places CPA so that a MIS-SET short horizon would miss it;
  bar: rho > 0.4 AND logged event fraction > 20%.
- M2 hold-your-fire (A6-repaired): level-separated, laterally-proximate
  pair where re-leveling climb/descend carries −50 inside the GT
  horizon (RA4 geometry — interventions have REAL negative
  consequences, not just fees). Pre-training gate: >= 5 probe states
  with gt_spread > 5 spanning distinct segments. Bars: command rate
  < 0.1/step; NOOP net-argmax >= 90% of probe states; rho > 0.4;
  NOOP-level |Hurwicz(NOOP) − GT(NOOP)| < 2 at >= 80% of probe states.
- M2-XL (A5, eval-only): scripted pass at 20+ aircraft (101+
  candidates): NOOP net-argmax and command-rate statistics at scale.
  No training, no GT rollouts — cheap. Bar: report + pre-registered
  regression check vs M2 (P(argmax != NOOP) rising with candidate
  count is the failure signature).
- M3 lateral-only crossing: verticals masked; clean >= 0.80, rho > 0.4.
- M4 give-way (A15-repaired): oracle = GT-rollout ACCEPTABLE SET
  (aircraft whose best (instruction, timing) GT return is within
  eps = 1.0 of global best); strict-subset gate: through-flight best
  gap > 5 from global best, else re-seed. Bars: first command targets a
  crossing member (through-flight instructed = hard MISS, NOT
  absorbable by the escape hatch); commands/ep <= 2; per-state greedy
  argmax vs acceptable set on frozen probe states (no episode-level
  fractions — post-warmup epsilon is exactly 0).
- M5 delivery (A7-repaired): OFF-ROUTE SPAWN — a heading/route_parallel
  clearance is REQUIRED to reach the exit. Gates: scripted NOOP
  delivers 0 (pre-registered); scripted single-clearance delivers
  cleanly. Bars: deliveries/ep >= 0.8; command rate over OCCUPIED steps
  < 0.2 (denominator = steps with >= 1 aircraft in obs, via step_hook).
- M6 ordering curve: M1 frozen states + full GT tables every N
  episodes; rho > 0.4 reached AND held for the final quarter.

Composite gate: M0 bitwise-green, RA-CF green, M1, M2, M3, M5 MEET;
M4 MEET or documented-miss with cause assigned outside fix C (escape
hatch NOT valid for the through-flight bar); M6 held; M2-XL reported.
Any other MISS → no full run; iterate at micro scale.

## 4. Out of scope for run 13

- Stage 2 / search distillation (D3c: separate future project).
- Ensemble epistemics (JK standing no).
- Pricing changes (12d audit: pricing correct; RA battery green).
- Tracker-feed change beyond the D6 live-only re-scoping.
- Full C2 recovery as a gate (D7 scoping stands).
