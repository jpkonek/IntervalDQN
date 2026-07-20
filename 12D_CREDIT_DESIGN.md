# 12D CREDIT DESIGN — Faithfulness Audit Synthesis and Run Composition

Date: 2026-07-14
Inputs: credit_diag_20260714_131032.json, CREDIT_DIAGNOSTICS.md, train_seed42_20260713_192907.jsonl,
bluebird_controller_dqn.py, lunarlander_interval_dqn.py, micro_level_allocation.py,
WIDTH_MECHANISM_PROBES.md, PROBE_BATTERY.md, MORNING_BRIEF_9JUL.md.
Method: 6 audited LL-unfaithful properties -> 6 proposed fixes -> 2 independent adversarial
verdicts each (12 verdicts total). Design criterion throughout: faithfulness to the LL
reference implementation, whose candidate ordering is healthy.

Headline: **zero fixes were killed; all 12 verdicts were AMEND.** What died were specific
*claims inside* fixes (Section 2.7 ledger) and three *alternatives* (Section 2.8). Two of the
six fixes turned out to be the same mechanism proposed from two findings and are merged.
The recommended 12d composition is **three changes**, not six (Section 3).

---

## 1. Consolidated faithfulness audit table

| # | Unfaithful property | Severity | Lenses | Key evidence | Proposed fix | Verdicts |
|---|---|---|---|---|---|---|
| P1 | n-step credit window (12) shorter than consequence-emergence horizon (>=24): 1.1% of final gap in-window vs LL 89% at H=6 | critical | reward-geometry, target-construction | d2: gap_best -0.10 at H<=12, +10.03 at H=24; NSTEP note (6->12 already failed) | **A: Terminal-lookback flush** (MC tails to violation terminal, reach 48, bootstrap window stays 12) | amend / amend |
| P2 | Command fee (-0.1) is the ONLY in-window action signal: exact anti-ordering, Q(s,a)=Q(s,NOOP)-0.1 | critical | reward-geometry, target-construction | d2: measured gap == fee, anti-signed, noise p95 ~1e-16; LL fuel cost lands same-step as benefit | **B: NSTEP 12 -> 24** (window covers fee AND benefit) | amend / amend |
| P3 | 92% of candidates (781/852) never receive a Bellman target; LL's 4 actions all densely targeted | critical | target-construction, exploration-and-width | d1: n_taken=71, n_untaken=781; identical gather code, catastrophically different allocation | **C: CF-replay** (2-step counterfactual transitions for K untaken candidates via env clones) | amend / amend |
| P4 | Double-DQN target argmax ranges over 1+5N (16-126) mostly-untrained candidates; E[max of N noise] ~ sqrt(2 ln N) corrupts every backup hop | critical | target-construction, exploration-and-width | d1: coverage_taken=0.056; LL argmax over 4 continuously trained columns | **D: Trained-support bootstrap** (argmax restricted to {NOOP, a'_taken}) | amend / amend |
| P5 | Coverage/t feedback starved (fed only on violated episodes) while update_t ratchets every gradient step: t frozen at caps [0.99,0.95,0.95] from ep50; width penalty NEVER fired in 3588 episodes | critical | target-construction, exploration-and-width | 12c log: max stratum coverage 0.754 < 0.85 for entire run; width 0.74->0.04->6.7->12.9 unopposed | **E: t-feed swap** (drive t on per-batch bootstrap-midpoint coverage, LL semantics) | amend / amend |
| P6 | Width calibrated to realized return dispersion (-50 at varying depth): width ~14.8 vs gap 0.10 — 150x sub-resolution for the quadratic loss | critical | reward-geometry, exploration-and-width | d1 mean_width_taken=14.78 vs d2 gap 0.10; LL width ~4.7 vs gaps of tens | **F: t-driver retarget to TD-target coverage** — **same mechanism as E; merged (Section 2.5)** | amend / amend |

Not in the audit table but live: the **vertical binary-gate faithfulness bug** in
`pair_conflict_f` (smooth lateral ramp, binary vertical gate at CONFLICT_FL=20; a climb gets
zero shaping response for ~7 steps then a cliff). Flagged 14 July, **not among the six audited
fixes** — no adversarially-reviewed fix exists for it. It is an open decision (Section 6, D5).

---

## 2. Fixes with verdict summaries

Every fix below survived two adversarial passes. Amendments listed are BINDING on the
recommended composition — the fixes as originally written contained falsified claims (2.7).

### 2.1 Fix A — Terminal-lookback flush (P1) — SURVIVED, **DEFERRED from 12d**

Mechanism: history deque (TERM_LOOKBACK) in ControllerWindower; on violation terminal, every
state in the last L steps receives an exact realized-discounted-tail MC item (disc=0,
target_l==target_u), including the -50. Bootstrap window stays n=12. Censored episodes emit
zero lookback items. Duplicate (s,a) items (bootstrap + MC) intentional.

Verdicts: amend / amend. No fatal flaw. Interval-semantics audit clean: realized returns are
absolutely anchored (no E1-class runaway), genuinely stochastic across episodes (no
determinized-rollout width poisoning), identical scalar path existing terminal items take.

Binding amendments (union of both auditors):
- **term_cap arithmetic wrong by 20x**: line 572 gives max(min(1000, 50000), 20000) = **20000**,
  not 1000 (verified in source). Churn risk was overstated; the real risks are (i) stale
  behavior-policy tails from long-dead policies persisting in a 20k pool — log per-item age,
  and (ii) sampled terminal fraction saturating at the 0.5 cap much earlier.
- **Selftest must be rewritten, not extended** — the n=12 block's assertions
  (len==T, pool split n/T-n) break by design under duplication.
- **Sampling asymmetry of duplicate pairs**: MC copy boosted in term_buf (up to 50%/batch),
  bootstrap copy diluted in ~80k reg_buf — effective ~10x favor to the pessimistic scalar can
  INVERT the predicted widening (width collapses toward the MC point). Log the ratio;
  pre-commit fallback (down-weight or route lookback duplicates to reg_buf).
- **Epsilon story is vacuous**: `_epsilon` is exactly 0.0 post-warmup (verified,
  lines 961-967) and Hurwicz is width-blind (1.06x) — the "resolving-action exploration
  episodes" the ordering story relies on do not exist at convergence. Scope the gate to
  rho_taken + conflict-state level drop, or add an epsilon floor (Section 6, D2).
- **TERM_LOOKBACK=48 too short**: D2 lead reaches 49 (state 11); micro decision-to-violation
  leads median 50, 51% > 48. Use 64 (gamma^64*50 = 7.1, still 70x the fee) or set from the
  measured lead distribution.
- **Semantics reframe**: MC tails at avoidable-violation leads teach Q^behavior, not Q* — the
  fix restores REACH, not LL's target semantics; expect early conservatism, then possible
  ordering decay/oscillation as violations vanish and the MC channel dries up. Track
  rho-vs-violation-rate across checkpoints.
- **Required micro gate**: rerun micro_level_allocation.py with lookback enabled (~2h,
  pre-registered rho>0.4 gate) — parts A/B of the original validation only test plumbing.

Why deferred (not killed): overlaps P1/P2 territory with Fix B, which is the strictly smaller,
precedented move (third application of the same dial). Running A and B together compounds the
terminal-pool economy changes (48-64 items/violation on top of 24) in ways neither audit
priced. Revival trigger in Section 3.3.

### 2.2 Fix B — NSTEP 12 -> 24 (P2, and partially P1) — SURVIVED, **RECOMMENDED**

Mechanism: one constant (or `--nstep 24`). ControllerWindower is n-agnostic; terminal collapse
writes exact realized -50 tails into windows within 24 steps of the violation; fee becomes a
1% rounding term inside a +10 sign-correct target instead of 100% of it.

Verified premise (recomputed by both auditors from the JSON): median in-window gap
-0.101/-0.100 at H=6/12 (== fee, anti-signed) vs **+10.03 at H=24, 9/15 states above fee**.
The D2 fixed rule flips FAIL -> PASS at window 24.

Verdicts: amend / amend. No fatal flaw. Targets remain realized rewards + target-net
bootstrap — no determinized rollouts, no live-output reference, self-reference actually
*decreases* (gamma^24 = 0.481 halves bootstrap weight).

Binding amendments:
- **"No other code changes needed" is FALSE**: selftest hard-asserts `n12 == 12` (verified,
  line ~2004) and its T=17 synthetic cannot host n=24 (zero full windows; pool-split assert
  fails). Parameterize the selftest block (T = NSTEP + 5, drop the ==12 assert, keep the
  n-parametric brute-force) — OR launch via `--nstep 24` only and defer the constant edit.
- **Mechanism variable corrected (strengthens the fix)**: the right quantity is
  `noop_violation_step` (median 22, 10/15 <= 24 — verified fields exist in the JSON), not
  `noop_los_at` (median 31, mostly OUTSIDE the window, which the original write-up cited).
- **Residual anti-signal accounting corrected**: of 6 fee-only states at n=24, three are
  NOOP-optimal (fee-only ordering is CORRECT there). True residual anti-ordering: 3/12
  action-helps states (75% sign-correct). Restate the >=50% fallback gate over action-helps
  states only, so an n=36 fallback isn't mis-triggered.
- **Mandatory knee-location rerun** before committing 24 vs 36:
  `credit_diagnostics.py --d2 --side atc` at horizons [12,18,24,36,48] (~25 min).
- **New risk 7 — delay-credited-as-avoidance optimism**: best arm ALSO violates within H=96 in
  15/15 states; the 24-window books delay past the boundary as full avoidance (in-window +25
  to +28 vs true +1 to +15), corrected only through the D3-collapsed bootstrap chain. Monitor
  gap MAGNITUDE vs ground truth at frozen probe states, not just sign.
- **Two-sided width gate**: gamma^24 strengthens per-hop width contraction ~1.44x while
  terminal point-targets stay ~50% of batches — width can collapse, not only widen. Numeric
  abort gate: any stratum with t at cap AND realized coverage < target-0.05 for K consecutive
  intervals. (Merges with Fix E/F gates.)
- **Benign-state pessimism gate**: log sampled terminal fraction AND benign-state
  NOOP-vs-issued Q gap each eval, with a numeric regression threshold vs 12c.
- Retract the warmup-epsilon off-policy risk (warmup is ~5 episodes; negligible).

Killed alternative recorded inside this fix: **fee removal/reduction — REJECTED.** It converts
the -0.1 anti-signal into a ~0.000 null (in-window shaping differential <= 2e-3), deleting
anti-ordering without creating ordering, and weakens the JK-approved spam deterrent.

### 2.3 Fix C — CF-replay (P3) — SURVIVED, **DEFERRED from 12d; micro gate runs anyway**

Mechanism: at each decision step, deepcopy the env for K=2 untaken candidates (Hurwicz
runner-up + random), step action + 1 NOOP lag, price with the identical CBP formulas, push
2-step transitions (disc=gamma^2) directly into the buffer. Restores "every decision-relevant
candidate gets a real (s,a,r,s') target."

Verdicts: amend / amend. Strongest kill attempt — that lag-1 successor states are
token-indistinguishable so gamma^2*V cannot carry ordering — FAILED on encoder inspection:
selected FL/heading are explicit token features, so issued commands are fully visible in the
bootstrap state. Width semantics clean (bootstrap-interval targets, only 2 realized reward
terms — as determinized as the real trajectory itself).

Binding amendments (both auditors independently converged on the first):
- **Bootstrap-depth asymmetry (the big one)**: taken candidates train toward gamma^12
  bootstraps, cf candidates toward gamma^2 — any target-net bias b yields a taken-vs-untaken
  offset ~0.09b; at |Q|~40-50 a 2% bias equals the entire 0.10 fee, polluting exactly the
  within-state ordering C2 measures, and cf-trained candidates equilibrate ~1.36x wider as
  pure discount arithmetic (fake epistemic signal biasing Hurwicz). Fix at zero clone cost:
  also push a 2-step cf-style item for the TAKEN candidate reusing the already-built env_a
  CBP action branch, equalizing backup depth across all compared candidates.
- **last_issued contamination**: the proposed insertion point is AFTER run_episode mutates
  last_issued with the taken action — snapshot before line ~1307 and build cf masks from the
  copy plus the cf command only.
- **DeliveryClock observe discipline**: clone's clock must observe() before clone step 1 or
  step-1 deliveries are mispriced; the step-0 reconciliation assert provably cannot catch
  this — add a 2-step reconciliation or a clock-step invariant.
- **Terminal-pool contamination**: skip cf pushes when the clone violates at step 0
  (pre-divergence states are bit-identical across k: zero-spread, boosted, fee-only
  anti-ordering data at exactly the C2 measurement states); route/tag remaining cf terminals
  away from term_buf.
- **aux does NOT carry candidate scores** (verified: generate_action returns
  cand_idx/tokens/callsigns/mean_width/c_used) — extend it or budget an extra candidate_q call.
- **Micro-gate deconfounding**: a null C2 at 100% micro coverage implicates EITHER
  coverage-not-binding OR lag-1 target construction; pre-register the discriminator (Spearman
  of target-net V(s''_k) vs micro GT; rerun with lag 8-12) and size the run past ~12-50
  target refreshes so propagation latency cannot fake a refutation.

Why deferred: heaviest mechanism of the six (clone pricing path, clock contract, mask
snapshotting, depth symmetrization — four independent silent-killer surfaces), and its central
claim (coverage is the binding constraint) is exactly what the cheap micro gate adjudicates.
**Run the micro full-coverage gate (~2h) in parallel with 12d prep regardless** — its outcome
decides whether CF-replay leads 12e.

### 2.4 Fix D — Trained-support bootstrap (P4) — SURVIVED, **RECOMMENDED**

Mechanism: replay tuple grows to 8 fields (next_a_idx via a one-step hold in the windower);
Double-DQN target argmax restricted to {NOOP (column 0, densest-trained), a'_taken (trained by
construction from its own next transition)}. Flag `--bootstrap_support {full, taken_noop}`.
Removes max-over-126 hallucination noise at every backup hop.

Verdicts: amend / amend. Layout consistency, reissue-mask skip, and NOOP-at-column-0 all
verified against source by both auditors. No self-referential reference; restricted argmax is
a subset of full, so targets are weakly lower with a NOOP floor — bounded pessimism, no
overestimation spiral. Coherence bonus: the new fixed point (act-then-NOOP continuation) is
essentially the quantity D2's ground-truth rollouts certify carries the +10 signal at H=24.

Binding amendments:
- **Honest framing**: this is NOT a restoration of LL's Bellman-optimality backup — it is
  n-step SARSA-with-a-NOOP-floor, a different fixed point, justified as the only
  trained-support option under the single-net constraint. Record it as an algorithm change.
- **All 7-tuple producers/consumers must be updated**, not just the windower push sites:
  _selftest_windower's four hand-built cases, _e1_synthetic_items, buffer.push(*it) selftest
  batches (~lines 2291, 2428), terminal-boost pool test. Add a tuple-length assert in
  sample()/push() so a missed site fails loudly. Re-verify E1 bitwise tests under `full`.
- **flush_censored must push the HELD window** (NOOP fallback) — otherwise one window per
  censored episode is silently dropped.
- **Drop the fragile ns-array-equality hold trigger**: tokens are rebuilt at t+1; stamp the
  held window unconditionally on the next add() and assert callsign-tuple identity instead.
- **Micro retrain with taken_noop is a REQUIRED gate**, not optional: the inflation probe
  alone cannot distinguish ordering-destroying noise from ordering-neutral uniform inflation.
- **Logging**: per-episode restricted-argmax winner share (NOOP vs a') — the only
  observability for NOOP-dominance stalls; NOOP-fallback fraction (censored tails);
  occasional no_grad full-vs-restricted target inflation on live batches. If winner share
  collapses to ~0, consider an epsilon floor (Section 6, D2).
- **Epsilon claim corrected**: post-warmup epsilon is exactly 0; the true propagation channel
  is STATE differences at s'_n through the dense NOOP column, not re-taken actions.
- bootstrap_hits metric changes meaning under taken_noop — within-run trend only, no 12c
  cross-run comparison.

### 2.5 Fix E+F (merged) — t-feed swap to bootstrap-midpoint coverage (P5 + P6) — SURVIVED, **RECOMMENDED**

E (P5) and F (P6) are the same mechanism proposed from two findings: stop driving t from
starved, violation-conditioned realized returns; drive it from per-batch bootstrap-midpoint
coverage (~128 hits/gradient-step — LL's exact statistic, verified at
lunarlander_interval_dqn.py:309-313/399-405); demote realized-return coverage to a logged
per-stratum diagnostic. ATC already computes the statistic and throws it away (line 1089
comment: "LOGGING ONLY"). This single change addresses both the frozen-t/dead-width-economy
pathology (P5) and the 150x width/gap sub-resolution (P6), because the loss's coverage/width
terms are defined w.r.t. bootstrap targets — calibrating t to a different, far more dispersed
quantity forced the global 13-15 width inflation.

Verdicts: amend x4 (two per finding). All four auditors verified the source claims. Stability
audits clean: reference is the frozen target net between syncs (not E1 topology); both
feedback arms negative; width collapse self-corrects once width falls below residual scale;
determinized rollouts enter no width-bearing term. Honest scoping: predicts NO D2 movement
and no C2 flip on its own — it removes the resolution ceiling so a gap-restoring fix can
express itself. It is necessary independently: any all-candidate-target fix still needs a
live width economy or the 0.74 -> 12.9 runaway recurs under it.

Binding amendments (union of all four verdicts):
- **Rate-couple update_t to fresh evidence per tracker** (flagged independently by both P5
  auditors): update_t currently fires for ALL trackers every gradient step; a stratum that
  vanishes from batches reproduces the ratchet-on-frozen-deque pathology in miniature. Only
  step a tracker's t when it received new hits this step (or scale by n_new/batch_share);
  log per-stratum steps-since-last-feed.
- **Flag-gate the feed** (`--t_feed {bootstrap, realized}`) so `realized` actually reproduces
  12c; record exactly ONE feed site (train_step OR interval_loss_sampled, not both).
- **Safe-stratum width-collapse guard**: LL's equilibrium-above-zero relies on reward noise
  ATC's safe strata lack (~1e-3/step shaping); t can pin at the 0.05 floor and width run to
  the softplus floor, degenerating adaptive_c and Hurwicz. Hard smoke-gate FAIL at
  safe-stratum median width < 0.02; pre-decided response: ABSOLUTE minimum-width epsilon in
  the width_mask gate (e.g. width_reg disabled below 0.05) — anchored absolutely, never to
  live batch stats (E1 lesson).
- **Equilibrium expectation corrected**: the 12c log shows bootstrap coverage 0.84-0.93
  across widths 7-15, so the 0.85-crossing under the boosted sampling mixture is plausibly
  ~7-8, not the claimed 2-6 — expected gap/width improvement may be ~2x, not 3-10x. The
  decisive pre-launch instrument is the **coverage-vs-scaled-width sweep** (Section 4). If
  strata 0-1 cross 0.85 at width >= 7, the resolution ceiling is target-dispersion-limited
  (realized -50 mass in up to 50% of sampled targets — the LL property this fix does NOT
  restore: LL one-step targets carry no -50 mass), and the escalation path is the
  gap-structure fixes, not calibration.
- **Stratum-localization corrected**: 12-step terminal collapse originates up to ~18-23 nm
  separation — terminal -50 mass lands heavily in the MID stratum, not just risky. All gates
  and width predictions per-stratum; log hit rates split by window-contains-terminal.
- **Gate on gap/width at the D2 conflict states directly** (>= 3x improvement target), not a
  loose [0.2, 8] width band that passes at 0.0125 — still sub-resolution.
- **Semantics bookkeeping**: rename/annotate checkpoint fields (strat_coverages/coverage now
  mean bootstrap self-coverage), add `t_driver` field to checkpoints, reset trackers on
  driver switch, require FRESH runs (load_checkpoint restores strat_t; resume transient is
  mild — coverage on 14-wide intervals is only 0.88-0.93 — but headline numbers need fresh).
- **Drift alert made real**: emit per-stratum |bootstrap_cov - realized_cov| with the 0.2
  threshold in the log, noting realized_cov is violation-conditioned (meaningful mainly for
  stratum-0/terminal-adjacent comparison).
- Memoize the 3 stratum coverages once per train_step (current per-sample deque sums are a
  ~256k-float-add/step CPU tax).
- Selftest addition: visit-weighted aggregate of stratified tracker coverage == the
  unstratified bootstrap_coverage on a synthetic batch.
- De-flake validation asserts: direction-matches-sign(coverage - target) instead of
  "moves both ways in 500 steps"; force two-sided regimes with a huge-width and a tiny-width
  probe agent; statistical (not monotonic) width decrease after first gate-open.

Calibration-semantics note for JK (also Section 6, D6): this changes what the intervals MEAN.
Realized-return coverage was the credal ground truth; after the swap, width = TD-residual
scale (LL semantics) and realized coverage will run well below 0.85 by design. That is the
LL-faithful contract, but it is a contract change and is being made explicit, not slipped in.

### 2.6 What was NOT killed — and what that means

No fix died under adversarial review. That is itself informative: the audit found six
mutually consistent mechanisms, four of which (B, D, E+F) are cheap and compose, and two of
which (A, C) are heavier and partially redundant with the cheap set. Deferral of A and C is a
conservatism decision (Section 3), not a verdict — both remain live with defined revival
triggers, and C's decisive gate runs anyway.

### 2.7 Falsified-claims ledger (what actually died)

| Claim (in original fix write-up) | Status | Corrected by |
|---|---|---|
| Fix A: "term_buf cap 1000 -> ~21 violations of history" | FALSE — cap is 20000 (line 572, verified) | both A auditors |
| Fix A: "all pre-existing selftest assertions still pass" | FALSE — breaks by design | A auditor 1 |
| Fix A: "epsilon-exploration episodes provide resolving actions" | FALSE — epsilon == 0.0 post-warmup (verified) | A auditor 1 |
| Fix A: TERM_LOOKBACK=48 covers the leads | FALSE for micro (51% > 48) and D2 state 11 (lead 49) | A auditor 2 |
| Fix A: "restores the LL property" | Overclaim — restores reach, teaches Q^behavior not Q* | A auditor 2 |
| Fix B: "no other code changes needed; selftest is n-parametric" | FALSE — hard `assert n12 == 12` (verified) | both B auditors |
| Fix B: "violation lands within 24 steps for most states" (noop_los_at) | Wrong variable — noop_violation_step (median 22, 10/15 <= 24) is the correct, stronger fact | B auditor 2 |
| Fix B: 6/15 residual anti-signal states | Mis-counted — 3 are NOOP-optimal; true residual 3/12 action-helps | B auditor 2 |
| Fix C: "aux already carries the candidate scores" | FALSE (generate_action returns 5 fields, no scores) | both C auditors |
| Fix C: cf and taken candidates comparable | FALSE without depth symmetrization (gamma^2 vs gamma^12: fee-sized offset from 2% V bias; 1.36x width artifact) | both C auditors |
| Fix C: step-0 reconciliation validates pricing | Incomplete — cannot catch the clone-clock step-1 bug | both C auditors |
| Fix D: "restores the LL property" | Overclaim — substitutes SARSA-with-NOOP-floor for optimality backup | D auditor 2 |
| Fix D: validation-A "replay next_a from trajectory order" | Unrecoverable from the shuffled stratified buffer | D auditor 1 |
| Fix E/F: "--t_feed realized reproduces 12c exactly" | FALSE as written (feed inserted unconditionally) | E auditor 1 |
| Fix F: equilibrium width 2-6, gap/width 3-10x | Contradicted by the fix's own cited log (coverage 0.84-0.93 at widths 7-15) — plausibly ~7-8 and ~2x | F auditor 2 |
| Fix F: -50 mass localizes to risky stratum | FALSE — terminal collapse reaches ~18-23 nm, i.e. MID stratum | F auditor 2 |
| Fix F: resume "slams t down from caps" | Overstated — transient is mild | F auditor 2 |

### 2.8 Killed alternatives (rejected before or during audit)

| Alternative | Fatal flaw |
|---|---|
| Remove/reduce CMD_COST | Converts -0.1 anti-signal to ~0.000 null (shaping differential <= 2e-3 in-window); deletes anti-ordering without creating ordering; weakens JK-approved spam deterrent |
| Ensemble epistemic width | User veto (single-net constraint) |
| Architecture change | Every LL check has exonerated the architecture; last resort by user constraint |
| Raise n directly into 24-96 band as sole fix without gates | Behavior-policy conflation + width-channel hazards unpriced; superseded by gated NSTEP=24 with knee rerun and n=36 fallback |

---

## 3. Recommended 12d composition

**Three changes, in this implementation order. Conservative by design: the two heavy
mechanisms (A, C) are deferred with defined revival triggers; the three included fixes are
each <~30 lines of mechanism, individually flag-gated for ablation, and jointly cover the
three distinct failure stages (calibration economy, target-window content, target-argmax
corruption).**

### 3.1 The composition

| Order | Fix | Flag | Role in the causal chain |
|---|---|---|---|
| 0 | Shared substrate: rewrite `_selftest_windower` ONCE — n-parametric (T = NSTEP + 5, drop `n12 == 12`) AND 8-field-tuple aware; add tuple-length assert in buffer push/sample | — | Both B and D touch this file region; one rewrite, not two |
| 1 | **E+F: t-feed swap** (all Section 2.5 amendments) | `--t_feed bootstrap` | Restores the width economy and gap/width resolution — precondition for ANY ordering signal to be expressible; necessary under every other fix |
| 2 | **D: trained-support bootstrap** (all Section 2.4 amendments) | `--bootstrap_support taken_noop` | Stops re-noising of the H>=24 signal at every backup hop; makes the bootstrap chain a viable carrier |
| 3 | **B: NSTEP 12 -> 24** (all Section 2.2 amendments; 24-vs-36 decided by the knee rerun) | `--nstep 24` | Puts sign-correct signal (median +10.03) INTO the window that fixes 1-2 can now calibrate and propagate |

Rationale for exactly this set: the three fixes attack three DIFFERENT stages that are each
independently verified broken (frozen t + dead width economy; fee-only window content;
hallucination-max targets), and each is cheap, flag-gated, and validated offline. B without
E+F risks the width channel (gamma^24 contraction with no live economy); B without D pushes a
bigger signal into a corrupted argmax; D without B propagates a window that contains only the
anti-signed fee. The composition is the minimal set where each member's predicted effect is
not blocked by another audited defect.

### 3.2 Launch command shape

```
python bluebird_controller_dqn.py --t_feed bootstrap --bootstrap_support taken_noop --nstep 24
```
Fresh run (no 12c checkpoint resume — tracker semantics changed). All three flags default to
12c behavior, so single-flag ablations are one command each if 12d results need attribution.

### 3.3 Deferred fixes and revival triggers

- **C (CF-replay)** — revive as the lead 12e change IF the parallel micro full-coverage gate
  (~2h, runs during 12d prep) shows ordering emerges under full coverage (rho > 0.4 with the
  pre-registered discriminator of 2.3), OR if 12d moves rho_taken but rho_untaken stays null
  (coverage hole then binding). Implement WITH depth symmetrization from day one.
- **A (terminal lookback)** — revive IF the knee rerun shows the emergence knee at 36-48
  (i.e., n=24 leaves most action-helps states fee-only) and n=36 is judged too weak a
  bootstrap (gamma^36 = 0.33); lookback then supplies reach without moving n. Implement with
  TERM_LOOKBACK=64 and the duplicate-pair sampling-ratio fallback pre-committed.

Neither deferral discards audit work: both fixes enter 12e (if triggered) with amendments
already adjudicated.

---

## 4. Pre-launch validation diagnostics (all cheap, all gate the run)

| Fix | Diagnostic | Cost | PASS / FAIL criterion |
|---|---|---|---|
| E+F | **Coverage-vs-scaled-width sweep** on frozen 12c checkpoint: scale all candidate widths x0.1-1.0 (midpoints fixed), recompute per-stratum bootstrap-midpoint coverage on **boosted-sampled** batches, read the 0.85 crossing | minutes | PASS: strata 0-1 crossing at width < 7 (meaningful gap/width gain available). FAIL: crossing >= 7 -> calibration is not the ceiling; keep the fix (still necessary) but expectation-set to ~2x and log the escalation note |
| E+F | 500-step synthetic-buffer probe (fresh agent + forced-wide agent + forced-narrow agent): feed density ~128/step; t direction matches sign(coverage - target) at every step; gate opens for the wide agent; statistical width decrease after first gate-open; `--t_feed realized` reproduces the 12c signature (monotone t to cap, gate never fires) | ~3 min CPU | All asserts; plus --selftest E1 no-contamination and the new aggregate-coverage-consistency assert |
| E+F | TD-residual probe: ~500 transitions via **ControllerReplayBuffer.sample()** (boosted), report per-stratum **85th percentile** of |target_mid - online_mid|, terminal vs bootstrapped windows SPLIT | minutes | One-sided: FAIL (fix refuted as ceiling-remover) if safe-stratum bootstrapped-window p85 is already ~13-15 |
| D | Checkpoint inflation probe at successors of the 15 D2 conflict states + fresh probe episodes (NOT shuffled-buffer reconstruction): mean(full-argmax target - restricted target), and fraction of full-argmax picks landing on never-targeted candidates | minutes | PASS: inflation >> 0.1 fee AND >= ~80% picks on untrained. FAIL: inflation ~0 or picks mostly trained -> do not launch taken_noop |
| D | **Micro retrain with --bootstrap_support taken_noop** (REQUIRED, not optional), sized >= ~50 target-net refreshes past full sampling | fast (micro) | PASS: C2 Spearman off zero (pre-registered rho > 0.4 preferred). FAIL: null -> hallucination-max not binding on micro; still launch 12d (D remains structurally justified) but flag attribution risk |
| B | Knee-location rerun: credit_diagnostics --d2 --side atc, horizons [12,18,24,36,48] on 12c ep2000 checkpoint | ~25 min | Decides 24 vs 36: require >= 50% of ACTION-HELPS states sign-correct in-window at the chosen n |
| B | Parameterized --selftest + --smoke --nstep 24 (3 episodes): budget reconciliation identity, finite loss, sampled terminal fraction logged (expect near the 0.5 cap) | minutes | All asserts pass; terminal fraction logged for the 12d baseline |
| C (parallel, informational) | Micro full-coverage CF-replay gate with the 2.3 discriminator (target-net V Spearman on failure; lag 8-12 rerun) | ~2h | Adjudicates whether coverage is binding -> 12e lead decision |

Run-time monitors with numeric gates (12d):
per-stratum t off caps and responsive (abort: t at cap AND realized cov < target-0.05
sustained); safe-stratum median width >= 0.02 (else engage absolute width floor);
gap/width at D2 states (target >= 3x vs 12c's 0.007); benign-state NOOP-vs-issued Q gap vs
12c baseline; restricted-argmax winner share (NOOP vs a'); full-vs-restricted target
inflation drift; gap magnitude vs ground truth at frozen probes (optimism); per-stratum
|bootstrap_cov - realized_cov| < 0.2; per-stratum steps-since-last-tracker-feed.

Success criterion scoping (per amendments): the primary 12d gate is **rho_taken moving off
-0.10/0 plus a conflict-state Q level drop for late/NOOP actions** — NOT full C2 Spearman
recovery, which the audits agree also needs the coverage hole (C) and possibly the vertical
gate closed. Ordering is a lagging indicator early in the run (restricted bootstrap resolves
to ~NOOP-continuation until the policy takes resolving actions).

---

## 5. Interactions and composition risks

1. **B x E+F (width channel, opposing pushes)**: gamma^24 strengthens bootstrap width
   contraction ~1.44x/hop while E+F newly opens the shrink gate — joint pressure toward
   collapse in low-residual strata. Covered by: two-sided width gate, absolute width floor,
   per-stratum monitoring. Conversely, B raises target variance (wider 24-step return
   dispersion) exactly where E+F's tracker will now respond — the economy absorbs it; watch t
   oscillate rather than peg.
2. **B x E+F (coverage-measurement mixture)**: terminal fraction stays ~pinned at the 0.5 cap
   under both n; but B changes WHICH windows are terminal-collapsed (reach 24), shifting the
   -50 mass in the coverage sample further into the MID stratum. All E+F gates are
   per-stratum for this reason; a pooled 0.85 is uninterpretable.
3. **B x D (fixed point)**: taken_noop over 24-step windows = 24-step SARSA-with-NOOP-floor.
   Synergistic for chain length (fewer hops for the H>=24 residue) but MORE off-policy suffix
   per window; the off-policy content is common-mode across compared candidates in
   expectation (variance, not ordering bias) — still, this is the least-priced pairwise
   interaction in the audits. Mitigation: the single-flag ablation paths exist; if 12d
   ordering moves but magnitudes are wild, ablate D first.
4. **D x E+F (coverage feed composition)**: restricted targets are systematically <= full
   targets (NOOP floor), shifting target_mid down; the tracker recalibrates t to the new
   target distribution automatically — that is the point of E+F — but 12c-vs-12d coverage
   and bootstrap_hits comparisons are void. Within-run trends only.
5. **Attribution**: three simultaneous changes. Accepted deliberately because each has an
   independent offline validation and an independent run-time signature (t/width economy for
   E+F; inflation drift and winner share for D; in-window sign-correct fraction for B), plus
   one-flag ablation paths. The alternative (three sequential runs) spends ~3x wall-clock to
   learn what the signatures can separate.
6. **Deferred-fix interactions pre-priced**: if A revives later, it compounds with B on the
   terminal-item economy (up to 64+24 items/violation, terminal fraction pinned, duplicate
   (s,a) targets) — re-audit that pairing before enabling both; if C revives, it MUST ship
   with depth symmetrization or it re-introduces a fee-sized ordering artifact into a system
   that by then may be ordering-sensitive.
7. **Selftest is shared substrate**: B and D both rewrite `_selftest_windower` and the
   hand-built tuple sites; do it once (step 0) or the second change silently invalidates the
   first's validation.

---

## 6. Open decisions for JK

- **D1 — n=24 vs n=36.** Data decides (knee rerun, action-helps sign-correct fraction), but
  the tradeoff is yours: n=36 covers more of the lead distribution (leads up to 49) at
  gamma^36 = 0.33 bootstrap weight — weaker carrier for the >36 residue, more off-policy
  suffix. Recommendation: 24 unless the knee rerun shows < 50% action-helps sign-correct.
- **D2 — epsilon floor post-warmup.** Three separate audit threads (A, D, and the width-blind
  Hurwicz fact) note that post-warmup action variety is exactly zero, which throttles how
  fast taken-action mechanisms (D now, A/C later) can propagate credit. A small persistent
  epsilon (e.g. 0.02-0.05) is a behavior-policy change to the JK-approved setup and slightly
  degrades headline policy metrics. Recommendation: launch 12d WITHOUT it; add only if the
  restricted-argmax winner share collapses to ~0 (the pre-registered trigger).
- **D3 — CF-replay deferral.** Sign off on running C's micro full-coverage gate in parallel
  (~2h) and holding the mechanism for 12e. If you would rather have coverage in 12d despite
  the four silent-killer surfaces, say so — the audits amended it to viable, and this doc's
  deferral is a conservatism call, not a technical one.
- **D4 — Terminal-lookback deferral** and its revival trigger (knee at 36-48 + n=36 judged
  too weak). Same status as D3: viable, deferred.
- **D5 — Vertical binary-gate bug** (`pair_conflict_f`: binary CONFLICT_FL=20 gate; climbs
  invisible to shaping for ~7 steps). NOT audited in this cycle; at ~1e-3/step the shaping
  differential carries no ordering weight, so it does not block 12d — but it caps what any
  potential-based term can ever price vertically. Decide: commission a separate small
  faithfulness fix (smooth vertical ramp mirroring the lateral one, CBP makes it safe) with
  its own audit, or accept the cap for 12d/12e.
- **D6 — Calibration contract change.** E+F redefines interval width as TD-residual scale
  (LL semantics) and demotes realized-return coverage — the credal ground-truth reading of
  0.85 — to a diagnostic that will run well below target by design. This is the LL-faithful
  choice and the audits endorse it, but it changes what published width/coverage numbers
  mean. Confirm you accept the contract change, and whether any external write-up needs the
  realized-coverage diagnostic reported alongside.
- **D7 — Success-gate scoping.** Confirm the primary 12d gate is rho_taken + conflict-state
  level drop (per the amendments), with full C2 recovery explicitly OUT of scope for 12d.
  If C2 recovery is the gate you want, 12d must include C, and D3 flips.

---

## Appendix: expected 12d signatures (falsifiable predictions)

- t oscillates around 0.85 per fed stratum within ~1-2k gradient steps; width_mask fires
  regularly; mean_width stops the 12c runaway and settles finite (realistic: ~7-8 pooled,
  lower in stratum 2 — NOT the original 2-6 claim).
- Net Hurwicz-midpoint gap at the 15 D2 conflict states tracks the sign of the H=24
  ground-truth gaps within ~2-3 target-refresh cycles of violations replaying.
- rho_taken moves from -0.10 toward positive; rho_untaken may stay null (coverage hole open).
- Full-vs-restricted target inflation ~0 by construction; pre-fix positive and >> 0.1.
- Realized coverage (diagnostic) runs well below 0.85 — expected, not a regression.
- D2 ground-truth gap-vs-horizon curve: UNCHANGED (environment property). Anyone reading D2
  movement as a fix effect is confused about what D2 measures.
