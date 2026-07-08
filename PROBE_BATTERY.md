# Controller-frame probe battery — pre-registered diagnostics (8 July 2026)

Purpose (JK): stop reacting to run outcomes; test faithfulness, reward
correctness, and ported-lesson coverage DIRECTLY. Each probe states its
expectation BEFORE it runs; a miss is a finding, not a tuning cue.
Probes marked [ENV] need no checkpoint (runnable before/without training);
[CKPT] run against any checkpoint (including mid-run).

## B. Reward-structure correctness (the objective itself)

B1. [ENV] ORACLE-POLICY ORDERING — the single most important probe.
    Score scripted policies under the sector objective (full episodes,
    2 seeds x 1200 s and 2400 s):
      - DELIVERER: route_parallel to every aircraft each step's clearance
        budget allows (approximation of intent-to-deliver)
      - NOOP: no clearances
      - HOLDER: keep every aircraft turning (the run-8 exploit, scripted)
      - WEAVER: alternate L/R on a rotating aircraft (run-8's texture)
    PRE-REGISTERED ORDERING: DELIVERER > NOOP > HOLDER, WEAVER;
    HOLDER < NOOP by at least the fuel differential; WEAVER <= HOLDER
    (adds fees for nothing). If any pair misorders, the objective is
    wrong and run-12b must not launch on it.

B2. [ENV] EXPLOIT RED-TEAM: (a) myopic oracle — a policy that argmaxes the
    TRUE immediate sector reward each step via deepcopy counterfactuals;
    run a full episode. Where 1-step greed leads is a lower bound on what
    the learner can find. EXPECTATION: myopic play produces plausible
    (boring) control, NOT a degenerate high scorer; its episode return
    should land between NOOP and DELIVERER. A weird strategy scoring
    above DELIVERER = objective exploit, fix before 12b.
    (b) scripted exit-window candidates (park 1.5 nm short of exit;
    oscillate across the fix) — none should out-score completing the
    delivery.

B3. [ENV] MARGINAL-CREDIT MEASUREMENT (the CTDE decision's data): at ~50
    probe states, exact counterfactual 1-step and H-step (H=10) sector-
    reward deltas per candidate clearance vs NOOP (determinized deepcopy).
    Report the distribution of |delta| vs the ambient per-step reward
    variance. EXPECTATION: top-candidate H-step |delta| distinguishable
    from ambient noise at H=10 for states with conflicts inside horizon;
    if not even the ORACLE deltas separate from noise, global-frame
    credit is information-starved and CTDE (or longer n-step) is needed
    regardless of what run 12a's curve says.

B4. [ENV] NEEDLE-THREADING FIELD: scripted approach trajectories to an
    exit (dead-center crossing; 1.5 nm offset miss -> excursion; veer-off
    2 nm early). Cumulative objective for each. EXPECTATION: crossing >
    veer-off > excursion-miss by margins >> shaping noise; if veer-off ~
    crossing, the run-11 boundary-aversion trap survives in the new
    objective.

## A. Spec faithfulness (does the system behave like the concept)

A1. [CKPT] CLEARANCE SEMANTICS: re-issue rate — fraction of clearances
    repeating the same (aircraft, instruction) as that aircraft's still-
    active clearance. EXPECTATION: falls with training (persistence is
    learned); high terminal rate = the fee is being wasted on no-ops the
    concept says are free.

A2. [CKPT] ATTENTION FIDELITY: per-aircraft attention mass vs (i) token
    mask-impact on Q (ground truth for "how much the scope reading uses
    aircraft i"), (ii) conflict proximity, (iii) clearance target choice.
    EXPECTATION: attention correlates with mask-impact (rho > 0.5); if
    not, the interpretability story is decorative and must not be cited.

A3. [CKPT] GLOBAL-VIEW UTILIZATION: chosen-clearance change rate when
    deleting the 4th..Nth-nearest aircraft token vs 1st-3rd nearest.
    EXPECTATION: nonzero sensitivity beyond 3-nearest (else the
    controller has learned only a pilot-frame policy in a controller
    body — frame not yet exploited; informative, not a bug).

A4. [CKPT] INTERVAL SEMANTICS (port of the LL battery): OOD width scaling
    (shuffled/garbage token sets vs real: expect >5x), width vs sector
    risk stratum (expect monotone after stratified-t training), and
    inversion count == 0.

## C. LunarLander lessons re-audit

C1. [NOW] Outcome-rate reporting: eval protocol reports violation rate BY
    TYPE + delivery rate + clean rate per condition (LL's crash/land-rate
    discipline), not TTV means alone. Adopt in all future evals.

C2. [CKPT] Q-VS-GROUND-TRUTH ORDERING + UNBIASED COVERAGE (LL's cheap-env
    verification, enabled here by determinized clones): at ~30 probe
    states, roll out EVERY candidate clearance to H=15 under the frozen
    policy via deepcopy; rank-correlate ground-truth returns with the
    net's Hurwicz scores (EXPECTATION: rho > 0.4 by end of 12a; near 0 =
    the net hasn't learned the objective, whatever the TTV curve says).
    Simultaneously: fraction of ground-truth returns inside each
    candidate's interval = a coverage measure with NO survivorship bias
    (fixes the standing measurement problem; becomes the primary
    calibration instrument).

C3. [JSONL] TERMINAL-GROUNDING WATCH (LL's structural advantage): fraction
    of episodes ending censored vs terminal, tracked per bin. EXPECTATION:
    if clean-1200s episodes grow, the tracker's diet starves again ->
    C2's rollout coverage takes over as primary.

## B1-v2 — ACTIVE (blessed 8 July); implemented as diagnose_controller.py
## --probe b1

Status: B1 as registered above MISSED twice (see MORNING_BRIEF.md §2-3). The
failure accounting localizes the misses to the instrument (discounted-from-t0
basis + oracle competence at plateau density), not clearly to the objective.
JK blessed this replacement on 8 July 2026; it REPLACES the original B1 as
the gate. Amendments below record exactly what was implemented.

Basis (implemented): per episode, the per-step training rewards r_t
(objective v2 + CBP lag-1) are collected from bcd.run_episode(cbp=True)
itself via its step_hook (the system's own pricing — no mirror copy on this
path; the sector_step/sector_step_cbp mirrors remain validated and are still
used by b2-b4). Score = MEAN RETURNS-TO-GO over on-trajectory states — mean
over t of G_t = sum_{k>=t} gamma^(k-t) r_k, the quantity the learner's
targets estimate — instead of the from-t=0 episode return (which makes
everything after ~700 s invisible: gamma^118 ~= 0.027). Episode G_disc and
undiscounted G still reported as side columns, plus the competition-metrics
side table.

Pairs (2 seeds: 10043, 20042):
  1. DELIVERER > NOOP  [KEPT, premise fixed]: GATED at LIGHT density.
     AMENDMENT vs draft: not a 600 s scenario — transits take 830-1500 s,
     so a 600 s episode ends before ANY entry can complete; the premise fix
     is a DENSITY knob, not a duration cut. The standard InfiniteEnv
     hardcodes Infinite.setup's spawn defaults (2 starters, 0.01/s initial,
     0.1/s max) and exposes no density keys, but the same package's
     CustomInfiniteEnv (already used by B4) passes them through. Light env
     = CustomInfiniteEnv with make_controller_env's exact config mutations
     + spawn 0.001/s (initial == max, no ramp), 2 starters, 1800 s.
     Calibrated 8 July (NOOP density probe, both seeds): mean 5.6-6.7
     concurrent, plateau 7-8, max 9-10, NOOP transits complete — inside the
     6-10 target with full-length transits possible. The plateau-density
     (standard 1200 s) DELIVERER > NOOP comparison is REPORTED but not
     gated — it measures the capability wall (vertical instructions), not
     the objective.
  2. HOLDER < NOOP  [KEPT, new basis, standard 1200 s]: sign check done in
     advance for the new basis: an earlier -50 occupies a larger, less-
     discounted share of visited states (mean-RTG of a violation at step T
     ~ -50*(1-g^T)/(T*(1-g)): T~24 -> ~-36 vs T~77 -> ~-20), so HOLDER <
     NOOP remains the correct expected sign and fees only widen it.
  3. DELIVERER > DELIVERER+REISSUE  [REPLACES WEAVER<=HOLDER; standard
     1200 s]: paired-trajectory fee test. Pass A records DELIVERER's exact
     (step, callsign, action) schedule; pass B replays it verbatim and the
     probe asserts DETERMINISM (identical per-step aircraft-count
     fingerprint incl. callsign sets, violation step and kind, and a
     bit-identical reward stream); pass C replays the schedule and, on
     recorded no-issue steps, re-issues a still-active clearance.
     AMENDMENT (forced choice, documented): L10/R10 are RELATIVE heading
     commands (a repeat accumulates another 10 degrees — the same
     mechanism DELIVERER's escalating turn uses), so the only dynamically
     inert re-issue is route_parallel (ABSOLUTE: change_heading_to =
     current segment bearing), and only while that recomputed bearing
     still equals the aircraft's selected heading; the shadow verifies
     this per re-issue with the env's own action builder (read-only)
     before issuing. The score gap must equal the discounted extra-fee sum
     to <1e-6 relative ON BOTH BASES (mean-RTG gap vs mean of the fee
     suffix sums; from-t0 G_disc gap vs sum gamma^t * fee_t), and the gate
     additionally requires >= 1 re-issue (else vacuous). Pre-registered
     fallback if pass A vs pass B is NOT deterministic: analytic
     construction — price pass A's recorded stream twice, adding the fee
     on the recorded eligible no-issue steps — and report the
     non-determinism prominently. (Machinery smoke-tested 8 July on the
     light env: determinism bit-identical; 48/48 re-issues inert; gap
     identity to ~1e-15 relative on both bases.)

Note: with --mask_reissue the LEARNER cannot express the re-issue policy;
pair 3 tests the objective's pricing, not reachable behavior. That is the
point — B1 is a reward-correctness probe.

Same 2 seeds (10043, 20042). The 2400 s leg is dropped (dead weight: every
scripted episode ends at its violation <= 768 s). JSON output carries a
b1v2 tag in the filename.

## Process
- Implement as diagnose_controller.py with --probe selectors; run the
  full battery at every checkpoint milestone (not once).
- Every probe prints its pre-registered expectation next to its result.
- B1/B2 gate run 12b regardless of 12a's learning curve.
