# Design Revisions — post run-6 audit (2026-07-04)

Context: run 6 (outcome anchoring, realized-coverage tracker, progress shaping,
γ=0.97) showed (a) a Hurwicz flip — optimism outlives pessimism, monotone in c;
(b) realized coverage stalled at ~0.25–0.44 with t pinned at its 0.95 cap;
(c) still no agent beats NOOP; (d) first genuine LoS events at 1200 s densities.
Working hypothesis (JK): the flip is an information artifact — the agent cannot
see that exiting is the least risky option, so "risk-seeking" happens to
implement the right behavior blindly.

## Revise now (run 7)

1. **Observations: switch to the `relative` encoder, k_nearest=3.**
   Adds the exit-relative block (fl_diff_exit, exit_dist_diff, boundary
   distance/heading — the exact variables needed to represent "exit = low
   risk") and 9 features per neighbour incl. distance trend. The per-neighbour
   controllable_flag is ±1 for real aircraft and 0 for padding, which
   disambiguates the zero-padding pathology (absent neighbour previously
   indistinguishable from collision-range neighbour on the distance feature).
   Obs dim ≈ 54 (k=3); network unchanged apart from input width.

2. **Asymmetric outcomes: violation −50, exit +10 (DQN default).**
   Violations are lexicographically bad in the true objective (they end the
   episode; exits merely help). Symmetric ±10 let optimism trade separation
   risk for exit bonuses — observed as the single LoS at c=1.0 under 1200 s
   stress. MCTS already used −50; this unifies the two objectives.

3. **Coverage controller: fix the controller/actuator misalignment via
   n-step interval Bellman targets (realized-G built into the target).**
   Run 6 proved the saturation: t sat at its 0.95 cap all run while realized
   coverage stalled ~0.3. Cause: the tracker (sensor) measures REALIZED
   coverage, but t (actuator) modulates a loss whose targets are 1-step
   BOOTSTRAPPED samples — whose coverage is already ~0.95. The knob turns a
   different valve than the gauge reads.
   Fix (JK's formulation, supersedes the earlier two-loss-term proposal):
   replace 1-step targets with n-step interval targets along the stored
   per-aircraft streams —
   [sum_{k<n} gamma^k r_k + gamma^n L', same + gamma^n U'] for continuing
   windows; when the flight TERMINATES inside the window the bootstrap term
   vanishes and the target IS the realized return (exact G). One coherent
   target semantics (no second loss, no arbitration weight); censored
   streams still train (bootstrap at the window edge); the U-on-U optimism
   loop is damped structurally (bootstrap mass gamma^n per hop, ~6 real
   rewards per target). Standard uncorrected-n-step off-policy caveat
   accepted (Rainbow-style). n default 6 (36 s), --nstep 1 reproduces the
   old behavior for ablation. Streams already exist (built for the realized
   tracker) — replay stores window views over them.

4. **Training distribution must include the dense regime: duration 1200 s.**
   Run 6 trained at 600 s but the safety-critical LoS phenomenology only
   appears near 1200 s (spawn ramp). Episodes end at first violation, so the
   cost grows only with competence.

5. **Consistency: hybrid MCTS must reconstruct the leaf net's encoder.**
   bluebird_interval_mcts.make_env hardcodes extra_minimal/k=2; with a
   relative-encoder checkpoint the leaf bootstrap would read garbage. Read
   encoder_cls/k from the checkpoint metadata (store them at train time).

## Deliberate keeps (re-examined, unchanged)

- **γ = 0.97** — with −50 penalties, a violation 20 steps out still carries
  weight ~27; the annuity pathology stays dead. Revisit only if the MC-loss
  term shifts scales.
- **Progress shaping 0.05** — full-flight total ≈ 5, still well under the
  outcome terminals; potential-based, policy-invariant.
- **Exit bonus +10 (private value only)** — the decongestion externality
  (my exit makes everyone else safer) remains unpriced by per-aircraft
  rewards; acceptable for now, revisit if system-level metrics demand it.

## Autopsy verdict (seed-10043 LoS, 2026-07-04, diagnose_interval_mcts --heavy)

The conflict WAS avoidable: open-loop beam search from step 65 finds
9x LEFT_10 then NOOP keeps AIR-00 violation-free (worst sep 9.84 nm).
The planner missed it for two compounding reasons, neither a machinery bug
(all backup/selection/clone diagnostics PASS):
- alert gate (15 nm radius) engaged the pair too late — they closed from
  36 nm, so by engagement the escape needed ~90 degrees of accumulated turn;
- a 9-ply-deep specific action sequence is undiscoverable at 24 sims with
  NOOP rollouts (tree depth ~2-3 plies; heading clearances persist, so a
  NOOP rollout never continues a turn).

MCTS-side remedies (no retraining needed, test in next stress suite):
- gate on time-to-closest-approach, not current distance;
- add larger turn magnitudes to action_config (e.g. heading_left/right
  [10, 30]) so deep escapes become shallow — also worth considering for the
  DQN action space in run 8.

## Deferred research questions (after run 7 evidence)

- **State-conditioned / adaptive c** (the Helsinki slides' adaptive-c): the
  flip suggests optimal c is density-dependent. Test AFTER the information
  and calibration fixes — if the flip persists with exit-risk visible and
  calibrated intervals, adaptive c is the answer; if it reverses, it was an
  information artifact as hypothesized.
- MCTS c_act sweep under density stress; global density/count feature
  (needs custom encoder); pricing the exit externality.
