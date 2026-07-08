# Controller-Frame Architecture — interval DQN/MCTS as the ATCO (run 12+)

Decision (JK, 7 July 2026): the decision-maker is a SINGLE agent with a
GLOBAL view, playing the air traffic controller. The pilot-frame era
(decentralized per-aircraft agents, runs 1-11) ends with run 11. Bluebird's
own docs name the frames: centralized = "an air traffic controller
perspective"; decentralized = "each agent operates as an individual pilot".
We were in the pilot frame; the goal was always the controller.

## 1. Role and interface

- One agent observes the entire sector each step (the radar scope) and
  issues AT MOST ONE clearance per step: (aircraft, instruction) or global
  NOOP. One radio call per 6 s sweep = 10 instructions/min ceiling —
  realistic ATCO bandwidth, and heading clearances persist, so low bandwidth
  suffices. This matches the shape of bluebird's own centralized action
  space (1 + N x A) and the competition interface (generate_action can emit
  the chosen clearance and NOOP for everyone else).

## 2. Sector state encoding (replaces slot concatenation)

- Per-aircraft token: the existing relative-encoder feature vector
  (bug-vetted; exit-relative + boundary + neighbour blocks) + raw
  kinematics (lat/lon normalized to sector frame, heading, FL, speed).
- Shared token MLP -> small multi-head self-attention (1-2 layers, d~64)
  -> (a) per-aircraft contextual embeddings, (b) attention-pooled sector
  summary. Permutation-invariant, any aircraft count, no slots, no padding
  ambiguity, no cap.
- Bonus: attention weights are an inspectable "who is the controller
  looking at" signal — the attention thread of this project becomes a
  first-class, readable quantity.
- Sizing: CPU-trainable; parameter budget comparable to the current MLP
  (target < 100k params to start).

## 3. Action space and Q-heads (pointer-style; dissolves the slot problem)

- For each aircraft i: head maps its contextual embedding -> interval
  Q [lower, upper] for each instruction in {L10, R10, route_parallel}
  (macro turns as a later flag). Plus one global-NOOP head from the sector
  summary.
- Action set at a state = {global NOOP} U {(i, instr)}: variable size,
  handled natively because Q is computed per aircraft — no fixed slots.
- Selection: Hurwicz over clearance intervals (adaptive width-conditioned c
  carries over unchanged). "Which aircraft deserves attention" is no longer
  a bolted-on gate: it IS the argmax.

## 4. Controller objective (supersedes the 8-term pilot stack)

Four components, each with an aviation meaning (LL-unification applied at
sector level):

  a. SECTOR POTENTIAL, paid as gamma*Phi' - Phi:
     Phi_sector = - alpha * SUM_i (along-track distance to exit, nm)
                  - beta  * SUM_i (centreline offset)
                  - delta * SUM_pairs f(separation margin / CPA)
     Additive by construction -> per-term deltas are loggable, and credit
     for a clearance can be traced to the terms it could touch.
     Telescoping kills slow-roll and dawdle farming; off-route/near-conflict
     are debts repaid on recovery, not incomes or taxes.
  b. FUEL/DELAY: -eps_fuel per aircraft-in-sector per step (global).
     This is what makes universal holding bleed (a perfect hold pays zero
     potential difference); congestion-conditioned pricing (workload
     management, ATC goal 2) is a later refinement of eps_fuel(density).
  c. INSTRUCTION ECONOMY: -0.1 per issued clearance (max one/step).
  d. TERMINALS (global; shared fate is now NATIVE):
     -50 when ANY violation ends the episode; +10 per aircraft delivered
     (credited to the single controller return at the delivery step).

  Initial weights (to be tuned in the pilot run): alpha 0.05/nm,
  beta 0.02, delta sized so one pair inside 15 nm converging ~ one
  aircraft-minute of fuel, eps_fuel 0.005/aircraft-step.
  ATC priority mapping: (1) separation = pair-margin potential + terminal
  + pessimistic execution (the credal machinery's native role);
  (2) workload = density in state + fuel structure; (3) emergency =
  dormant (env models none; enters later as state-triggered weight
  overrides).

## 5. Interval/credal machinery — carries over, mostly simplified

- Interval heads, interval loss with sampled targets, n-step windows,
  terminal-boost replay, coverage-driven t: unchanged in form.
- MAJOR SIMPLIFICATION: there is ONE trajectory per episode (the
  controller's), not N per-aircraft streams. Realized returns-to-go are
  computable for every step at episode end; the only censoring is the time
  limit. The per-aircraft windower/stream machinery, its churn handling,
  and most survivorship caveats are deleted.
- Stratified-t: strata by SECTOR risk (min pairwise separation bucket at
  the decision step) instead of per-aircraft neighbour distance.
- Realized-coverage tracker, bucketed calibration diagnostics, adaptive-c:
  as before, at controller level.

## 6. MCTS in the controller frame — v3, a large simplification of v2

- The joint-action problem DISSOLVES: one clearance per step means the tree
  is directly over clearances. Sequential per-aircraft factoring, frozen
  actions, per-aircraft returns, replan-priority: all deleted.
- v2's good parts survive intact: deepcopy determinized rehearsal, network
  intervals at nodes with running-mean bound backup (JK's rule), annealed
  c(n) selection, leaf bootstrap, base-policy rollouts (the controller DQN
  flies the imagined future), shared-fate returns (now just "the return").
- Attention/gating layer: deleted — candidate clearances are ranked by the
  net's own interval relevance; search budget goes to the top-k clearances.

## 7. Migration plan

- Run 12a (PILOT, ~1000-1500 eps @ 1200 s): sole purpose = measure
  attribution noise in the global frame. Acceptance: TTV slope clearly
  positive over NOOP within the run; loss/width dynamics sane. If it
  stalls, fallback path = CTDE middle ground (global reward + factored
  execution) before abandoning the frame.
- Run 12b (full, 3600 s, ~6000 eps) on 12a's evidence, with the standard
  battery: calibration (controller-level realized coverage), behavior
  (GIF + instruction economy + throughput/deliveries), TTV + uncapped
  protocol, attention-weight inspection.
- Then MCTS v3 validation on the 12b winner (planner value-add over its
  own base policy remains the headline planner metric).

## 8. What survives / what is deleted

- SURVIVES: interval loss + coverage/stratified/boost machinery, bucketed
  calibration + semantics diagnostics (reframed), radar renderer, violation
  detection, reward-layer pattern, run/eval/suite harness bones, MCTS
  backbone (v2 minus factoring), memory of every measured pathology.
- DELETED: decentralized generate_action as the agent (kept only as a
  baseline), per-aircraft reward plumbing and streams, the 8-term pilot
  reward stack (superseded by Section 4), v2's factoring and gating layers,
  bluebird slot-sampling centralized mode (bypassed by the set encoder).

## 9. Open questions for JK before 12a launches

1. Weights in Section 4 are engineering guesses — bless or adjust.
2. Delivery bonus at +10 per aircraft with -50 any-violation: preserves the
   old magnitudes; in the controller return, deliveries recur — is 5:1
   violation:delivery the intended exchange rate? (Lexicographic intent
   says violations should stay practically unbuyable.)
3. Pilot-run acceptance thresholds (Section 7) — agree or tighten.
