# Interval MCTS v2 — locked specification (5 July 2026)

Decisions locked with JK. Evidence base: MCTS_ARCHITECTURE.md +
diagnostics/design_campaign/SYNTHESIS.md (D1-D5, seed 10043).

## The inversion (the defining change)
The interval DQN (adaptive-c) flies EVERY aircraft by default, live.
The planner is an override: it spends its budget only on aircraft where
thinking is expected to change the outcome. (D1: planner idle for 18
consecutive steps while an unwatched aircraft drifted out; D4: the failing
aircraft ranked top-relevance the whole time.)

## Components

1. BASE POLICY (live and imagined, identical — Q3 consistency):
   the trained interval DQN with adaptive width-conditioned c, per aircraft,
   w_mid auto-calibrated per checkpoint. Deploy and rehearse the same
   controller; safety margins live in the penalty structure, never in a
   deliberately-degraded imagined self.

2. ATTENTION (replaces pure geometric gating):
   candidates per step =
     (a) geometric floor: TTC gate (cpa_dist 8 nm / cpa_time 300 s) or
         proximity < 15 nm  [permanent, encodes physics not model opinion]
     (b) interval relevance: one batched net pass over all aircraft;
         relevance = width(chosen action) + LARGE if argmax at c=0 differs
         from argmax at c=1 (decision ambiguity). Top-k by relevance join
         the candidate set.  [ranking use only — ordinal bar, per sober
         assessment]
     (c) random audit: with prob p_audit (default 0.10/step) one
         non-candidate aircraft is added.  [insurance against confident
         wrongness; rate revisited against bucketed-calibration data]
   Budget: max_planned slots ordered (a) then (b) then (c).

3. REHEARSAL (per attended aircraft, sequential as before):
   - in-tree steps: real simulator; all other aircraft flown by the base
     policy (not NOOP, not frozen) — D2b validated (phantom excursions
     vanish; correcting turns elicited).
   - NO rollouts. Each simulation walks the tree, expands one node,
     evaluates it with the network's interval (leaf bootstrap at every
     expansion). ~10x cheaper per simulation; frozen-future distortion
     eliminated.

4. SHARED FATE (Q1): any violation observed during rehearsal ends the
   rehearsal immediately. Penalty at that step's discount: -50 if the
   planned aircraft is involved, -10 if not. (Large enough to matter;
   small enough that own-action credit survives. D1: background
   violations previously carried exactly zero weight.)

5. NODE VALUES + BACKUP (JK's rule): a child is born with the network's
   interval for its state as prior. Each simulation backs up
   [path rewards + gamma^d * Q_l(leaf), path rewards + gamma^d * Q_u(leaf)]
   into every node on the path as RUNNING MEANS of lower and upper
   (the nim-ancestor rule; envelopes retained only behind
   --backup envelope for ablation).

6. SELECTION (annealed c, replaces Hurwicz-c_search + UCB double-count):
   in-tree score = l + c(n) * (u - l), with
   c(n) = c_act + (c0 - c_act) / sqrt(1 + n_child),  c0 = 1.0.
   No separate UCB term. Root commits at c_act (default 0.1) with
   NOOP-preferring exact-tie break (unchanged).

## Validation battery (all niced; run-8 training must not be disturbed)
- Seed 10043 full episode: MUST beat 456 s; expectation clean 600 s
  (D2b showed the base-policy future alone elicits the saving turn).
- Seeds 10042, 10044: no regression vs 582/600 s.
- BASELINES on same seeds: (i) NOOP, (ii) pure base policy (DQN adaptive-c,
  no planner) — v2's value-add over its own base policy is the headline
  planner metric from now on.
- Latency: target mean < 5 s/decision (no rollouts); report distribution.
- Bucketed calibration instrument (new, feeds the audit-rate decision):
  coverage of realized H-step returns bucketed by traffic density and by
  predicted width, measured on the validation episodes.
- Leaf checkpoint: best_run6.pt now; swap to run-8 winner on selection.

## Out of scope for v2 (deliberate)
- Horizon extension (D3: costliest lever, superseded by leaf values).
- Per-situation calibration training fixes (run 9+; measured first).
- Vertical actions (Springfield prep, separate track).
