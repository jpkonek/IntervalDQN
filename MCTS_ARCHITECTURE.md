# Interval MCTS — architecture map and design questions (6 July 2026)

A plain map of what the planner actually does, one block per design choice.
Status: VERIFIED = machinery proven correct by diagnostics; SUSPECT = design
choice under foundational review.

## The pipeline, per live step

```
live sector state
   |
   v
[1] ATTENTION   which aircraft get thinking time this step?
   |            gate: predicted close approach (TTC) or proximity; budget cap 4
   |            STATUS: SUSPECT (heuristic geometry, not value-of-thinking)
   v
[2] FACTORING   plan ONE aircraft at a time, in risk order.
   |            while planning aircraft i: already-decided aircraft replay
   |            their chosen command once; everyone else does nothing, forever.
   |            STATUS: SUSPECT (the imagined future is one nobody intends)
   v
[3] REHEARSAL   tree over aircraft i's own commands (5 actions), ~2-3 plies
   |            deep at 24 simulations; beyond the tree, aircraft i holds
   |            heading (does nothing) to the horizon (15 steps = 90 s).
   |            STATUS: machinery VERIFIED; rollout policy SUSPECT
   |            (a turn is always evaluated WITHOUT its recovery turn)
   v
[4] SCORING     each rehearsed future is scored by aircraft i's own shaped
   |            rewards, plus -50 if AIRCRAFT I violates, +10 if it exits.
   |            STATUS: SUSPECT (see "objective mismatch" below)
   v
[5] INTERVALS   each option keeps a range over its rehearsed outcomes
   |            (model-free: min/max envelope; hybrid: average bounds +
   |            the trained network's range at the horizon).
   |            STATUS: arithmetic VERIFIED; semantics drifted from the
   |            slides' conception (range over own future choices, not
   |            uncertainty about value) — hybrid partially restores it
   v
[6] COMMIT      execute the option whose worst case is best (pessimistic
                Hurwicz, c_act = 0.1). Replan everything next step.
                STATUS: VERIFIED mechanically; meaningfulness depends on [5]
```

## The seed-10043 evidence (why foundations are in question)

- Legacy gates: LoS at 456 s.
- TTC + macro-turns: LoS avoided (per-aircraft return -792 -> +8) but AIR-02
  excursion at 384 s — WORSE on the competition metric.
- Replan-priority fix: IDENTICAL excursion at 384 s. Attention was not the
  mechanism.

## Candidate foundational problems (to be settled by data, then discussion)

**P1 — Objective mismatch (per-aircraft returns cannot see the real game).**
The competition ends at the FIRST violation by ANYONE. A per-aircraft return
treats "my LoS at t=76" and "my excursion at t=64" as comparable costs at
different discounts — and the discounting can make the EARLIER violation the
cheaper one. The planner may be knowingly choosing the excursion as
"least bad" while the metric says any earlier violation is strictly worse.
Data needed: for each commanded action, did the chosen branch's rehearsals
contain a violation, which kind, at what depth — i.e. was the excursion a
deliberate trade?

**P2 — The default future is frozen, so recovery is invisible.**
Rollouts assume "act once, then nothing" for the planned aircraft, and
"nothing, forever" for everyone undecided. So every avoidance turn is priced
as if no controller will ever issue the recovery command — turning away from
a conflict looks like drifting to the boundary, because in the rehearsal it
IS. A one-line alternative exists: make the default rehearsal policy
"everyone flies their route" (the route-parallel command) instead of
"everyone freezes". Data needed: rerun key decisions with route-following
rollouts and compare the pictured futures.

**P3 — Horizon blindness prices distant violations at a discount.**
-50 * gamma^14 ~ -32 vs nearer shaped noise; and anything past 90 s does not
exist. Data needed: horizon sensitivity (15 vs 25 vs 40) at the decisions
that mattered.

**P4 — Attention is geometric, not value-driven.**
The intervals natively express "how much does the choice matter here"
(wide, overlapping action-ranges = thinking helps; identical ranges =
thinking is wasted). The gate ignores them in favour of distance/TTC
geometry. Data needed: for gated-out aircraft, compute the intervals anyway —
would interval-driven attention have selected differently?

**P5 — Interval semantics (the lineage question).**
Model-free envelopes measure "spread over what I might do next", not
"uncertainty about what this is worth". The slides' conception is the
latter. The hybrid (network bounds at the leaves, averaged up the tree)
is the faithful reading; the envelope is a stand-in that made sense before
a trained network existed. Decide which object the framework means.
