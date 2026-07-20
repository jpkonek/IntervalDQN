# Credit-pipeline diagnostics — where does candidate-ordering information die?

JK directive (14 July 2026): candidate ordering (C2) is null in 12b, 12c,
AND the credit-dense micro-problem, while policies still learn. Trace WHERE
ordering information is lost along target → gradient → representation →
readout. Prioritize stages 1–3; stage 4 (supervised-probe architecture
test) runs afterward, reframed as: identify which feature of LL — where
everything works — we failed to implement faithfully in 12c.

EVERY diagnostic runs on BOTH: 12c ep2000 checkpoint (the patient) and an
LL per_c checkpoint (the positive control — same instruments, healthy
pipeline). Pre-registered predictions below; misses are findings.

## Pre-registered LL-reference hypothesis (the stage-4 candidate answer,
## stated before stages 1–3 run)

LL's credit pipeline differs from 12c in three ways at once: (i) DENSE
action-differentiated immediate reward (thrust changes next-step shaping
measurably); (ii) ONE-step targets (consequence horizon ≈ credit horizon);
(iii) 4 actions, all heavily sampled (no untaken-candidate problem).
12c inverts all three: ~21 candidates, immediate action differential
~1e-3 (B3), consequences 50–80 steps out vs a 12-step window. If D1–D3
localize the loss to these stages, the unfaithful feature is the
REWARD/TARGET GEOMETRY, not the architecture — consistent with every LL
excursion to date exonerating the architecture.

## D1 — Taken vs untaken candidate split (coverage stage)

At ~20 probe states (12c: on-policy at density; LL: on-policy): per-
candidate ground-truth H-step rollout returns (C2 machinery). Estimate
per-candidate visitation by rolling the frozen policy from each probe
state with small epsilon (N=30 short rollouts) and counting first-step
action frequencies. Split candidates into TAKEN (top of visitation) vs
UNTAKEN (never chosen in N rollouts). Report separately: rank correlation
(net Hurwicz vs GT), calibration (GT inside interval), and MEAN WIDTH.
- H-coverage: 12c rho_taken >> rho_untaken ≈ 0; LL: both healthy (all 4
  actions frequently sampled → little difference).
- WIDTH-EXPOSURE CHECK (the action-level certain-basin question): if
  untaken candidates are NOT systematically wider than taken ones, width
  does not track training exposure at the action level — the state-OOD
  narrowing pathology and the ordering failure are one mechanism.

## D2 — Consequence-horizon vs credit-horizon (target stage)

At ~15 conflict states (12c: min pair separation < 12 nm, LoS ahead under
NOOP; LL: pre-terminal states ~30–80 steps from crash under a degraded
policy): ground-truth Q-gap between the best conflict-resolving candidate
and NOOP as a function of rollout horizon H ∈ {6, 12, 24, 48, 96} (12c)
/ {1, 3, 6, 12, 24} (LL), determinized deepcopy, frozen policy after the
first action. Report gap(H) alongside per-state target noise.
- H-mismatch: 12c gap ≈ noise for H ≤ 12 (the n-step window sees nothing),
  emerging only at H ≥ 24–48. LL: gap significant already at H = 1–3
  (dense shaping differentiates immediately).
- This directly measures how much action signal the training target can
  possibly contain.

## D3 — Bootstrap-chain fidelity (propagation stage)

On recorded episodes (12c JSONL replays / fresh greedy rollouts;
LL rollouts): evaluate V(s_t) = max Hurwicz at every step; correlate with
realized return-to-go G_t, binned by steps-to-terminal (0–10, 10–20, ...,
80+). Also mean |V − G| per bin.
- H-chain: 12c correlation collapses beyond ~20 steps from terminal (the
  chain cannot carry conflict credit to decision time ~50–80 steps out).
  LL: correlation stays high at all depths (short effective horizon +
  dense reward anchor the chain).

## Stage-4 (deferred until D1–D3 report): supervised probe on
## counterfactual dataset — run ONLY as the LL-faithfulness identifier:
## whichever stage D1–D3 indicts, the supervised probe confirms the
## architecture is innocent of it (JK's prior: it will be).

## Decision relevance
If D1–D3 land as predicted, the faithful-to-LL fix is target-side:
rollout/planner-generated targets for non-taken candidates and/or
immediate counterfactual action-differentiated reward (CBP was this idea
but ~1e-3 too weak) and/or per-candidate epistemic width so untried
actions stay wide (restores Hurwicz exploration semantics). These are
run-13-era design choices; MCTS v3 depends on the same fix.
