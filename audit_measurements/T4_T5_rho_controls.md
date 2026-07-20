# T4 — Does the width-free rho curve tell the same story as c=0.5?

**What this decides:** whether the rho readings on M1/M3 are an
artifact of interval-width collapse (the audit flagged that at
c_train=0.5 the pre-registered rho-vs-rho_mid width control is an
algebraic identity: l + 0.5(u-l) = (l+u)/2, so it controls nothing).
The genuine width-sensitive controls are c=0 (pure lower bound) and
c=1 (pure upper bound). Every round-1 M1 checkpoint (205216 run, 16
ckpts) and every round-1 M3 checkpoint (205320 run, 32 ckpts) was
re-scored at c in {0, 0.5, 1} with the module's own instruments
(rho_eval on the h=69 M1 probe; the m3 harness rho_eval_subset +
lateral_keep on the capped h=69 probe), pricing smooth/0.5.

**Answer: yes — c=0 tracks c=0.5 closely on both modules.** M1 ep800:
c0 +0.28, c0.5 +0.33, c1 +0.27. The M1 curve wanders 0.02-0.33 at all
three c values in near-lockstep. M3 is noisier at c=1 (upper heads are
the untrained-vertical-contaminated side even under the lateral
subset), but c=0 and c=0.5 agree within ~0.1 at most checkpoints
(ep800: c0 +0.36, c0.5 +0.34, c1 +0.13). Median interval width falls
only mildly over training (M1 ~10.4 -> 10.6, i.e. flat; M3 11.7 ->
7.4), so there was no width collapse driving the score in the first
place — and the ordering story is the same with width removed.

**Decision:** the NO-VERDICT on M1/M3 rho bars cannot be blamed on the
width term; whatever the instrument reads, it reads at every c. Full
per-checkpoint tables (all c, width quantiles 25/50/75) in
`rho_c_sweep.json`.

# T5 — Untrained null bands for the rho instruments (30 random inits)

**What this decides:** whether ANY of the observed rho values on these
20-state probe sets mean anything. The audit's single untrained scores
(0.29 M1 / 0.411 M3) hinted the instrument floor is high; the proper
30-init band (fresh random ControllerAgent nets, token_dim 60, n_instr
5, torch seeds 7000-7029, same instruments, same probes) settles it:

| instrument | c | null mean | sd | 2.5-97.5% band |
|---|---|---|---|---|
| M1 rho_eval | 0.0 | -0.004 | 0.400 | [-0.675, +0.650] |
| M1 rho_eval | 0.5 | -0.000 | 0.389 | [-0.615, +0.634] |
| M1 rho_eval | 1.0 | +0.000 | 0.398 | [-0.687, +0.596] |
| M3 lateral rho | 0.0 | +0.035 | 0.437 | [-0.661, +0.723] |
| M3 lateral rho | 0.5 | +0.054 | 0.448 | [-0.600, +0.847] |
| M3 lateral rho | 1.0 | +0.039 | 0.445 | [-0.606, +0.794] |

The null is centred at ~0 (good: no systematic bias) but its spread is
enormous — a single random net scores anywhere in +-0.6-0.85 because
the 20 "states" are timestamps of one trajectory (effective n ~ 1: one
draw of net-vs-scenario geometry, not 20 independent draws).

**Decision:** the pre-registered rho > 0.4 bar sits INSIDE the 95%
null band of both instruments. No round-1 rho reading on M1 (max
+0.33) clears the M1 band's upper edge (+0.63), and M3's trained
values (0.06-0.70) live entirely inside its band (upper edge +0.85).
The instrument as built cannot certify OR refute ordering at this
probe design — the audit's "NO-VERDICT (instrument-limited)" is now
quantified, and any redesign needs genuinely independent states (or
paired/trajectory-level statistics), not more episodes. Band archived
in `null_bands.json` (per-c, per-instrument, all 30 values).
