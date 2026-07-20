# OVERNIGHT AUDIT MEASUREMENTS — 20 July 2026 (00:05-00:20)

Pre-registered measurement pass on ROUND-1 artifacts while the six
round-2 arms train undisturbed (their processes were never touched; all
work ran sequentially at nice 15 from audit_measurements/). Pricing for
every measurement: bcd.set_conflict_pricing("smooth", 0.5). No project
source file was edited. Per-task detail files sit next to this report:
T2_T3_cf_decomposition.md, T4_T5_rho_controls.md, T6_basis_ceiling.md,
T8_m4_forensics.md, plus one JSON per measurement.

## T2 — CF-target decomposition (decides: where M2's doom comes from)

Replaying the 20 NOOP probe states through the production CF path
(cf_branch_rollout, H=36, each round-1 M2 checkpoint as its own frozen
continuation) shows the NOOP training target is manufactured negative
by two separate, now directly measured mechanisms:
- real in-window LoS caused by the continuation policy re-leveling the
  pair (ep800: 6 branches violate inside the episode, from probe states
  as early as step 35; targets -18 to -41);
- branches running past the 100-step episode end (4-6 per checkpoint;
  some "violations" at absolute steps 109-130 — phantom events the
  censored live returns can never see) plus gamma^36 = 0.334 bootstrap
  tails of -2 to -16 on the non-violating branches.
Mean full target y across states: -15.2 (ep200), -13.7 (ep400), -12.3
(ep600), -14.8 (ep800) on a scenario whose true NOOP value is ~0.
DECIDES: the doom equilibrium is target-borne and conditional; both
holes must be closed, fixing either one alone leaves double-digit
negative targets from the other.

## T3 — Pure-MC floor (decides: how to score the running m2b arm)

The disc=0 target is G_cf alone; the measured per-state profile is the
best m2b can converge to. PRE-REGISTERED SUCCESS CRITERION (written
before m2b finishes): Q(NOOP) approaching roughly -1 to -2 at early
probe states and -18 to -46 at mid/late states whose branches violate,
with 10-45% of branches violating in-window (checkpoint-dependent:
25%, 30%, 10%, 45% at ep200/400/600/800) — NOT zero. Near-zero
everywhere = m2b not learning its targets; uniform -20s = tail
contamination surviving some other way. The audit's expected profile is
confirmed; judge m2b against the band, not one checkpoint.

## T4 — Width-free rho control (decides: can width collapse explain the rho readings?)

No. Re-scoring all 16 round-1 M1 checkpoints and all 32 round-1 M3
checkpoints at c=0 (pure lower bound), c=0.5 (the vacuous-identity
point) and c=1 (pure upper bound): the c=0 curve tracks c=0.5 within
~0.05 on M1 (ep800: +0.28 / +0.33 / +0.27) and within ~0.1 on M3
(ep800: +0.36 / +0.34 / +0.13). Median widths stay ~8-13 throughout —
no width collapse existed to correct for. DECIDES: the M1/M3
NO-VERDICT cannot be rescued or blamed via the width term.

## T5 — Untrained null bands (decides: whether ANY rho reading on these probes means anything)

30 fresh random-init nets (token_dim 60 matched, torch seeds 7000-7029)
scored by the same instruments: null mean ~0 but 95% band roughly
[-0.62, +0.63] on M1 and [-0.60, +0.85] on M3-lateral (sd 0.39-0.45).
The pre-registered 0.4 bar sits INSIDE both null bands, every trained
M1 value (max +0.33) sits inside the band, and M3's whole trained range
does too. Root cause: the 20 "states" are timestamps of one trajectory
(effective n ~ 1). DECIDES: the instrument as built can neither certify
nor refute ordering; redesign needs independent states or paired
statistics, not more episodes. Band archived in null_bands.json.

## T6 — Basis ceiling (decides: is M2's rho > 0.4 bar reachable in principle?)

No. Full GT tables for the 20 M2 probe states under both bases
(NOOP-continuation vs ep800-frozen-policy continuation, H=80 capped,
CBP): mean Spearman between the bases is +0.180 (sd 0.39). A perfect
learner of the training basis therefore tops out around 0.18 on the
NOOP-basis bar — under half the 0.4 threshold — and that number itself
sits inside the T5 null band. The ep800 net actually tracks neither
basis (+0.113 vs NOOP — exactly reproducing the run's recorded value,
validating the rebuild; +0.014 vs its own training basis). Mechanism:
39/220 frozen-basis branches (6/20 NOOP branches) end in a real LoS
caused by the continuation policy. DECIDES: judging frozen-policy-
trained values against a NOOP-basis table is a quantified category
error; re-base the instrument or the training, and note the bases will
converge only once the policy stops wrecking the pair.

## T8 — M4 forensics (decides: does the FULL-MISS re-score stand?)

Yes, on the run's own records. DEVIATION FLAGGED: no M4 checkpoint
exists (train_arm passes rho_fn=None, and run_training_arm only saves
checkpoints on rho evaluations — same gap in round 2), so nothing
net-side was recomputed; evidence is from summary + jsonl. Measured:
greedy first command (climb10 to AIR-00) is one draw at an empirical
acceptable-first rate of 0.645 over the last 200 training episodes (the
claimed 0.60-0.67 coin flip); the same policy gave the protected
through-flight its 3rd command at 12 s, 10 of 60 commands total, and
argmaxes to it at 8/20 frozen probe states (b3 needs 0); b2 needs <= 2
commands, got 60. DECIDES: b1 as written does not measure the A15
intent; FULL MISS stands; bar redesign remains queued for JK ruling.

## Cross-cutting takeaways

1. Every "ordering" number in the battery is currently read through an
   instrument whose null band swallows the bar (T5) AND whose GT basis
   disagrees with the training basis at rho ~ 0.18 (T6). Fix the
   instrument before interpreting any round-2 rho curve.
2. The M2 doom mechanism is fully accounted for by targets (T2); the
   pure-MC arm has a measured, non-zero success profile (T3) — do not
   score m2b against zero.
3. M4 needs a checkpoint hook (or rho_fn) before its next run if
   forensics beyond the summary are ever wanted (T8 deviation).

## Files

- cf_decomposition_by_ckpt.json / puremc_floor.json (T2/T3)
- rho_c_sweep.json / null_bands.json (T4/T5)
- basis_ceiling.json (T6)
- m4_forensics.json (T8)
- scripts: t2_t3_cf_decomposition.py, t4_t5_rho_sweep_nulls.py,
  t6_basis_ceiling.py, t8_m4_forensics.py (re-runnable, read-only
  against project sources); t6_run.log (run log)
