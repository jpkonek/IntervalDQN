# T2 — What the CF targets actually tell the M2 net (round-1 checkpoints)

**What this decides:** whether the "doom" in the M2 net's NOOP values is
manufactured by the training targets themselves — and it is. Replaying
the 20 frozen NOOP probe states through the production CF path
(bcd.cf_branch_rollout, the checkpoint agent as frozen continuation,
H=36, CBP pricing smooth/0.5) shows the NOOP target is strongly negative
at almost every state, through two separate mechanisms, both previously
hypothesized by the audit and both now directly measured:

1. **Real in-window violations caused by the continuation policy.** The
   scenario is NOOP-clean for all 100 steps, yet the frozen net keeps
   commanding during the branch and re-levels the FL280/FL290 pair into
   a real loss of separation. At ep800 this happens from probe states as
   early as step 35 (LoS at absolute step 69, well inside the episode).
   The NOOP candidate is then priced -18 to -47.
2. **Uncapped branch windows running past the 100-step episode end.**
   Branches from late states simulate up to absolute step ~130 of a
   100-step episode. Some "violations" occur at absolute steps 109-130,
   i.e. after the episode the training returns are censored at — phantom
   events the live objective can never realize. Non-violating late
   branches also pick up large bootstrap tails (disc = gamma^36 = 0.334
   times a Q midpoint of -10 to -48) — the tail-re-injection mechanism.

## Per-checkpoint summary (20 probe states each)

| ckpt | branches violated | ...in-window (<step 100) | ...post-episode (phantom) | branches past episode end | mean G_cf | mean full target y |
|------|------|------|------|------|--------|--------|
| ep200 | 5/20 | 4 | 1 (step 109) | 4/20 | -11.5 | -15.2 |
| ep400 | 6/20 | 3 | 3 (steps 110-123) | 5/20 | -10.0 | -13.7 |
| ep600 | 2/20 | 2 | 0 | 6/20 | -5.1 | -12.3 |
| ep800 | 9/20 | 6 | 3 (steps 118-130) | 6/20 | -12.3 | -14.8 |

Early states (steps 6-30) price NOOP at G_cf -0.5 to -2.1 (command fees
and shaping only) but still get full targets y of -3 to -8 via the
bootstrap tail. Mid/late states swing between -5 and -47 depending on
whether the frozen continuation happens to wreck the pair that
generation. The target a NOOP-clean state "should" carry is ~0.

**Decision:** the M2 doom equilibrium is conditional, not architectural
— the targets, not the net, carry the -20s. Any fix must close BOTH
holes (continuation-induced violations and the uncapped window); fixing
one leaves -10-to--20 targets from the other. Full rows in
`cf_decomposition_by_ckpt.json` (per state: G_cf, disc, Q-midpoint tail
at the stored next action, violation step, past-end flag).

Method notes: probe states rebuilt with the exact
collect_probe_states walk build_probe_set_capped uses (seed 167,
deterministic); tail term uses agent.candidate_q — legitimate as the
target-net copy since load_agent syncs target_net = q_net.

# T3 — Pre-registered success profile for the pure-MC arm (m2b)

**What this decides:** what "m2b worked" must look like. With disc = 0
the target is G_cf alone, so the best the running pure-MC arm can
converge to is the measured G_cf profile above — **not zero**.

**PRE-REGISTERED READING (before m2b finishes):** m2b SUCCESS =
Q(NOOP) at the probe states approaching the measured G_cf profile:
roughly **-1 to -2 at early states, -18 to -46 at the mid/late states
whose branches violate, with 10-45% of branches violating in-window
(25-45% at the checkpoints bracketing convergence)**. A Q(NOOP) profile
near zero everywhere would mean m2b is NOT learning its own targets;
a uniformly -20s profile would mean tail contamination persists some
other way. The audit's expected profile (~-1..-2 early, -18..-46
mid/late, 30-45% violating) is confirmed by measurement here at ep400
(30%) and ep800 (45%); ep600's dip to 10% shows the floor itself is
policy-generation-dependent, so m2b should be judged against the BAND,
not one checkpoint's profile. Numbers in `puremc_floor.json`.
