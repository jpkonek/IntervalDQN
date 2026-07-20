# T6 — Is the rho > 0.4 bar reachable in principle under the training basis?

**What this decides:** whether M2's blocked B3 bar (rho > 0.4 against
the NOOP-basis GT table) can be met AT ALL by a learner whose CF-replay
targets are priced under the D1 Branch-A frozen-policy basis. Answer:
**no — the two bases only agree at rho ~ 0.18, so the bar as written is
unreachable in principle, independent of learning quality.**

Setup: the 20 frozen M2 probe states (seed 167), all 11 candidates
each, two CBP-priced GT rollouts per candidate with H = min(80, steps
to episode end), pricing smooth/0.5, via diagnose_controller.
rollout_return (the C2 machinery):
- GT_noop — candidate then all-NOOP continuation (exactly the
  build_probe_set_capped table the bars judge against);
- GT_frozen — candidate then the round-1 ep800 M2 agent, greedy,
  re-issue-mask-aware (what CF-replay training actually prices).

## Results (per-state detail in basis_ceiling.json)

| quantity | mean | sd | range over 20 states |
|---|---|---|---|
| Spearman(GT_noop, GT_frozen) | **+0.180** | 0.389 | -0.65 to +1.00 |
| ep800 net rho vs GT_noop (bar basis) | +0.113 | 0.306 | -0.42 to +0.72 |
| ep800 net rho vs GT_frozen (training basis) | +0.014 | 0.310 | -0.56 to +0.56 |

Cross-check: net-vs-noop mean +0.113 reproduces the round-1 run's own
recorded rho at ep800 (0.113) exactly — the probe rebuild is faithful.

Why the bases diverge: in 39 of 220 frozen-basis branches the ep800
continuation policy commands the level-separated pair back into a real
loss of separation (including 6 of 20 NOOP branches), so GT_frozen
ranks candidates by what the damaged continuation does afterwards, not
by the candidate's own merit. The basis agreement is best exactly where
it matters least (the last few states, h <= 16, where both bases just
see the imminent geometry).

**Decision:** a PERFECT learner of the training basis would score about
0.18 (the basis-to-basis correlation) on the NOOP-basis bar — less than
half the pre-registered 0.4. On top of that, both readings sit inside
the T5 null band (+-0.6), so the bar could not certify a pass even if
reached. The B3 bar needs either (a) the instrument re-based to the
frozen-policy basis (as RUN13_FIX_C_DESIGN Section 1 already argues
under "rho_untaken instrument basis MATCHES training basis"), or (b)
training re-based to NOOP-continuation targets — judging one basis
against the other is a category error, now quantified. Note the ep800
net tracks NEITHER basis (+0.014 against its own training basis):
basis mismatch is the ceiling, not the only problem.

Caveat: GT_frozen is conditional on the ep800 policy; a healthier
continuation policy would raise the basis agreement (that is the point
— the bases converge only when the policy stops wrecking the pair).
