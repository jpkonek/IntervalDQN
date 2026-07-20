# T8 — M4 round-1 forensics: the evidence behind the FULL-MISS re-score

**What this decides:** whether the green b1 bar on M4 round-1 (600 ep,
summary 20260719_230841) was a real pass or a coin flip, and whether the
policy actually respects the protected through-flight. The recorded
outputs settle it: FULL MISS stands.

**DEVIATION (flagged):** the task asked for summary/jsonl+ckpt, but M4
round-1 saved **no checkpoints** — micro_battery_m4.train_arm passes
rho_fn=None and run_training_arm only checkpoints on rho evaluations.
Nothing net-side could be recomputed; all evidence below is extracted
from the run's own summary + jsonl (m4_seed134_20260719_210020.jsonl).
Round-2 has the same gap: if per-checkpoint M4 forensics are wanted
later, the module needs a rho_fn or a checkpoint hook.

## Evidence table

| Claim (audit) | Measured | Verdict |
|---|---|---|
| b1 green is a single Bernoulli draw at chance ~0.60-0.67 | Training first-command target = acceptable member in 491/600 episodes overall (0.818) but **0.645 over the last 200 episodes** (AIR-00 174, AIR-01 317, AIR-02 109 first-commands overall) | Confirmed: the bar as written is one draw at p ≈ 0.65 |
| Through-flight instructed as 3rd command, 12 s in | Greedy eval: command 3 of 60 = L10 to AIR-02 at step 2 = 12 s | Confirmed exactly |
| Through-flight got 10 of 60 commands | Commands per aircraft: AIR-00 33, AIR-01 17, **AIR-02 10** (of 60) | Confirmed exactly |
| Argmax targets the through-flight at 8/20 frozen probes | Probe split: noop 0, acceptable 12, **through 8**, other 0 | Confirmed exactly |

Supporting context from the same records: greedy eval G = -5.99, 60
commands (bar b2 needs <= 2 — missed by 30x), zero NOOP argmaxes at any
probe state, and the b3 through-flight bar (== 0 states) missed at 8.
The first greedy command (climb10 to AIR-00 at step 0) is what b1
scored green — a 1-in-1 sample from the ~0.65 rate above.

**Decision:** b1 as written does not measure the A15 intent (one green
draw at p≈0.65, from a policy that harasses the protected aircraft 10
times in the same episode and argmaxes to it at 40% of probe states).
The FULL-MISS re-score is backed by the run's own records; bar redesign
remains queued for JK ruling. Full command list and per-probe rows in
`m4_forensics.json`.
