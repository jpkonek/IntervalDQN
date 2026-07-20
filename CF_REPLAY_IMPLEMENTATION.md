# CF-REPLAY IMPLEMENTATION — RUN 13 FIX C, M0 + BRANCH MACHINERY

Status: build-order step 1 (RUN13_FIX_C_DESIGN.md Section 2) complete,
19 July 2026. Everything is flag-gated behind `--cf_replay`
(default OFF) in [bluebird_controller_dqn.py](bluebird_controller_dqn.py).
Plain-English summary first, then the exact mechanics, deviations,
selftest results, and open risks.

## What was built (plain English)

When the flag is on, the trainer occasionally pauses at a live decision
point, clones the whole simulator, and asks "what if I had issued THAT
clearance instead?" It plays each what-if forward for the same number
of steps the live learning window uses, letting the current network fly
the continuation, and prices every step with the exact same reward code
the live episode uses. The resulting what-if returns become extra
training targets for actions the agent never actually took — the
coverage hole the run-13 evidence says is binding. With the flag off,
the file behaves bit-for-bit as before (proved by selftest, not by
inspection).

## Mechanics (spec Section 1 → code)

- **Flag + wiring**: `--cf_replay` plus `--cf_k`, `--cf_budget_rate`,
  `--cf_boost`, `--cf_max_frac`, `--cf_age_cap`. Config lands on
  `ControllerAgent(cf_*)`; provenance is persisted in checkpoints
  (`cf_replay`, `cf_counts`, `cf_gen_ep`, plus the args in the extra
  block). The cf pool itself is NOT persisted (env unpicklable; items
  are cheap relative to their staleness cap).
- **Shared pricing path (A3)**: the reward layer of `run_episode` was
  factored VERBATIM into `priced_env_step` (same operations, same
  order); live steps and CF branch steps both call it, so "priced via
  the same CBP code path" holds by construction, and M0 checks it
  bitwise rather than by trust.
- **Hook placement (A13.1)**: `cf_process_state` runs inside
  `run_episode` AT s, after action selection but BEFORE the live
  `last_issued` update and before the live env step, so the CF
  candidate mask and the branch continuation see exactly the live
  masked action set at s. `last_issued` is deep-copied at branch time
  into every branch (`cf_branch_rollout`), which maintains its own
  copy thereafter.
- **State selection (A5-corrected)**: `cf_select_state` — stratum 0
  (`snapshot_stratum` 0 = min proximate-pair separation < 10 nm, the
  CONFLICT stratum, semantics NOT inverted) always; stratum 1 at
  p = 0.3; state-level random floor p = 0.05 across all strata (band +
  floor folded into one draw: stratum-1 effective p = 1 − 0.7·0.95 =
  0.335). Stratum 0 consumes no RNG draw.
- **Candidate selection**: `cf_choose_candidates` — mandatory NOOP
  probe first (A1); then interval-overlap-with-argmax ambiguity, then
  lowest CF-side count cells (a SEPARATE `cf_counts` table, A12.3),
  then a candidate-level random floor p = 0.1 that swaps in a uniform
  leftover. Excluded (A13.2): re-issue-masked candidates,
  BEFORE_ENTRY-target candidates, and the live taken action (it gets a
  live windower target). K = max(3, `--cf_k`) including NOOP.
- **BEFORE_ENTRY analytic targets (A13.2)**: for every BE candidate at
  a selected state, the item `G_cf(NOOP) − CMD_COST` is banked off the
  shared NOOP probe (env ignores the clearance; only the step-0 fee
  differs on this deterministic env) at zero sim cost.
- **Branch rollout (D1 Branch A)**: `cf_branch_rollout` deep-copies the
  env at s, applies the candidate, then runs the training net frozen at
  branch time, greedy (epsilon 0), with the deep-copied `last_issued`,
  for H_cf = nstep steps total (A4 matched depth — G_cf spans exactly
  the live window's n rewards, disc = gamma^n; NO gamma^12-vs-gamma^36
  tail mismatch). A branch violation collapses exactly like the live
  windower (disc 0, empty next state, NOOP-only mask). The
  DeliveryClock rides the deepcopy: it was already observed AT s by the
  live loop, so branch step 0 does not observe again and every later
  step observes exactly once before stepping (12d silent-killer
  surfaces 1 and 2).
- **Bootstrap (deliverable 2)**: stored `next_a_idx` = `a_cont(s_H)`,
  the continuation policy's ACTUAL action at s_H (one extra forward, no
  env step). `train_step` forces the target-side action of cf rows to
  this stored index — `y = G_cf + gamma^H * Q_target(s_H, a_cont)` —
  bypassing both the Double-DQN argmax and the taken_noop restricted
  argmax. Verified numerically against pre-step nets by selftest.
- **RNG isolation (A12.3)**: `agent.cf_rng` (a dedicated
  `random.Random`) feeds ALL CF-side draws; `generate_action` grew an
  `rng` parameter defaulting to the module-level `random` (bitwise
  identical for every existing caller). CF code never writes the live
  `explore_counts`; the CF-side table `cf_counts` is separate.
- **Budget (A2, D2)**: `CFBudget` banks `--cf_budget_rate`
  step-equivalents per live training step ACROSS episodes (A2.3),
  charges branch env steps at 1.0 (including the CBP clones' steps —
  at cbp lag 1 a priced branch step costs 4 step-equivalents + 2
  deepcopies, mirroring the live CBP cost) and every deepcopy at 1.5
  step-equivalents (A2.2). A state is processed only if the whole
  estimated K-candidate cost is affordable, so states are never
  half-funded. Spent/banked totals are logged per episode.
- **CF pool + sampler (A14)**: a SEPARATE additive deque on
  `ControllerReplayBuffer` (never carved out of the term/reg pools —
  avoids the documented tiny-capacity footgun and never evicts live
  experience). Items carry a generation-episode stamp and an in-place
  replay counter. Sampling is share-proportional with boost
  `min(cf_max_frac, cf_boost * share)` (terminal-boost pattern); the
  1:3 cap (`cf_max_frac` 0.25) is a never-expected-to-bind upper
  bound. Realized cf batch fraction, sampled-age median and max
  per-item replay count are logged per episode. Age cap: items older
  than `--cf_age_cap` (400) episodes are evicted at episode end.
- **Tracker quarantine (A12.1, deliverable 3)**: cf rows skip
  `bootstrap_hits`, `_tripwire_devs` and the `bsupport_*` logs;
  parallel `cf_bootstrap_hits` / `_cf_tripwire_devs` series are fed
  instead and surfaced in the per-episode `cf` JSONL block
  (`bootstrap_cov`, `accuracy_proxy`). The pre-registered tripwire
  revival trigger and its ep500 baseline therefore keep reading the
  live-only series (D6).
- **Pricing instrumentation (A3)**: per-CF-rollout event flag
  (delivery or violation inside the window; `event_frac` logged, 20%
  floor is a run-time flag for the operator) and the per-state spread
  of G_cf across the K candidates, bootstrap excluded
  (`gcf_spread_med`).

## Deviations from the spec (flagged)

1. **Recompute-on-expiry NOT implemented (drop-on-expiry instead).**
   Design Section 1 pre-registers "recompute path via retained in-RAM
   env clones" / "recompute-on-expiry for up to 2000 retained clones,
   else drop". This build stamps generations and enforces the A_max =
   400 eviction, but expired items are DROPPED, not recomputed, and no
   clone retention store exists yet. Reason: retention/recompute is a
   sizeable subsystem (clone lifecycle, memory watermarking, recompute
   scheduling against the same budget) orthogonal to M0 correctness;
   the deliverable list for this build step required stamps + age cap.
   Consequence: under heavy staleness the pool thins instead of
   refreshing. Must be added (or formally re-registered as drop-only)
   before the full run 13 launch.
2. **K is a flag, not solved from the coverage target.** The design
   derives K per state from the >= 20% stratum-0 coverage target and
   the realized episode mix. This build exposes `--cf_k` (floor 3) and
   `--cf_budget_rate` and logs realized coverage inputs
   (`states_selected`, `items`, per-stratum `cf_counts`, bank/spend);
   the solve — and the A2.4 micro-vs-full abort rule — belong to the
   milestone that certifies the mechanism (M1+), not to M0. The 10x
   abort comparison is NOT auto-enforced yet; the JSONL carries the
   numbers it needs.
3. **Branch length arithmetic**: A2's cost model says one state costs
   "K*(1+H_cf) priced steps"; matched depth (A4/M0) requires G_cf to
   span exactly nstep rewards. This build runs H_cf = nstep branch
   steps TOTAL (the apply step is step 0 of the window), because the
   bitwise M0 identity `G_cf == windower n-step return` is only
   satisfiable that way. Budget charging reflects actual steps stepped
   (plus CBP clone costs), so no budget is under-counted; the "+1" in
   the audit's arithmetic is interpreted as the apply step already
   inside the window.
4. **BE analytic targets inherit the NOOP branch's mask/next-action.**
   A real BE-candidate branch would differ from the NOOP branch in one
   invisible way: `last_issued[BE aircraft]` would be set, which could
   mask that candidate at s_H. The audit's own prescription (store the
   analytic target off the shared NOOP rollout) accepts this; noted
   here for completeness.
5. **Censored branches** (env time-limit inside the branch window —
   possible only if the sim refuses to step past the scenario end,
   which the InfiniteEnv does not do in practice; CBP clones already
   step past the end routinely): no special handling exists. If a
   branch ever returns fewer than H rewards without a violation, the
   item bootstraps at the last reached state with disc = gamma^m and
   a_cont there — the same semantics the live windower gives censored
   tails, except with a real continuation action instead of the NOOP
   fallback.

## Selftests (all runnable via `--selftest`; battery extended)

| Test | Deliverable | Result |
| --- | --- | --- |
| `_selftest_cf_flag_off` | 4(a) flag-OFF bitwise no-contamination (train_step loss/weights/RNG/logs + sampler draw-for-draw vs a legacy reimplementation) | PASS |
| `_selftest_cf_budget_selection` | budget banking, 1.5x clone pricing, A5 stratum semantics (s0 always, s1 0.333~0.335, s2 0.050), NOOP-first, exclusions, overlap/count ranking, floor | PASS |
| `_selftest_cf_pool` | A14 sampler (boosted share draw, aligned cf flags, ages, per-item replay counts), age eviction, severed-pool parity | PASS |
| `_selftest_cf_train_quarantine` | 2 + 3: forced a_cont bootstrap verified against pre-step nets (bitwise 0 deviation; argmax alternative differs), tracker quarantine (12 live / 4 cf rows, bsupport live-only) | PASS |
| `_selftest_cf_env_suite` — A13.1 part | 4(c) mask-threading assert (CF step-0 masked set == live masked set), BE analytic targets, live env/RNG/table side-effect freedom | PASS (1 masked candidate excluded, 3 probes incl. NOOP, 5 analytic BE targets, env fingerprint + python/numpy/torch RNG states + count tables untouched) |
| `_selftest_cf_env_suite` — A12.3 part | 4(b) flag-ON isolation: CF generation active, optimizer channel severed, live trajectory/weights bitwise identical | PASS (76 live steps bitwise identical incl. weights/trackers while cf generated 518 items from 16 states) |
| `_selftest_cf_env_suite` — M0 part | 4(d) M0 CF-identity per A4: teacher-forced recorded-action replay, bitwise G_cf == windower return, synthesized ns/mask/next_a == stored 8-tuple, clock once-per-step, separate terminal collapsed-semantics assert | PASS (84 bootstrap windows bitwise + 6 terminal windows collapsed; T=90, n=6, violation at 540 s, greedy episode with 90 recorded commands) |
| Full battery (`--selftest`, all 26 sections incl. legacy env tests) | regression (incl. the priced_env_step refactor) + all of the above in one run | PASS — 38 OK asserts, 0 FAIL, "SELFTEST PASSED" (19 July 2026) |
| `--smoke` (flag OFF) | end-to-end training regression | PASS; episodes 1-2 bit-match the flag-ON smoke (identical G, widths) until cf rows start training |
| `--smoke --cf_replay --cf_budget_rate 20` | end-to-end flag-ON: generation, banking, sampling, logging | PASS; JSONL `cf` block live: 1 state, 3 rollouts, 8 analytic BE items, pool 11, batch_frac 0.25, median_sampled_age 2, spent 144 steps + 75 clones, gcf_spread 0.22, cf bootstrap_cov + cf accuracy_proxy populated |

Two implementation findings surfaced by M0 (exactly the kind of silent
killer it exists to catch):

- **env.step mutates its action dict.** The BlueBird wrapper's action
  formatter injects entries for aircraft that spawn during the step
  into the CALLER'S dict. Live code never noticed (each dict is
  stepped once), but M0's teacher-forced scripts share dicts across
  overlapping windows — replaying a mutated dict at an earlier state
  KeyErrors on the injected callsign. `cf_branch_rollout` now copies
  every scripted dict before stepping. Production CF branches build
  fresh dicts per step and were never exposed.
- **Clock counter convention.** The M0 recorder captures the delivery
  clock POST-observe (run_episode observes before it selects), so the
  branch's final counter equals the live counter at s_{i+n} MINUS the
  not-yet-performed observe there; the selftest encodes exactly that
  relation, which is the once-per-step invariant A4 asks for.

## Open risks

- **Analytic BE items can dominate the cf pool.** At sector-entry-heavy
  states most candidates target BEFORE_ENTRY aircraft, so a selected
  state can bank far more analytic items than rollout items (isolation
  test: 518 items from 16 states, mostly analytic). They are exact and
  cheap, but they dilute rollout items inside the share-proportional cf
  draw. If M1-era rho needs the rollout items to surface more often,
  either cap analytic items per state or split them into a third pool —
  a pre-registration decision, not taken silently here.
- **Tiny-pool oversampling is visible, not prevented** (per A14: no
  max-replay bound by design). The smoke shows max_replay_count 15 on
  an 11-item pool — exactly the situation the per-item counts and
  median-age tripwire exist to expose.
- The recompute-on-expiry gap (deviation 1) — pre-launch item.
- The A2.4 coverage-abort rule is logged-not-enforced (deviation 2).
- `cf_episode_stats` counters are cumulative; per-episode rates need
  differencing in the JSONL consumer.
- M0's next_a synthesis matches the stored live action only because
  the M0 episode runs with epsilon 0, no count bonus and no weight
  updates (batch gate); the selftest constructs exactly that regime.
  Live training trajectories legitimately diverge from the frozen
  continuation (count bonus, epsilon) — by design (A4).
- The frozen-continuation-vs-live-suffix persistence asymmetry
  (A13.3, the signed taken-vs-untaken bias check) is battery scope
  (M-milestones), not built here.
- RA-CF gate (build-order step 2) not run yet — this build only
  delivers the machinery + M0.

## Reproduce

```bash
cd /Users/jk13942/Documents/GitHub/IntervalDQN
# full battery incl. the four synthetic cf tests + the cf env suite
.venv/bin/python bluebird_controller_dqn.py --selftest
# flag-on smoke (tiny end-to-end run with cf generation live)
.venv/bin/python bluebird_controller_dqn.py --smoke --cf_replay --cf_budget_rate 20
# flag-off smoke (regression twin of the above)
.venv/bin/python bluebird_controller_dqn.py --smoke
# run-13 style launch (D5 ruling nstep 36) — NOT launched by this build
.venv/bin/python bluebird_controller_dqn.py --train --cf_replay --nstep 36 --episodes 4000
```
