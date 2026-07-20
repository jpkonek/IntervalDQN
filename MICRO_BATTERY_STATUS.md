# RUN-13 MICRO BATTERY — BUILD STATUS (19 July 2026)

## ⚠ ROUND-1 RESULTS AUDIT CORRECTIONS (20 July, results-audit workflow)

- **M4 round-1 (600 ep) is re-scored as a FULL MISS.** The green
  first-command bar (b1) is a single Bernoulli trial passed at chance
  ~0.60-0.67 (the 40-ep smoke failed the same bar — a coin flip across
  draws). The same policy instructed the protected through-flight as
  its 3rd command (12 s in), gave it 10 of 60 commands, and argmaxes to
  it at 8/20 frozen probes. The b1 bar as written does not measure the
  A15 intent; redesign queued for JK ruling.
- **M1/M3 rho bars are NO-VERDICT (instrument-limited), not "rising"
  or "missed":** untrained nets score 0.29 (M1) / 0.411 (M3 — above
  the bar!) on these probe sets; the 20 "states" are timestamps of one
  trajectory (effective n≈1); and at c_train=0.5 the rho-vs-rho_mid
  width control is an algebraic identity (vacuous). Clean-rate MEETs
  stand. Instrument redesign queued for JK ruling.
- **M5 deliveries MEET survived refutation** — the one unambiguous
  round-1 positive. The cmd-rate miss (0.476) also stands and is
  cross-cutting (same compulsion drives M2's continuation doom and
  M4's through-flight harassment).
- **M2 mechanism (verified on ckpts):** doom is TWO mechanisms, both
  must be fixed — (i) 9/20 probe states: REAL −50s inside CF windows
  (frozen-greedy continuation re-levels the pair; training windows
  UNCAPPED past episode end while eval GT is capped); (ii) 11/20:
  gamma^36 bootstrap tail re-injects doom. Plus the D1 Branch-A basis
  mismatch (bars judge NOOP-basis, training prices frozen-policy
  basis). Self-sustaining equilibrium, conditional not architectural.
- Round-2 M2 arms restarted 00:5x 20 July with pid-tagged outputs
  (timestamp collision had cross-contaminated both; ~70 eps lost).
  Effective k=3 in both (CF_K_FLOOR clamp) → ~5.6 funded states/ep at
  rate 40 vs demand ~51 (89% skip): read m2a as "quiet coverage 0.055
  w/ halved conflict funding", NOT "floor 0.5". Full reading rules in
  the audit synthesis (task output wu7yg7z23).

Scenario builds for M2, M2-XL, M3, M4, M5 of RUN13_FIX_C_DESIGN.md
Section 3 (post-audit v2; amendments A5, A6, A7, A10, A15). M0/M1/M1-far/
M6 and the CF-branch machinery in bluebird_controller_dqn.py
(--cf_replay) are SEPARATE work items and are not built here.

Code: `micro_battery_common.py` (shared harness) + `micro_battery_m2.py`,
`micro_battery_m2xl.py`, `micro_battery_m3.py`, `micro_battery_m4.py`,
`micro_battery_m5.py`. No existing file was edited; everything reuses
micro_level_allocation / diagnose_controller / bluebird_controller_dqn by
import (scripted episodes priced by bcd.run_episode itself; training IS
bcd.run_episode(train=True); GT via rollout_return / run_scripted).

All scenario finders and scripted gates were RUN on 19 July; training
arms were SMOKE-TESTED (<= 50 episodes — harness proof, NOT bar
attempts; the pre-registered bars are printed anyway, and at smoke scale
they MISS as expected).

## Status table

| Test  | Scenario found | Scripted gates | Smoke (harness) | Bars runnable now | Bars blocked on --cf_replay |
|-------|----------------|----------------|-----------------|-------------------|------------------------------|
| M2    | YES seed 167   | PASS (i, ii) + A6.2 probe gate PASS | 50 ep, clean end-to-end | B1 cmd-rate, B2 NOOP-argmax, B4 NOOP-level | B3 rho > 0.4 |
| M2-XL | YES (spawn 0.02, max 37 concurrent / 186 candidates) | eval-only: full pass DONE | n/a (no training by design) | all (report + trend check) | none |
| M3    | YES seed 134 (M1 scenario) | PASS (i, ii-lateral) | 50 ep, clean; mask held (0 verticals taken) | B1 clean rate | B2 rho (lateral subset) > 0.4 |
| M4    | YES seed 134 (3-starter env) | A15 oracle + strict-subset gate PASS (seed 126 correctly re-seeded) | 40 ep — see below | B1 first-command, B2 cmd/ep, B3 probe argmax (all evaluable without CF; ordering *quality* still expected to need CF) | none formally; expect MISSes until CF grounds candidate ordering |
| M5    | YES seed 109   | PASS (i: NOOP delivers 0; ii: single R10 delivers) | 15 ep — see below | B1 deliveries/ep, B2 occupied-step cmd rate | none |

## Per-test detail, seeds, reproduce commands

### M2 hold-your-fire (seed 167)

- Scenario: 2 starters, spawn 0.0, 600 s. Pair AIR-00/AIR-01 at FL280/290
  (|dFL| = 10, exactly one ±10 FL command from the LoS band), would-be
  lateral CPA 0.95 nm at step 93. NOOP clean for the full 100 steps.
  Scan: seeds [10043, 20042, 100..199] traced; qualifying candidates
  119, 167, 176, 193; 167 chosen (tightest would-be CPA).
- Gates: (M2-i) NOOP via bcd.run_episode: G = 0.00, clean, full duration
  -> PASS. (M2-ii) re-leveling endangerment: climb(lower AIR-00) causes
  LoS for issue steps [5..96]; descend(upper AIR-01) for [6..96]; LoS
  delay 1-82 steps after issue -> PASS.
- A10 CPA-matched GT horizon: h = 80, and each probe state's GT window
  is capped at the episode end (build_probe_set_capped). Finding that
  forced the cap: the uncapped GT rolled PAST the 100-step episode and
  priced a post-episode sector excursion (~step 109), dragging GT(NOOP)
  to -36 on a NOOP-clean scenario — a basis mismatch with the
  time-limit-censored training returns that would have made bar B4
  unfair by construction. With the cap GT(NOOP) = 0.00 at every state.
- A6.2 pre-training gate: 19/20 probe states hot (gt_spread > 5;
  spreads 5.0-48.6), spanning 5/5 trajectory segments -> PASS.
- Smoke (50 ep, agent_seed 42, nstep 36, cbp, gate/0.2 pricing): runs
  end-to-end, 134 s wall. Verdict block (SMOKE, no CF): B1 cmd rate
  0.640 MISS; B2 NOOP-argmax 0.350 MISS; B3 rho 0.295 MISS (expected —
  blocked on CF); B4 NOOP-level frac 0.000 (median |dev| 9.63) MISS.
  The greedy net at 50 ep commands compulsively on a pair that needs
  nothing — exactly the pathology the bars are built to catch, visible
  in-harness.
- Reproduce:
  `.venv/bin/python micro_battery_m2.py --gates --scenario_seed 167`
  `.venv/bin/python micro_battery_m2.py --gates --train --episodes 50 --scenario_seed 167 --rho_every 25`

### M2-XL eval-only scale pass (checkpoint 12d-final)

- Env: CustomInfiniteEnv spawn 0.02/s, 4 starters (starter count is
  CAPPED at 4 by the spawn machinery — measured; density comes from the
  spawn rate), 1800 s, seed 10043. All-NOOP measurement walk, violations
  ignored (density-probe convention); 300 states, max 37 concurrent
  aircraft = 186 candidates; 242 states with >= 101 candidates.
- Checkpoint: checkpoints/bluebird_controller/train_seed42_20260718_151733_final.pt
  (12d-final, ep 4000, c = 0.5). Any checkpoint via --ckpt.
- Results: overall NOOP net-argmax fraction 0.113. By candidate count:
  2-51: 0.208 | 52-101: 0.000 | 102-151: 0.052 | 152+: 0.190 (mean
  Hurwicz margin of best non-NOOP over NOOP +0.7 to +2.9; widths ~9-11).
  Greedy agent-driven episode at scale: 63 commands in 70 steps
  (0.90/step), sector_excursion at 420 s, G = -56.3.
- Pre-registered A5 regression check (P(argmax != NOOP) rising with
  candidate count): Spearman(n_cand, argmax != NOOP) = -0.225 -> MEET
  (no rising trend — the 12d net is uniformly non-NOOP at ALL scales,
  which is the M2 pathology, not the A5 scale-degradation signature).
  Report bar only; composite gate takes the report.
- Note: checkpoint pricing (smooth, delta 0.5) != module default
  (gate, 0.2) — load_agent's loud re-pricing note fires; eval-only, no
  training, intentional.
- Reproduce: `.venv/bin/python micro_battery_m2xl.py`
  (or `--ckpt <path> --spawn 0.02 --seed 10043`)

### M3 lateral-only crossing (seed 134)

- Scenario: the M1 level-allocation crossing (2 starters, same-FL 240/240
  pair, NOOP -> LoS at step 67). Finder = micro_level_allocation.
  find_conflict_seed, unchanged.
- Gates: (M3-i) NOOP -> loss_of_separation, G = -50.0 -> PASS.
  (M3-ii) lateral-only resolvability: single L10 on AIR-01 clean for
  issue steps [10..60] (51/68); all other single lateral commands fail
  -> PASS (solvable inside the masked action set).
- Mask: candidate-level in the harness (HarnessAgent.masked_instr =
  climb10, descend10) applied to greedy selection, epsilon sampling AND
  the Double-DQN bootstrap argmax. bcd is untouched. Mask verified: 0
  vertical clearances taken across all smoke episodes.
- A10 GT horizon: h = 69 (NOOP LoS step 67; micro default 15 would miss
  the conflict outcome from early states). 18/20 probe states hot.
- Smoke (50 ep): runs end-to-end, 132 s wall. B1 clean(last100) 0.62
  MISS (rising: 0.20 -> 0.62); B2 rho(lateral subset) 0.000 MISS
  (expected — blocked on CF). Full-set rho +0.435 reported (not gated;
  the untrained vertical heads happen to rank below NOOP, which the
  lateral-subset metric deliberately excludes).
- Reproduce:
  `.venv/bin/python micro_battery_m3.py --gates --scenario_seed 134`
  `.venv/bin/python micro_battery_m3.py --gates --train --episodes 50 --scenario_seed 134 --rho_every 25`

### M4 give-way (seed 134, 3-starter env)

- Scenario: 3 starters, spawn 0.0, 600 s. Crossing pair AIR-00/AIR-01
  (NOOP -> LoS at step 67); through-flight AIR-02. Scan shortlist (from
  the 3-starter sweep of seeds [10043, 20042, 100..169]): 126, 134, 138,
  116, 120, 144, 152, 168, 108 all produce a NOOP LoS pair + third
  aircraft.
- A15 acceptable-set oracle: grid over (3 aircraft x 5 instructions x
  issue-step stride 4), each cell ONE scripted single-clearance episode
  through bcd.run_episode (cbp), UNDISCOUNTED ep_return (A15's ~-50 gap
  arithmetic is in raw return units; a discounted basis would shrink
  late LoS below the gap threshold). Unissued cells (target not in obs)
  excluded. ~255 cells, ~10 min.
- Seed 126 FAILED the strict-subset gate (best single command for EVERY
  aircraft still ends in LoS; gaps 0.00) and was re-seeded per A15 —
  the gate doing its job. Seed 134 PASSES: best AIR-00 G = -0.10
  (climb10 @ step 4, clean), best AIR-01 G = -0.10 (L10 @ step 60,
  clean), best AIR-02 (through) G = -50.10 -> through-gap 50.0 > 5;
  acceptable set = {AIR-00, AIR-01} (strict subset, both crossing
  members).
- Bars: B1 first greedy command targets an acceptable member
  (through-flight instructed = HARD MISS, escape hatch invalid; no
  command at all = ordinary MISS); B2 commands/ep <= 2 (greedy eval);
  B3 zero probe states with a through-flight-targeting greedy argmax
  (per-state basis per A15.3 — post-warmup epsilon is exactly 0).
- Smoke (40 ep, 111 s wall): runs end-to-end; verdict block prints. All
  bars MISS at smoke scale as expected — including B1 as a HARD MISS
  (the barely-trained greedy net's first command targets the
  through-flight aircraft): the exact pathology the bar exists to
  catch, visible in-harness. (Filled by coordinator from
  m4_summary_seed134_20260719_191938.json.)
- Reproduce:
  `.venv/bin/python micro_battery_m4.py --gates --scenario_seed 134`
  `.venv/bin/python micro_battery_m4.py --gates --train --episodes 40 --scenario_seed 134`

### M5 delivery (seed 109)

- Scenario: 1 starter, spawn 0.0, 1800 s (300 steps; single-aircraft
  transits need 700-1500 s — B4 finding). Seed 109: NOOP delivers 0
  (sector_excursion at 684 s); single clearances DO deliver (3 working
  cells; earliest R10 @ step 10). Delivery is action-contingent — the
  A7 point; the degenerate never-commanding net scores 0, not 1.
- A7 DEVIATION (documented): CustomInfiniteEnv exposes NO off-route
  spawn knob (verified — spawn config only has rates/counts/threshold),
  so the finder is a seed search. No scanned seed has a working
  clearance at step <= 5; the finder therefore takes the BEST-EARLIEST
  seed (109: step 10 of 300, i.e. 60 s in — required correction is
  near-spawn, not a late boundary-pressure retrim). Scan detail: of
  seeds [20042, 100..139]: 12 NOOP-deliver-1 (skipped, not
  action-contingent), 17 with NO working single clearance, the rest
  working at steps 10-60.
- Gates: (M5-i) NOOP deliveries == 0 (pre-registered next to the 0.8
  bar) -> PASS; (M5-ii) single clearance delivers cleanly -> PASS.
- Command-rate denominator: OCCUPIED steps (>= 1 aircraft in obs),
  counted in HarnessAgent.generate_action — bcd's step_hook carries no
  obs count, so the doc's literal "via step_hook" is implemented at the
  generate_action call run_episode makes once per step (documented in
  micro_battery_common.py).
- Bars: B1 deliveries/ep >= 0.8 (trailing 100); B2 cmd rate over
  occupied steps < 0.2.
- Smoke (15 ep, 42 s wall): runs end-to-end; verdict block prints; both
  bars MISS at smoke scale as expected (15 episodes is inside warmup).
  (Filled by coordinator from m5_summary_seed109_20260719_191006.json.)
- Reproduce:
  `.venv/bin/python micro_battery_m5.py --gates --scenario_seed 109`
  `.venv/bin/python micro_battery_m5.py --gates --train --episodes 15 --scenario_seed 109`

## Deviations and findings log

1. GT horizon cap (M2/M3, extends A10): probe-state GT windows are
   capped at the episode's remaining steps (build_probe_set_capped in
   micro_battery_common.py). Uncapped windows price post-episode events
   the training returns are censored away from (measured on M2 seed 167:
   GT(NOOP) -36 from a phantom step-109 excursion). The A10 CPA rule is
   still satisfied — each scenario's decisive event is inside the
   episode (asserted by the gate sweeps).
2. M4 oracle return basis: UNDISCOUNTED ep_return (see M4 section);
   pre-registered here before any M4 training run at scale.
3. M5 occupied-steps counting at generate_action, not step_hook
   (step_hook has no obs field; same once-per-step call site).
4. M5 off-route spawn is approximated by best-earliest-clearance seed
   search (no env knob exists). If a true off-route spawn knob is added
   to bluebird_gymnasium later, re-run the finder.
5. M2-XL starters cap at 4 regardless of num_starter_aircraft; 20+
   concurrent comes from spawn 0.02/s (measured max 37 by step ~140).
6. Battery defaults: nstep = 36 (D5 ruling; bcd's code default is 12),
   pricing gate/0.2 (micro_level_allocation's default). 12d-final was
   trained smooth/0.5 — for run-13 arms that must match 12d pricing,
   pass `--vertical_ramp smooth --delta_conflict 0.5`.
7. --cf_replay is plumbed through every training arm but ABORTS until
   bcd.run_episode grows the parameter (stage-1 CF build, separate work
   item). Blocked bars: M2-B3, M3-B2 (and M4 ordering quality). The
   smoke MISSes on those bars are the expected 12d-era nulls.

## Smoke-scale caveat

50/40/15-episode smokes exist to prove the harnesses run end-to-end
(episode loop, mask, counters, probe evaluation, verdict blocks, JSONL +
summary writing). They are NOT bar attempts: warmup alone is 1500 steps
(~15-20 episodes) and the pre-registered bars were written for full
micro runs (hundreds of episodes) WITH --cf_replay where noted.
