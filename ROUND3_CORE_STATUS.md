# ROUND 3 CORE BUILD STATUS (training side, 20 July 2026)

Scope: build-order steps 2-4 from ROUND3_REVIEW.md plus the C2 tie-break
scaffold and the W_rec payment-site scaffold, per the BINDING amendments
in ROUND3_REREVIEW.md. File domain: bluebird_controller_dqn.py only.
Everything below is flag-gated and DEFAULT OFF; the flag-off code paths
are proven bitwise identical to the pre-round-3 code by the new
no-contamination selftests, and the full existing selftest battery
stays green.

## What was built

### 1. B1 — CF branch windows crossing the ENV time limit are EXCLUDED
Flag: `--cf_exclude_limit` (default OFF).

- `run_episode` threads the live branch step index and the episode's
  ENV time limit into `cf_process_state` (`branch_step=step_i`,
  `ep_maxstep=maxstep`) — inert integers when the flag is off.
- Gate: a state is EXCLUDED (never capped) when
  `branch_step + H > ep_maxstep` (H = the run's nstep). The boundary
  case `branch_step + H == ep_maxstep` passes: it matches live-window
  feasibility (the last full live window bootstraps at s_maxstep).
- The check sits BEFORE `cf_select_state`, so excluded states consume
  no CF RNG draw.
- Telemetry: `agent.cf_excluded_limit`, surfaced per episode as
  `cf.excluded_limit` in the JSONL (present whenever cf stats are).

PRE-REGISTERED coverage cost (per ROUND3_REVIEW B1): the exclusion
drops the last H states of every censored episode from CF coverage —
cost = H/T of the episode's states under the run's ACTUAL nstep. At
production nstep=36 on the run-13 micro horizon (T ~ 100 steps at
600 s / 6 s-per-step) that is ~36% of late states; excluded late states
fall back on censored live windows (mechanism i) until A4 — the
least-bad trade. Branch futures crossing would-be A2 recoveries run
uncollapsed this round; branch-side A2 semantics are explicit future
work (pre-registered, per the review).

### 2. A1 — roster-based completion terminal (spawn-0 micros)
Flag: `--completion_terminal` (default OFF).

- Roster = `num_starter_aircraft` from `env.config.scenario_config`.
  The flag REFUSES (loud assert) any config with a nonzero
  `initial_spawn_rate`/`max_spawn_rate` — with spawns, delivered ==
  roster can hold with aircraft still airborne. A1 is micro-only; A2
  owns the full sector.
- The completion check runs strictly AFTER the violation branch in
  `run_episode`'s step loop (the OUT_SECTOR corner: an undelivered
  aircraft leaving obs on exactly the step `detect_violation` fires
  `sector_excursion` must end the episode as a violation, never a win).
  Terminal assert: `delivered == roster and not violated`.
- On completion the episode ENDS as a true terminal: the windower
  collapses all pending windows to realized returns with disc = 0 (no
  bootstrap), kind `completion`; `flush_censored` is NOT called;
  `censored_episodes` does not move; `agent.completed_episodes` counts
  it; stats carry `"completed"`.
- Realized-coverage trackers ARE fed on completion episodes (see
  Deviations/decisions #1).

### 3. B2 — class-aware buffer/windower schema + per-class policy
Schema (always on — the row format changed): every replay row is now
9 fields, `(state, a_idx, R_n, next_state, ns_mask, disc, stratum,
next_a_idx, terminal_kind)`, `terminal_kind` in
`{live, censored, crash, completion, armed_recovery, unarmed_recovery}`
(module constant `TERMINAL_KINDS`).

- The tag ORIGINATES at the windower collapse site (re-review binding
  spec: the R_n sign is NOT a proxy): completed windows push `live`;
  terminal collapse pushes the caller's kind (`crash` default —
  violation; `completion` from A1); `flush_censored` pushes `censored`
  including the held window it stamps (so B3's censored identifier is
  `kind == "censored"`, exact where `next_a_idx == 0` alone is not).
- `armed_recovery`/`unarmed_recovery` are enum values ONLY this round:
  the collapse site, router, sampler and telemetry all accept them
  (selftested end-to-end), but nothing emits them until A2.
- CF rows carry the tag too (`crash` when the branch violated, else
  `live`), stamped at the branch collapse in `cf_process_state`.
- A `gen_episode` stamp rides every boosted-pool append (aligned
  `term_gen` deque, the cf_buf age-cap precedent), advanced once per
  ENV episode (`buffer.gen_episode`, incremented in `run_episode`'s
  train path — pinned to env episodes per the review's bookkeeping
  contract).

Policy flag: `--class_replay` (default OFF; flag-off routing/sampling
is bitwise legacy):
- crash rows keep the current terminal boost (term_buf);
- completion/win rows (and future recovery rows) go to the REGULAR
  pool — wins must never capture the 3x boost built for sparse -50
  anchors (~16x violation dilution otherwise);
- crash rows age-capped at `--crash_age_cap` (default 400 episodes,
  mirroring CF_AGE_CAP), evicted from the left once per episode;
- violation-row batch floor `--crash_floor` (PRE-REGISTERED draft
  constant 0.10): `k_t = max(boost_target, round(rest * 0.10))`,
  clamped to the available crash rows;
- per-class batch-composition telemetry: sampler tallies
  `last_kind_counts`, `train_step` accumulates, and `run_episode`
  logs `"batch_kinds"` per episode in the JSONL.

Interaction note (documented, intended): with A1 ON but
`--class_replay` OFF, completion rows land in the boosted pool via the
legacy disc==0 route — exactly the dilution B2 exists to prevent. Runs
that enable `--completion_terminal` should enable `--class_replay`
(build order lands B2 before any win rows exist for this reason).

### 4. C2 — NOOP-preference tolerance scaffold
Flag: `--noop_tolerance FLOAT` (default None = OFF). The THRESHOLD is a
measured quantity (ROUND3_REREVIEW C2 derivation: post-repair noise
band at the A1+B-only checkpoint as lower bound, minimum GT
resolving-command margin as upper) and ships UNSET this round.

- Lives ONLY inside `select_candidate`, evaluated on a RAW PRE-BONUS
  copy of the Hurwicz scores retained before the count bonus is added
  (the bonus, scale 0.5 at count 0 vs threshold ~0.1, is added
  pre-argmax and would override a post-bonus tolerance ~5x during
  training, then hand back control on decay). The re-issue mask still
  applies to the raw copy (masked candidates are not selectable; NOOP
  is never masked).
- Verdict: prefer NOOP unless the best raw candidate beats raw NOOP by
  MORE than the tolerance. If the bonus-inclusive argmax was CHANGED
  by the bonus, the exploration pick stands and is LOGGED
  (`tol_bonus_override_ep`) per the re-review placement amendment
  ("the bonus-inclusive argmax may override the verdict only as LOGGED
  exploration"); otherwise the pick flips to NOOP (`tol_invoked_ep`).
- STRATUM GUARD: the tolerance never applies on stratum-0 (in-band)
  states, and only where the caller supplies a stratum — i.e. the
  run_episode training path. (The re-review text says "never on
  stratum>=1 (in-band)"; stratum semantics in this codebase are
  0 = <10 nm conflict (A5-corrected), so in-band IS stratum 0 — built
  per the task's explicit resolution.)
- Per-episode logging: `"noop_tolerance": {invoked, bonus_override}`
  in the episode stats whenever the flag is set.
- Probe/eval instruments (`probe_argmax` etc. in the battery files)
  never see the tolerance: it is not mirrored anywhere outside
  `select_candidate`, and eval/probe callers pass no stratum.

### 5. W_rec payment-site scaffold
- `run_episode`'s terms budget gains a `recovery_win` column, paid 0.0
  ALWAYS this round; the reconciliation identity
  `cbp_shaping + fuel + cmd_cost + delivery_bonus + violation_term +
  recovery_win == ep_return` now includes it and is asserted every
  episode.
- A2 will pay W_rec HERE, under this column — never inside
  `priced_env_step`, so CF branch pricing structurally cannot see
  W_rec (pre-registered CF-sees-no-W_rec) and the belief-vs-payment
  decomposition survives.
- The column appears in every episode's stats/JSONL (new key,
  0.0 for now).

## Selftest table

Full battery: `.venv/bin/python bluebird_controller_dqn.py --selftest`
— ALL SECTIONS PASS (existing battery + 5 new sections; see the
reproduce command below for the live output).

New sections:

| Section | What it proves |
|---|---|
| `ROUND 3 terminal_kind schema` | Collapse site stamps the tag (completion + both recovery enums thread end-to-end); invalid kinds fail loudly at windower and buffer; legacy routing bitwise (disc==0 -> boosted) vs class routing (crash-only boost); gen stamps ride appends in lockstep; crash age cap evicts exactly the stale rows |
| `ROUND 3 class replay` | Flag-off sampler is draw-for-draw + RNG-state identical to a literal legacy reimplementation on a crash-starved pool (floor branch inert); flag-off train_step with junk crash params bitwise matches a default agent (loss/weights/RNG); flag-on floor lifts 1 -> 2 crash rows in a 24-batch; composition telemetry tallies and pops |
| `ROUND 3 cf exclude-limit gate` | Crossing branches excluded BEFORE any CF RNG draw or env access (env=None survives); boundary step+H==maxstep passes; flag-off inert with identical RNG consumption; `excluded_limit` telemetry |
| `ROUND 3 NOOP tolerance` | Default None is OFF (no divergence, no counters); raw-margin flip / above-threshold hold / masked-raw semantics; stratum-0 and no-stratum guard; bonus-driven pick preserved and logged as override, tolerance governs after bonus decay; pop clears |
| `ROUND 3 scripted-env suite` | run_episode end-to-end on a scripted spawn-0 env: completion ends the episode with realized kind-tagged collapse (both routings), trackers fed, censored count untouched; OUT_SECTOR-on-last-aircraft step is a VIOLATION (check order proven); nonzero spawn rate refused; `recovery_win` present, 0.0, identity reconciles; B1 threading through run_episode excludes exactly the 2 horizon-crossing states (cf pool 18 -> 12) |

Existing sections touched: `controller n-step windows` was updated for
the 9-field row (unpack arity) and STRENGTHENED with terminal_kind
asserts (live/crash/censored per window); no assertion was weakened.
No other existing selftest needed changes (legacy 8-positional
`push(*it)` call sites are served by the `terminal_kind=None`
fallback, which is selftest-only — production rows are always tagged
at the collapse site).

## Deviations / decisions flagged (nothing silently reinterpreted)

1. Completion episodes FEED the realized-coverage trackers
   (`record_realized_episode`), exactly like violation endings. The
   task spec said "realized returns (windower terminal collapse), no
   bootstrap" without naming the tracker feed; since a completion's
   returns-to-go are fully realized (the episode is provably over,
   terminal value 0), feeding them is the honest reading and matches
   the re-review's coverage-starvation repair direction. Flagged in
   case JK wants completion episodes counted-but-not-fed instead.
2. The disc-based `terminal_kind=None` fallback in `push`/`push_cf`
   exists ONLY for legacy synthetic callers (selftest items). It maps
   disc==0 -> crash, disc>0 -> live — deliberately NOT a class oracle
   (censored/completion would be mislabeled). Every production push
   site passes an explicit tag from the windower/branch collapse.
3. `buffer.gen_episode` advances unconditionally on the training path
   (an integer increment, no RNG, no behavioral effect flag-off) so
   crash rows are correctly aged if `--class_replay` is toggled on in
   a later run of the same process. Eviction itself runs only under
   the flag.
4. C2 tolerance is inactive wherever no stratum is supplied — i.e. on
   eval/probe paths. Activating it at eval would require threading the
   state's stratum through the eval call chain; deferred until the
   threshold is actually measured (it ships unset anyway).
5. The bonus-override rule (bonus-changed argmax stands, logged) is
   the re-review's placement amendment read literally. If JK prefers
   the tolerance to veto even bonus-driven picks, flip the
   `idx == raw_idx` branch — the counters already separate the cases.
6. `--completion_terminal` is wired through `run_training`, but the
   default training env has spawn 0.01 — the guard will refuse it
   there by design. The flag is consumed by the micro battery's
   spawn-0 envs via `run_episode(completion_terminal=True)`.
7. New JSONL/stats keys: `completed` (always), `recovery_win`
   (always, 0.0), `cf.excluded_limit` (whenever cf stats log),
   `noop_tolerance` (only when the flag is set), `batch_kinds` (only
   under `--class_replay`), `completed_episodes` (only under
   `--completion_terminal`). Schema additions only; no existing key
   changed meaning.
8. GT/null-band recompute after A1 (build-order step 3's fingerprint
   rule: A1 changes env-level termination) belongs to the instrument
   files and is NOT done here — the other agent owns it.
9. `evaluate()` does not thread `completion_terminal` (eval episodes on
   the default env run to the time limit as before). Micro-battery eval
   harnesses call `run_episode` directly and can pass the kwarg.
10. `cf.excluded_limit` counts every late-state hook call (every live
   step whose horizon crosses the limit), not only states that would
   have passed the selection rule — it is a coverage measure, read it
   as such. At smoke scale (T=20, nstep=12) it is large by
   construction (~11/episode); at production T it is ~H/T.

## Reproduce

```bash
cd /Users/jk13942/Documents/GitHub/IntervalDQN
# full battery (existing 38 sections + 5 new ROUND 3 sections)
.venv/bin/python bluebird_controller_dqn.py --selftest

# flags (all default OFF; shown here only to document spelling)
.venv/bin/python bluebird_controller_dqn.py --train \
    --cf_exclude_limit --class_replay --crash_age_cap 400 \
    --crash_floor 0.10 ...            # B1 + B2
# A1 (spawn-0 micro envs only; refuses the default env):
#   run_episode(..., completion_terminal=True)
# C2 (threshold is a measured quantity — do NOT set until derived):
#   --noop_tolerance <float>

# end-to-end wiring checks performed (both PASSED):
.venv/bin/python bluebird_controller_dqn.py --smoke --episodes 2 \
    --duration 120                              # flag-off default path
.venv/bin/python bluebird_controller_dqn.py --smoke --episodes 2 \
    --duration 120 --cf_replay --cf_exclude_limit --class_replay \
    --noop_tolerance 0.1   # flags-on: JSONL carries recovery_win,
                           # completed, noop_tolerance counters,
                           # batch_kinds, cf.excluded_limit
```
