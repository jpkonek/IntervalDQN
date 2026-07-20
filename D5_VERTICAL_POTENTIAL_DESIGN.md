# D5 — Smooth Vertical Conflict Potential (design for adversarial review)

Date: 2026-07-14. Status: DESIGN ONLY — no code changed, no runs launched.
Commissioned by JK as the reviewed fix for the vertical binary-gate
faithfulness bug flagged in 12D_CREDIT_DESIGN.md §1 / §6-D5.

Inputs read: bluebird_controller_dqn.py (pair_conflict_f L336-343,
shaping_terms L346-376, CBP machinery L466-499 + run_episode L1226-1441,
_selftest_telescoping L1837-1934, _selftest_cbp), CONTROLLER_ARCHITECTURE.md
§4, 12D_CREDIT_DESIGN.md, PROBE_BATTERY.md (B1-v2/B2), CREDIT_DIAGNOSTICS.md
(D2) + credit_diag_20260714_131032.json (the 15 frozen conflict states),
WIDTH_MECHANISM_PROBES.md (E1 attempt-1 runaway lesson),
micro_level_allocation.py, bluebird_gymnasium source
(actions/simple/climb_descent.py, state_repr/relative.py),
bluebird_dt/core/aircraft.py (fl, selected_fl, percentile-sampled rocd_cl).

Design criterion (JK): LL-faithfulness. In the LunarLander reference the
native shaping is smooth in EVERY controlled variable, so every action's
effect registers in the immediate reward. FL is a controlled variable of
this system (climb10/descend10 are 2 of 5 instructions) and it is the only
one whose potential response is an indicator function.

---

## 0. The bug, stated precisely (sharper than the flag note)

Current pricing (pair_conflict_f):

```
f(pair) = 1[|dFL| < 20] * max(0, (15 - d_nm)/15)^2
```

Lateral proximity is a smooth quadratic ramp; vertical proximity is a
binary gate at CONFLICT_FL = 20. Aircraft climb ~2.5-3 FL per 6 s sweep
(rocd_cl is percentile-sampled per aircraft — treat 2.75 FL/step as the
planning number, verify per scenario at implementation), so a +10 FL
clearance takes ~4 sweeps to fly and a co-level pair needs TWO climbs
(~7-8 sweeps) before |dFL| reaches 20.

The flag note said "zero shaping response for ~7 steps, then a cliff."
Under CBP (the training signal) it is worse than that, and the cliff never
arrives either:

- CBP at decision step t pays `gamma*(Phi(s_a) - Phi(s_noop))`, both
  branches measured 1+lag = 2 steps after s (lag 1, command latency).
  A climb issued at co-level: action branch has |dFL| ~ 2.75 at the
  measured state, NOOP branch has 0. BOTH are < 20, so the gate is on in
  both branches, f is identical (lateral geometry is common-mode), and the
  CBP differential is EXACTLY 0.
- At every later decision step the already-issued climb lives in the
  deepcopied NOOP branch too (global NOOP does not cancel an active
  command — the CBP_LAG note), so the ongoing climb is common-mode and
  pays 0 forever.
- The 20-FL cliff can produce a nonzero CBP term only when the measured
  branches straddle the gate, i.e. |dFL| at the NOOP-branch measured state
  lies in ~(17.25, 20) — a measure-zero sliver reachable only mid-maneuver
  of a second command or when the other aircraft is itself climbing.

Net: under the training signal, vertical resolution (and vertical
ENDANGERMENT — a descent onto co-altitude traffic, the mirror image) pays
identically zero at every decision step. The immediate reward for any
vertical clearance is exactly the fee, -0.1: the exact P2 anti-ordering
signature (D2: measured gap == fee at H<=12, noise p95 ~1e-16), with no
possibility of a vertical exception. D2 confirms the stakes: 6 of the 15
frozen conflict states have a VERTICAL ground-truth-best action (j=4/5),
and the environment's own docstring calls altitude "the missing primary
tool" (12b's lateral-only ceiling at ~600 s TTV).

---

## 1. Proposed functional form

### 1.1 Shape: product of smooth ramps

```
f(pair) = f_lat(d_nm) * g_vert(pair)

f_lat(d)   = max(0, (15 - d)/15)^2                       [unchanged]

g_vert     = (1 - w) * g_cur + w * g_cmd,   w = 0.5
g_cur      = max(0, (V_cur - |fl_a  - fl_b |)/V_cur)^2,  V_cur = 20
g_cmd      = max(0, (V_cmd - |sel_a - sel_b|)/V_cmd)^2,  V_cmd = 10
```

where `fl_*` is current flight level (already in AcSnap) and `sel_*` is the
commanded/selected FL (`simulator.aircraft[cs].selected_fl`, the exact
field the climb/descend actions mutate — verified in
bluebird_gymnasium/actions/simple/climb_descent.py: `selected_fl ± value`).
All constants are ABSOLUTE (E1 attempt-1 lesson: no reference to any live
or batch statistic anywhere in the term — see §4.7).

### 1.2 Why a product (not additive / min / soft-min)

LoS requires BOTH d < 5 nm AND |dFL| < 10. Hazard is a conjunction, so the
potential must vanish when EITHER separation is large:

- **Additive** f_lat + g_vert: prices every co-level pair in the sector
  regardless of lateral distance — at ~25 aircraft that is dozens of
  phantom-hazard pairs, a massive new ambient term (the exact pathology
  CBP was built to fight). Rejected outright.
- **min(f_lat, g_vert)**: correct zero-set but non-smooth at the
  crossover, and its gradient is blind to one dimension at a time — a
  climb registers zero whenever f_lat < g_vert, recreating a (state-
  dependent) version of the gate. Rejected.
- **Soft-min**: fixes the kink but keeps the one-dimension-at-a-time
  gradient suppression and adds a temperature parameter. Rejected.
- **Product**: smooth everywhere, nonzero gradient in both dimensions
  throughout the interior, and the vertical credit is automatically
  weighted by lateral proximity — resolving a 6 nm pair pays ~4x a 12 nm
  pair. That is correct pricing: vertical resolution matters exactly in
  proportion to how laterally threatened the pair is. Note the current
  gate IS a product with g_vert = an indicator; this design is its minimal
  smoothing, not a new structure.

The two commissioned corner cases:

- **Laterally close, vertically separating** (the bug): f_lat large and
  roughly common-mode; g_vert falls smoothly as the climb is commanded
  (g_cmd, instantly) and flown (g_cur, ~2.75 FL/sweep). Priced, immediately
  and monotonically.
- **Vertically close, laterally separating**: g_vert ~ 1, f_lat falls
  along the existing lateral ramp. Priced exactly as today. Unchanged.

### 1.3 Zero-set consistency with LoS

Violation = (d < 5) AND (|dFL| < 10). Genuinely-safe configurations must
carry zero potential; hazardous ones must not.

New zero-set: f = 0  iff  d >= 15  OR  (|dFL_cur| >= 20 AND |dFL_cmd| >= 10).

- Every priced pair is inside a buffered hazard region: lateral buffer 3x
  the LoS radius (unchanged), current-vertical buffer 2x the LoS band
  (unchanged support), commanded-vertical relief complete exactly when the
  pair is COMMANDED out of the violation band (>= 10 FL).
- No configuration that can produce LoS without further vertical
  convergence is priced as fully safe: a pair co-level but commanded apart
  keeps g_vert >= 0.5 * g_cur > 0 until physically separated (the current
  factor is the honesty anchor — see 1.4).
- A pair currently separated (>= 20 FL) but COMMANDED to converge (e.g. a
  descent through occupied levels) now carries hazard 0.5 * g_cmd > 0 — a
  region the old gate priced at zero until the aircraft physically closed
  within 20 FL. This closes the endangerment blind spot symmetric to the
  climb blind spot. (This is the one region where f_new > f_old; see §2.3.)

### 1.4 Current vs commanded separation: price BOTH, blended — justification

Three candidate policies were considered:

1. **Current-FL only** (w = 0). Zero-set honest, minimal change. But the
   command's full intent registers only as flown: the one-time CBP payment
   sees ~2.75 FL of a 10 FL clearance (Δg small), and under CBP there is
   no second payment (§0). Signal ~2.5x weaker than the blend, for no
   safety benefit.
2. **Commanded only** (w = 1). Maximal immediacy, but the zero-set is
   WRONG: a co-level converging pair with a just-issued climb reads as
   75-100% resolved while LoS (defined on CURRENT fl) can still fire for
   ~4 more sweeps. The potential would stop warning precisely during the
   physically riskiest phase. Violates the commissioned zero-set
   requirement. Rejected as sole pricing.
3. **Blend, w = 0.5** (recommended). Zero-set = "safe iff currently
   separated AND commanded separated" (union of supports — consistent per
   1.3). The commanded half makes the clearance register at the decision
   step (LL-immediacy: the action's effect is in the very next priced
   differential); the current half keeps physical risk priced until the
   climb is actually flown, at 2x-LoS buffer. w = 0.5 is the
   least-tunable choice; w is an explicit amplification dial if review
   prefers otherwise (§2.4).

Why V_cmd = 10 while V_cur = 20: the commanded factor prices INTENT
against the violation definition — one atomic +10 FL clearance commands
the pair exactly out of the LoS band, so it should complete the commanded
relief (g_cmd: 1 -> 0). The current factor keeps the 2x physical buffer.
A pedantic note pre-answered: commanded dFL moves in 10-FL quanta
(DEFAULT_INTERVAL_FL), so g_cmd realizes almost binary values — that is
not a re-created gate, because commanded separation changes ATOMICALLY at
issuance (there is no gradual "commanded progress" to be blind to); the
gradually-evolving variable (current FL) has the smooth factor. Smoothness
is required exactly in the variables that evolve smoothly.

Also note the value net can SEE commanded state: selected FL is an
explicit token feature (relative encoder feature 178 block; re-verified in
state_repr/relative.py, consistent with the 12D Fix-C audit) — so the
bootstrap channel can carry whatever the one-time payment does not.

---

## 2. Weights, normalization, and the CBP magnitude arithmetic

### 2.1 Scale compatibility with the DELTA_CONFLICT budget

DELTA_CONFLICT stays 0.2. Properties:

- Per-pair debt cap unchanged: f <= 1, so |phi_conflict per pair| <= 0.2.
- On the old support (|dFL| < 20, no commanded convergence): g_vert <= 1 =
  old gate value, so f_new <= f_old pointwise. The conflict potential can
  only shrink there; pairs at dFL 10-19.9 (previously full-counted, never
  able to violate without further vertical convergence) are strongly
  de-priced — a mispricing REMOVED, not added (risk R3 tracks it).
- New nonzero region (currently >= 20 FL apart, commanded convergent):
  bounded by 0.5 * DELTA * f_lat <= 0.1 per pair, and it prices genuine
  hazard. Ambient impact under CBP is nil (common-mode); eval-time
  absolute phi_conflict columns shift — 12c cross-run comparisons of that
  diagnostic column are void (they already are, under the 12d flags).
- Budget reconciliation identity: pure bookkeeping over whatever
  shaping_paid is; holds for ANY f (§4.3).

### 2.2 Expected per-step differential for one climb step (the honest number)

One climb issued to one aircraft of a co-level pair at lateral distance d,
CBP lag 1, measured state: action branch |dFL_cur| ~ 2.75, |dFL_cmd| = 10;
NOOP branch 0, 0. Lateral geometry common-mode.

```
p1 = gamma * DELTA * f_lat(d) * [g_vert(noop) - g_vert(action)]
   = 0.97  * 0.2   * f_lat(d) * [1 - (0.5*(17.25/20)^2 + 0.5*0)]
   = 0.97  * 0.2   * f_lat(d) * 0.628
   = 0.122 * f_lat(d)                    (per vertically-proximate pair)
```

Per-state analytic p1 at the 15 frozen D2 conflict states (single
min-sep pair, co-level assumption; the rerun must recompute from actual
state geometry and sum over all pairs involving the climbed aircraft):

| min_sep nm | f_lat | p1 | p1 - fee |
|---|---|---|---|
| 2.86 | 0.655 | 0.080 | -0.020 |
| 3.97 | 0.541 | 0.066 | -0.034 |
| 4.63 | 0.478 | 0.058 | -0.042 |
| 5.36 | 0.413 | 0.050 | -0.050 |
| 5.71 | 0.384 | 0.047 | -0.053 |
| 5.86 | 0.371 | 0.045 | -0.055 |
| 6.09 | 0.353 | 0.043 | -0.057 |
| 6.36 | 0.332 | 0.041 | -0.059 |
| 6.39 | 0.330 | 0.040 | -0.060 |
| 6.52 | 0.320 | 0.039 | -0.061 |
| 7.14 | 0.275 | 0.034 | -0.066 |
| 7.31 | 0.263 | 0.032 | -0.068 |
| 8.83 | 0.169 | 0.021 | -0.079 |
| 8.98 | 0.161 | 0.020 | -0.080 |
| 10.11 | 0.106 | 0.013 | -0.087 |

(For reference: current-FL-only pricing gives p1 = 0.050 * f_lat; the
old gate gives p1 = 0 identically.)

### 2.3 Comparison with the command fee, stated without spin

A single-pair correct climb's one-time payment is 0.013-0.080: it offsets
13-80% of the -0.1 fee but does NOT exceed it at any of the 15 measured
geometries. Multi-pair climbs (one aircraft leaving a co-level cluster of
k vertically-proximate neighbours inside 15 nm) sum: k = 2-3 close pairs
can reach p1 ~ 0.1-0.24 and go net-positive. This is a structural cap of
CBP, not of the ramp: under CBP every consequence of a command is paid
exactly once, as the 1+lag-step divergence — there is no "accrues over the
next few steps" channel for the training reward (§0, third bullet). The
same cap binds the LATERAL channel (B3's ~1e-3), which is why fix B
(nstep 24) exists.

So, precisely: **"a correct climb becomes net-positive within a few
steps" is satisfied at the TARGET level, not the raw-reward level**, and
D5 + the 12d composition is what delivers it: the climb's own n-step
target is `(p1 - 0.1) + ~0 + gamma^n V(s_n)` while the NOOP candidate's
learned value at the same states carries `-50 * gamma^k` with k =
noop_violation_step <= 24 in 10/15 D2 states (median 22). D5's specific
contribution is that the vertical candidate's immediate term is now
sign-correct and 10-50x the lateral differential (0.02-0.08 vs ~1e-3)
instead of EXACTLY the anti-signed fee — i.e. within-window, climbing the
right aircraft is now strictly cheaper than any lateral command of equal
fee, and different candidate climbs order correctly among themselves.

### 2.4 If review demands raw-reward net-positivity at H<=6 (priced option)

p1 > 0.1 at the median state (f_lat ~ 0.33) requires
`DELTA * 0.97 * 0.628 * 0.33 >= 0.1`, i.e. DELTA_CONFLICT ~ 0.5 (2.5x).
Costs: eval-time absolute objective rescaled (all ep_return histories
discontinuous), B1 margins re-based, per-pair debt cap 0.5 vs delivery 10
(still fine), and a JK-approved weight (spec §4) changed. I do NOT
recommend it in this cycle: the ordering-level fix (2.3) achieves the
learning goal without re-pricing the objective, and DELTA can be revisited
with B1 evidence if 12d/13 shows the vertical channel still starved.
Second dial, also not recommended: cbp_lag > 1 grows the flown divergence
(~2.75 FL per extra lag step) for ALL channels at the cost of extra clone
steps and weaker latency-matching semantics — out of D5 scope, recorded as
an interaction (§5.5).

---

## 3. What changes in code (scope statement for the reviewers)

Mechanism is deliberately tiny (~15 lines + selftest work):

1. `AcSnap` gains a `sel_fl` slot; `sector_snapshot` fills it from
   `env.get_simulator_env().aircraft[cs].selected_fl` (fallback: current
   fl if unavailable — degrades to current-only pricing, never crashes).
2. `pair_conflict_f(a, b)` implements §1.1. Flag-gated:
   `--vertical_ramp {gate, smooth}`, default `gate` (bit-identical 12c
   reproduction; the flag is the ablation path). Constants:
   `CONFLICT_FL = 20` reused as V_cur; new `CONFLICT_FL_CMD = 10`,
   `VERT_CMD_WEIGHT = 0.5`.
3. `snapshot_stratum` KEEPS the binary < 20 gate (deliberate mismatch,
   §4.6).
4. Selftest extensions (§4.1) and the validation battery (§6).

Nothing else: shaping_terms, CBP branches, run_episode, replay, network,
checkpoints are untouched (the potential is not a learned object).

---

## 4. Invariants that must be preserved (with the proof or the plan)

### 4.1 Telescoping (potential-based shaping correctness)

f remains a pure function of the two AcSnaps, so shaping_terms' matched-
pair payments `gamma*phi(t+1) - phi(t)` telescope exactly as before — the
identity is form-level, independent of f's internals. The selftest
(_selftest_telescoping) must PASS UNCHANGED in gate mode and be EXTENDED
for smooth mode:

- Add sel_fl to the synthetic AcSnaps (test both sel == fl and sel != fl).
- Add a climbing aircraft: FL ramping 2.5 FL/step across another's 20-FL
  boundary (exercises the ramp interior AND the old cliff region), plus a
  mid-run selected-FL change (commanded factor jumps; identity must still
  hold to <= 1e-9 because the jump is a state change like any other).
- The endpoint expression already calls pair_conflict_f on endpoint
  snapshots, so it stays correct automatically.

### 4.2 Matched-set convention

Untouched: pair matching (both aircraft IN_SECTOR at t and t+1) is
computed before f is evaluated and does not depend on f. Entries/exits
still cannot create phantom kicks.

### 4.3 Budget reconciliation identity

`cbp_shaping + fuel + cmd_cost + delivery_bonus + violation_term ==
ep_return` is bookkeeping over whatever shaping_paid was: it holds for any
f in both cbp modes. The existing assert (run_episode L1404-1409) is the
check; --smoke in both flag modes exercises it.

### 4.4 CBP invariants

- NOOP-zero: for a global-NOOP action both branches are the same
  determinized computation at any lag — true for any state function f.
  _selftest_cbp must pass unchanged in both flag modes.
- Live-env side-effect-freedom (env_fingerprint assert): untouched — no
  new env stepping is introduced. sel_fl is a read of existing simulator
  state carried by the same deepcopies (selected_fl is part of aircraft
  state, hence already bit-faithfully cloned — the CBP latency finding
  depends on exactly this).
- Determinism: f is deterministic; B1-v2 pair-3's bit-identical replay
  property is unaffected.

### 4.5 B1-v2 / B2 gate — re-pass plan and pre-registered expectations

Any objective change must re-run the gate (PROBE_BATTERY process note).
Plan: `diagnose_controller.py --probe b1` (b1v2) and `--probe b2` with
`--vertical_ramp smooth`, seeds 10043/20042, BEFORE any training run uses
the flag. Pre-registered:

- Pair 1 (DELIVERER > NOOP, light density): PASS preserved. Neither
  scripted policy issues vertical commands; the conflict term shrinks
  common-mode (dFL 10-20 pairs de-priced for both). Directional side
  prediction: |phi_conflict| diagnostic column shrinks for all policies.
- Pair 2 (HOLDER < NOOP): PASS preserved — ordering is dominated by
  violation timing (-50 mean-RTG geometry), which f does not move.
- Pair 3 (fee identity): PASS preserved — shaping is common-mode between
  passes B and C by the determinism property; the identity is about fee
  arithmetic.
- B2(a) myopic oracle: expectation stays "boring control, between NOOP
  and DELIVERER". New capability: the oracle can now SEE vertical relief;
  with DELTA = 0.2, p1 < fee at single pairs, so predicted behavior is
  climbs only at multi-pair clusters (log the oracle's climb count — first
  nonzero value ever possible). A weird high scorer built on vertical
  commands = objective exploit, blocks the flag exactly as B2 specifies.
- B2(b) exit-window parking: vertical term orthogonal; PASS preserved.

Any miss is a finding that blocks enabling the flag, per battery
discipline.

### 4.6 Stratification and diagnostics consistency

snapshot_stratum keeps the binary |dFL| < 20 vertical-proximity gate.
Rationale: strata are a bucketing instrument feeding stratified-t and
cross-run comparisons; silently re-bucketing mid-program corrupts every
per-stratum monitor 12d just installed. The mismatch (potential smooth,
strata binary) is deliberate and recorded here. Revisit only with a
version bump on the strata definition.

### 4.7 E1 attempt-1 lesson (anchor absolutely)

All references in the new term are absolute constants (15, 20, 10, 0.5,
0.2). No live batch statistic, no self-referential floor, no learned
quantity enters the potential. Runaway of the E1-attempt-1 kind is
impossible by construction. (Stated because the reviewers will check.)

### 4.8 Checkpoint / replay compatibility

The potential lives outside the network: old checkpoints load and evaluate
unchanged (eval ep_return shifts with the objective, as with objective_v2
— report flag mode alongside). Replay transitions priced under gate mode
must not be mixed with smooth-mode training: fresh runs only (12d already
requires fresh runs; add `vertical_ramp` to the checkpoint extra fields
and refuse resume across modes).

---

## 5. Interaction analysis with the 12d composition

12d context: `--t_feed bootstrap --bootstrap_support taken_noop
--nstep 24`, likely plus an exploration bonus.

1. **x nstep 24 (fix B)**: the pairing that actually delivers vertical
   ordering (§2.3): B puts the -50 in-window for 10/15 conflict states;
   D5 makes the vertical candidate's in-window immediate term sign-correct
   instead of exactly-fee. Without D5, B still prices climbs at
   fee-only; without B, D5's p1 alone does not clear the fee. Window
   content shift from D5 is +0.01-0.08 on climb steps — negligible against
   B's +10/-50 terms; the "fee becomes a 1% rounding term" story is
   unchanged.
2. **x bootstrap_support taken_noop (fix D)**: restricted argmax is over
   {NOOP, a'_taken}. When climbs get taken (exploration bonus, below),
   their successor states carry selected_fl in tokens, so the NOOP-
   continuation value the restricted bootstrap evaluates is exactly the
   quantity D2's rollouts certify carries the +10/-50 signal at H=24.
   D5 does not touch the argmax; no new interaction surface.
3. **x exploration bonus**: whatever mechanism increases vertical action
   sampling, TODAY each sampled climb is taught "strictly worse than NOOP
   by exactly 0.1" (deterministic anti-learning). With D5 the sampled
   climb at conflict geometry nets -0.02..-0.09 immediately plus the
   in-window terminal difference — exploration stops actively teaching the
   wrong sign on the vertical channel. Caution: if the bonus magnitude
   >> p1, the shaping ordering within exploration noise is swamped —
   acceptable (bonuses buy coverage, ordering comes from targets).
4. **x t_feed bootstrap (fix E+F)**: D5 shifts target middles by <= ~0.1
   on a minority of steps; the bootstrap-coverage tracker recalibrates t
   automatically — that is its job. The safe-stratum width-collapse guard
   is unaffected (D5 adds no width-bearing term). Per-stratum monitors
   remain valid because strata definitions are unchanged (§4.6).
5. **x cbp_lag**: D5 makes the vertical channel respond to the lag dial
   at all (gate mode: zero at any lag). Raising lag amplifies p1 by
   ~2.75 FL per extra step through g_cur. Not proposed; recorded so nobody
   tunes lag "for D5" without pricing the lateral-channel and cost
   consequences.
6. **x mask_reissue**: repeating climb10 on the same aircraft is masked
   until another instruction intervenes, though a second climb is
   dynamically meaningful (+10 FL more, relative command). Pre-existing
   quirk, out of D5 scope — but note D5 REDUCES its damage: under the old
   gate the second climb was the only vertical command that could ever
   touch shaping (the 20-FL cliff); under D5 the first climb pays.
7. **MCTS v3 (forward interaction)**: the planner prices short
   determinized rollouts with the ABSOLUTE shaping, where a climb pays the
   full ramp as flown (~0.15 * f_lat over 4 sweeps plus avoided
   deepening), not the one-shot CBP payment. A vertical-blind Phi caps the
   planner permanently; D5 removes that cap. This is the strongest reason
   D5 is worth doing even though it does not by itself flip raw-reward
   sign at the fee.

---

## 6. Pre-registered validation plan (all before any training run)

Gate order: selftests -> (b) micro scripted -> (a) D2-Phi -> (c) B1/B2.

**(0) Selftest battery**: full `--selftest` in BOTH flag modes. Gate mode
must be bit-identical to today (the flag default proves 12c
reproduction). Smooth mode: extended telescoping (§4.1), CBP NOOP-zero,
budget reconciliation via --smoke, plus a new unit selftest: hand-built
two-snapshot pair, assert pair_conflict_f and the resulting shaping match
the closed-form p1 of §2.2 to 1e-12.

**(b) Micro level-allocation scripted check** (~minutes, ScriptedShim +
run_episode step_hook, the micro_level_allocation.py machinery):
scripted single well-timed climb in the 2-aircraft co-level conflict.
Pre-registered:
- Gate mode (regression proof of the bug): shaping_paid at the climb
  decision step == 0.0 exactly; every step's vertical contribution 0.
- Smooth mode: shaping_paid at the climb decision step > 0 and equal to
  the analytic p1 from the actual pair geometry to <= 1e-9 (determinized);
  it lands IN r_t of the decision step itself (lag-1 measurement), i.e.
  positive shaping accrues within 1 step — stronger than the commissioned
  "~3 steps". NOOP steps still pay exactly 0.0.
- Full-episode absolute-Phi columns telescope to their endpoint
  expressions (reuse of the selftest identity on a real episode).

**(a) D2-Phi rerun** (~25 min, credit_diagnostics --d2 --side atc, same
15 frozen conflict states, horizons [6,12,24,48,96], smooth pricing):
- Tier 1 (MUST PASS): at every state whose GT-best action is vertical
  (6/15), gap(6) and gap(12) move off exactly-fee by that state's analytic
  p1 (recomputed from true geometry incl. all pairs of the climbed
  aircraft), sign-correct, magnitude within 2x of analytic (tolerance for
  lateral drift within the horizon). States with lateral best actions:
  gap(6) unchanged within 1e-3 (no phantom vertical credit).
- Tier 2 (the commissioned prediction, honestly scoped): gap(6) > 0 is
  predicted ONLY where summed-pair p1 > 0.1 or noop_violation_step <= 6 —
  from §2.2 arithmetic that is the multi-pair/tight-geometry subset, not
  the median state, at DELTA = 0.2. If JK requires Tier 2 at median
  states, §2.4 is the priced route (DELTA ~ 0.5) and this rerun must be
  repeated under it with a fresh B1 audit.
- Environment-property sanity: noop_violation_steps identical to the
  20260714 JSON (Phi does not move the dynamics); H>=24 gaps shift only by
  the (small) shaping deltas.

**(c) B1-v2/B2 gate** as §4.5, both seeds, pre-registered expectations
there. Any pair misordering blocks the flag.

**Run-time monitors when the flag first trains** (12e or a JK-approved
12d addition): per-episode climb/descend command counts and their mean
shaping_paid (expect sign-correct, ~1e-2); delivery rate and sector-
excursion kind mix vs baseline (R2/R3 watch); phi_conflict diagnostic
column trend (expect smaller magnitude).

---

## 7. Risks, and the strongest adversarial objection

Risks, with mitigations:

- **R1 Magnitude insufficiency** — the headline risk; treated as the
  adversarial objection below.
- **R2 Vertical spam / altitude farming**: telescoping symmetry means a
  climb-then-descend round trip pays net ~0 shaping and 2 fees — no
  farming income. Residual behavioral risk: gratuitous climbs away from
  exit FL could delay deliveries or exit the sector vertically
  (excursion terminal). Monitors in §6; also VERIFY at implementation
  whether OUT_SECTOR triggers on vertical bounds (cheap probe, currently
  unconfirmed — flagged, not assumed).
- **R3 De-pricing of dFL 10-19.9 pairs**: laterally converging pairs at
  12-19 FL separation lose most of their conflict debt. Argued correct
  (they cannot violate without vertical convergence, which is now priced
  when commanded or flown), but it changes behavior pressure around
  vertically-stratified streams; watch violation-kind mix and B1 margins.
- **R4 Commanded-factor gaming**: issue climb (collect p1), later descend
  back. The descend's own CBP term is the symmetric negative at issuance
  (commanded factor jumps hazardward) — anti-symmetric by construction,
  plus two fees. No exploit identified; B2's myopic oracle is the
  instrument that would find one.
- **R5 sel_fl availability**: selected_fl could be None/degenerate on
  spawn edge cases; the fallback (sel := fl) degrades to current-only
  pricing locally and cannot crash pricing. Covered by smoke.
- **R6 Attribution creep in 12d**: a fourth simultaneous change violates
  the 12D conservatism design. Recommendation: ship flag-gated default-
  OFF; enable in the next run boundary (12e / run 13) unless JK explicitly
  trades attribution for time. All validations in §6 run before either.

**Strongest adversarial objection (constructed against this design):**

> "D5 is cosmetic under CBP. Your own arithmetic (§2.2) caps the one-time
> payment at 0.122 * f_lat <= 0.08 across all 15 measured conflict states
> — below the 0.1 fee everywhere — so a correct climb's immediate reward
> stays net-negative and the H<=12 gap stays anti-signed. The thing that
> actually flips vertical ordering is the -50 entering the 24-step window
> (fix B), which ships regardless. You've spent design risk (commanded-FL
> semantics, a new AcSnap field, a B1 re-audit) to move a differential
> from 0.000 to 0.04 that the fee still dominates: the LL-faithfulness
> banner hides that you did not restore LL's property (shaping that
> outweighs the action cost at the decision step) and cannot, at DELTA
> = 0.2, at these geometries."

Answer, part concession:

- **Conceded**: at DELTA = 0.2, D5 does not make a single-pair correct
  climb net-positive in raw immediate reward, and the H<=6 D2 gap stays
  negative at median geometry (the doc says so in §2.3 and scopes Tier 2
  accordingly). Also conceded: the fee-vs-shaping timing pathology (P2) is
  fix-B territory; D5 alone does not fix ordering.
- **Rebutted on ordering**: P2's measured pathology is EXACT anti-ordering
  — every action's in-window differential is -0.1 with p95 noise ~1e-16,
  so within-window ranking among actions is pure fee-tie. D5 breaks the
  tie with sign-correct structure: climbs of the right aircraft at the
  right geometry become the cheapest commands in-window (by 0.02-0.08,
  i.e. 10-50x the lateral differentials), and candidate climbs order
  correctly AMONG THEMSELVES by f_lat * Delta-g. Ordering is comparative;
  "which resolution, on which aircraft, in which direction" is exactly
  what the -50-in-window channel (a scalar event at H~22) cannot price
  and the smooth term can.
- **Rebutted on scope**: the terminal-in-window channel covers 10/15
  states and teaches avoidance at the violation horizon; 5/15 (noop
  violation at 27-49 steps) remain fee-only under B alone — for those, the
  smooth differential is the ONLY in-window vertical signal that exists.
- **Rebutted on the planner**: MCTS v3 and B2's myopic oracle price
  ABSOLUTE shaping over determinized rollouts, where the climb pays the
  full ramp (~0.15 * f_lat + avoided deepening), not the one-shot CBP
  payment. A vertical-blind Phi permanently caps every rollout-based
  component of this program; that cap is removed by D5 and by nothing in
  the 12d composition.
- **Priced escalation**: if the review's bar is literally raw-reward
  net-positivity at the decision step, the honest instrument is DELTA
  ~ 0.5 (§2.4) — offered as a flagged variant with its costs listed, not
  smuggled in.

---

## 8. Recommendation

Implement as §3 (flag-gated, default gate), run the §6 battery, and enable
`--vertical_ramp smooth` at the next clean run boundary. Parameters:
product form, w = 0.5, V_cur = 20, V_cmd = 10, quadratic ramps, DELTA
unchanged at 0.2. Open decisions for JK: (i) accept ordering-level
(Tier 1) success semantics vs demand Tier 2 and take the DELTA = 0.5
re-pricing with a fresh B1 audit; (ii) 12d inclusion vs 12e (attribution
discipline says 12e; the validations are run-independent either way).
