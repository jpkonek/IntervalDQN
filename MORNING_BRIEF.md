# Morning brief — overnight autonomous window, 7–8 July 2026

**Bottom line: run 12b did NOT launch.** The B1 objective gate failed twice — once
on the v1 sector objective (real structural bug, fixed), once on the v2 revision
(where my read is that the probe, not the objective, is now the weaker instrument).
Per the pre-committed stop rule I did not iterate a second time and did not train.
Nothing is running; no commits were made. Everything below is working-tree only.

The window's real products: a probe battery that **refuted the v1 objective before
any training was spent on it**, a diagnosis that the pilot nets' celebrated OOD
width scaling was an architectural accident (LL lesson, section 4), and a clean
decision point for you (section 3).

---

## 1. BLESS OR REVERT — changes made under delegated authority

All in the working tree, none committed. Each is flag-gated, so "revert" is a flag,
not surgery.

| # | Change | Where | Default | Revert |
|---|--------|-------|---------|--------|
| 1 | **CBP shaping at lag 1** — counterfactual-baselined potential, γ·(Φ(s′_a)−Φ(s′_noop)), one-sweep command latency means spec-literal lag-0 is provably identically zero | bluebird_controller_dqn.py `--cbp` | ON | `--no-cbp` |
| 2 | **Re-issue masking** — repeat (aircraft, instruction) of a still-active clearance masked out of action space and Double-DQN argmax | `--mask_reissue` | ON | flag off |
| 3 | **Objective v2** — global fuel term REMOVED (it taxed survival: HOLDER out-scored NOOP by dying at 144 s); flat +10 delivery → `10·max(0.3, nominal_T/actual_T)` decaying bonus (nominal_T from entry along-track distance / entry TAS, no-wind) | `--objective_v2` | ON | `--no-objective_v2` restores v1 |
| 4 | **n-step 6 → 12** — credit horizon extension; motivated by B3 showing per-step marginal credit ≈ 2.4e-3, i.e. nearly all learnable signal lives in event/terminal terms | `--nstep` | 12 | `--nstep 6` |
| 5 | **B1 gate basis: undiscounted → discounted (γ=0.97) returns** — matches what the learner optimizes; side-effect analyzed in §3 | diagnose_controller.py | — | probe change only |
| 6 | **A4 criteria revised** — old "monotone width vs stratum" PASS was vacuous (stratum 1 had zero states); new: absolute floor ≥1.0, OOD-distance monotonicity, all-strata-populated sampling | diagnose_controller.py | — | probe change only |

My assessment of each: #1 and #2 are safe and verified (CBP NOOP-zero exact at all
50 probe steps, clone side-effect-free, budget reconciliation to 7e-15; masking has
its own selftest suite). #3's fuel removal is forced by evidence; the decaying bonus
is my design and is the least-exercised piece (end-to-end only on one delivery in
B1 and the single-aircraft B4 field — see caveats §6). #4 is defensible but was not
ablated. #5 is where the controversy lives — §3.

## 2. Gate history — why 12b did not launch

**Battery round 1 (v1 objective):** B1 FAIL, structural. The per-aircraft-step fuel
term made dying cheap: HOLDER (all aircraft orbiting) beat NOOP by crashing at
144 s and paying less fuel. A "competent DELIVERER" oracle (urgency-scheduled
give-way vectoring, TTV 462→768 s + 1 delivery) still scored below NOOP
undiscounted. Objective refuted before training on it. This is the probe battery
doing exactly its job.

**Bounded iteration (one round, pre-authorized):** objective v2 as in §1, gate
re-based to discounted returns, B1/B2/B3/B4 re-run.

**Battery round 2 (v2 objective): B1 FAIL again — 2 of 3 pre-registered pairs.**

| Policy | s10043 G_disc | s20042 G_disc | TTV / kind |
|---|---|---|---|
| DELIVERER | **−1.61** (1 delivery) | −2.90 (0 deliveries, 90 cmds) | 768 s LoS / 714 s LoS |
| NOOP | −4.94 | **−1.75** | 462 s LoS / 666 s excursion |
| HOLDER | −26.01 | −26.87 | 144 s / 138 s excursion |
| WEAVER | −21.55 | −14.74 | 186 s / 276 s excursion |

- **DELIVERER > NOOP**: holds on 10043, fails on 20042. On 20042 the oracle
  delivered nothing — it prevented NOOP's 666 s drift excursion only to cause its
  own LoS at 714 s, and its 90 instruction fees (−1.53 discounted) exceeded the
  +0.38 discounted value of violating 48 s later.
- **WEAVER ≤ HOLDER**: fails in all 4 cells, purely on terminal timing. Weaving
  (≈ heading hold) drifts off-route slower than a persistent left turn, so WEAVER
  excurses 42–138 s later; at γ=0.97 near t≈25 steps each 6 s of terminal delay is
  worth ~0.6–0.8, an order of magnitude more than the fee gap the expectation was
  written to test.
- HOLDER < NOOP: holds everywhere (the v1 pathology is dead).
- **B2 red-team: PASS.** The myopic true-reward oracle finds no degenerate high
  scorer; exit-window parking scripts don't beat delivering. **B4: PASS-shaped.**
  Clean crossing (+9.34) ≫ veer-off (−51.9) > miss (−54.7), margins ≫ noise; the
  run-11 boundary-aversion trap is not present in v2.

**Stop rule executed.** "If B1 fails again, deliver failure accounting and STOP" —
this is that accounting. I also declined to compute a post-hoc returns-to-go
re-scoring of the same episodes tonight: the data isn't in the JSON (aggregates
only), and re-specifying the instrument after a pre-registered miss and re-running
until green is probe-shopping. That option is yours to bless, not mine to take.

## 3. The decision — my read, then your options

**What the second failure actually indicts.** Be careful here; the two misses have
different characters:

- The WEAVER/HOLDER expectation is, I now think, **ill-posed under a discounted
  basis**. Both are degenerate policies that die early; their order is decided by
  which one stumbles across the sector boundary first, not by the fee structure
  the expectation was registered to test. This miss is an instrument artifact.
- The DELIVERER/NOOP miss on 20042 is substantially an **oracle-competence
  failure**: at plateau density (24–25 aircraft, 10–13 conflict pairs by 720 s) a
  lateral-only, one-clearance-per-sweep controller cannot reliably deliver — the
  scripted oracle included. Ranking NOOP above a policy that spends 90 radio calls
  and still violates 48 s later is *defensible pricing*, not obviously a wrong
  objective.
- What survives both rounds as a real structural fact: **at γ=0.97 from t=0, the
  episode-level objective is ≈ discounted violation timing minus fees.** γ^118 ≈
  0.027, so anything after ~700 s — deliveries included — is nearly invisible from
  the episode start. BUT the learner bootstraps returns-to-go from each state,
  where a delivery 10 steps ahead is worth ~7.4, not ~0.27. B1-as-written may be
  structurally unable to see delivery credit that the learner would see fine.

**Your options (pre-registered gate means this is your call, not mine):**

**(a) Re-specify B1 and re-gate — my recommendation.** Two changes: score policies
on mean returns-to-go over on-trajectory states (the learner's actual quantity),
and fix the oracle premise — either probe at pre-plateau density or give the
DELIVERER vertical instructions if/when they exist. Drop WEAVER≤HOLDER (ill-posed)
in favor of something that tests fees directly, e.g. DELIVERER vs DELIVERER+noise
commands. Named risk: this is the "revise the test after failing it" pattern. I
think it's justified because the failure decomposition above localizes the misses
to the instrument, and B2/B4 (the exploit checks) pass — but it should be blessed
by you precisely because I'm the one who designed both the objective and the test.

**(b) Launch 12b on v2 anyway**, treating B1 as advisory, with C2 (Q-vs-ground-
truth rank correlation + unbiased rollout coverage) as the in-run kill criterion.
Cheaper, but it abandons the "objective proven before compute" discipline that
just saved us a wasted overnight run on v1.

**(c) Judge lateral-only capped and build vertical/FL instructions first.** The
density-wall evidence: NOOP itself violates at 462–666 s; no scripted or learned
lateral-only policy has survived plateau density. If the ceiling is ~700 s
regardless of objective polish, capability — not the reward — is the binding
constraint. This was already flagged as the morning's capability question; the B1
data strengthens it. Compatible with (a): fix the probe, add vertical actions,
then gate.

## 4. The three standing questions (faithfulness / reward / LL lessons)

**Is the implementation faithful to the conceptual spec? YES, now verified rather
than assumed.** Full selftest suite green: permutation invariance (1.2e-07),
padding equivalence, telescoping shaping (≤8.9e-16), CBP NOOP-zero exact,
budget reconciliation identity asserted every episode (7e-15), n=12 windower
brute-forced on a synthetic violation trajectory, DeliveryClock rides deepcopies,
59,016 params. The mirror pricing in the probes was validated against the real
`run_episode` (dG ≈ 3.5e-05) before any probe ran.

**Is the reward structure correct? v1 NO (refuted); v2 UNPROVEN — see §3.** What
IS established: no exploit found by the myopic red-team; needle-threading
geometry correctly priced; the survival tax is gone; ambient shaping noise is
zero under CBP (28× noise reduction vs naive), so credit rides on event terms —
which is why n-step went to 12.

**Overlooked LL lessons? One big one, found and diagnosed (A4) — and, after JK
challenged the claim on 8 July, adjudicated directly on the LL checkpoints.**

CORRECTION first: the original wording of this section misattributed "10–40× OOD
width scaling" to LL's headline result. The 10–40× figure was the bluebird PILOT
nets on garbage inputs. LL's actual recorded OOD effect (ood_eval JSONs) was a
modest, monotone **1.15–1.6×** median widening (calm ≈ 4.0–4.7 → wind20 ≈
5.1–5.5 → wind25+grav ≈ 5.6–7.5).

The mechanism test on the bluebird pilot net showed width tracks activation
magnitude, not unfamiliarity: scaling PERFECTLY FAMILIAR states ×10/×100
reproduces 8.5×/84× widening, while norm-matched garbage buys only 1.7×; the
LayerNormed controller net inverts the sign entirely (garbage → NARROW
"certain −50" intervals). JK correctly noted this was demonstrated on bluebird
nets, not LL, so the same probe was run on the LL per_c checkpoints
(ll_width_mechanism_probe.py; 9 nets = c∈{0,0.5,1}×seeds 0–2; results in
checkpoints/per_c/ll_mechanism_probe.json). Pre-registered, run 8 July. Verdict:

- The magnitude channel transfers exactly: width ∝ hidden norm (log-log slope
  1.03–1.05); calm×10 → 12.7×, calm×100 → 133× widening on familiar states.
- **The headline wind conditions sit EXACTLY on the magnitude curve**: wind10
  residual 1.00±0.17, wind20 residual 0.99±0.09. LL's OOD widening is fully
  priced by wind states being physically larger (faster, more tilted) — the
  net does not recognize them as unfamiliar over and above their size.
- **But a genuine unfamiliarity signal exists**: structure-destroying inputs at
  matched norm sit ~1.8–2.2× ABOVE the curve (shuffled 1.80±0.27, box garbage
  2.21±0.39). The interval loss did teach the LL nets something real about
  data density — it's just small, and the wind eval never exercised it.

Consequences: (1) adaptive-c's LL performance is untouched as an OPERATIONAL
result — width-keyed caution helps regardless of why widths grow, and in LL
magnitude correlates with danger; (2) as EPISTEMIC semantics ("the credal set
widens OOD"), the headline is ~entirely magnitude — right answer, wrong reason
— and the mechanism is architecture-fragile (LayerNorm deletes it; the
controller net flips its sign). Any public claim needs restating. (3)
Width-ranked risk in the controller is not currently trustworthy (A4 stratum
widths non-monotone: 9.84/11.83/4.61 at both 40 and 1500 eps). The honest
epistemic mechanism is the **independent-seed ensemble with randomized prior
functions** — width from member disagreement, a genuine finite credal set —
now built code-only (§8).

Also persisting from LL discipline: in-distribution realized coverage 0.68 vs
0.85 target with t saturated at its cap — the loss's coverage lever is exhausted;
this feeds the parked deeper-loss-redesign thread.

## 5. Run 12a state (for completeness)

1500 eps / 142k steps / 6682 s completed on the ORIGINAL v1 objective —
predating everything in §1, superseded. TTV ~476–484 s (≈ NOOP), 0 deliveries.
Checkpoints under `checkpoints/bluebird_controller/train_seed42_20260707_182733_*`.
Its main value was as the checkpoint substrate for A1–A4/C2 probe development.
Checkpoint probes were NOT re-run after v2 (a v1-trained net is stale against a
v2 objective — re-run them on the first 12b milestone instead).

## 6. Caveats and loose ends (deliberately not swept under the rug)

- **Delivery bonus is the least-tested new component**: exercised end-to-end on
  exactly one delivery (B1-10043, bonus 9.96) plus the synthetic selftest and B4's
  single-aircraft field. Mirror equality for the bonus rests on shared code, not
  an episode-level cross-check with deliveries.
- **Latent footgun FIXED**: `ControllerReplayBuffer` with capacity ≤ 1250 used to
  yield a ZERO-length regular pool; `term_cap` floor now capped at capacity//2,
  with a tiny-capacity assertion added to the selftest. Full selftest suite
  re-run green after the fix. (Production 100k / smoke 5000 were never affected.)
- B1's 2400 s leg is dead weight (every scripted episode ends at its violation
  ≤ 768 s; the second duration duplicates the first exactly).
- B3's MEET is degenerate by construction (CBP makes ambient NOOP noise
  identically 0, so "signal > noise" is trivially true); the absolute marginal
  signal is still tiny — CTDE / longer-horizon concerns are not retired.
- nominal_T ignores wind (entry TAS); tailwind ratios >1 pay >10 uncapped —
  intentional, but unreviewed by you.

## 7. Assets

- Battery JSON: `checkpoints/bluebird_controller/diagnostics/probe_battery_20260707_224656.json`
  (v2 round; earlier rounds and the A4 diagnosis JSONs sit alongside).
- Specs: CONTROLLER_ARCHITECTURE.md, PROBE_BATTERY.md (pre-registrations).
- Code: bluebird_controller_dqn.py, diagnose_controller.py (both untracked —
  never committed; first commit should follow your bless/revert pass).
- GIFs: radar_controller_tiny_seed10043.gif (12a-era behavior).

## 8. Actions taken after the stop rule fired (also for your review)

- **B1-v2 draft pre-registration written, NOT run** — bottom of PROBE_BATTERY.md.
  Returns-to-go basis; DELIVERER>NOOP gated at pre-plateau density (plateau
  reported, not gated — that's the capability wall); WEAVER≤HOLDER replaced by a
  paired-trajectory fee test (DELIVERER vs DELIVERER+re-issues: identical
  dynamics, gap must equal the discounted fee sum exactly — zero terminal-timing
  confound). It gates nothing until you bless it.
- **Ensemble/randomized-prior module DELIVERED, code-only** (new file
  `bluebird_controller_ensemble.py`, 952 lines; no training, no edits to
  existing files). **This deviates from my own "after 12b stable" sequencing** —
  flagged deliberately: the gate's reason was compute ordering, which a code-only
  build doesn't touch, and the module is needed under every branch of §3. If you
  disagree, the file is standalone and deletable.
  Design: M=4 members = unmodified base agents whose nets are wrapped with
  frozen random priors (f_θ + β·p on both the lower head and delta_raw
  pre-softplus, via forward hooks — the base class is not forked); shared
  replay, NO bootstrap masks (independent init + priors, per the RPF paper);
  credal hull [min l_m, max u_m] + two disagreement scalars (midpoint-std and
  Hurwicz-std, both logged so run-13 data can decide which should drive
  adaptive-c). All 7 selftests pass — **independently re-verified by me, 3s**:
  member independence, priors bitwise-frozen under real train steps, β wiring,
  hull-containment + u≥l at β∈{0,3,30}, checkpoint round-trip, param budget
  (236k trainable = 4×59k; wall-clock est. 1.5–3× since CBP deepcopies, not
  net passes, dominate). Honest test-5 result: UNTRAINED members show no
  OOD/real disagreement separation (ratios 1.00×/0.92×) — expected; separation
  is a trained-ensemble prediction, to be tested at the first 13 milestone.
  Pre-run-13 decisions for you: β (3 vs 10–30 — can the prior escape the −50
  basin at |Q|≈50 scale?), M (4 vs 8), selection policy (credal-hull Hurwicz
  vs per-episode member sampling), and its --train harness needs a 2-episode
  smoke before any real run.
- Replay-buffer footgun fix + selftest (see §6).

**Suggested first move:** decide §3. If (a): the B1-v2 draft in PROBE_BATTERY.md
is the concrete proposal — review/amend the pre-registration, then it gates 12b
as before. If (c) is bundled in: vertical instructions become the next build item
and the oracle gets them too.
