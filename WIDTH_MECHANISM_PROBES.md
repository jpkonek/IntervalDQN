# Width-mechanism probes — isolating the ~2× above-curve unfamiliarity signal

JK directive (8 July 2026): the LL mechanism probe (ll_width_mechanism_probe.py)
showed LL's wind-OOD widening is fully priced by input magnitude, BUT
structure-destroyed inputs at matched norm sit 1.8–2.2× ABOVE the norm→width
curve — a genuine learned unfamiliarity signal. Task: isolate the mechanism
generating it, then refine/build-upon/exploit it in future LL and ATC versions.

Every probe here states its predictions BEFORE running (probe-battery
discipline). A miss is a finding.

## Hypotheses

- **H1 — loss-carved width valley.** Width is squeezed only where data lives:
  the explicit width penalty (width_reg·w², applied when coverage > target)
  and/or the interval loss's (1−t)·max-dist term fire on-distribution only.
  Off-manifold, delta_raw keeps its un-squeezed generalization value → wider.
  Genuinely epistemic residue; amplifiable by construction.
- **H2 — activation-pattern novelty + Jensen dispersion.** Unfamiliar states
  hit ReLU regions unvisited in training, where raw head outputs SCATTER more;
  softplus convexity converts raw dispersion into higher mean width even with
  no raw-mean shift (E[softplus(x)] ≥ softplus(E[x])).
- **H3 — misgeneralized aleatoric structure.** The net learned real width-vs-
  feature structure; shuffling recombines features whose additive width
  contributions sum high. Not epistemic; would not rank novelty reliably.
- **H4 — rare-state resemblance.** Shuffled states resemble rare wide-width
  training states (crash transitions). Density in disguise.

## Probes and pre-registered predictions

**P1 [CKPT] Emergence over training.** Measure shuffled/box residual (vs each
net's own calm×k norm curve) on UNTRAINED nets (3 seeds) and on the c0.5
snapshot ladder (ep200/400/600/800/final × seeds 0–2).
- All hypotheses predict residual ≈ 1.0 untrained (no manifold learned yet);
  any large untrained residual = measurement artifact, stop and fix.
- H1/H3: residual grows over training. Timing vs the coverage/width-reg era
  (tracker_t in ckpts) localizes which loss term carves it.

**P2 [CKPT] Raw-head decomposition (mean vs dispersion).** At matched norm
(shuffled/box vs calm), record delta_raw mean and std pre-softplus.
mean-driven share = (softplus(E raw_ood) − softplus(E raw_calm)) / (width_ood
− width_calm).
- H1/H3 (net genuinely says "wider"): raw MEAN shifts up; mean-driven share
  > 0.5.
- H2 (Jensen artifact): raw mean ≈ unchanged, raw std ↑; share < 0.3.

**P3 [CKPT] Activation-pattern novelty.** Per-state 256-bit ReLU on/off
pattern; Hamming distance to nearest calm-state pattern; Spearman rho between
pattern novelty and width WITHIN the shuffled condition.
- H2 (and H1's geometric reading): rho > 0.3.
- H3: rho ≈ 0 once marginals are controlled.

**P4 [CKPT] Additive-marginal model.** Fit width ≈ global + Σ_d f_d(s_d)
(16 quantile bins/dim) on real states; predict shuffled widths.
- H3: prediction ≈ actual (ratio actual/pred ≈ 1.0, so the "signal" is
  additive recombination — deflationary).
- H1/H2: actual exceeds additive prediction by ≥ 1.3×.

**P5 [CKPT] Dose–response + per-state separability.** (a) Interpolate
calm→shuffled (α ∈ {0,.25,.5,.75,1}); (b) shuffle only k ∈ {1,2,4,8} dims.
- Any learned mechanism: monotone width growth in α and k. Flat = artifact.
- Per-state AUROC (real vs shuffled by width): if ≈ 0.5–0.6, the signal is
  population-level only and cannot gate per-decision caution; if > 0.75, it
  can. This bounds HOW the mechanism can be exploited regardless of which
  hypothesis wins.

**P6 [TRAIN, small] width_reg causal sweep.** Train LL 800 eps (≈4 min each),
c=0.5, width_reg ∈ {0.0, 0.1} × seeds {0,1} (0.01 nets already exist =
per_c); measure the residual.
- H1a (explicit penalty carves the valley): residual grows with width_reg
  (0 < 0.01 < 0.1).
- H1b (interval-loss t-term carves it): residual ≈ flat across the sweep,
  but still > 1 at width_reg = 0.
- H2/H3: flat in width_reg.

**ATC corollary [no run needed]** — from the existing A4 controller data:
garbage collapses delta_raw MEAN 41.4 → 2.5 (a learned-function mean effect
into the "certain −50" basin), which is H3-flavored misgeneralization, not
dispersion. Prediction to check later on run-12b+ nets: the controller has NO
latent width valley to amplify; single-net exploitation there requires adding
the off-manifold term explicitly (E1) or the ensemble.

## RESULTS (8 July 2026; JSONs: checkpoints/per_c/ll_mechanism_probe2*.json)

- **P1**: untrained residual 1.07–1.20 (measurement clean, as pre-registered).
  Signal fully formed by ep200 (1.75–2.01), flat thereafter. Learned EARLY.
- **P2**: mean-driven share = 1.00 on all seeds (raw mean 4.5→9.7 / 3.5→11.2 /
  5.8→10.4). NOT Jensen dispersion — the net genuinely outputs "wider".
  H2's dispersion mechanism REJECTED.
- **P3**: shuffled states 23–29 bits from nearest calm activation pattern
  (calm-self ≈ 1); Spearman(novelty, width) = 0.22–0.42 within shuffled vs
  ≈ 0 within calm. The elevation is organized by ReLU-region novelty.
- **P4**: additive-marginal model underpredicts shuffled widths 1.8–3.2×.
  H3's additive form REJECTED (caveat: naive marginal fit is poor on calm
  itself — R² 0.24/−0.61/−1.23 from correlated-feature double-counting — so
  this is not a maximally fair H3 test, but the margin is large).
- **P5**: monotone in interpolation α, roughly monotone in shuffled-dim k;
  per-state AUROC 0.67–0.82 (shuffled), 0.81–0.89 (box) — population-strong,
  per-decision moderate.
- **P6**: width_reg 0.0 → residual 1.75/1.72; 0.01 (per_c) → 1.6–2.5;
  0.1 → 1.18/1.65. NO dependence on the explicit penalty; signal fully
  present at width_reg = 0. **H1b confirmed.**

**VERDICT: H1b.** The interval loss's own coverage/width trade (t-weighted
min/max-dist terms, t ≈ 0.93 throughout training) globally pressures
delta_raw upward, trims it precisely on the data manifold, and leaves
piecewise-linear extrapolation through novel ReLU regions elevated —
a genuine, dose-responsive, novelty-correlated mean effect. It is epistemic-
flavored but UNTARGETED: nothing in the loss aims at off-manifold behavior,
and its sign is contingent on learned slopes — which is why the LayerNormed
ATC controller (different geometry) collapses into a narrow −50 basin
instead. Conclusion: engineer the property explicitly (E1) rather than rely
on the accident.

## E1 pre-registration (before the E1 validation runs)

Implementation: `ood_width_lambda`/`ood_width_floor` in IntervalDQNAgent
(default OFF): per-dim batch-permutation negatives each train step.

**Attempt 1 (FAILED gate 1, kept for the record):** hinge to floor × current
batch mean width (λ=0.05, floor-mult 4). Self-referential floor → runaway
feedback through the shared trunk: on-dist width 4 → 2033, R(50) −405/−713.
Lesson (transfers to ATC): the off-manifold width target must be ANCHORED
(absolute, or to a frozen/EMA-lagged reference), never to the live width.

**Attempt 2 (the registered design):** bounded relative hinge to an ABSOLUTE
floor: λ·relu(1 − w_neg/floor)², λ = 0.5, floor = 15 (≈4× healthy on-dist
width). Term capped at λ, gradient dies once satisfied — runaway impossible
by construction. 800 eps, c = 0.5, seeds 0–1.
Gates:
  1. Training health: final R(50) within noise of baseline (~200–260);
     on-dist calm width < 1.3× baseline nets.
  2. Signal: shuffled residual ≥ 3; AUROC(shuffled) ≥ 0.9.
  3. GENERALIZATION (the memorization check): box-garbage residual ≥ 3 and
     AUROC(box) ≥ 0.9, though box is NOT the training-negative distribution;
     graded k∈{1,2,4,8} response.
  4. wind10/20 residual expected ≈ 1 (E1 targets structural OOD; near-
     manifold epistemic remains the ensemble's job). Reported, not gated.
  5. Magnitude channel (calm×k) unchanged — E1 does not claim to fix it.

## E1 v2 RESULTS vs gates (8 July; JSONs: ll_mechanism_probe2_e1v2.json,
## ll_mechanism_probe_e1v2_full.json)

1. Health: R(50) end ≈ 170–255 (seed0 noisy-dips, seed1 254.7; baseline
   ~200–260) — PASS. On-dist width 4.5–7.0 vs baseline 3.5–5.9 (~1.3–1.5×
   on seed0) — GRAZE/mild width cost; tune λ/floor if it matters.
2. Signal, as-registered metric: shuffled residual-vs-norm-curve 1.86/2.27 —
   FAIL by the letter, but the METRIC IS CONFOUNDED for E1 nets: the hinge
   generalizes to scaled states, lifting the whole calm×k curve (slope 1.04
   → 1.13/1.17; calm×2..×10 resid 1.1–1.3), so the baseline the residual
   divides by has itself risen. Honest gauges: absolute ratio and AUROC.
   Shuffled ×calm = 2.68/3.22 (baseline 2.11±0.45), floor achieved almost
   exactly (15.07/13.65 vs floor 15); AUROC 0.76/0.90 (baseline 0.67–0.82;
   note shuffled has an intrinsic AUROC ceiling < 1: some per-dim-permuted
   rows are accidentally near-manifold and SHOULD stay narrow).
3. Generalization (memorization check): **STRONG PASS** — box garbage (not
   the training-negative distribution) ×calm = 5.17/4.74 (baseline
   2.71±0.45), exceeds the floor (36.3/21.4), AUROC 0.98/0.99 (baseline
   0.81–0.89). Graded k-response present (k=1 already 10.4/7.7 vs calm
   6.0/4.6). The net learned "structurally incoherent ⇒ wide", not the
   shuffle trick.
4. wind10/20: ×calm 1.1–1.6, resid 0.70–0.92 — as pre-registered, E1 does
   NOT manufacture near-manifold sensitivity; wind stays priced by
   magnitude. Near-manifold epistemics remain the ensemble's job.
5. Magnitude channel: intact (unclaimed), slightly steepened.

**E1 verdict: qualified success.** Off-manifold width is now a TRAINED,
generalizing property with near-perfect separation on unambiguous garbage,
at the cost of ~1.3× on-dist width and a residual metric that needs
re-basing for E1-trained nets. IP reading for the broader program: the
floor is a prior credal commitment — "far from data, the credal set must be
at least this wide" — a knob with imprecise-probability semantics, not just
a regularizer.

## ATC transfer spec (E1-ATC — build GATED on the 12b objective decision)

- Negatives at token level, two generators: (i) permute per-field blocks
  (kinematics, neighbor blocks) ACROSS aircraft within the batch —
  incoherent aircraft; (ii) swap whole aircraft tokens across scenario
  states — plausible aircraft, impossible joint traffic picture (the
  failure mode that matters operationally).
- ANCHORED absolute floor (attempt-1 lesson: never key the floor to live
  widths): measure the healthy on-dist width of a trained 12b-era net
  first, set floor ≈ 3–4×; bounded hinge λ·relu(1 − w/floor)².
- Purpose: directly attacks the "certain −50 basin" (A4: garbage → narrow,
  lower ≈ −50). With LayerNorm the magnitude channel is dead, so E1 is the
  ONLY single-net off-manifold widener available to the controller;
  complements ensemble-RPF (near-manifold disagreement) rather than
  competing with it.
- Ship default-OFF behind flags; validate with this file's gate battery
  plus the A4 conditions before any reliance.

## E1-near pre-registration (8 July, JK-approved: can E1 reach
## realistic-but-unfamiliar?)

Question: LL's recorded OOD widening (1.05→1.44×, monotone over
wind/gravity/combos — JK's read of ood_eval*.json, verified correct) is
carried by the magnitude channel (wind states sit ON the norm→width curve).
Can E1 training with NEAR-manifold negatives create widening that size does
NOT explain, on the wind conditions specifically?

Design: `ood_neg_mode` ∈ {mix = 50/50 blend of state and shuffled state,
noise = state + 1σ per-dim gaussian}; λ=0.5, floor=15, c=0.5, seeds 0–1,
800 eps (4 runs).
Gates:
  1. Health: R(50) ~200–260; on-dist width < ~1.5× baseline.
  2. SUCCESS = wind10/20 residual vs the net's own norm curve ≥ 1.3
     (baseline and far-E1 nets: 0.70–1.16), i.e. wind widening beyond what
     magnitude predicts. Secondary: wind20 ×calm ≥ 2 (baseline 1.39±0.35).
  3. FAILURE = wind residual ≈ 1 with health intact → E1 cannot reach
     near-manifold; the realistic-but-unfamiliar gap stands and the
     ensemble question returns to the table with data.
  4. Watch: near negatives may fight the on-dist interval loss (they
     overlap real data). If health breaks, one retry at λ=0.2/floor=10 is
     authorized; beyond that, report failure-by-interference.

## E1-near RESULTS (8 July; ll_mechanism_probe_e1near.json)

Gate 1 health: PASS (R(50) 222–270; calm widths 3.9–6.6 vs baseline 3.5–5.9).
Gate 2 success criterion (wind residual ≥ 1.3): **FAIL** — 6 of 8 wind cells
below 1.3 (range 0.66–1.33; only two cells touch exactly 1.33); secondary
criterion wind20 ×calm ≥ 2 also missed (max 1.93). Directional nuance,
honestly reported: mix-negatives nudged wind residuals from the far-E1 nets'
0.70–0.92 up to 1.08–1.33 — the trained rim moves TOWARD the manifold — but
wind-visited states sit far closer in than 50/50 blends, and negatives any
closer would overlap real data and fight the on-dist interval loss.
Meanwhile structural-OOD widening stays strong (shuffled 2.2–3.1×, box
5.9–7.9×, box resid up to 6.9).

**VERDICT (per pre-registration): E1 cannot reach realistic-but-unfamiliar
at safe negative distances. The gap is now MEASURED as uncovered by any
single-net mechanism tested: magnitude channel (accidental, deleted by LN
in ATC, wrong-direction there anyway), loss-carved valley (~2×, structural
inputs only), E1-far (garbage only), E1-near (marginal). Per the
pre-registered stop rule, the ensemble question returns to JK's table with
data. The remaining untested single-net option is the architectural channel
(deliberately unnormalized scalar features — aircraft count / density —
whose beyond-training-range values extrapolate width upward); plausible for
ATC where density IS the main unfamiliarity axis, untested, and inherently
limited to the chosen scalars.**

## Width-prior arm pre-registration (8 July — JK's "start wide, narrow with
## data" initialization mechanism, single-net)

Motivation (JK): absence-of-narrowing off-data is the CORRECT structure; the
finding that current nets don't implement it (P1: untrained nets start
NARROW; off-manifold direction is architecture-contingent) means the
structure should be built in, not hoped for. Single-net constraint stands
(ensemble deferred).

Design: frozen random net's |raw output| added to the width head
pre-activation (width_prior_beta in IntervalQNetwork; prior shared between
q and target nets). The net starts wide everywhere; the trainable head
must LEARN to cancel the prior where data falls. β=200 calibrated so
untrained width ≈ 12 (~3× calibrated on-dist width). Known imperfection,
recorded up front: |prior| spatial spread is large (p10→p90 ≈ 8×), so the
wide-start floor is uneven (~4 to ~28). No E1 term (mechanism isolated).
c=0.5, seeds 0–1, 800 eps.

Gates (same bar E1-near failed):
  1. Health: R(50) ~200–260; on-dist calm width < 1.5× baseline (the
     coverage tracker + width terms must succeed in narrowing the prior
     on-manifold — if they can't, that's failure-by-interference).
  2. SUCCESS: wind10/20 residual vs own norm curve ≥ 1.3 (and/or wind20
     ×calm ≥ 2). This is the first mechanism whose DESIGN targets
     near-manifold: wind states are off-data, so cancellation shouldn't
     reach them — unless it generalizes too smoothly.
  3. Pre-named failure mode: the trainable head cancels the prior SMOOTHLY
     BEYOND the data (both nets share architecture/smoothness), wind
     residual ≈ 1. If so: single-net verdict = cancellation-generalization
     is the binding obstacle; levers = higher-frequency prior input
     encoding, or revisit ensemble (JK's future option).
  4. Report: shuffled/box (should stay wide — prior uncancelled), calm×k.

Width-prior RESULT (β=200, plain prior; ll_mechanism_probe_widthprior.json):
health PASS (R 254/216; calm width 4.11/3.27 — prior fully cancelled
on-manifold), wind gate **FAIL by the pre-named mode 3**: wind residuals
0.82–0.95 (baseline territory); even shuffled/box residuals (1.83–2.69)
match plain nets — cancellation generalized essentially everywhere.

Fourier-prior arm (final single-net lever, same gates): prior input =
sin/cos(6·x) so the prior is rougher than the trainable head can track
off-data (local corr 0.93 vs ~1.0 at 0.1σ perturbations); β=120 (untrained
width ≈ 12). Named risk: imperfect cancellation ON-data → inflated calm
widths / coverage damage — that outcome = failure-by-interference and
closes the single-net program with the tradeoff quantified.

## ⚠ CORRECTION (8 July, late) — SUPERSEDES THE "CLOSING VERDICT" BELOW

JK challenged the claim that training merely inherits the architecture's
magnitude response. Three follow-up measurements (wind_knn_excess.json and
inline probes) overturned the wind attribution:

1. WITHIN the training distribution, trained nets have a NEGATIVE
   width-vs-norm relation (Spearman −0.19..−0.31; untrained: +0.42..+0.57).
   Training does not endorse the magnitude slope in-range — it REVERSES it.
   The +1.04 ×k slope is extrapolation beyond the data range only.
2. Wind states are moderately structurally novel (10–22 bits of activation-
   pattern novelty vs ~1 for calm; full shuffles: 23–29), and most sit
   INSIDE the calm norm range — where the trained relation predicts
   NARROWER, yet they measure wider.
3. Correct instrument (width ÷ 10-NN-neighbor-predicted width): baseline
   nets show wind excess 1.18/1.31/2.21 — a GENUINE near-manifold
   unfamiliarity response in the plain single net, seed-noisy. The old
   norm-curve-residual gate was near-powerless at wind-scale norm shifts
   (predicted-vs-null differ by ~15%) and, for E1 nets, divided away the
   signal via the lifted curve.

CORRECTED CONCLUSIONS:
- LL's recorded OOD dilation is NOT a magnitude accident. It is
  substantially force (1) — the loss-carved novelty elevation — expressed
  at moderately-novel wind states, plus learned regional width structure.
  Magnitude (2) governs only the far field (×10/×100, garbage-at-scale:
  12–133× — THOSE numbers are size, not epistemics).
- The "single-net program closed / realistic-but-unfamiliar uncovered"
  verdict below is RETRACTED as instrument-artifact. Under kNN-excess, the
  plain net already covers it partially; none of the engineered arms
  (e1_far 1.15/1.45, e1_mix 1.25/1.48, wp200 1.50/1.98, wpf6 1.38/1.62)
  clearly beats baseline seed spread. E1's garbage result (AUROC .98) stands.
- What survives unrevised: large seed variance (reliability problem); the
  ATC controller measured the OPPOSITE off-manifold behavior (−50 basin),
  so the LL mechanism did not transfer as configured — the sharpened
  question is WHY (LayerNorm? attention pooling? doom-dominated outcome
  landscape?), answerable by adding these instruments (kNN-excess,
  pattern novelty, in-range width-norm slope) to the A4 battery on the
  first properly trained 12b net.
- Helsinki narrative: partially REHABILITATED — "intervals dilate OOD" has
  a genuine learned component in LL; restate the mechanism (novelty
  elevation from on-manifold trimming) and quarantine the far-field
  magnitude numbers.

## LayerNorm-ablation pre-registration (8 July — last LL experiment before
## training; JK's "have we set ATC up correctly" question made testable)

LL is the verified positive control for the novelty-elevation mechanism.
Train LL with an ATC-style LayerNorm trunk (--layernorm; LN after each
hidden Linear), 2 seeds, everything else stock. Measure with the CORRECTED
instruments: wind kNN-excess, shuffled/box kNN-excess + AUROC, pattern
novelty, in-range width-vs-norm slope.
- If LN-LL RETAINS the elevation (wind kNN-excess within baseline's
  1.18–2.21 spread): LayerNorm exonerated → controller architecture stays;
  the ATC inversion is pinned on the degenerate 12a outcome landscape
  (near-constant returns → global downward width pressure), answerable
  only by training 12b on an objective with real outcome diversity.
- If LN-LL KILLS/inverts the elevation: LN convicted → modify the
  controller BEFORE 12b (LN off the width-head path, or unnormalized
  width side-path).
Either outcome directly determines the 12b build. A4 additions carried
forward regardless: kNN-excess (token space), pattern novelty, in-range
slope, and GLOBAL WIDTH-PRESSURE DIRECTION (mean delta_raw drift on/off
manifold across training checkpoints — the landscape-hypothesis
diagnostic).

## 12b milestone readings (a4v2 probe; frozen state set)

ep500 (probe_battery_a4v2_20260708_163712.json): landscape hypothesis
HALF-confirmed. (i) The 12a-style global downward width pressure is GONE —
mean delta_raw on fixed real states rises 4.86→8.85 over ep25→550, garbage
rises in parallel — the diverse-outcome objective pushes width up
everywhere, as predicted. (ii) BUT no novelty elevation yet: aircraft-swap
states (31 bits novel) sit exactly at neighbor-predicted width (excess
1.03, AUROC 0.565), and the certain basin persists in ordering (garbage
0.76×, shuffled 0.62× real width; AUROC 0.22–0.33 ANTI-separation; present
from ep25). In-range width-norm slope +0.209 (width_scalars channel live).

PRE-REGISTERED for next milestone (~ep1000–1500): in LL the elevation was
fully formed by 25% of training. If by ep1500 (37%) swap kNN-excess is
still ≈1 and garbage still < 1× real, conclude outcome diversity fixes the
PRESSURE DIRECTION but not the OOD ORDERING in this architecture/token
regime → E1-ATC term (token-level negatives, anchored floor) becomes the
required lever, as run 12c or a fine-tune stage — do NOT restart 12b for
it (policy learning is independent of width ordering).

ep1000/ep1500 readings + **DECISION (pre-registered criterion fired)**:
swap kNN-excess dead flat across 500/1000/1500 (1.03/1.01/1.02, novelty
27–31 bits — the elevation is NOT emerging on the operationally relevant
near-OOD); shuffled-field floor still broken (0.62→0.70→0.90). Honest
trend note: the basin is slowly FILLING as outcome diversity accumulates
(garbage excess 0.94→1.04; garb/real raw deficit 0.79→0.87) — but filling
to neighbor-predicted (≈1.0) is not elevation (LL gives 1.2–2.2 on novel
states). CALL: outcome diversity fixes pressure direction, not OOD
ordering → E1-ATC (token-level aircraft-swap negatives, anchored absolute
floor, bounded hinge) is the required lever, as run 12c or a fine-tune
stage; 12b runs to completion untouched. C2 side-note: rho ~0 at
ep1000/1500 (0.04) with coverage creeping (0.11→0.15) while policy
improves — B3's credit-starvation prediction; CTDE/longer-horizon thread
stays open for JK.

RESULT (ln_ablation_probe.json): **LayerNorm EXONERATED.** LN-LL trains
healthily (R 243/234) and RETAINS the full mechanism: wind kNN-excess
1.18/1.84 (inside baseline's 1.18–2.21), shuffled 1.95/2.32, box
1.53/1.88, AUROCs comparable. Bonus: LN deletes the in-range size
relation entirely (Spearman 0.01/0.09) while preserving the novelty
elevation — an LN trunk is arguably the CLEANER interval net (width
semantics uncontaminated by the far-field magnitude artifact).
Consequence (pre-registered branch): controller architecture stands
unchanged; the ATC inversion is pinned on the degenerate 12a outcome
landscape (near-constant returns → global downward width pressure);
the test is training 12b on the objective with real outcome diversity
and running the corrected A4 instruments at milestones. LL is mined out.

## SINGLE-NET PROGRAM — CLOSING VERDICT (8 July 2026, RETRACTED — see ⚠)

Fourier-prior RESULT (ll_mechanism_probe_wpf6.json): health PASS (R
249/237), wind gate **FAIL** (residuals 0.78–1.10). Cancellation
generalized even against the rougher prior; pushing roughness further
degrades on-data calibration (the tradeoff is structural).

Six single-net mechanisms, one shared bar (wind residual ≥ 1.3 = widening
on realistic-but-unfamiliar beyond what input size predicts):

| mechanism | wind residual | covers |
|---|---|---|
| magnitude channel (accidental) | ≡1 by construction | "big" inputs only; carries LL's recorded 1.05–1.44× |
| loss-carved valley (~2×) | 0.99–1.00 | structural inputs only |
| E1-far (engineered) | 0.70–0.92 | garbage, strongly (AUROC .98) |
| E1-near | 0.66–1.33 | marginal |
| width-prior β200 | 0.82–0.95 | nothing beyond baseline |
| Fourier width-prior β120 f6 | 0.78–1.10 | nothing beyond baseline |

Why, in one sentence: any width signal inside the net must coexist with
the loss that calibrates widths on data, and a smooth trainable head's
cancellation/carving generalizes across exactly the near-manifold band
where realistic-but-unfamiliar states live — so the signal survives only
far off-manifold (garbage), never nearby. Distinguishing "near-manifold
but unseen" requires a function whose generalization is DECOUPLED from
the width head's: independent ensemble members (deferred by JK, this
dataset ready if revisited), an explicit density model, or — ATC-only,
learned-slope, no guarantee — the unnormalized scalar channels (count/
density), which remain the directed next build.

## Exploitation branches (chosen by probe outcome)

- **E1 — contrastive off-manifold width term (if H1 or H2).** Add negative
  samples to the interval loss: for shuffled/perturbed copies of each batch,
  penalize SMALL widths (push delta_raw up off-manifold). Carves the valley
  deliberately instead of relying on the accident. Same recipe in ATC at
  token level (swap aircraft feature blocks across the batch). This is the
  concrete re-entry point for the parked deeper-loss-redesign thread.
- **E2 — dispersion as a second signal (if H2).** Expose raw-head dispersion
  (or MC-dropout spread) alongside width; cheap ensemble-lite.
- **E3 — don't build on it (if H3/H4).** The signal is misgeneralized
  aleatoric structure; single-net width cannot rank novelty. Ensemble/RPF
  (already built) becomes the only honest mechanism; E1 still usable but as
  pure regularization, without epistemic interpretation.
- **E-val — regardless of winner**: any exploited signal must pass P5-style
  per-state AUROC and a calm×k FALSE-POSITIVE check (magnitude alone must NOT
  trigger it once inputs are normalized).
