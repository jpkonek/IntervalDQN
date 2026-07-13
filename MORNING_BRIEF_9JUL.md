# Morning brief — run 12b complete (9 July 2026, ~06:45)

**Run 12b finished cleanly: 4000 episodes / ~19 hours on the gate-validated
objective (v2 + CBP + masking + width scalars).** No crashes, no watchers
fired, all milestone passes ran on schedule. Everything below is measured,
uncommitted, and waiting on three decisions (§4).

## 1. Policy verdict: modest gain, hard lateral ceiling

- Final eval (8 eps, 2400 s, seeds 10042+): TTV **629 ± 95 s** at c=0,
  635 ± 56 s adaptive; **8/8 violated** (5 excursion / 3 LoS at c=0);
  deliveries ≈ 0.1/ep at eval. 12a was ~480 s; NOOP is 462–666 s.
- Training curve: TTV climbed to ~576 s by ep1000 and stayed FLAT for the
  remaining 3000 episodes while deliveries/200eps went 1 → 16 → ~10.
  Command rate settled ~0.6/step (fee discipline works; no circling —
  both GIFs show transit-like flow, episodes end on single-aircraft
  boundary excursions at ~24 aircraft).
- Reading: the objective now prices the right things (the gate proved
  that), but a lateral-only, one-clearance-per-sweep controller cannot
  beat plateau density. The 600 s ceiling is a CAPABILITY wall, not a
  reward bug. This is the strongest evidence yet for vertical/FL
  instructions as the next policy lever.

## 2. Width system verdict (the a4v2 trend across the whole run)

| instrument | ep500 → final | meaning |
|---|---|---|
| swap kNN-excess | 1.03 / 1.01 / 1.02 / 1.05 / **1.04** | novelty elevation NEVER emerged on the operational near-OOD — five flat readings; the pre-registered E1 call stands, now with a complete run behind it |
| shuffled-field | 0.62 → 0.90 → 0.68 → **0.84** | certain-basin floor broken throughout (noisy, no resolution) |
| garbage excess / raw | 0.94 → 1.04 → **1.26**; raw_garb 11.47 > raw_real 10.20 at end | honest counter-trend: by run end the garbage condition's raw output crossed ABOVE real — the basin partially filled late. Still far from LL's 2×+ elevation and swap/shuffled unmoved, so it doesn't change the E1 decision — but it says outcome diversity does slowly push the right direction on the crudest condition |
| width-pressure ladder | raw_real 4.86 → 10.20 monotone | the 12a global downward pressure is definitively gone (landscape hypothesis: direction CONFIRMED, ordering NOT) |
| in-range density slope (scalar channel) | 0.21 → 0.36 → 0.23 | JK's channel learned a real positive slope, then plateaued/noisy — alive but not yet load-bearing |

## 3. Calibration vs ordering — the run's most consequential split

- **C2 rank correlation is a flat NULL across the entire run**: 0.16 /
  0.04 / 0.04 / 0.13 / **0.004** final. The net never learned to ORDER
  candidate clearances by consequence — B3's credit-starvation prediction
  (per-candidate credit ~2e-3 vs a −50 terminal) held to the end.
- **Unbiased rollout coverage nearly tripled**: 0.11 → 0.15 → 0.16 →
  **0.33** (in-training realized coverage 0.65 vs 0.85 target, t at cap).
  The intervals are slowly calibrating even though ordering isn't.
- Consequence that outranks the width story: **MCTS v3 is blocked on
  this.** A planner rehearsing with a rho≈0 value ordering rehearses
  noise. Whatever fixes candidate-level credit (CTDE, denser event terms,
  vertical actions changing the credit landscape, or E1-style auxiliary
  signals) is a prerequisite for the tree, not an optional refinement.

## 4. Decisions queued for you

1. **E1 sequencing** — the term is built, selftested, flag-gated
   (--e1_width, λ=0.5, floor=30 = 3× measured median width; hinge-gradient
   caveat: may need λ higher). My recommendation: **fine-tune from 12b
   final** (~1500 eps) rather than fresh 12c — the policy plateau is
   stable, E1 shapes the width function, and a fresh run spends ~19 h
   relearning the same plateau. Fresh buffer either way (not persisted).
2. **Vertical/FL instructions** — now the binding policy constraint (§1).
   Nontrivial build: action space, token features, oracle updates, B1
   re-gate. If approved I'd sequence it BEFORE the ensemble revisit,
   since it also plausibly changes the credit landscape (§3).
3. **Ensemble revisit** — the near-manifold gap is now confirmed
   uncovered across an entire training run (five flat swap readings).
   Module is built and idle. Your single-net directive stands until you
   say otherwise; this is the standing evidence, not a nag.

Assets: final ckpt train_seed42_20260708_141121_final.pt (+161 milestone
ckpts); diagnostics/a4v2_12b_*.log, c2_12b_*.log, eval_final_*.log,
probe_battery_a4v2_*.json; radar_12b_ep1500_seed10043.gif,
radar_12b_final_seed20042.gif; WIDTH_MECHANISM_PROBES.md carries the full
pre-registration/decision trail. Nothing committed to git.
