# Data / results I need to verify before cutting slides

The slide claims I want to check are all in the ML half (slides 19–41 in the current deck). I've looked in `Credal DQN/` and `Credal LunarLander/` and found numbers that don't match the slide text. Below: every numeric claim on a slide, where I looked, what I found, and the *specific* question I have for you.

If the answer is "you're looking in the wrong folder — go look in X," tell me X and I'll redo the audit.

---

## Folders / files I've already checked

| Path | Used for |
|---|---|
| `Credal DQN/HANDOFF.md`, `STATUS.md` | gates G1–G5 on collision; PIP ratio |
| `Credal DQN/results/collision/g1_aggregate.json` | per-seed robust values |
| `Credal DQN/scripts/eval_g1_maxmin.py` | PIP reference constant 102.95 |
| `Credal LunarLander/SESSION_HANDOFF LUNAR LANDER.md` (root copy) | seed 42 / seed 123 tables |
| `Credal LunarLander/results/per_c/` and `per_c_windy/` | directory listings only |
| `Pessimistic Iterative Planning with RNNs for Robust POMDPs.pdf` | PIP Table 2: 103.91 median, 102.10 min on Aircraft |
| `FINAL_SUMMARY.md` | EMBER side project, not used for the talk |
| `ADVERSARIAL_BENCHMARKS.md` | EMBER side project, not used for the talk |

## Folders / files I have NOT inspected (might be the right place)

| Path | What it might contain |
|---|---|
| `Invterval PP/` (sic) | older/newer pipelines, named "Interval PP" with typo |
| `Interval PP/` | possibly the canonical evaluation pipeline |
| `IP n4/` | possibly results on $n = 4$ problems |
| `Finite tail CPM/` | likely theoretical, but worth confirming |
| `review_package/` | I peeked at `code/losses.py` only — the rest might be where the slide numbers live |
| `INTERVAL_RM_v5_DESIGN.md`, `INTERVAL_RM_v6_IMPLEMENTATION.md` | v5/v6 of the interval reward-machine work; may have its own eval |
| `DEMO_SCRIPT.md`, `DEMO_SCRIPT_SHORT.md` | the *demo* script — most likely candidate for canonical talk numbers |
| `lunarlander_interval_dqn_per_c.py`, `eval_lunarlander_*.py` at repo root | per-c evaluation drivers — separate from the `Credal LunarLander/` folder |
| `Credal LunarLander/results/per_c/c{0.0,0.2,…}_seed{0..4}/` | each per-c-per-seed json — I have not opened them |
| Anything under `Credal DQN/results/collision/seed{0..4}/g3_curve.json`, `g4_width.json`, etc. | the actual per-gate raw results, not just the aggregate |

---

## Itemised questions per slide

### Slide 23 — "PIP reported cost ≈ 103"

**Slide says**: "Pessimistic Iterative Planning … expected cost ≈ 103. This is the number we want to reach."

**What I found**:
- PIP paper Table 2: median 103.91, min 102.10 on `Aircraft`.
- `Credal DQN/scripts/eval_g1_maxmin.py` uses constant 102.95 (from Figure 3 of the paper).

**Question for you**: Are 103 / 102.95 / 103.91 / 102.10 all referring to the same Aircraft-collision number? Which exact value should the slide quote?

---

### Slides 24, 25, 26 — "our policy: cost 76 nominal, 116 worst-case, 108 best seed"

**Slide 24 (nominal)**: "cost 76"
**Slide 25 (worst-case)**: "cost 116"
**Slide 26 (best seed)**: "cost 108 … within 5% of PIP"

**What I found**:
- `Credal DQN/results/collision/g1_aggregate.json` gives per-seed robust values: **5068, 1124, 11418, 15313, 6822** (with FSC extraction at c=0).
- `Credal DQN/HANDOFF.md` reports "Best MC G1 robust value: **242** vs PIP's 103. Within 2.4×."

The slide numbers (76 / 108 / 116) don't appear in either of those.

**Question for you**: Where do 76 / 108 / 116 come from? Specific candidates:
- Was the collision evaluation re-run after the HANDOFF was written?
- Are those numbers from `Credal DQN/results/collision/seed*/g3_curve.json` at a particular c-value, not from G1?
- Are the slide numbers from a *different* benchmark (Avoid, Evade, Intercept from the PIP paper) and not Aircraft?
- Or are they aspirational/illustrative numbers used in the slide draft, not measured?

---

### Slide 27 — "Imprecision enters at two points"

This is a *conceptual* slide; no numbers to verify.

---

### Slides 32, 33 — "cautious c=0 reward +302" / "daring c=1 reward −1006"

**Slide 32 (cautious)**: "reward +302 — lands cleanly under unseen wind."
**Slide 33 (daring)**: "reward −1006 — chases best-case — takes risks; crashes hard."

**What I found**:
- `Credal LunarLander/SESSION_HANDOFF LUNAR LANDER.md` reports:
  - seed 42, strong wind, c=0.0: **254.6** reward, 90% solve
  - seed 42, c=1.0: **242.5** reward, 86.7% solve, **0% crash**
- The same handoff says "Wind perturbation doesn't substantially degrade any well-trained agent." And the *Ensemble (N=5)* agent crashes 100% of the time with reward −529.4 — that's a different agent, not c=1.

The +302 and −1006 numbers don't appear in the seed 42 / seed 123 tables.

**Questions for you**:
- The −1006 crash result — is that for the **ensemble** (which crashes 100% of the time with reward −529.4), or for the **interval c=1** (which actually still solves 86.7%)? Different stories.
- Is +302 from a different seed/condition than the ones in the handoff?
- Or are these numbers from a yet-to-find evaluation script (e.g., `lunarlander_interval_dqn_per_c.py` at the repo root)?

---

### Slide 34 — "Width is calibrated to OOD distance, +35% wider at extreme OOD"

**Slide claim**: line chart of interval width vs deployment condition, with the credal-trained network's width up to **+35% wider** than the calm baseline at the most extreme OOD.

**What I found**:
- `Credal DQN/HANDOFF.md` reports G4 (width-tracks-credal-uncertainty) **FAILED** on collision: Spearman ρ ∈ [−0.08, 0.23], need ≥ 0.5.
- `Credal LunarLander/SESSION_HANDOFF LUNAR LANDER.md` doesn't claim a width-OOD correlation; it emphasises sample efficiency and variance.

**Questions for you**:
- Is there a separate set of LunarLander experiments — perhaps under `Credal LunarLander/results/per_c_windy/` or `eval_lunarlander_ood.py` — that measured interval width across OOD conditions and produced the chart on slide 34?
- The chart data on the existing slide (`talk/slides.html` line ~852) hard-codes 9 conditions with specific width values: 4.04, 4.36, 4.47, 4.58, 5.14, 5.86, 4.04, 4.49, 6.49. Where did those numbers come from?

---

### Slides 35, 36, 37 — wind = 0, 15, 25 progression

**Slide 35 (wind = 0, ID)**: standard reward +241, credal +271.
**Slide 36 (wind = 15, mild OOD)**: standard +225, credal +260.
**Slide 37 (wind = 25, severe OOD)**: standard **−4** (fails), credal +232.

**What I found**: The `SESSION_HANDOFF LUNAR LANDER.md` reports only one wind condition (strong wind = (20, 2)), two seeds (42, 123). It does not show a per-wind sweep at 0 / 15 / 25.

**Questions for you**:
- Where are the per-wind-value results from? Were they generated by `lunarlander_interval_dqn_per_c.py` or `eval_lunarlander_per_c.py` (both at repo root)?
- For each of the three wind conditions, can you point me at a JSON / CSV / log file containing the reported rewards? I want to read those directly and cite the actual numbers.

---

### Slides 38, 39 — gravity = −11.5, −11.99

**Slide 38 (mild OOD)**: standard reward +258, credal +268.
**Slide 39 (severe OOD)**: standard **+29** (near-fails), credal +244.

**What I found**: The `SESSION_HANDOFF LUNAR LANDER.md` does not mention gravity perturbations at all. It does say gravity is held constant during training, but doesn't quote evaluation-under-gravity numbers.

**Questions for you**:
- Where are the gravity-perturbation results? Same questions as slides 35–37: which script generated them, and where are the per-condition results?

---

### Slide 40 — "Adaptive c wins on 8/11 conditions, 5 seeds, 30 episodes per condition"

**Slide says**: bar chart of three agents (calm-trained best fixed c, range-trained best fixed c, range-trained adaptive c) across 6 conditions: calm ID, wind 10, wind 20, wind 25, high gravity, combo OOD. Italic note: "5 seeds; 30 episodes per (model, condition) pair." Body text: "Adaptive c wins on 8/11 deployment conditions tested."

**What I found**: I haven't found the underlying numbers. The chart on the slide hard-codes:

| Condition | calm_best | credal_best | adaptive_credal |
|---|---:|---:|---:|
| calm (ID) | 259 | 262 | 258 |
| wind 10 | 231 | 233 | 249 |
| wind 20 | 141 | 179 | 198 |
| wind 25 | 90 | 160 | 130 |
| high gravity | 197 | 219 | 228 |
| combo OOD | 44 | 65 | 41 |

**Questions for you**:
- "5 seeds, 30 episodes per pair" — is that real? The seed-42/seed-123 LunarLander runs in the handoff are 2 seeds. If 5 seeds are real, where are seeds 0, 1, 2 (or 0–4, depending on labelling)?
- "8/11 conditions" — slide chart only shows 6 conditions. What are the other 5?
- Are the chart numbers (259 / 262 / 258 …) from a single eval json, or aggregated from per-seed files? Can you point at the source?

---

## Summary of what I need from you

Looking at the questions above, three pointers would resolve almost all of them:

1. **The canonical evaluation script(s) for LunarLander at multiple wind values and multiple gravity values.** I suspect this is `lunarlander_interval_dqn_per_c.py` or `eval_lunarlander_per_c.py` or `eval_lunarlander_ood.py` at the repo root (not under `Credal LunarLander/`). Knowing which one, and the directory where its outputs live, will pin down slides 32–40.

2. **The canonical evaluation script and results for the collision benchmark (slides 22–26)** — specifically the source of the 76 / 108 / 116 numbers. Are these from a later evaluation than the one summarised in `Credal DQN/HANDOFF.md`, or from a different (Avoid/Evade/Intercept) benchmark, or aspirational placeholders?

3. **Confirmation about the "5 seeds, 30 episodes / condition" claim on slide 40.** Either point me at five-seed evaluation outputs, or tell me the truth is "2 seeds" and we need to fix the slide caption.

Once those three things are clarified I can:
- Update `ml_section_script.md` with accurate numbers.
- Tell you which slides need to be cut vs reframed vs left alone.
- Verify that "wins on 8/11 conditions" is supported by the actual evaluation runs.

If nothing else lives in those folders and the existing `HANDOFF.md` / `SESSION_HANDOFF LUNAR LANDER.md` are the most recent, then the cut list in the corrected script stands and I should proceed to revise the slides.
