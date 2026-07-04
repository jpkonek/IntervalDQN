# IP-ML project audit

Date: 2026-05-12

## Scope

I read `HANDOFF.md`, `talk/ml_section_script.md`, `talk/data_verification_requests.md`, the current `talk/slides.html`, the collision JSON outputs, the LunarLander aggregate JSON outputs, and the relevant training/evaluation code in the audit package.

The aim here is not to assess whether the project is a competitive ML contribution. It is to identify which claims are currently supported, which are overstated, and which should be rewritten to avoid making wild or technically false claims.

## Bottom line

The project has a defensible modest core:

> Interval-valued Q-networks trained with a coverage/tightness interval loss are implementable; the same broad interval-head/loss/action-selection idea can be used in a finite-state collision benchmark and in LunarLander; the resulting interval widths and Hurwicz-style policies are empirically interesting enough to motivate further work.

The current script is much safer than the HTML slide deck. The slide deck still contains several claims that are false, stale, or too strong. The main corrections are:

1. Do **not** describe the LunarLander aggregate comparisons as comparisons against “standard DQN” or “vanilla DQN”. The JSONs used for the quoted aggregate numbers are calm-trained interval-DQN runs versus wind-range-trained interval-DQN runs.
2. Do **not** say the same neural architecture handles both environments. Collision uses a GRU-based recurrent encoder. LunarLander uses a feed-forward MLP interval agent.
3. Do **not** say the intervals are calibrated epistemic uncertainty, calibrated OOD distance, or learned imprecise probabilities. The supported claim is weaker: the learned interval width is a deployment-time interval-width signal that trends upward on parts of the chosen OOD grid.
4. Do **not** say the method matches/reaches PIP without caveats. The best selected collision run has MC point estimate 107.77 versus PIP reference 102.95, but this is best-of-25, has large MC uncertainty at the worst parameter setting, and is not reliable across seeds.
5. Do **not** say LunarLander is smooth continuous dynamics without qualification. Gymnasium LunarLander has continuous observations and discrete actions, with contact indicators and contact dynamics.
6. Do **not** claim the ML section demonstrates the four mathematical properties from the first half. The RL loss is an engineering interval-scoring loss, not the CPM-derived loss.

## Claims that are supported

### 1. Collision: best selected run near the PIP Aircraft reference

Source files:

- `Credal DQN/results/collision/per_c_dh64_trunc/c*_seed*/g1_mc_at_c.json`
- Best run: `Credal DQN/results/collision/per_c_dh64_trunc/c0.8_seed2/g1_mc_at_c.json`

Across the 25 per-checkpoint MC evaluation JSONs, the minimum `robust_value_mc` is:

| field | value |
|---|---:|
| c | 0.8 |
| seed | 2 |
| robust_value_mc | 107.77 |
| worst_sl | 0.6 |
| pip_reference_N9 | 102.95 |
| diff_vs_pip_pct | 4.6819% |
| n_rollouts_per_sl | 100 |
| SEM at worst_sl | 23.6147 |

Safe wording:

> In the best selected run out of 25, the MC point estimate of worst-case cost is 107.8, compared with the PIP reference value 102.95. This is a selected-best result and the MC standard error at the worst parameter setting is large, so it should not be presented as a stable performance claim.

Unsafe wording:

> We match PIP.
>
> We reach PIP.
>
> We are within 5% of PIP.
>
> Comparable to PIP.

The last two can be made safe only with the selected-best and uncertainty caveats.

### 2. Collision: performance is not reliable across seeds

The 25 `robust_value_mc` values are:

`107.77, 128.04, 216.23, 225.65, 235.00, 235.00, 240.00, 240.00, 240.00, 246.20, 248.93, 250.00, 250.00, 250.00, 260.00, 260.00, 284.31, 293.58, 293.87, 304.00, 312.67, 356.38, 627.71, 794.68, 843.56`.

By fixed `c`:

| c | values | median | mean |
|---:|---|---:|---:|
| 0.0 | 250.00, 240.00, 250.00, 284.31, 304.00 | 250.00 | 265.66 |
| 0.2 | 260.00, 216.23, 627.71, 235.00, 225.65 | 235.00 | 312.92 |
| 0.5 | 260.00, 248.93, 240.00, 293.87, 843.56 | 260.00 | 377.27 |
| 0.8 | 312.67, 246.20, 107.77, 128.04, 240.00 | 240.00 | 206.94 |
| 1.0 | 356.38, 235.00, 794.68, 250.00, 293.58 | 293.58 | 385.93 |

Safe wording:

> The method can find policies in the right cost range on this benchmark, but training/selection is not stable in the current implementation.

### 3. LunarLander: the 7/11 adaptive claim is supported under the right comparator

Source files:

- Calm-trained interval-DQN: `checkpoints/per_c/ood_eval_extended.json`
- Wind-range-trained interval-DQN: `Credal LunarLander/results/per_c_windy/ood_eval.json`

The claim verified from the aggregate JSONs is:

> Wind-range-trained adaptive-c beats both fixed-c baselines — the best calm-trained fixed-c agent and the best wind-range-trained fixed-c agent — on 7 of 11 evaluation conditions.

The exact condition table is:

| condition | calm best fixed-c | wind-trained best fixed-c | wind-trained adaptive-c | adaptive beats both? |
|---|---:|---:|---:|:---:|
| ID calm, normal gravity | 259.3 | 261.6 | 257.5 | no |
| wind 5 | 248.5 | 263.7 | 266.1 | yes |
| wind 10 | 233.8 | 238.8 | 249.0 | yes |
| wind 15 | 185.6 | 208.8 | 220.1 | yes |
| wind 20 | 140.6 | 178.9 | 197.9 | yes |
| wind 25 | 89.5 | 159.9 | 129.5 | no |
| low gravity -3 | 223.7 | 215.9 | 226.3 | yes |
| gravity -11.9 | 184.2 | 218.9 | 168.9 | no |
| gravity -11.99 | 197.4 | 206.1 | 228.0 | yes |
| wind 15 + gravity -11.9 | 145.3 | 170.4 | 174.3 | yes |
| wind 25 + gravity -11.99 | 44.2 | 65.1 | 41.4 | no |

Related counts:

| comparator | count |
|---|---:|
| wind-trained adaptive-c beats both calm best fixed-c and wind-trained best fixed-c | 7/11 |
| wind-trained adaptive-c beats calm best fixed-c alone | 8/11 |
| wind-trained adaptive-c beats calm adaptive-c | 9/11 |
| wind-trained adaptive-c beats calm best among fixed-c and adaptive-c | 7/11 |

The script currently says “against the calm-trained baseline alone, it is 9 out of 11.” That is only true if “calm-trained baseline” means calm-trained adaptive-c. It is false if the comparator is the best calm-trained fixed-c policy. Rewrite this sentence.

### 4. LunarLander: interval width increases on much of the OOD grid, but calibration is not established

For wind-range-trained `c=0.0`, the mean interval widths are:

| condition | mean width |
|---|---:|
| ID calm, normal gravity | 4.162 |
| wind 5 | 4.359 |
| wind 10 | 4.466 |
| wind 15 | 4.579 |
| wind 20 | 5.140 |
| wind 25 | 5.857 |
| low gravity -3 | 4.040 |
| gravity -11.9 | 4.663 |
| gravity -11.99 | 4.491 |
| wind 15 + gravity -11.9 | 5.042 |
| wind 25 + gravity -11.99 | 6.490 |

Relative to the ID width 4.162:

| condition | relative change |
|---|---:|
| wind 25 | +40.7% |
| wind 25 + gravity -11.99 | +55.9% |
| low gravity -3 | -2.9% |
| gravity -11.99 | +7.9% |

Safe wording:

> On this evaluation grid, mean interval width generally increases under stronger wind and under the hardest combined shift. This is a descriptive width trend, not a validation of calibrated uncertainty.

Unsafe wording:

> Learned imprecision is calibrated to OOD distance.
>
> The model knows when it is wrong.
>
> This is epistemic uncertainty.
>
> The intervals are imprecise probabilities.

## Claims that need correction

### A. “Standard DQN” / “vanilla DQN” comparator

Problem: the aggregate LunarLander numbers in the source-of-truth JSONs are not from a scalar standard DQN baseline. They are from calm-trained interval-DQN agents in `checkpoints/per_c/`, evaluated by `eval_lunarlander_ood.py`.

The file `lunarlander_interval_dqn.py` does define `StandardDQNAgent`, but the aggregate OOD files used for slides 34–40 are generated from `IntervalDQNAgent` checkpoints.

Replace:

> standard DQN / vanilla DQN

with:

> calm-trained interval-DQN baseline

or, if discussing the codebase rather than the quoted aggregate numbers:

> the codebase also contains a scalar DQN agent, but these aggregate comparisons are not using it.

### B. “Same architecture handles both environments”

Problem: false.

Collision uses `CredalIntervalQNet` in `Credal DQN/credal_interval_dqn.py`, with a GRU cell over observation/action history. LunarLander uses `IntervalQNetwork` in `lunarlander_interval_dqn.py`, a feed-forward MLP with an interval output head.

Replace:

> the same architecture handles both settings

with:

> the same interval-head/loss/action-selection pattern is reused, with environment-specific encoders.

### C. “GRU encoder” for LunarLander

Problem: false. LunarLander has no GRU in the inspected aggregate pipeline.

Replace:

> GRU encoder over observation history

with:

> feed-forward MLP for LunarLander; recurrent encoder only in the collision benchmark.

### D. “Continuous smooth dynamics” for LunarLander

Problem: too strong / partly false. LunarLander has continuous observation components, discrete actions, two leg-contact indicators, and contact dynamics.

Replace:

> continuous, smooth dynamics

with:

> continuous-state physics simulator with discrete actions and contact indicators.

### E. “Calibrated” / “epistemic” / “OOD distance”

Problem: not established.

The project has not validated frequentist coverage against true Q-values, posterior uncertainty, or a formal epistemic/aleatoric decomposition. The collision handoff explicitly says width-correlation failed: Spearman rho was only about `[-0.08, 0.23]`.

Replace:

> epistemic uncertainty / calibrated imprecision / calibrated OOD distance

with:

> learned interval-width signal / descriptive uncertainty proxy / interval-valued value estimate.

### F. “Loss functions incentivise learning imprecise probabilities”

Problem: too strong in the RL half.

The first half may be about proper losses whose Bayes optima are credal predictive models. But the RL experiments here train interval-valued Q estimates, not credal sets over transition probabilities, not lower previsions over all gambles, and not imprecise probabilities in the strict Walley sense.

Safe replacement:

> loss functions can incentivise interval-valued predictive objects, and in RL this can be used to train interval-valued Q estimates.

If you want to use “imprecise probability” in the talk, say explicitly:

> This is not yet a full imprecise-probability RL algorithm. It is a pragmatic interval-Q surrogate inspired by the same scoring-rule idea.

### G. “The four properties are delivered”

Problem: unsupported. The ML section does not demonstrate the mathematical four-property CPM story from the first half. It demonstrates an engineering surrogate.

Replace:

> even this simplification delivers the four properties

with:

> even this simplification gives a concrete test case for the broader idea: train a model to output non-point-valued predictive objects and use them at decision time.

### H. PIP comparison language

Problem: too strong unless caveated. PIP is a robust-POMDP method using formal robust planning machinery and recurrent finite-state controllers. Your method is not doing that.

Safe replacement:

> PIP is the formal robust-POMDP baseline. My method is not a replacement for it. The best selected interval-Q run is numerically close to the PIP reference on this one small benchmark, but the result is unstable across seeds.

### I. OOD labels for LunarLander wind 5/10/15

Problem: ambiguous. For the calm-trained baseline, wind 5/10/15 are OOD. For the wind-range-trained agent, wind 5/10/15 are in-range or boundary cases, since training samples wind from `[0,15]`.

Replace:

> OOD wind 5/10/15

with:

> shifted relative to calm training; in-range for the wind-trained agent when wind is within `[0,15]`.

## Recommended talk framing

A safe version of the project claim is:

> I am not proposing a new deep-RL architecture or a state-of-the-art robust-control method. The point is narrower. The first half of the talk argues that loss functions can be designed so that their optima are non-point-valued predictive objects. The RL experiments ask whether the same design instinct is operationally useful: train a Q-network to output an interval rather than a scalar, penalise both non-coverage and excessive width, and use a simple Hurwicz parameter at decision time. On two small benchmarks, this is implementable and gives suggestive behaviour. In collision, a best selected run gets an MC point estimate near the PIP Aircraft reference but is not stable across seeds. In LunarLander, a wind-range-trained interval agent with adaptive action selection improves over fixed-c interval baselines on 7 of 11 tested conditions, and its interval widths tend to increase on stronger shifts. These are preliminary empirical observations, not claims of calibrated imprecise probabilities or state-of-the-art robust RL.

## Recommended slide edits

### Slide 18/19

Remove:

> even this simplification delivers the four properties

Use:

> even this simplification gives a concrete RL test case for interval-valued prediction under a proper-loss-style training objective.

### Slide 23

Use:

> PIP reference on Aircraft: about 103 expected cost. I use 102.95 as the numeric comparison value because that is the constant used in the evaluation script.

Avoid implying that PIP is merely a neural baseline. It is a formal robust-POMDP planning baseline.

### Slides 24–25

If keeping the 76 and 116 numbers, label them as single rendered trajectories, not aggregate performance numbers.

Use:

> One illustrative rollout, not a robustness estimate.

### Slide 26

Replace:

> within 5% of PIP

with:

> best selected run: 107.8 MC worst-case cost versus PIP reference 102.95; selected-best and high-variance.

Add either a small table or speaker caveat:

> Median across all 25 runs is 250.0; median at c=0.8 is 240.0.

### Slide 27

Replace:

> how unsure the model is

with:

> learned interval width signal.

### Slide 28

Replace:

> continuous, smooth dynamics

with:

> continuous observation vector, discrete actions, contact indicators, and wind/gravity shifts.

### Slide 30

Replace:

> same architecture handles both settings

with:

> same interval-head/loss/action-selection pattern, different encoders.

Remove GRU language from LunarLander unless explicitly referring back to collision.

### Slides 32–33

Use aggregate medians if the slide is making a performance claim. If keeping rollout GIFs, label them as illustrative single trials.

### Slide 34

Replace:

> learned imprecision is calibrated to OOD distance

with:

> interval width increases on the stronger wind and combined-shift conditions.

Use exact numbers:

- ID width: 4.162
- wind 25 width: 5.857, +40.7%
- wind 25 + gravity -11.99 width: 6.490, +55.9%

### Slides 35–39

Replace “standard DQN” with “calm-trained interval baseline” unless the plotted data are regenerated from an actual scalar DQN baseline.

Do not use “collapses”, “robust”, or “8x” unless tied to exact aggregate data and comparator.

### Slide 40

Use:

> Wind-range-trained adaptive-c beats both fixed-c baselines on 7 of 11 conditions.

Optional additional statement:

> It beats the calm-trained adaptive-c policy on 9 of 11 conditions.

Do not say “against the calm baseline alone, 9/11” unless the comparator is explicitly the calm adaptive-c policy.

### Slide 41

Replace:

> reaches PIP at small finite scale

with:

> best selected collision run is numerically close to PIP, but stability and calibration remain open.

## Methodological gaps to state openly

1. No validation that interval endpoints have correct frequentist coverage for true Q-values.
2. No proof that the RL interval loss has Bayes optima corresponding to imprecise probabilities or lower previsions.
3. No comparison with strong distributional RL baselines.
4. No scalar vanilla-DQN aggregate baseline for the quoted LunarLander OOD results, unless there are additional files not in this audit package.
5. No stable collision result across seeds.
6. No robust statistical uncertainty analysis beyond the stored MC SEMs.
7. Reproducibility scripts contain hard-coded absolute local paths in several places; a third party may not be able to rerun them without path edits.

## Final assessment

The project should be framed as conceptually motivated empirical exploration, not as a sophisticated ML method. The safe claim is not “we have a new robust RL algorithm”. The safe claim is:

> Scoring-rule ideas for non-point-valued prediction can be turned into a simple interval-Q training objective; the resulting agents give suggestive behaviour on two small RL examples; this motivates a more principled theory of RL losses whose optima are imprecise-probabilistic objects.

That is coherent and defensible. The current slides need edits before presenting, because several claims are currently too strong or technically false.
