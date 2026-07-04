# Speaker script — ML half of the talk

18 slides (deck slides 27–44). Modest delivery, clear narrative, slides carry the visuals, the script carries the nuance.

## Tone and scope

I am a mathematician giving an empirical-ML section. I am not claiming a new robust-RL architecture or a state-of-the-art result. The point is narrower:

- **Conceptual claim**: the same scoring-rule design instinct from the first half — losses whose Bayes optima are *non-point-valued* predictive objects — can be turned into a workable RL training rule. Train a $Q$-network to output an interval per action; reward coverage; penalise width. Use the resulting interval at deployment via the Hurwicz rule.
- **Empirical claim, narrow**: this is implementable. On the lunar-lander OOD grid the range-trained adaptive-$c$ policy beats both fixed-$c$ interval baselines and the scalar DQN baseline on most conditions, and the interval width trends wider on stronger out-of-distribution conditions.
- **Not claimed**: state-of-the-art robust RL, calibrated frequentist coverage on true $Q$-values, that this is the CPM-derived loss from the first half. These are open.

### Sources for every quoted number

- LunarLander OOD eval (calm-trained interval-DQN): `checkpoints/per_c/ood_eval_extended.json`.
- LunarLander OOD eval (range-trained credal interval-DQN): `Credal LunarLander/results/per_c_windy/ood_eval.json`.
- LunarLander OOD eval (scalar DQN baseline, 5 seeds × 30 episodes): `checkpoints/standard_per_seed/standard_ood_eval.json`.
- Training-dynamics dashboard data: `talk/training_dashboard_log_v2.json` (1 seed × 800 episodes, re-instrumented to log per-episode loss term decomposition).

---

## Pre-ML bridge slides (deck slides 18–26)

*Not part of the speaker script proper — short notes for the transition into the ML half.*

**Deck slide 18 — Control problems.** Brief: the broader class of problems we're going to specialise to in the ML half.

**Deck slide 19 — Reinforcement learning as a control problem.** Standard agent / environment diagram. Set up: the agent picks actions $a_t$ from a policy; the environment returns next state and reward; the agent has internal value estimates $Q(s, a)$ trained from experience. One sentence on each.

**Deck slide 20 — Deep $Q$-learning, the standard recipe.** Explainer for a non-ML audience. A neural network maps state $s$ to one scalar $Q$-value per action. Trained by minimising the squared error between $Q(s,a)$ and a one-step Bellman target $r + \gamma \max_{a'} Q(s', a')$. Action picked by $\arg\max$. This is the scalar-output baseline we are about to modify — change the output head to two scalars per action, change the loss to an interval-scoring loss, and the rest of the machinery stays the same.

**Deck slide 21 — A different theoretical role for loss functions.** The pivot. Classical scoring rules evaluate forecasts after the fact. In machine learning, the loss is part of training — it determines what gets learned. So a loss is a control lever: a way of incentivising a system to represent uncertainty the way we want.

**Deck slide 22 — What we want a loss function to incentivise.** Coverage of the target plus tightness of the interval — the two axes our loss will trade off.

**Deck slide 23 — Bridge: the interval $Q$ loss used in practice.** The actual loss formula. Acknowledge that this is an engineering interval-scoring loss, not derived from the first-half CPM theory — closing that gap is an open problem.

**Deck slide 24 — Standard DQN: how the loss closes the control loop.** Diagram of vanilla DQN's training loop. Environment supplies $r$ and $s'$; Bellman target $T = r + \gamma \max_{a'} Q(s', a')$; MSE loss pulls $Q(s, a)$ toward $T$; gradient update; policy is greedy $\arg\max$. The whole loop produces one point estimate per (state, action); no representation of how confident the model is. This is the baseline diagram — same layout as the next slide so the audience can read what changes.

**Deck slide 25 — Credal interval DQN: what changes.** Same diagram, with red stars marking everything that's been added or modified. *Modified:* the $Q$-net now outputs $[Q_\ell, Q_u]$ per action; the Bellman target is an *interval* $[T_\ell, T_u]$ (both bounds bootstrap from the next state's Hurwicz argmax action $a^*$); the loss is an interval-scoring loss averaged over $N$ samples drawn uniformly from $[T_\ell, T_u]$; the policy uses Hurwicz selection. *New:* a coverage tracker measures the running rate at which sampled targets fall inside the predicted interval; an adaptive $t$ raises/lowers the coverage knob in response, closing a self-regulating feedback loop. Everything else — environment, gradient update — is unchanged from vanilla DQN.

**Deck slide 26 — Training dynamics: the coverage feedback loop.** Animated five-panel dashboard of an actual single-seed credal-LunarLander training run: reward, mean interval width, observed coverage, the tracker's $t$, and the loss decomposed into miss term and width term. Use this to make the feedback loop visually concrete — coverage stays parked at $0.85$ because the tracker keeps adjusting $t$; width responds to $t$; the loss is dominated by the miss term because $t \approx 0.93$ weights it heavily.

---

## Slide 1 — Hook  *(deck slide 27)*

> A drone has to land somewhere with wind that nobody measured beforehand. A car has to drive on a road whose friction nobody measured. An aircraft has to avoid an intruder whose pilot is going to do whatever they're going to do.
>
> These are decisions under uncertainty about the rules of the environment itself, not just uncertainty about the current state. That's the setting for the rest of the talk.

## Slide 2 — From scoring rules to RL  *(deck slide 28)*

> The first half argued that loss functions can be designed so that their optima are non-point-valued predictive objects — credal predictive models — and that this is the right way to extract honest uncertainty from data.
>
> The question in this half: can we apply the same instinct in reinforcement learning? Specifically: can a $Q$-network's loss be designed so that the network outputs an interval per action, $[Q_\ell, Q_u]$, and the width of that interval is something useful — a learned signal of when the model is more or less sure?
>
> This is what the rest of the slides take seriously, on the standard `LunarLander` benchmark with wind as a parameter.

## Slide 3 — Two ways imprecision enters the model  *(deck slide 29)*

> This is the conceptual centre of the talk. Imprecision in this framework enters at *two* genuinely different points, and I want them held apart.
>
> **① Modeller's input.** An uncertainty set on the environment, supplied at problem-setup time — like saying "the wind during deployment could be anywhere in $[0, 15]$". This is what the modeller asserts about the world *before* training. In our pipeline, it's used as a domain-randomisation sampler: each training episode draws a fresh wind value from the set and the agent trains under those dynamics.
>
> **② Learned interval $Q$.** The interval $[Q_\ell(s, a), Q_u(s, a)]$ that the trained network outputs at deployment. This varies with the input — wider in states the network is less confident about. It's a model-internal signal that a scalar $Q$-network throws away entirely.
>
> The first kind is structured input — what the modeller hands the agent. The second kind is *learned* — extracted from the training data by the loss function we'll see in two slides. Both feed the one risk knob the agent uses at deployment. The rest of the talk shows what each of these does.

## Slide 4 — The setting: lunar lander  *(deck slide 30)*

> The environment is the standard `LunarLander` benchmark — continuous 8-dimensional state, four discrete actions, with wind as a parameter that varies between $[0, 15]$ across episodes during training.
>
> The comparison throughout is against a scalar DQN trained on the same recipe but without the interval head. We're not doing formal robust planning — what we have is domain randomisation on top of an interval-scoring loss.

## Slide 5 — The architecture: interval $Q$-network  *(deck slide 31)*

> Standard DQN architecture: a feed-forward MLP on the lander. The only change from a vanilla DQN is the output head — two scalars per action instead of one, parameterising the interval $[Q_\ell, Q_u]$.
>
> The width is the learned imprecision signal. The conceptual contribution is not in this architecture; it's in the loss that drives it.

## Slide 6 — The training loss  *(deck slide 32)*

> Given an interval prediction $[\ell, u]$ and a target $v$ — for us, a bootstrapped Bellman target — the loss has two terms.
>
> If $v$ is outside the predicted interval, pull the nearer endpoint toward $v$. That's the coverage term, weight $t$.
>
> Whether or not $v$ is inside, pull the farther endpoint toward $v$. That's the width-tightening term, weight $1-t$.
>
> The coverage parameter $t$ is adjusted online to target observed coverage around $0.85$.
>
> This is a coverage-and-width prediction-interval loss in the lineage of Khosravi et al.\ 2011 (LUBE) and Pearce et al.\ 2018 (Quality-Driven). Per-example interval scoring rules go back to Winkler 1972 in the statistics literature.
>
> What's distinctive isn't the form of the loss — it's the use as a $Q$-network training signal in RL with moving, possibly-interval-valued targets.

## Slide 7 — A risk knob at deployment  *(deck slide 33)*

> Given the interval, the action selection rule is the Hurwicz combination — $Q_\ell + c (Q_u - Q_\ell)$, with $c \in [0, 1]$. Hurwicz 1951, from decision theory under ignorance.
>
> At $c = 0$ the policy acts on the lower bound — pessimistic. At $c = 1$ it acts on the upper bound — optimistic. One trained network gives a one-parameter family of policies; no retraining when the risk attitude changes at deployment.

## Slide 8 — Cautious  *(deck slide 34)*

> Same network, $c = 0$. Acts on the lower bound. In this trial it lands cleanly under unseen wind, reward $+302$.
>
> I want to flag: this is an above-median trajectory I picked to illustrate the cautious-policy behaviour. The aggregate medians are on slide 16.

## Slide 9 — Daring  *(deck slide 35)*

> Same network weights, only $c$ has changed. In this trial the daring policy crashes hard — reward $-1006$. Not the typical $c=1$ rollout either; I picked a bad one to show the failure mode the knob exposes. Same caveat — aggregate medians on slide 16.

## Slide 10 — Width grows under shift  *(deck slide 36)*

> Two panels. *Left:* a wind sweep — real-valued x-axis, wind power 0 to 25. Both controllers' widths grow with wind. The shaded band marks the range-trained agent's training distribution $[0, 15]$. *Right:* discrete OOD conditions — low gravity, high gravity, the combined wind + gravity condition. Up to $+60\%$ wider at the combo.
>
> Two honest caveats. First, this is one environment with five seeds — a qualitative trend, not a calibration claim. The width is not validated as a frequentist OOD detector. Second, the mechanism could be aleatoric — bootstrap target distributions becoming noisier under OOD inputs — rather than epistemic uncertainty. We have not distinguished those.

## Slide 11 — Wind = 0 (in-distribution)  *(deck slide 37)*

> Both controllers land cleanly. Standard DQN reward $+241$ in the illustrated trial; credal interval $+271$. Single rollouts.
>
> I want to be precise: the GIF on the left is a single trial from a single scalar-DQN seed; the GIF on the right is a single trial from a credal interval-DQN seed. The 5-seed aggregate picture is on slide 16.

## Slide 12 — Wind = 15 (boundary)  *(deck slide 38)*

> Mild OOD for the scalar agent — beyond its calm training distribution. Boundary for the credal agent — the upper end of its training range. Both land. Scalar $+225$, credal $+260$ in these trials.

## Slide 13 — Wind = 25 (far OOD)  *(deck slide 39)*

> Severe OOD for both. The scalar agent in this trial fails — reward $-4$. The credal agent lands with $+232$.
>
> Single trials, again. The aggregate medians on slide 16 will show that the average story is less dramatic than this single comparison — the scalar baseline's 5-seed median at wind 25 is around $-84$, the credal-adaptive's is around $+130$. So the qualitative direction is the same; the magnitudes are smaller in expectation.

## Slide 14 — Gravity OOD on a different axis  *(deck slide 40)*

> Gravity at $-11.99$, near the floor of the simulator's gravity range. Neither agent saw varied gravity during training, so this is genuine out-of-distribution on a different axis from the one we trained against. Scalar agent at $+29$ in this trial, credal agent at $+244$.

## Slide 15 — Combo OOD (wind + gravity)  *(deck slide 41)*

> The hardest condition in the eval grid for both controllers — both OOD axes shifted at once. Scalar DQN crashes with reward $-57$ in the illustrated trial; the credal interval DQN with adaptive $c$ lands with reward $+216$.
>
> I want to be explicit about the credal selection. Across the 45 rollouts I scanned (5 model seeds × 9 rollout seeds), 19 trials solved this condition — about $42\%$. The trial in the GIF is the *lowest-reward solving trial* — a "typical successful" landing rather than a cherry-picked extreme. The 5-seed aggregate median is around $+41$ at this condition; some seeds reliably solve, others don't, and a fair chart of the average story is on the next slide.

## Slide 16 — Aggregate performance  *(deck slide 42)*

> Across six representative deployment conditions, all four of: scalar DQN (5 seeds × 30 episodes per condition), calm-trained interval-DQN best fixed-$c$, range-trained interval-DQN best fixed-$c$, range-trained adaptive-$c$.
>
> Two readings:
>
> First, the scalar baseline is well below all three interval baselines at every condition — even at in-distribution calm, where scalar DQN's 5-seed median is $+11$ while calm-trained interval-DQN is $+259$. That gap is partly a training-stability difference: vanilla DQN at this episode budget on this benchmark trains less reliably than the interval-DQN does at the same budget and recipe. I want to be honest about that — it's not pure OOD robustness.
>
> Second, among the interval variants, the range-trained adaptive-$c$ policy beats both fixed-$c$ baselines on 7 of 11 deployment conditions across the full evaluation grid. It's doing well; it's not dominating. Five seeds, no hyperparameter search.

## Slide 17 — Where this sits in the literature  *(deck slide 43)*

> Each ingredient has older roots. Interval value functions in robust and bounded-parameter MDPs — Givan–Leach–Dean 2000, Iyengar 2005, Nilim–El Ghaoui 2005. Distributional value learning in RL — C51, QR-DQN, the Bellemare and Dabney work from 2017–18. Coverage-and-width losses for neural prediction intervals — Khosravi et al.\ 2011 (LUBE), Pearce et al.\ 2018 (QD). Interval $Q$-learning for exploration — Mancuso & Asgharbeygi 2020. Hurwicz criterion — 1951.
>
> What appears distinctive in this project is the *combination*: a DQN-style value learner with an explicitly interval-valued output and a coverage-and-width loss, then using the learned width as a decision-relevant imprecision signal — bridging the first-half scoring-rule story to RL.

## Slide 18 — Takeaway  *(deck slide 44)*

> Three things to leave with.
>
> First, the conceptual bridge. The first half argued that loss design can drive imprecise-probability outputs from neural networks. This half is empirical evidence that the same instinct works in reinforcement learning: train a $Q$-network with an interval-scoring loss, get a deployment-time uncertainty signal — the interval width — that a scalar $Q$-network throws away, plus a one-parameter family of policies via the Hurwicz rule.
>
> Second, narrow empirics. On the lunar lander, the range-trained adaptive-$c$ policy beats both fixed-$c$ interval baselines on 7 of 11 conditions and clears the scalar DQN baseline at every condition. The interval width trends wider on out-of-distribution conditions. Single environment, five seeds, no hyperparameter search.
>
> Third, open. The interval-scoring loss is an engineering choice, not derived from the first-half theory. The deployment-time width signal correlates with OOD on this benchmark but is not validated as a calibrated uncertainty measure. Closing both gaps — deriving an RL loss whose Bayes optima are CPM-style objects, and validating the width as a frequentist OOD signal — is the natural next mathematical step.

---

# Speaker's notes / appendix

*Not part of the delivered script. Backup material for Q&A and post-talk follow-up.*

## A1. Pearce et al.\ 2018 (Quality-Driven loss) — comparison

**Citation.** Pearce, Brintrup, Zaki, Neely (2018). High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach. ICML / PMLR 80.

**Their setup.** Tabular regression with calibrated prediction intervals. Benchmarked on 10 UCI regression datasets. QD-Ens (5-network ensemble) reports about 11.6% narrower PIs than MVE-Ens at comparable PICP. Honest caveat in their §6.1: training is "fragile" — needs lower LR, more epochs.

**Their loss.**

$$\text{Loss}_{\text{QD}} = \text{MPIW}_{\text{capt}} + \lambda \cdot \frac{n}{\alpha(1-\alpha)} \cdot \max\!\bigl(0, (1-\alpha) - \text{PICP}\bigr)^2$$

**Comparison.**

|  | Pearce QD | Ours |
|---|---|---|
| Aggregation | Batch-level (PICP) | Per-example |
| Coverage penalty | Squared deviation of batch PICP from target | Per-example, fires when $v$ outside $[\ell, u]$ |
| Width term | $\text{MPIW}_{\text{capt}}$ — linear, captured only | $\max((v-\ell)^2,(v-u)^2)$ — squared, asymmetric |
| Coverage knob | Fixed $\lambda$ | Adaptive $t$ online toward $0.85$ |
| Target type | Fixed observations | Bootstrapped, possibly interval-valued |
| Domain | Regression | RL |
| Epistemic via | Ensemble | Single network (mechanism unclear) |

**If asked "is your loss better than QD?"** — I haven't run that comparison. The loss-design choices are motivated by the non-stationary-Bellman-target setting in RL, not by an apples-to-apples comparison with QD on regression. Both losses are in the same coverage-and-width family.

## A2. Pre-formulated answers to likely audience questions

**Q. Why not distributional RL (C51, QR-DQN, IQN)?**

Distributional RL would give a full return distribution per action — a strictly richer object than the interval. The reason I'm using intervals is that the conceptual bridge from the first half is to imprecise-probability objects (credal sets, intervals), not to precise return distributions. This is a research-direction choice, not a claim that intervals are better than distributions.

**Q. Hurwicz at $c = 0$ is just the worst-case rule. Is this Γ-maximin under a different name?**

$c = 0$ is exactly Γ-maximin, the standard robust-optimisation rule. The contribution of the full Hurwicz combination is the $c \in (0, 1)$ family, which is rare in mainstream RL but standard in the imprecise-probability decision-theory literature (Walley 1991, Troffaes 2007).

**Q. Why 85% coverage rather than 95%?**

A choice motivated by the noisier-target regime in RL — Bellman targets are themselves moving estimates, so demanding 95% coverage on them is less meaningful than in regression. The number isn't derived; an ablation across coverage targets would strengthen the empirical work.

**Q. Why didn't you ensemble?**

Computational cost in RL — training $N$ Q-networks together is expensive in sample efficiency and wall-time. Bootstrapped DQN (Osband et al.\ 2016) is the relevant precedent for ensemble-based uncertainty in RL. Pearce et al.\ explicitly need an ensemble to capture epistemic uncertainty; we observe single-network width variation on OOD without one, but we cannot yet rule out that this is just aleatoric variance of the Bellman target distribution at OOD inputs.

**Q. Vanilla DQN seems badly trained at 800 episodes — how is that a fair baseline?**

It is the same recipe and budget as the interval DQN runs — same number of seeds, same number of episodes, same hyperparameters except for the loss and head. The fact that interval DQN trains more reliably than scalar DQN at this budget is itself a finding consistent with the way distributional / interval methods often regularise learning. If the comparison were 5,000 episodes, scalar DQN would catch up more. We do not claim training-time efficiency; the chart should be read as "at a fixed standard budget, interval methods clear scalar DQN by a wide margin on this benchmark."

**Q. Are the intervals calibrated imprecise probabilities?**

No, not in the strict Walley sense. We haven't validated frequentist coverage against true $Q^\pi$, we haven't shown stability under distribution shift, we haven't represented them as lower/upper expectations on a credal set. The intervals are a learned, input-dependent width signal that the loss has driven the network to express. "Imprecise" is a fair gloss; "calibrated imprecise probabilities in Walley's sense" is not. The bridge to a calibrated credal-set semantics is open.

**Q. How does a standard DQN interact with imprecise state transitions?**

It doesn't, in any meaningful sense. The DQN objective takes its expectation over the *one* transition distribution that generated the training data; there is no slot in the architecture or the loss for $P$ itself being uncertain. Three scenarios:

1. *Single-environment training.* The Q-values are calibrated to that one dynamics. Outside it, they are silently miscalibrated; the agent does $\arg\max$ over wrong numbers with no internal signal that something is off.
2. *Domain-randomised training.* The MSE loss averages over the training distribution; the network learns a single "marginalised" Q. More robust as a point estimator, still no honest uncertainty signal — there is nothing in the output to say "I might be in an unusual environment right now."
3. *Deployment under OOD.* Same architecture, no new information; Q-values silently miscalibrate.

The principled object for imprecise transitions is the interval value function from robust MDPs (Iyengar 2005, Nilim–El Ghaoui 2005): $\underline Q(s, a) = \inf_{P \in \mathcal P} \mathbb E_P[\cdots]$ and the matching sup. A scalar Q cannot represent this. The interval head has the right output type to express such a range; whether our interval-scoring loss actually drives the network toward the Iyengar inf/sup over a meaningful credal set is the open theoretical question. Empirically on the lander we observe that width tracks variability of the Bellman target across the training range — qualitatively in the spirit of the robust-MDP picture, not a derivation of it.

**Q. Is the credal lunar lander a credal MDP?**

Strictly speaking, no. The modeller's input — wind ∈ [0, 15] — is a credal set in the modeller's-imprecision sense, and the output object is interval-valued. But the training algorithm is domain randomisation with an interval-scoring loss, not a credal Bellman backup. The Bellman target uses one specific next-state value drawn from one specific dynamics (the one we happened to sample this episode), not a sup/inf over the credal set $\mathcal P$. The intervals reflect the spread of Bellman targets across the *sampled* training distribution, not the formal robust-MDP inf/sup over $\mathcal P$. So the intervals are heuristic, not robust-MDP-derived.

## A3. Bibliography

- Khosravi, Nahavandi, Creighton, Atiya (2011). Lower upper bound estimation method for construction of neural network-based prediction intervals. *IEEE TNN* 22(3). [LUBE.]
- Pearce, Brintrup, Zaki, Neely (2018). High-Quality Prediction Intervals for Deep Learning. *ICML / PMLR 80*. [Quality-Driven loss.]
- Lakshminarayanan, Pritzel, Blundell (2017). Deep Ensembles. *NeurIPS*. [MVE-Ens baseline.]
- Winkler (1972). A decision-theoretic approach to interval estimation. *JASA* 67(337). [Per-example interval scoring rule.]
- Gneiting & Raftery (2007). Strictly proper scoring rules. *JASA* 102(477).
- Givan, Leach, Dean (2000). Bounded-parameter Markov decision processes. *Artificial Intelligence* 122(1-2).
- Iyengar (2005). Robust dynamic programming. *Math. of OR* 30(2).
- Nilim & El Ghaoui (2005). Robust control of MDPs with uncertain transition matrices. *Operations Research* 53(5).
- Bellemare, Dabney, Munos (2017). A distributional perspective on RL. *ICML*. [C51.]
- Dabney, Rowland, Bellemare, Munos (2018). Distributional RL with quantile regression. *AAAI*. [QR-DQN.]
- Osband, Blundell, Pritzel, Van Roy (2016). Deep exploration via bootstrapped DQN. *NeurIPS*.
- Mancuso & Asgharbeygi (2020). Interval Q-learning: balancing deep and wide exploration. *ALA Workshop*.
- Hurwicz (1951). Optimality criteria for decision making under ignorance. Cowles Commission DP, Statistics, No. 370.
- Walley (1991). *Statistical Reasoning with Imprecise Probabilities*. Chapman and Hall.
- Troffaes (2007). Decision making under uncertainty using imprecise probabilities. *IJAR* 45(1).
