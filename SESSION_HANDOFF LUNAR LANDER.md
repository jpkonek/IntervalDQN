# Session Handoff — Interval DQN for Robust RL

## Project Goal

First proof of concept that **credal/interval-valued Q-networks** work in standard RL environments. Nobody has applied interval neural networks (from the imprecise probability literature) to RL value functions. This is an open niche identified between Cuzzolin's CreINNs work (classification) and the RL literature.

## Architecture: Interval DQN

Single neural network outputs **interval [lower, upper] for each action's Q-value**.

### Key components (faithful to original interval MCTS design):
- **Interval representation**: `[lower, lower + softplus(delta_raw)]` — ensures upper >= lower
- **Hurwicz action selection**: `Q = lower + c * (upper - lower)`, where c in [0,1]
  - c=1 (optimistic) -> explore: prefer actions with high upper bounds
  - c=0 (pessimistic) -> safe: prefer actions with high lower bounds
- **Interval loss with sampled targets**: Sample N=5 points across the Bellman target interval `[r + gamma*L_next, r + gamma*U_next]`, average the interval loss across them. This propagates width through the Bellman chain.
- **Adaptive coverage**: Parameter t auto-tunes to maintain ~85% target coverage
- **Width regularisation**: When coverage exceeds target, penalise width^2 to keep intervals tight

### Key design choice: Hurwicz drives exploration directly
- **No epsilon-greedy** (except brief warmup). c_train (moderate optimism) during data collection naturally selects uncertain actions.
- **Same model, different behaviour**: Train with c=0.5, evaluate at c=0.0/0.2/0.5/1.0
- This is analogous to UCB — optimism in the face of uncertainty — but from a single forward pass.

### Critical architecture detail: Per-step training
- Training happens **after every environment step** (not batched after episodes)
- **Double DQN** with **hard target updates** every 100 steps
- This was essential — batched post-episode training failed for all agents on LunarLander

## What's been built and tested

### 1. Grid World Non-Stationary Demo (`grid_demo_nonstationary.py`)
- **Status**: Complete, 6-seed robustness test done
- **Result**: Interval agent commits to safe path in 6/6 seeds after regime switch. Standard agent oscillates (safe only 47-90% of phase 2 rounds).
- **Limitation**: Benefit is decision-time caution, not better exploration. Both agents use epsilon-greedy for training.

### 2. CartPole with Dynamics Shift (`cartpole_interval_dqn.py`)
- **Status**: Complete, 3-seed runs done at c_train=1.0 and c_train=0.8
- **Result (c_train=1.0, 3 seeds)**:
  - Phase 1: Interval >= Standard >= Ensemble (~250 reward)
  - Phase 2 early: Interval c=0 (249) > Standard (238) > Ensemble (226)
  - Phase 2 late: Interval c=0 (259) > Standard (234) > Ensemble (231)
- **Limitation**: Differences are within noise (large std, 3 seeds). CartPole may be too easy.

### 3. MountainCar (`mountaincar_interval_dqn.py`)
- **Status**: Built, FAILED — all agents score -200 (nobody reaches flag)
- **Diagnosis**: Sparse reward means no Bellman target variability for intervals to track. Wide initialization (Option A) collapses too fast. Environment abandoned in favour of LunarLander.

### 4. LunarLander (`lunarlander_interval_dqn.py`) — BEST RESULTS
- **Status**: Complete, 2-seed runs (42, 123) with c_train=0.5
- **Setup**: Train 800 episodes without wind, evaluate under no wind / moderate wind (10,1) / strong wind (20,2)
- **Architecture**: Double DQN, per-step training, hard target updates every 100 steps, hidden=128, lr=1e-3

#### Results — Seed 42, Strong Wind (20, 2):

| Agent | Reward | Solve% | Crash% | Std |
|---|---|---|---|---|
| Standard DQN | 127.1 | 36.7% | 0.0% | 124.4 |
| Interval c=0.0 | 254.6 | 90.0% | 0.0% | 73.9 |
| **Interval c=0.2** | **277.0** | **100.0%** | **0.0%** | **17.8** |
| Interval c=0.5 | 269.5 | 96.7% | 0.0% | 40.9 |
| Interval c=1.0 | 242.5 | 86.7% | 0.0% | 71.6 |
| Ensemble (N=5) | -529.4 | 0.0% | 100.0% | 100.9 |

#### Results — Seed 123, Strong Wind (20, 2):

| Agent | Reward | Solve% | Crash% | Std |
|---|---|---|---|---|
| Standard DQN | 216.0 | 73.3% | 0.0% | 67.9 |
| Interval c=0.0 | 229.5 | 60.0% | 0.0% | 48.5 |
| **Interval c=0.2** | **269.9** | **93.3%** | **0.0%** | **35.8** |
| Interval c=0.5 | 265.7 | 96.7% | 0.0% | 45.9 |
| Interval c=1.0 | 259.4 | 93.3% | 0.0% | 72.0 |
| Ensemble (N=5) | -181.6 | 0.0% | 83.3% | 81.5 |

#### Key findings:
1. **Interval c=0.2 consistently achieves ~270+ reward and 93-100% solve rate** across all conditions
2. **Standard DQN is more variable**: 37-73% solve, 68-124 std deviation
3. **Interval agent learns faster**: Solving by ep 150, standard needs 250-300
4. **Lower variance**: Interval c=0.2 std ~18-36 vs Standard's ~68-126
5. **Ensemble is broken** — uses pessimistic selection (mean - std) which self-defeats training. Needs fix but not priority.
6. **Wind perturbation doesn't substantially degrade any well-trained agent** — the story is performance + stability, not robustness-under-shift

### Honest Assessment

The LunarLander results show the interval approach provides:
- **Higher solve rate** (93-100% vs 37-73% for standard)
- **Lower variance** (std ~20-40 vs ~70-130)
- **Faster learning** (solving by ep 150 vs ep 300)
- **Risk-adjustable behaviour** from a single model (c=0 cautious, c=0.2 balanced, c=1 exploratory)

The story is NOT "robustness under perturbation" (wind barely affects well-trained agents). The story IS:
- **Better sample efficiency** (Hurwicz exploration > epsilon-greedy)
- **More stable final performance** (lower variance)
- **Controllable risk** from a single model

## Research Positioning

### The gap (confirmed by literature search)
- Cuzzolin's group: CreINNs, CBDL, Credal Learning Theory — all classification/regression
- EDL (Evidential Deep Learning): exists for RL but uses Dirichlet families, credibility questioned
- **Nobody has done interval/credal Q-networks for RL**

### Paper pitch
"Credal Q-Networks: Single-Pass Uncertainty for Robust RL via Imprecise Probabilities"

### What the results show
1. Intervals enable **better sample efficiency** via Hurwicz exploration (LunarLander)
2. Intervals enable **lower-variance** final performance
3. Single forward pass — **one model, many policies** via c parameter
4. Intervals enable **committed safe decisions** under regime change (Grid World)

## Technical Detail: How the Interval Loss Works

```python
def interval_loss(prediction, target, t):
    # prediction: [lower, upper]
    # target: true scalar value
    # t: coverage parameter (adaptive, targets ~0.85)

    min_dist_sq = min((target - lower)^2, (target - upper)^2)
    max_dist_sq = max((target - lower)^2, (target - upper)^2)
    outside = (target < lower) or (target > upper)

    loss = t * (min_dist_sq if outside else 0) + (1-t) * max_dist_sq
```

- High t -> penalise targets outside interval -> intervals widen for coverage
- Low t -> penalise max distance -> intervals narrow for precision
- Adaptive t: if coverage < 85%, increase t; if coverage > 85%, decrease t

With sampled targets: instead of a single target point, sample N points across `[r + gamma*L_next, r + gamma*U_next]` and average the loss. This propagates width backwards.

## Open Issues

1. **Ensemble baseline broken**: Pessimistic selection (mean - std) during training causes self-defeating exploration. Needs separate exploration policy (epsilon-greedy for training, pessimistic for eval).
2. **Only 2 seeds tested for LunarLander**: Need 5+ seeds for statistical significance.
3. **t parameter tuning**: User flagged this — "we should also carefully consider which t parameter value we choose for the loss function, to incentivise having intervals of appropriate width for the problem." Currently using adaptive t targeting 85% coverage. May benefit from problem-specific tuning.
4. **Wind perturbation too weak**: LunarLander with wind_power=20 doesn't substantially degrade well-trained agents. For a robustness story, need a more disruptive perturbation or different environment.

## Files

```
grid_demo_nonstationary.py    # Complete, working, 6-seed results
cartpole_interval_dqn.py      # Complete, working, results noisy but positive
mountaincar_interval_dqn.py   # Built, failed (sparse reward)
lunarlander_interval_dqn.py   # Complete, BEST RESULTS (2 seeds)
SESSION_HANDOFF.md            # This file
```

## Environment

```bash
/Users/jk13942/Library/CloudStorage/OneDrive-UniversityofBristol/Documents/GitHub/IP-ML/.conda/bin/python
# Has: torch 2.2.2, gymnasium 1.2.3
```
