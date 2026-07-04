# IntervalDQN

Interval-valued DQN for robust reinforcement learning, using imprecise probability
theory. A single network predicts an interval [lower, upper] for each action's
Q-value; action selection uses the Hurwicz criterion
`Q = lower + c * (upper - lower)` with a risk parameter `c` in [0, 1]
(c=0 pessimistic/safe, c=1 optimistic/exploratory).

Code migrated from the IP-ML repository. See
[SESSION_HANDOFF LUNAR LANDER.md](<SESSION_HANDOFF LUNAR LANDER.md>) for the full
project background and design notes.

## Layout

- `lunarlander_interval_dqn.py` — main implementation: Interval DQN vs Standard DQN
  vs Ensemble DQN on LunarLander-v3 with wind/gravity perturbations
- `lunarlander_interval_dqn_per_c.py`, `launch_lunarlander_per_c.sh` — per-c training
  sweep (c in {0.0, 0.2, 0.5, 0.8, 1.0} x 5 seeds)
- `lunarlander_standard_dqn_per_seed.py`, `launch_standard_per_seed.sh` — standard DQN
  baseline per-seed training
- `eval_lunarlander_ood.py`, `eval_lunarlander_per_c.py`,
  `eval_lunarlander_standard_ood.py` — out-of-distribution evaluation (wind, gravity)
- `render_comparison.py` — side-by-side landing GIFs (standard vs interval agent)
- `cartpole_interval_dqn.py`, `mountaincar_interval_dqn.py` — Interval DQN on other
  classic-control environments
- `Credal LunarLander/` — credal (set-valued) extension built on the interval DQN
- `checkpoints/per_c/` — trained per-c models (local only, gitignored)
- `talk/` — Helsinki talk (`slides-helsinki.html` + assets) and the GIF/dashboard
  scripts that generate its figures from trained checkpoints
- `eval_results_100ep*.json`, `*_seed42.png` — saved evaluation results and plots

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Building `box2d-py` (LunarLander physics) requires `swig`: `brew install swig`.

## Run

```bash
source .venv/bin/activate

# Train + evaluate all three agents on LunarLander
python lunarlander_interval_dqn.py

# Single run
python lunarlander_interval_dqn.py --single --seed 42
```
