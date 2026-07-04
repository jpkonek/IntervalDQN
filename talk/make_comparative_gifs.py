"""Comparative GIFs: calm-trained interval DQN vs credal-trained interval DQN
at three OOD strengths.

Pairs (each pair = same wind, both at c=0 cautious):
  Pair A — moderate OOD (wind = 12):
    A1_calm_w12.gif    calm-trained, c=0
    A2_credal_w12.gif  credal-trained (wind ∈ [0,15]), c=0

  Pair B — boundary of credal training distribution (wind = 18):
    B1_calm_w18.gif    calm-trained, c=0
    B2_credal_w18.gif  credal-trained, c=0

  Pair C — extreme OOD for both (wind = 25):
    C1_calm_w25.gif    calm-trained, c=0
    C2_credal_w25.gif  credal-trained, c=0
"""

from __future__ import annotations
import math
import sys
import warnings
import pathlib
import random
import numpy as np
import torch
import imageio.v2 as imageio

sys.path.insert(0, "/Users/jk13942/Documents/GitHub/IP-ML")
from lunarlander_interval_dqn import IntervalDQNAgent

import gymnasium as gym


OUT_DIR = pathlib.Path("/Users/jk13942/Documents/GitHub/IP-ML/talk/gifs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_env(wind_power: float = 0.0, gravity: float = -10.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gym.make(
            "LunarLander-v3",
            render_mode="rgb_array",
            enable_wind=(wind_power > 0.0),
            wind_power=wind_power,
            gravity=gravity,
            turbulence_power=0.0,
        )


def load_calm(c: str, model_seed: int):
    """Calm-trained interval DQN (original LunarLander)."""
    ckpt_path = f"/Users/jk13942/Documents/GitHub/IP-ML/checkpoints/per_c/c{c}_seed{model_seed}/ckpt_final.pt"
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    agent = IntervalDQNAgent(c_train=ck["args"]["c_train"], warmup_steps=0)
    agent.q_net.load_state_dict(ck["q_net_state_dict"])
    agent.target_net.load_state_dict(ck["target_net_state_dict"])
    agent.q_net.eval(); agent.target_net.eval()
    agent.total_env_steps = 1_000_000
    return agent


def load_credal(c: str, model_seed: int):
    ckpt_path = f"/Users/jk13942/Documents/GitHub/IP-ML/Credal LunarLander/results/per_c_windy/c{c}_seed{model_seed}/ckpt_final.pt"
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    agent = IntervalDQNAgent(c_train=ck["args"]["c_train"], warmup_steps=0)
    agent.q_net.load_state_dict(ck["q_net_state_dict"])
    agent.target_net.load_state_dict(ck["target_net_state_dict"])
    agent.q_net.eval(); agent.target_net.eval()
    agent.total_env_steps = 1_000_000
    return agent


def rollout_to_frames(agent, wind_power, rollout_seed, c_eval,
                      max_steps=600, downsample=2):
    np.random.seed(rollout_seed); random.seed(rollout_seed); torch.manual_seed(rollout_seed)
    env = make_env(wind_power=wind_power)
    state, _ = env.reset(seed=rollout_seed)
    frames = []
    total = 0.0
    for t in range(max_steps):
        action = agent.select_action(state, c=c_eval, force_epsilon=0.0)
        if t % downsample == 0:
            f = env.render()
            if f is not None:
                frames.append(f)
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            for _ in range(8):
                f = env.render()
                if f is not None:
                    frames.append(f)
            break
    env.close()
    return frames, total


def find_typical_rollout(load_fn, wind, c_eval, model_seeds, rollout_seeds,
                         representative_threshold=None, prefer_low=False):
    """Search (model_seed × rollout_seed) for a 'typical' rollout — one whose
    reward is close to the median of all attempts. Reduces cherry-picking."""
    rolls = []
    for ms in model_seeds:
        agent = load_fn(ms)
        for rs in rollout_seeds:
            frames, total = rollout_to_frames(agent, wind, rs, c_eval=c_eval)
            rolls.append((ms, total, len(frames), frames, rs))
    rewards = sorted([r[1] for r in rolls])
    median = rewards[len(rewards) // 2]
    if prefer_low:
        # Find a representative bad rollout: closest to median among the bad half
        bad = sorted([r for r in rolls if r[1] <= median], key=lambda r: -r[1])
        chosen = bad[0]
    elif representative_threshold is not None:
        # Find a rollout above threshold, closest to typical
        good = [r for r in rolls if r[1] >= representative_threshold]
        if good:
            chosen = min(good, key=lambda r: abs(r[1] - median))
        else:
            chosen = max(rolls, key=lambda r: r[1])
    else:
        # Most representative — closest to median
        chosen = min(rolls, key=lambda r: abs(r[1] - median))
    print(f"    median over {len(rolls)} rolls = {median:.1f}; picked ms={chosen[0]} rs={chosen[4]} R={chosen[1]:.1f}")
    return chosen


def save_gif(frames, path, fps=30):
    print(f"  saving {path.name} ({len(frames)} frames)")
    imageio.mimsave(path, frames, format="GIF", fps=fps, loop=0)


def main():
    rs_pool = [1, 3, 5, 7, 11, 17, 23, 29, 31]
    seeds = [0, 1, 2, 3, 4]

    pairs = [
        ("A", 12.0, "moderate OOD (wind = 12)"),
        ("B", 18.0, "extreme OOD (wind = 18, beyond credal range)"),
        ("C", 25.0, "extreme OOD (wind = 25)"),
    ]

    for letter, wind, label in pairs:
        print(f"\n=== Pair {letter}: {label} ===")
        print(f"\n  [{letter}1] CALM-trained (no-wind), c=0:")
        pick = find_typical_rollout(lambda ms: load_calm("0.0", ms),
                                     wind=wind, c_eval=0.0,
                                     model_seeds=seeds, rollout_seeds=rs_pool)
        save_gif(pick[3], OUT_DIR / f"{letter}1_calm_w{int(wind)}.gif")

        print(f"\n  [{letter}2] CREDAL-trained (wind ∈ [0,15]), c=0:")
        pick = find_typical_rollout(lambda ms: load_credal("0.0", ms),
                                     wind=wind, c_eval=0.0,
                                     model_seeds=seeds, rollout_seeds=rs_pool)
        save_gif(pick[3], OUT_DIR / f"{letter}2_credal_w{int(wind)}.gif")

    print(f"\n=== Comparative GIFs ===")
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix == ".gif" and (f.name.startswith("A") or f.name.startswith("B") or f.name.startswith("C")):
            print(f"  {f.name}: {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
