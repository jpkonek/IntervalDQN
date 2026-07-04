"""Evaluate a credal-trained LunarLander interval DQN.

For each candidate wind in a grid, run N episodes at the same c as training
(deployment matches training). Worst-case wind = wind with lowest mean reward.
Robust value = mean reward at worst wind.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
import warnings
import pathlib
import random

import numpy as np
import torch

sys.path.insert(0, "/Users/jk13942/Documents/GitHub/IP-ML")
from lunarlander_interval_dqn import IntervalDQNAgent

import gymnasium as gym


def make_env(wind_power: float = 0.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gym.make("LunarLander-v3",
                        enable_wind=(wind_power > 0.0),
                        wind_power=wind_power,
                        turbulence_power=0.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--c-eval", type=float, required=True,
                   help="Hurwicz c at deployment (set to training c by default).")
    p.add_argument("--n-wind", type=int, default=11,
                   help="Number of wind values in the grid.")
    p.add_argument("--wind-lo", type=float, default=0.0)
    p.add_argument("--wind-hi", type=float, default=15.0)
    p.add_argument("--n-rollouts-per-wind", type=int, default=30)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")

    agent = IntervalDQNAgent(c_train=ckpt["args"]["c_train"],
                             warmup_steps=0, warmup_epsilon=0.0,
                             width_reg=0.01, target_coverage=0.85)
    agent.q_net.load_state_dict(ckpt["q_net_state_dict"])
    agent.target_net.load_state_dict(ckpt["target_net_state_dict"])
    agent.q_net.eval()
    agent.total_env_steps = 1_000_000  # past warmup → ε=0 for eval

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    wind_grid = list(np.linspace(args.wind_lo, args.wind_hi, args.n_wind))
    per_wind = []

    print(f"Eval ckpt: {args.ckpt} at c={args.c_eval}, "
          f"{args.n_rollouts_per_wind} rollouts × {args.n_wind} wind values",
          flush=True)
    t0 = time.time()
    for wind in wind_grid:
        env = make_env(wind_power=float(wind))
        rewards = []
        crashes = 0
        solves = 0
        for _ in range(args.n_rollouts_per_wind):
            state, _ = env.reset()
            total = 0.0
            for _ in range(1000):
                action = agent.select_action(state, c=args.c_eval, force_epsilon=0.0)
                state, r, terminated, truncated, _ = env.step(action)
                total += r
                if terminated or truncated:
                    break
            rewards.append(total)
            if total < -100:
                crashes += 1
            elif total >= 200:
                solves += 1
        env.close()
        per_wind.append({
            "wind": float(wind),
            "mean_reward": float(np.mean(rewards)),
            "sem": float(np.std(rewards) / np.sqrt(len(rewards))),
            "crash_rate": crashes / args.n_rollouts_per_wind,
            "solve_rate": solves / args.n_rollouts_per_wind,
        })
        print(f"  wind={wind:5.2f}: mean_R {per_wind[-1]['mean_reward']:7.1f} "
              f"± {per_wind[-1]['sem']:.1f}  crash {crashes}/{args.n_rollouts_per_wind}  "
              f"solve {solves}/{args.n_rollouts_per_wind}", flush=True)
    elapsed = time.time() - t0

    worst = min(per_wind, key=lambda p: p["mean_reward"])
    print(f"\nWorst wind = {worst['wind']:.2f}  mean_R = {worst['mean_reward']:.2f}",
          flush=True)

    summary = {
        "ckpt": args.ckpt,
        "seed": args.seed,
        "c_eval": args.c_eval,
        "wind_grid": wind_grid,
        "per_wind": per_wind,
        "worst_wind": worst["wind"],
        "robust_value_mc": worst["mean_reward"],
        "wall_s": elapsed,
    }
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
