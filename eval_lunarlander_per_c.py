"""
Per-c evaluation of LunarLander interval DQN models.

For each (c, seed) checkpoint produced by `lunarlander_interval_dqn_per_c.py`,
run N evaluation episodes on the standard (no-perturbation) LunarLander
with action selection at the *training* c. Aggregate mean reward, std,
solve rate, crash rate per c.

This mirrors the pattern used on collision/credal-LunarLander: per-c
training, evaluate at the training c, look at the c-curve.
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import random
import statistics
import sys
import warnings

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lunarlander_interval_dqn import IntervalDQNAgent, make_env, episode_outcome


def evaluate_one(ckpt_path: str, c_eval: float, n_episodes: int = 100,
                 seed: int = 0) -> dict:
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    agent = IntervalDQNAgent(c_train=ck["args"]["c_train"], warmup_steps=0)
    agent.q_net.load_state_dict(ck["q_net_state_dict"])
    agent.target_net.load_state_dict(ck["target_net_state_dict"])
    agent.q_net.eval(); agent.target_net.eval()
    agent.total_env_steps = 1_000_000  # past warmup ⇒ ε = 0

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = make_env()
    rewards = []
    outcomes = {"solved": 0, "crashed": 0, "partial": 0}
    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0.0
        for _ in range(1000):
            action = agent.select_action(state, c=c_eval, force_epsilon=0.0)
            state, r, te, tr, _ = env.step(action)
            total += r
            if te or tr:
                break
        rewards.append(total)
        outcomes[episode_outcome(total)] += 1
    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "solve_rate": outcomes["solved"] / n_episodes,
        "crash_rate": outcomes["crashed"] / n_episodes,
        "rewards": rewards,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-values", type=str, default="0.0,0.2,0.5,0.8,1.0")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--results-root",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "checkpoints/per_c"))
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    c_values = [float(c) for c in args.c_values.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    root = pathlib.Path(args.results_root)
    out_path = pathlib.Path(args.out) if args.out else root / "c_curve.json"

    per_c: dict = {}
    for c in c_values:
        per_seed = []
        for seed in seeds:
            run_dir = root / f"c{c}_seed{seed}"
            ckpt = run_dir / "ckpt_final.pt"
            if not ckpt.exists():
                print(f"  c={c} seed={seed}: NO CHECKPOINT", flush=True)
                continue
            res = evaluate_one(str(ckpt), c, args.n_episodes, seed)
            per_seed.append({"seed": seed, **res})
            print(f"  c={c} seed={seed}: R={res['mean_reward']:7.1f} "
                  f"± {res['std_reward']:5.1f}  solve={res['solve_rate']:.0%}  "
                  f"crash={res['crash_rate']:.0%}", flush=True)
        if per_seed:
            vals = [p["mean_reward"] for p in per_seed]
            per_c[str(c)] = {
                "c": c,
                "n_seeds": len(per_seed),
                "values": vals,
                "median": statistics.median(vals),
                "mean": statistics.mean(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "per_seed": per_seed,
            }

    print("\n=== c-curve (LunarLander interval DQN, original no-perturbation env) ===",
          flush=True)
    print(f"{'c':>6} {'n':>3} {'median':>10} {'mean':>10} {'min':>10} {'max':>10} {'stdev':>10}",
          flush=True)
    cs_sorted = sorted(per_c.keys(), key=lambda x: float(x))
    for c_str in cs_sorted:
        d = per_c[c_str]
        print(f"{d['c']:>6} {d['n_seeds']:>3} {d['median']:>10.2f} "
              f"{d['mean']:>10.2f} {d['min']:>10.2f} {d['max']:>10.2f} "
              f"{d['stdev']:>10.2f}", flush=True)

    out = {"c_values": c_values, "seeds": seeds, "per_c": per_c}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
