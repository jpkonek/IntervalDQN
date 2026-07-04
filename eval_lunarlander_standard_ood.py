"""
Out-of-distribution evaluation of *standard* (scalar) LunarLander DQN baselines.

Mirrors `eval_lunarlander_ood.py` exactly — same 11 conditions, same n=30
episodes per (seed, condition), same aggregate JSON structure — but loads
`StandardDQNAgent` checkpoints from `checkpoints/standard_per_seed/seed*/`.

The output JSON has a single policy key `"scalar_dqn"` whose value matches
the per-condition structure of `ood_eval_extended.json` exactly (so it can
be merged with that file or plotted alongside it without further surgery).

Usage:
  python eval_lunarlander_standard_ood.py \\
      --results-root checkpoints/standard_per_seed \\
      --seeds 0,1,2,3,4 \\
      --n-episodes 30 \\
      --out checkpoints/standard_per_seed/standard_ood_eval.json
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
from lunarlander_interval_dqn import StandardDQNAgent, episode_outcome

# Re-use the canonical 11-condition grid from eval_lunarlander_ood.py.
from eval_lunarlander_ood import DEFAULT_CONDITIONS


def make_env(wind_power: float = 0.0, gravity: float = -10.0,
             turbulence_power: float = 0.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gym.make(
            "LunarLander-v3",
            enable_wind=(wind_power > 0.0),
            wind_power=wind_power,
            turbulence_power=turbulence_power,
            gravity=gravity,
        )


def load_standard_agent(ckpt_path: str) -> StandardDQNAgent:
    """Loads either a plain state-dict ckpt or our wrapped per-seed ckpt."""
    obj = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    agent = StandardDQNAgent()
    if isinstance(obj, dict) and "q_net_state_dict" in obj:
        agent.q_net.load_state_dict(obj["q_net_state_dict"])
        agent.target_net.load_state_dict(obj["target_net_state_dict"])
    else:
        agent.q_net.load_state_dict(obj)
        agent.target_net.load_state_dict(obj)
    agent.q_net.eval(); agent.target_net.eval()
    return agent


def run_episode(agent: StandardDQNAgent, env, max_steps: int = 1000):
    state, _ = env.reset()
    total = 0.0
    for _ in range(max_steps):
        action = agent.select_action(state, epsilon=0.0)
        state, r, te, tr, _ = env.step(action)
        total += r
        if te or tr:
            break
    return total


def evaluate_one(ckpt: str, conditions, n_eps: int, seed: int) -> dict:
    agent = load_standard_agent(ckpt)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    out = []
    for cond in conditions:
        env = make_env(wind_power=cond.wind_power, gravity=cond.gravity)
        rewards, solves, crashes = [], 0, 0
        for _ in range(n_eps):
            r = run_episode(agent, env)
            rewards.append(r)
            outcome = episode_outcome(r)
            if outcome == "solved": solves += 1
            elif outcome == "crashed": crashes += 1
        env.close()
        out.append({
            "condition": cond.label,
            "wind_power": cond.wind_power,
            "gravity": cond.gravity,
            "is_in_dist": cond.is_in_dist,
            "ood_strength": cond.ood_strength,
            "n_episodes": n_eps,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "solve_rate": solves / n_eps,
            "crash_rate": crashes / n_eps,
        })
    return {"per_condition": out}


def stat_block(values):
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--results-root", required=True,
                    help="Root containing seed<n>/ckpt_final.pt")
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    root = pathlib.Path(args.results_root)
    out_path = pathlib.Path(args.out)

    per_seed_per_cond = {}
    for seed in seeds:
        ckpt = root / f"seed{seed}" / "ckpt_final.pt"
        if not ckpt.exists():
            print(f"  [scalar_dqn] seed {seed}: NO CKPT {ckpt}", flush=True)
            continue
        res = evaluate_one(str(ckpt), conditions=DEFAULT_CONDITIONS,
                            n_eps=args.n_episodes, seed=seed)
        for entry in res["per_condition"]:
            cond = entry["condition"]
            per_seed_per_cond.setdefault(cond, []).append(entry)
        print(f"  [scalar_dqn] seed {seed}: done", flush=True)

    agg = {}
    for cond, rows in per_seed_per_cond.items():
        rewards = [r["mean_reward"] for r in rows]
        solves = [r["solve_rate"] for r in rows]
        crashes = [r["crash_rate"] for r in rows]
        agg[cond] = {
            "wind_power": rows[0]["wind_power"],
            "gravity": rows[0]["gravity"],
            "is_in_dist": rows[0]["is_in_dist"],
            "ood_strength": rows[0]["ood_strength"],
            "reward":  stat_block(rewards),
            "solve":   stat_block(solves),
            "crash":   stat_block(crashes),
        }

    # Wrap in a single-policy aggregate to match ood_eval_extended.json shape.
    aggregate = {"scalar_dqn": agg}

    out_path.write_text(json.dumps(aggregate, indent=2))
    print(f"\nWrote {out_path}", flush=True)

    # Print table.
    print("\n=== Scalar DQN: reward by condition (median across seeds) ===")
    cond_labels = [c.label for c in DEFAULT_CONDITIONS]
    header = "policy".ljust(14) + "".join(c[:14].rjust(15) for c in cond_labels)
    print(header)
    row = "scalar_dqn".ljust(14)
    for cl in cond_labels:
        v = agg.get(cl, {}).get("reward", {}).get("median")
        row += (f"{v:>15.1f}" if v is not None else " " * 15)
    print(row)


if __name__ == "__main__":
    main()
