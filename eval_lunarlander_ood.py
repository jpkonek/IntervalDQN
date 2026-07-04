"""
Out-of-distribution evaluation of LunarLander interval-DQN per-c models.

Two scientific questions, both tested rigorously across 5 seeds × 5 c values:

  Q1.  Does the predicted interval *width* grow under OOD conditions?
       → calibration of the uncertainty signal.

  Q2.  Does the cautious policy ($c=0$) outperform the daring one ($c=1$)
       at far-OOD conditions, while doing comparable in-distribution?
       → does the signal lead to better decisions under OOD.

Optionally also evaluates an *adaptive c* policy:
  $c(s) = c_{lo} + \sigma(-k(\bar{w}(s) - w_{mid}))(c_{hi} - c_{lo})$,
where $\bar w$ is the mean predicted Q-width across actions at state s.
Wide intervals → low c (cautious); narrow → higher c. Same shape as the
original `lunarlander_interval_dqn.adaptive_c` static method.

OOD axes evaluated:
  • wind_power   ∈ {0, 5, 10, 15, 20}  with enable_wind=(power>0)
  • gravity      ∈ {-10, -3, -11.9}    (in-distribution + low-grav + high-grav)

Per (model, condition), records: mean reward, solve rate, crash rate,
and mean predicted interval width across visited states.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import pathlib
import random
import statistics
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lunarlander_interval_dqn import IntervalDQNAgent, episode_outcome


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


@dataclass
class Condition:
    label: str
    wind_power: float = 0.0
    gravity: float = -10.0
    is_in_dist: bool = False
    ood_strength: float = 0.0   # ordinal "distance from training dist"


# Default OOD grid. Gravity constrained to LunarLander's [-12, 0].
# Calibrated to test both calm-trained models (every non-calm point is OOD)
# and credal-trained models (wind ∈ [0, 15] in-distribution; wind 20-25 OOD;
# any gravity ≠ −10 OOD).
DEFAULT_CONDITIONS = [
    Condition("ID  (calm, normal grav)",        0.0, -10.0,  True, 0),
    Condition("OOD wind = 5",                   5.0, -10.0,  False, 1),
    Condition("OOD wind = 10",                 10.0, -10.0,  False, 2),
    Condition("OOD wind = 15",                 15.0, -10.0,  False, 3),
    Condition("OOD wind = 20",                 20.0, -10.0,  False, 4),
    Condition("OOD wind = 25",                 25.0, -10.0,  False, 5),
    Condition("OOD low gravity (-3)",           0.0,  -3.0,  False, 1),
    Condition("OOD grav -11.9",                 0.0, -11.9,  False, 2),
    Condition("OOD grav -11.99",                0.0, -11.99, False, 3),
    Condition("OOD wind 15 + grav -11.9",      15.0, -11.9,  False, 5),
    Condition("OOD wind 25 + grav -11.99",     25.0, -11.99, False, 9),
]


def adaptive_c(mean_width: float, c_low: float = 0.0, c_high: float = 0.3,
               w_mid: float = 4.0, k: float = 2.0) -> float:
    """Wide intervals → low c (cautious); narrow → c_high. Clamped to avoid
    overflow when widths drift large under OOD."""
    z = max(min(k * (mean_width - w_mid), 50.0), -50.0)
    t = 1.0 / (1.0 + math.exp(z))
    return c_low + t * (c_high - c_low)


def mean_width_at(agent: IntervalDQNAgent, state) -> float:
    s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        L, U = agent.q_net(s)
    return float((U - L).mean().item())


def run_episode(agent: IntervalDQNAgent, env, c_eval, *, adaptive: bool = False,
                max_steps: int = 1000):
    """Returns (total_reward, mean_width_over_trajectory, terminated_well)."""
    state, _ = env.reset()
    total = 0.0
    widths = []
    for t in range(max_steps):
        w_now = mean_width_at(agent, state)
        widths.append(w_now)
        if adaptive:
            c_use = adaptive_c(w_now)
        else:
            c_use = c_eval
        action = agent.select_action(state, c=c_use, force_epsilon=0.0)
        state, r, te, tr, _ = env.step(action)
        total += r
        if te or tr:
            break
    return total, float(np.mean(widths)) if widths else 0.0


def load_agent(ckpt_path: str) -> tuple[IntervalDQNAgent, dict]:
    """Load either an old plain-state-dict ckpt (e.g. seed42_c0.5_interval.pt)
    or a new wrapped-dict ckpt produced by lunarlander_interval_dqn_per_c.py."""
    obj = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    if isinstance(obj, dict) and "q_net_state_dict" in obj:
        c_train = obj["args"]["c_train"]
        agent = IntervalDQNAgent(c_train=c_train, warmup_steps=0)
        agent.q_net.load_state_dict(obj["q_net_state_dict"])
        agent.target_net.load_state_dict(obj["target_net_state_dict"])
        meta = {"c_train": c_train, "format": "wrapped"}
    else:
        agent = IntervalDQNAgent(c_train=0.5, warmup_steps=0)
        agent.q_net.load_state_dict(obj)
        agent.target_net.load_state_dict(obj)
        meta = {"c_train": 0.5, "format": "plain"}
    agent.q_net.eval(); agent.target_net.eval()
    agent.total_env_steps = 1_000_000
    return agent, meta


def evaluate_one(ckpt: str, c_eval: float, conditions, n_eps: int, seed: int,
                 use_adaptive: bool = False) -> dict:
    agent, _ = load_agent(ckpt)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    out = []
    for cond in conditions:
        env = make_env(wind_power=cond.wind_power, gravity=cond.gravity)
        rewards, widths, solves, crashes = [], [], 0, 0
        for _ in range(n_eps):
            r, w = run_episode(agent, env, c_eval, adaptive=use_adaptive)
            rewards.append(r); widths.append(w)
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
            "mean_width": float(np.mean(widths)),
            "solve_rate": solves / n_eps,
            "crash_rate": crashes / n_eps,
        })
    return {"per_condition": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-values", type=str, default="0.0,0.2,0.5,0.8,1.0")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--results-root", required=True,
                    help="Root containing c<v>_seed<n>/ckpt_final.pt")
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--include-adaptive", action="store_true",
                    help="Also evaluate the adaptive-c policy (width-driven c).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    c_values = [float(c) for c in args.c_values.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    root = pathlib.Path(args.results_root)
    out_path = pathlib.Path(args.out)

    aggregate = {}  # (c_label) → {condition_label: stats over seeds}

    def stat_block(values):
        return {
            "n": len(values),
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values),
        }

    # Define the policy variants we evaluate (each gets its own row in agg).
    policies = [{"label": f"c={c}", "c_eval": c, "adaptive": False} for c in c_values]
    if args.include_adaptive:
        policies.append({"label": "adaptive_c", "c_eval": None, "adaptive": True})

    for pol in policies:
        # Pick which models to use for this policy: per-c-trained at the
        # nearest c (for static c_eval) or use c=0.5 models for adaptive
        # (matches the original adaptive-c experiment setup).
        if pol["adaptive"]:
            train_c_for_eval = 0.5
        else:
            train_c_for_eval = pol["c_eval"]

        per_seed_per_cond = {}
        for seed in seeds:
            run_dir = root / f"c{train_c_for_eval}_seed{seed}"
            ckpt = run_dir / "ckpt_final.pt"
            if not ckpt.exists():
                print(f"  [{pol['label']}] seed {seed}: NO CKPT {ckpt}", flush=True)
                continue
            res = evaluate_one(str(ckpt),
                               c_eval=pol["c_eval"] if not pol["adaptive"] else 0.0,
                               conditions=DEFAULT_CONDITIONS,
                               n_eps=args.n_episodes, seed=seed,
                               use_adaptive=pol["adaptive"])
            for entry in res["per_condition"]:
                cond = entry["condition"]
                per_seed_per_cond.setdefault(cond, []).append(entry)
            print(f"  [{pol['label']}] seed {seed}: done", flush=True)

        agg = {}
        for cond, rows in per_seed_per_cond.items():
            rewards = [r["mean_reward"] for r in rows]
            widths = [r["mean_width"] for r in rows]
            solves = [r["solve_rate"] for r in rows]
            crashes = [r["crash_rate"] for r in rows]
            agg[cond] = {
                "wind_power": rows[0]["wind_power"],
                "gravity": rows[0]["gravity"],
                "is_in_dist": rows[0]["is_in_dist"],
                "ood_strength": rows[0]["ood_strength"],
                "reward":  stat_block(rewards),
                "width":   stat_block(widths),
                "solve":   stat_block(solves),
                "crash":   stat_block(crashes),
            }
        aggregate[pol["label"]] = agg

    out_path.write_text(json.dumps(aggregate, indent=2))
    print(f"\nWrote {out_path}", flush=True)

    # Print table.
    print("\n=== Reward by policy × condition (median across seeds) ===")
    cond_labels = [c.label for c in DEFAULT_CONDITIONS]
    header = "policy".ljust(14) + "".join(c[:14].rjust(15) for c in cond_labels)
    print(header)
    for pol_label, agg in aggregate.items():
        row = pol_label.ljust(14)
        for cl in cond_labels:
            v = agg.get(cl, {}).get("reward", {}).get("median")
            row += (f"{v:>15.1f}" if v is not None else " " * 15)
        print(row)

    print("\n=== Mean predicted interval width by policy × condition ===")
    print(header)
    for pol_label, agg in aggregate.items():
        row = pol_label.ljust(14)
        for cl in cond_labels:
            v = agg.get(cl, {}).get("width", {}).get("median")
            row += (f"{v:>15.3f}" if v is not None else " " * 15)
        print(row)


if __name__ == "__main__":
    main()
