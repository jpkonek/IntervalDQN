"""
Per-c interval DQN on the *original* LunarLander setup.

Same architecture, hyperparameters, env, and training loop as
`lunarlander_interval_dqn.py`. The only difference: c_train varies across
runs so we can build a c-curve and check whether the high-c overestimation
pathology we observed on collision/credal-LunarLander also appears on the
no-perturbation single-environment setting.

Imports the existing `IntervalDQNAgent` and `train_episode_interval` to
guarantee identical machinery; only the orchestration around them is new.
The previously trained checkpoint at
`checkpoints/seed42_c0.5_interval.pt` is preserved — this script writes to
`checkpoints/per_c/` instead.

Usage:
  python lunarlander_interval_dqn_per_c.py --seed 0 --c-train 0.0 \
      --output-dir checkpoints/per_c/c0.0_seed0
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import random
import sys
import time
import warnings

import numpy as np
import torch
import gymnasium as gym

# Pull the original machinery in unchanged.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lunarlander_interval_dqn import (
    IntervalDQNAgent,
    train_episode_interval,
    make_env,
    evaluate_agent,
    episode_outcome,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--c-train", type=float, default=0.5)
    p.add_argument("--total-episodes", type=int, default=800)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--warmup-epsilon", type=float, default=0.8)
    p.add_argument("--width-reg", type=float, default=0.01)
    p.add_argument("--target-coverage", type=float, default=0.85)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def save_ckpt(path: pathlib.Path, agent: IntervalDQNAgent, episode: int,
              args: argparse.Namespace, log_rows: list):
    torch.save({
        "episode": episode,
        "q_net_state_dict": agent.q_net.state_dict(),
        "target_net_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "tracker_t": agent.t,
        "total_env_steps": agent.total_env_steps,
        "args": vars(args),
        "log_tail": log_rows[-50:] if log_rows else [],
    }, path)


def main():
    args = parse_args()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"[c={args.c_train} seed={args.seed}] LunarLander interval DQN, "
          f"original (no-perturbation) env, {args.total_episodes} ep",
          flush=True)

    agent = IntervalDQNAgent(
        c_train=args.c_train,
        warmup_steps=args.warmup_steps,
        warmup_epsilon=args.warmup_epsilon,
        width_reg=args.width_reg,
        target_coverage=args.target_coverage,
    )
    env = make_env()  # standard LunarLander, no wind, normal gravity

    log_rows = []
    log_file = log_path.open("w")
    t0 = time.time()
    rewards_window: list[float] = []

    for ep in range(args.total_episodes):
        reward = train_episode_interval(env, agent)
        rewards_window.append(reward)
        if len(rewards_window) > 50:
            rewards_window = rewards_window[-50:]

        if (ep + 1) % args.log_every == 0:
            mean_r50 = float(np.mean(rewards_window))
            entry = {
                "episode": ep + 1,
                "env_steps": agent.total_env_steps,
                "reward": float(reward),
                "mean_reward_50": mean_r50,
                "tracker_t": float(agent.t),
                "wall_s": time.time() - t0,
            }
            log_rows.append(entry)
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            print(f"[c={args.c_train} seed={args.seed}] "
                  f"ep {ep+1}/{args.total_episodes}  "
                  f"R(50) {mean_r50:7.1f}  t {agent.t:.3f}  "
                  f"steps {agent.total_env_steps}", flush=True)

        if (ep + 1) % args.checkpoint_every == 0:
            save_ckpt(out_dir / f"ckpt_ep{ep+1:05d}.pt", agent, ep + 1, args, log_rows)
            save_ckpt(out_dir / "ckpt_latest.pt", agent, ep + 1, args, log_rows)

    save_ckpt(out_dir / "ckpt_final.pt", agent, args.total_episodes, args, log_rows)
    save_ckpt(out_dir / "ckpt_latest.pt", agent, args.total_episodes, args, log_rows)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(log_rows, f)
    log_file.close()

    env.close()
    print(f"[c={args.c_train} seed={args.seed}] DONE  ep {args.total_episodes}  "
          f"wall {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
