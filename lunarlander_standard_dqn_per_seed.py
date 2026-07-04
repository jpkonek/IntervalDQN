"""
Per-seed *standard* (scalar) DQN training on the original LunarLander.

Companion to `lunarlander_interval_dqn_per_c.py`: identical training loop,
identical env, identical hyperparameters — but using `StandardDQNAgent`
(scalar Q-output, no interval head, no Hurwicz machinery) rather than
`IntervalDQNAgent`. Produces the proper baseline column that the talk's
LunarLander aggregate slides have been missing.

Usage:
  python lunarlander_standard_dqn_per_seed.py --seed 0 \\
      --output-dir checkpoints/standard_per_seed/seed0

Single existing `checkpoints/seed42_c0.5_standard.pt` is left untouched;
this script writes to `checkpoints/standard_per_seed/`.
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lunarlander_interval_dqn import (
    StandardDQNAgent,
    train_episode_standard,
    make_env,
    evaluate_agent,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-episodes", type=int, default=800)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def save_ckpt(path: pathlib.Path, agent: StandardDQNAgent, episode: int,
              args: argparse.Namespace, log_rows: list):
    torch.save({
        "episode": episode,
        "q_net_state_dict": agent.q_net.state_dict(),
        "target_net_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
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

    print(f"[std seed={args.seed}] LunarLander standard DQN, "
          f"original (no-perturbation) env, {args.total_episodes} ep",
          flush=True)

    agent = StandardDQNAgent()
    env = make_env()  # standard LunarLander, no wind, normal gravity

    log_rows = []
    log_file = log_path.open("w")
    t0 = time.time()

    for ep in range(args.total_episodes):
        # Match the epsilon schedule used in lunarlander_interval_dqn:run_experiment
        epsilon = max(0.01, 1.0 - ep / 400)
        ep_reward = train_episode_standard(env, agent, epsilon)

        row = {
            "episode": ep + 1,
            "reward": float(ep_reward),
            "epsilon": float(epsilon),
            "wall_s": time.time() - t0,
        }
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()
        log_rows.append(row)

        if (ep + 1) % args.log_every == 0:
            recent = [r["reward"] for r in log_rows[-args.log_every:]]
            print(f"[std seed={args.seed}] "
                  f"ep {ep+1}/{args.total_episodes}  "
                  f"reward(last {args.log_every} avg) = {np.mean(recent):7.1f}  "
                  f"eps = {epsilon:.3f}  "
                  f"wall = {(time.time()-t0)/60:.1f} min",
                  flush=True)

        if (ep + 1) % args.checkpoint_every == 0:
            save_ckpt(out_dir / f"ckpt_ep{ep+1:05d}.pt", agent, ep + 1, args, log_rows)
            save_ckpt(out_dir / "ckpt_latest.pt", agent, ep + 1, args, log_rows)

    env.close()
    log_file.close()

    save_ckpt(out_dir / "ckpt_final.pt", agent, args.total_episodes, args, log_rows)
    save_ckpt(out_dir / "ckpt_latest.pt", agent, args.total_episodes, args, log_rows)

    # Quick post-training evaluation at calm conditions (sanity check).
    eval_env = make_env()
    eval_stats = evaluate_agent(eval_env, agent, n_episodes=30)
    eval_env.close()

    with (out_dir / "metrics.json").open("w") as f:
        json.dump({
            "seed": args.seed,
            "total_episodes": args.total_episodes,
            "eval_calm": eval_stats,
            "wall_total_s": time.time() - t0,
        }, f, indent=2)

    print(f"[std seed={args.seed}] DONE  ep {args.total_episodes}  "
          f"eval_calm_mean = {eval_stats['mean']:.1f}  "
          f"solve = {eval_stats['solve_rate']:.2f}  "
          f"wall = {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
