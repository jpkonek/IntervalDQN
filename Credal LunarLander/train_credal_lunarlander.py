"""
Credal interval DQN on LunarLander with a credal set over wind_power.

Setup:
- Credal set: wind_power ∈ [WIND_LO, WIND_HI]. Each episode samples a wind
  uniformly from this interval and trains under the resulting POMDP.
- Architecture: identical to lunarlander_interval_dqn.IntervalDQNAgent.
- Training: per-c (c_train = c_target = c), per-seed; the c-curve is built
  by training one model per c value and evaluating each at *its own* c.

Mirrors the collision/Credal DQN per-c experimental design but on a
continuous-state POMDP that PIP cannot evaluate.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import pathlib
import random
import warnings

import numpy as np
import torch

# Import the existing interval DQN agent.
sys.path.insert(0, "/Users/jk13942/Documents/GitHub/IP-ML")
from lunarlander_interval_dqn import IntervalDQNAgent, episode_outcome

import gymnasium as gym


def make_env(wind_power: float = 0.0):
    """Active-wind LunarLander. enable_wind=True is required — without it
    the wind_power kwarg is silently ignored by Gymnasium's LunarLander-v3."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gym.make("LunarLander-v3",
                        enable_wind=(wind_power > 0.0),
                        wind_power=wind_power,
                        turbulence_power=0.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-episodes", type=int, default=800)
    p.add_argument("--c-train", type=float, default=0.5)
    p.add_argument("--wind-lo", type=float, default=0.0)
    p.add_argument("--wind-hi", type=float, default=15.0)
    p.add_argument("--checkpoint-every", type=int, default=200,
                   help="Episodes between intermediate checkpoints.")
    p.add_argument("--log-every", type=int, default=10,
                   help="Episodes between log entries.")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def save_checkpoint(path, agent, episode, args, log_rows):
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


def train_episode_credal(agent, args, rng):
    """One episode: sample wind ~ Uniform[wind_lo, wind_hi], make env, train."""
    wind = float(rng.uniform(args.wind_lo, args.wind_hi))
    env = make_env(wind_power=wind)
    state, _ = env.reset()
    total_reward = 0.0
    for _ in range(1000):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.push(state, action, reward, next_state, float(done))
        agent.total_env_steps += 1
        agent.train_step()
        state = next_state
        total_reward += reward
        if done:
            break
    env.close()
    return total_reward, wind


def main():
    args = parse_args()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"[c={args.c_train} seed={args.seed}] credal LunarLander, "
          f"wind ∈ [{args.wind_lo}, {args.wind_hi}], {args.total_episodes} ep",
          flush=True)

    agent = IntervalDQNAgent(c_train=args.c_train,
                             warmup_steps=5000, warmup_epsilon=0.8,
                             width_reg=0.01, target_coverage=0.85)

    log_rows = []
    log_file = log_path.open("w")
    t0 = time.time()
    rewards_window = []

    for ep in range(args.total_episodes):
        reward, wind = train_episode_credal(agent, args, rng)
        rewards_window.append(reward)
        if len(rewards_window) > 50:
            rewards_window = rewards_window[-50:]

        if (ep + 1) % args.log_every == 0:
            mean_r50 = float(np.mean(rewards_window))
            entry = {
                "episode": ep + 1,
                "env_steps": agent.total_env_steps,
                "wind_this_ep": wind,
                "reward": float(reward),
                "mean_reward_50": mean_r50,
                "tracker_t": float(agent.t),
                "wall_s": time.time() - t0,
            }
            log_rows.append(entry)
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            print(f"[c={args.c_train} seed={args.seed}] ep {ep+1}/{args.total_episodes} "
                  f"R(50) {mean_r50:7.1f}  t {agent.t:.3f}  steps {agent.total_env_steps}",
                  flush=True)

        if (ep + 1) % args.checkpoint_every == 0:
            save_checkpoint(out_dir / f"ckpt_ep{ep+1:05d}.pt",
                            agent, ep + 1, args, log_rows)
            save_checkpoint(out_dir / "ckpt_latest.pt",
                            agent, ep + 1, args, log_rows)

    save_checkpoint(out_dir / "ckpt_final.pt",
                    agent, args.total_episodes, args, log_rows)
    save_checkpoint(out_dir / "ckpt_latest.pt",
                    agent, args.total_episodes, args, log_rows)
    log_file.close()
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(log_rows, f)
    print(f"[c={args.c_train} seed={args.seed}] DONE  ep {args.total_episodes}  "
          f"wall {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
