"""
Render side-by-side GIF comparison of Standard DQN vs Interval DQN landing.

Creates animated GIFs showing the lander's behaviour under different conditions:
- Normal (no wind)
- Strong wind
- Low gravity
- Extreme combo

Usage:
    python render_comparison.py --seed 42 --c_train 0.5
"""

import torch
import numpy as np
import gymnasium as gym
import warnings
import os
import sys
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

from lunarlander_interval_dqn import (
    load_agents, IntervalDQNAgent, make_env
)


def record_episode(agent, env, c=None, max_steps=500):
    """Record frames from one episode. Returns (frames, total_reward, outcome)."""
    state, _ = env.reset()
    frames = [env.render()]
    total_reward = 0

    for _ in range(max_steps):
        if c is not None and isinstance(agent, IntervalDQNAgent):
            action = agent.select_action(state, c=c, force_epsilon=0.0)
        else:
            action = agent.select_action(state, epsilon=0.0)

        state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += reward
        if terminated or truncated:
            # Add a few extra frames so we see the landing/crash
            for _ in range(15):
                frames.append(env.render())
            break

    outcome = "SOLVED" if total_reward >= 200 else ("CRASHED" if total_reward <= -100 else "PARTIAL")
    return frames, total_reward, outcome


def add_label(frame, text, position="top", color=(255, 255, 255)):
    """Add text label to a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    # Use default font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except (IOError, OSError):
        font = ImageFont.load_default()

    if position == "top":
        xy = (10, 5)
    else:
        xy = (10, frame.shape[0] - 25)

    # Draw text with black outline for readability
    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (-2,0), (2,0), (0,-2), (0,2)]:
        draw.text((xy[0]+dx, xy[1]+dy), text, fill=(0, 0, 0), font=font)
    draw.text(xy, text, fill=color, font=font)

    return np.array(img)


def make_side_by_side(frames_left, frames_right, label_left, label_right,
                       reward_left, reward_right, outcome_left, outcome_right):
    """Combine two frame sequences into side-by-side frames."""
    # Pad shorter sequence
    max_len = max(len(frames_left), len(frames_right))
    while len(frames_left) < max_len:
        frames_left.append(frames_left[-1])
    while len(frames_right) < max_len:
        frames_right.append(frames_right[-1])

    combined = []
    for i, (fl, fr) in enumerate(zip(frames_left, frames_right)):
        # Add labels
        color_l = (255, 100, 100) if outcome_left == "CRASHED" else (100, 255, 100) if outcome_left == "SOLVED" else (255, 255, 100)
        color_r = (255, 100, 100) if outcome_right == "CRASHED" else (100, 255, 100) if outcome_right == "SOLVED" else (255, 255, 100)

        fl = add_label(fl, f"{label_left}", "top", (200, 200, 255))
        fl = add_label(fl, f"R={reward_left:.0f} {outcome_left}", "bottom", color_l)
        fr = add_label(fr, f"{label_right}", "top", (200, 255, 200))
        fr = add_label(fr, f"R={reward_right:.0f} {outcome_right}", "bottom", color_r)

        # Add separator
        sep = np.ones((fl.shape[0], 4, 3), dtype=np.uint8) * 128
        combined.append(np.concatenate([fl, sep, fr], axis=1))

    return combined


def save_gif(frames, path, duration=50):
    """Save frames as animated GIF."""
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=duration, loop=0)
    print(f"  Saved: {path} ({len(frames)} frames)")


def render_condition(standard_agent, interval_agent, condition_name,
                     gravity, wind_power, turb_power, c_eval=0.2,
                     save_dir=".", n_attempts=5):
    """Record best episodes for a condition and save comparison GIF."""
    env_std = gym.make("LunarLander-v3", render_mode="rgb_array",
                       gravity=gravity, wind_power=wind_power,
                       turbulence_power=turb_power)
    env_int = gym.make("LunarLander-v3", render_mode="rgb_array",
                       gravity=gravity, wind_power=wind_power,
                       turbulence_power=turb_power)

    # Record multiple attempts, pick most representative
    std_episodes = []
    int_episodes = []
    for _ in range(n_attempts):
        f, r, o = record_episode(standard_agent, env_std)
        std_episodes.append((f, r, o))
        f, r, o = record_episode(interval_agent, env_int, c=c_eval)
        int_episodes.append((f, r, o))

    env_std.close()
    env_int.close()

    # Pick median-reward episode for each
    std_episodes.sort(key=lambda x: x[1])
    int_episodes.sort(key=lambda x: x[1])
    std_best = std_episodes[len(std_episodes)//2]
    int_best = int_episodes[len(int_episodes)//2]

    # Make side-by-side
    combined = make_side_by_side(
        std_best[0], int_best[0],
        "Standard DQN", f"Interval DQN (c={c_eval})",
        std_best[1], int_best[1],
        std_best[2], int_best[2]
    )

    path = os.path.join(save_dir, f"comparison_{condition_name}.gif")
    save_gif(combined, path)

    # Also save individual summaries
    std_rewards = [e[1] for e in std_episodes]
    int_rewards = [e[1] for e in int_episodes]
    std_outcomes = [e[2] for e in std_episodes]
    int_outcomes = [e[2] for e in int_episodes]
    print(f"    Standard: rewards={[f'{r:.0f}' for r in std_rewards]}, "
          f"outcomes={std_outcomes}")
    print(f"    Interval: rewards={[f'{r:.0f}' for r in int_rewards]}, "
          f"outcomes={int_outcomes}")


def main(seed=42, c_train=0.5, c_eval=0.2):
    save_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Loading agents (seed={seed}, c_train={c_train})...")
    standard_agent, interval_agent, ensemble_agent = load_agents(seed, c_train)
    standard_agent.q_net.eval()
    interval_agent.q_net.eval()
    print("Loaded.\n")

    conditions = [
        ("normal",        -10.0,  0.0, 0.0),
        ("strong_wind",   -10.0, 20.0, 2.0),
        ("extreme_wind",  -10.0, 40.0, 4.0),
        ("low_gravity",    -3.0,  0.0, 0.0),
        ("low_grav_wind",  -3.0, 20.0, 2.0),
        ("high_grav_wind",-11.9, 20.0, 2.0),
    ]

    for name, grav, wind, turb in conditions:
        print(f"\n{name.upper()} (gravity={grav}, wind={wind}, turb={turb}):")
        render_condition(standard_agent, interval_agent, name,
                        grav, wind, turb, c_eval=c_eval, save_dir=save_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c_train", type=float, default=0.5)
    parser.add_argument("--c_eval", type=float, default=0.2)
    args = parser.parse_args()

    main(seed=args.seed, c_train=args.c_train, c_eval=args.c_eval)
