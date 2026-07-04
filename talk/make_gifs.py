"""Generate LunarLander rollout GIFs for the talk slides.

Six GIFs in three pairs, picked by trying multiple (model_seed, rollout_seed)
pairs and selecting the most representative demonstration per condition.

  Pair 1 — training under uncertainty matters at OOD:
    01_standard_at_wind.gif      Standard DQN (calm-trained) at wind=20
    02_credal_at_wind.gif        Credal-trained at c=0 at wind=20

  Pair 2 — same model, two attitudes at OOD wind:
    03_cautious_at_wind.gif      Credal at c=0  at wind=25
    04_daring_at_wind.gif        Credal at c=1  at wind=25

  Pair 3 — adaptive c, in-distribution vs OOD:
    05_adaptive_calm.gif         Adaptive c at wind=0
    06_adaptive_wind.gif         Adaptive c at wind=20
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
from lunarlander_interval_dqn import IntervalDQNAgent, StandardDQNAgent

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


def load_standard():
    agent = StandardDQNAgent()
    sd = torch.load("/Users/jk13942/Documents/GitHub/IP-ML/checkpoints/seed42_c0.5_standard.pt",
                    weights_only=False, map_location="cpu")
    agent.q_net.load_state_dict(sd)
    agent.target_net.load_state_dict(sd)
    agent.q_net.eval(); agent.target_net.eval()
    return agent


def load_credal(c: str, model_seed: int = 0):
    ckpt_path = f"/Users/jk13942/Documents/GitHub/IP-ML/Credal LunarLander/results/per_c_windy/c{c}_seed{model_seed}/ckpt_final.pt"
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    agent = IntervalDQNAgent(c_train=ck["args"]["c_train"], warmup_steps=0)
    agent.q_net.load_state_dict(ck["q_net_state_dict"])
    agent.target_net.load_state_dict(ck["target_net_state_dict"])
    agent.q_net.eval(); agent.target_net.eval()
    agent.total_env_steps = 1_000_000
    return agent


def adaptive_c_fn(mean_width, c_low=0.0, c_high=0.3, w_mid=4.0, k=2.0):
    z = max(min(k * (mean_width - w_mid), 50.0), -50.0)
    t = 1.0 / (1.0 + math.exp(z))
    return c_low + t * (c_high - c_low)


def rollout_to_frames(agent, wind_power, rollout_seed, c_eval=None,
                      gravity=-10.0, max_steps=600, downsample=2,
                      use_adaptive=False):
    np.random.seed(rollout_seed); random.seed(rollout_seed); torch.manual_seed(rollout_seed)
    env = make_env(wind_power=wind_power, gravity=gravity)
    state, _ = env.reset(seed=rollout_seed)
    frames = []
    total = 0.0
    for t in range(max_steps):
        if isinstance(agent, IntervalDQNAgent):
            if use_adaptive:
                with torch.no_grad():
                    L, U = agent.q_net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                    w = float((U - L).mean().item())
                use_c = adaptive_c_fn(w)
                action = agent.select_action(state, c=use_c, force_epsilon=0.0)
            else:
                action = agent.select_action(state, c=c_eval, force_epsilon=0.0)
        else:
            action = agent.select_action(state, epsilon=0.0)
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


def best_failing_rollout(agent_factory, wind, gravity, c_eval, model_seeds, rollout_seeds,
                         use_adaptive=False):
    """Try (model_seed × rollout_seed) combos; return the most-failing.
    agent_factory(model_seed) → agent."""
    print(f"  searching for failure at wind={wind} grav={gravity} c={c_eval} "
          f"adaptive={use_adaptive} (model seeds {model_seeds}, rollout seeds {rollout_seeds})")
    best = None
    for ms in model_seeds:
        agent = agent_factory(ms)
        for rs in rollout_seeds:
            frames, total = rollout_to_frames(
                agent, wind, rs, c_eval=c_eval, gravity=gravity,
                use_adaptive=use_adaptive)
            tag = f"ms={ms} rs={rs}: R={total:7.1f} frames={len(frames)}"
            if best is None or total < best[1]:
                best = (ms, total, len(frames), frames, rs)
                tag += "  (new worst)"
            print(f"    {tag}")
    return best


def best_succeeding_rollout(agent_factory, wind, gravity, c_eval, model_seeds, rollout_seeds,
                            use_adaptive=False, prefer_threshold=100):
    """Try combos; return the most-succeeding (above threshold)."""
    print(f"  searching for success at wind={wind} grav={gravity} c={c_eval} "
          f"adaptive={use_adaptive} (model seeds {model_seeds}, rollout seeds {rollout_seeds})")
    successes = []
    all_rolls = []
    for ms in model_seeds:
        agent = agent_factory(ms)
        for rs in rollout_seeds:
            frames, total = rollout_to_frames(
                agent, wind, rs, c_eval=c_eval, gravity=gravity,
                use_adaptive=use_adaptive)
            all_rolls.append((ms, total, len(frames), frames, rs))
            print(f"    ms={ms} rs={rs}: R={total:7.1f} frames={len(frames)}")
            if total >= prefer_threshold:
                successes.append((ms, total, len(frames), frames, rs))
    if successes:
        return max(successes, key=lambda r: r[1])
    return max(all_rolls, key=lambda r: r[1])


def save_gif(frames, path, fps=30):
    print(f"  saving {path.name} ({len(frames)} frames)")
    imageio.mimsave(path, frames, format="GIF", fps=fps, loop=0)


def main():
    rs_pool = [1, 3, 5, 7, 11, 17, 23, 29, 31, 37, 41, 43]
    cred_seeds = [0, 1, 2, 3, 4]   # all five trained credal models per c

    print("=== Pair 1: training under uncertainty matters at OOD ===")
    print("\n[1] Standard DQN (calm-trained, only one seed available) at wind=20:")
    std = load_standard()
    pick = best_failing_rollout(lambda _: std, wind=20.0, gravity=-10.0, c_eval=None,
                                 model_seeds=[42], rollout_seeds=rs_pool)
    save_gif(pick[3], OUT_DIR / "01_standard_at_wind.gif")
    print(f"  → ms={pick[0]} rs={pick[4]} R={pick[1]:.1f}")

    print("\n[2] Credal-trained at c=0, wind=20:")
    pick = best_succeeding_rollout(lambda ms: load_credal("0.0", ms),
                                    wind=20.0, gravity=-10.0, c_eval=0.0,
                                    model_seeds=cred_seeds, rollout_seeds=rs_pool)
    save_gif(pick[3], OUT_DIR / "02_credal_at_wind.gif")
    print(f"  → ms={pick[0]} rs={pick[4]} R={pick[1]:.1f}")

    print("\n=== Pair 2: cautious vs daring at OOD wind ===")
    print("\n[3] Credal c=0 (cautious) at wind=25:")
    pick = best_succeeding_rollout(lambda ms: load_credal("0.0", ms),
                                    wind=25.0, gravity=-10.0, c_eval=0.0,
                                    model_seeds=cred_seeds, rollout_seeds=rs_pool)
    save_gif(pick[3], OUT_DIR / "03_cautious_at_wind.gif")
    print(f"  → ms={pick[0]} rs={pick[4]} R={pick[1]:.1f}")

    print("\n[4] Credal c=1 (daring) at wind=25 — looking for failure:")
    pick = best_failing_rollout(lambda ms: load_credal("1.0", ms),
                                 wind=25.0, gravity=-10.0, c_eval=1.0,
                                 model_seeds=cred_seeds, rollout_seeds=rs_pool)
    save_gif(pick[3], OUT_DIR / "04_daring_at_wind.gif")
    print(f"  → ms={pick[0]} rs={pick[4]} R={pick[1]:.1f}")

    print("\n=== Pair 3: adaptive c, ID vs OOD ===")
    print("\n[5] Adaptive c at calm (in-distribution):")
    pick = best_succeeding_rollout(lambda ms: load_credal("0.5", ms),
                                    wind=0.0, gravity=-10.0, c_eval=None,
                                    model_seeds=cred_seeds, rollout_seeds=rs_pool,
                                    use_adaptive=True)
    save_gif(pick[3], OUT_DIR / "05_adaptive_calm.gif")
    print(f"  → ms={pick[0]} rs={pick[4]} R={pick[1]:.1f}")

    print("\n[6] Adaptive c at wind=20 (out-of-distribution):")
    pick = best_succeeding_rollout(lambda ms: load_credal("0.5", ms),
                                    wind=20.0, gravity=-10.0, c_eval=None,
                                    model_seeds=cred_seeds, rollout_seeds=rs_pool,
                                    use_adaptive=True)
    save_gif(pick[3], OUT_DIR / "06_adaptive_wind.gif")
    print(f"  → ms={pick[0]} rs={pick[4]} R={pick[1]:.1f}")

    print(f"\n=== GIFs ===")
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix == ".gif":
            print(f"  {f.name}: {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
