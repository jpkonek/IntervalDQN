"""Comparison-grid GIFs: standard DQN vs credal interval DQN, across two
progressions (wind, gravity) at three severities (in-distribution, mild OOD,
severe OOD).

Wind progression (gravity = -10 throughout):
  wind = 0   -> in-distribution for both
  wind = 15  -> in-distribution for credal, OOD for standard
  wind = 25  -> severe OOD for both

Gravity progression (calm wind throughout):
  gravity = -10    -> in-distribution for both
  gravity = -11.5  -> mild OOD for both
  gravity = -11.99 -> severe OOD for both (LunarLander caps at -12)

Standard DQN: a calm-trained scalar Q-network (single seed=42 ckpt available).
Credal interval DQN: range-trained on wind ∈ [0, 15] with adaptive c at deployment.
"""

from __future__ import annotations
import math, sys, warnings, pathlib, random
import numpy as np, torch, imageio.v2 as imageio

sys.path.insert(0, "/Users/jk13942/Documents/GitHub/IP-ML")
from lunarlander_interval_dqn import IntervalDQNAgent, StandardDQNAgent

import gymnasium as gym


OUT_DIR = pathlib.Path("/Users/jk13942/Documents/GitHub/IP-ML/talk/gifs")


def make_env(wind_power: float = 0.0, gravity: float = -10.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gym.make(
            "LunarLander-v3", render_mode="rgb_array",
            enable_wind=(wind_power > 0.0),
            wind_power=wind_power, gravity=gravity, turbulence_power=0.0,
        )


def load_standard():
    agent = StandardDQNAgent()
    sd = torch.load("/Users/jk13942/Documents/GitHub/IP-ML/checkpoints/seed42_c0.5_standard.pt",
                    weights_only=False, map_location="cpu")
    agent.q_net.load_state_dict(sd)
    agent.target_net.load_state_dict(sd)
    agent.q_net.eval(); agent.target_net.eval()
    return agent


def load_credal(c: str, model_seed: int):
    p = f"/Users/jk13942/Documents/GitHub/IP-ML/Credal LunarLander/results/per_c_windy/c{c}_seed{model_seed}/ckpt_final.pt"
    ck = torch.load(p, weights_only=False, map_location="cpu")
    agent = IntervalDQNAgent(c_train=ck["args"]["c_train"], warmup_steps=0)
    agent.q_net.load_state_dict(ck["q_net_state_dict"])
    agent.target_net.load_state_dict(ck["target_net_state_dict"])
    agent.q_net.eval(); agent.target_net.eval()
    agent.total_env_steps = 1_000_000
    return agent


def adaptive_c_fn(w, c_low=0.0, c_high=0.3, w_mid=4.0, k=2.0):
    z = max(min(k * (w - w_mid), 50.0), -50.0)
    t = 1.0 / (1.0 + math.exp(z))
    return c_low + t * (c_high - c_low)


def rollout(agent, wind, gravity, rollout_seed, max_steps=600, downsample=2,
            adaptive=False):
    np.random.seed(rollout_seed); random.seed(rollout_seed); torch.manual_seed(rollout_seed)
    env = make_env(wind_power=wind, gravity=gravity)
    s, _ = env.reset(seed=rollout_seed)
    frames = []; total = 0.0
    for t in range(max_steps):
        if isinstance(agent, IntervalDQNAgent):
            if adaptive:
                with torch.no_grad():
                    L, U = agent.q_net(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))
                    w_pred = float((U - L).mean().item())
                c_use = adaptive_c_fn(w_pred)
            else:
                c_use = 0.0
            a = agent.select_action(s, c=c_use, force_epsilon=0.0)
        else:
            a = agent.select_action(s, epsilon=0.0)
        if t % downsample == 0:
            f = env.render()
            if f is not None: frames.append(f)
        s, r, te, tr, _ = env.step(a)
        total += r
        if te or tr:
            for _ in range(8):
                f = env.render()
                if f is not None: frames.append(f)
            break
    env.close()
    return frames, total


def find_typical_credal(c, wind, gravity, model_seeds, rollout_seeds, label):
    """Scan model_seeds × rollout_seeds, pick the trial closest to median reward."""
    rolls = []
    for ms in model_seeds:
        agent = load_credal(c, ms)
        for rs in rollout_seeds:
            frames, total = rollout(agent, wind, gravity, rs, adaptive=True)
            rolls.append((ms, rs, total, frames))
    medR = sorted([r[2] for r in rolls])[len(rolls)//2]
    chosen = min(rolls, key=lambda r: abs(r[2] - medR))
    print(f"  [{label}] median R={medR:.1f}  picked ms={chosen[0]} rs={chosen[1]} R={chosen[2]:.1f}")
    return chosen[3], chosen[2]


def find_typical_standard(wind, gravity, rollout_seeds, label):
    agent = load_standard()
    rolls = []
    for rs in rollout_seeds:
        frames, total = rollout(agent, wind, gravity, rs)
        rolls.append((rs, total, frames))
    medR = sorted([r[1] for r in rolls])[len(rolls)//2]
    chosen = min(rolls, key=lambda r: abs(r[1] - medR))
    print(f"  [{label}] median R={medR:.1f}  picked rs={chosen[0]} R={chosen[1]:.1f}")
    return chosen[2], chosen[1]


def save_gif(frames, path, fps=30):
    print(f"    saving {path.name} ({len(frames)} frames)")
    imageio.mimsave(path, frames, format="GIF", fps=fps, loop=0)


def main():
    rs_pool = [1, 3, 5, 7, 11, 17, 23, 29, 31]
    cred_seeds = [0, 1, 2, 3, 4]

    # Conditions across both progressions, with the ID condition shared.
    conditions = [
        # (label_for_filename, wind, gravity, severity, axis)
        ("id",       0.0, -10.0,  "ID",       "shared"),
        ("w15",     15.0, -10.0,  "wind 15",  "wind"),
        ("w25",     25.0, -10.0,  "wind 25",  "wind"),
        ("g_n11_5",  0.0, -11.5,  "grav -11.5", "gravity"),
        ("g_n11_99", 0.0, -11.99, "grav -11.99", "gravity"),
    ]

    results = {}  # (model, label) -> (frames, reward)

    for filename, wind, gravity, severity, axis in conditions:
        print(f"\n=== Condition: {severity} (wind={wind}, grav={gravity}) ===")

        out_std = OUT_DIR / f"PC_std_{filename}.gif"
        out_cred = OUT_DIR / f"PC_credal_{filename}.gif"

        print(f"  -- Standard DQN --")
        frames, R = find_typical_standard(wind, gravity, rs_pool, label="std")
        save_gif(frames, out_std)
        results[("std", filename)] = R

        print(f"  -- Credal interval DQN (adaptive c, c_train=0.5) --")
        frames, R = find_typical_credal("0.5", wind, gravity, cred_seeds, rs_pool,
                                          label="credal")
        save_gif(frames, out_cred)
        results[("credal", filename)] = R

    print("\n=== Summary ===")
    for (model, lab), R in results.items():
        print(f"  {model:>8} {lab:>10}: R = {R:7.1f}")
    print(f"\nGIFs in {OUT_DIR}:")
    for f in sorted(OUT_DIR.glob("PC_*.gif")):
        print(f"  {f.name}: {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
