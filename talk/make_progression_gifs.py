"""Credal LunarLander progression GIFs:
  in-distribution (wind = 5)  →  boundary (wind = 15)  →  severe OOD (wind = 25)

Same trained credal model at adaptive c throughout. Shows how the agent
behaves as it leaves the training distribution.
"""

from __future__ import annotations
import math, sys, warnings, pathlib, random
import numpy as np, torch, imageio.v2 as imageio
sys.path.insert(0, "/Users/jk13942/Documents/GitHub/IP-ML")
from lunarlander_interval_dqn import IntervalDQNAgent
import gymnasium as gym


OUT_DIR = pathlib.Path("/Users/jk13942/Documents/GitHub/IP-ML/talk/gifs")


def make_env(wind_power: float = 0.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gym.make(
            "LunarLander-v3", render_mode="rgb_array",
            enable_wind=(wind_power > 0.0),
            wind_power=wind_power, gravity=-10.0, turbulence_power=0.0,
        )


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


def rollout(agent, wind, rollout_seed, max_steps=600, downsample=2, use_adaptive=True):
    np.random.seed(rollout_seed); random.seed(rollout_seed); torch.manual_seed(rollout_seed)
    env = make_env(wind_power=wind)
    s, _ = env.reset(seed=rollout_seed)
    frames = []; total = 0.0; widths = []
    for t in range(max_steps):
        with torch.no_grad():
            L, U = agent.q_net(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))
            w = float((U - L).mean().item())
        widths.append(w)
        c = adaptive_c_fn(w) if use_adaptive else 0.0
        a = agent.select_action(s, c=c, force_epsilon=0.0)
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
    return frames, total, float(np.mean(widths)) if widths else 0.0


def find_typical(load_fn, wind, model_seeds, rollout_seeds, use_adaptive=True):
    rolls = []
    for ms in model_seeds:
        agent = load_fn(ms)
        for rs in rollout_seeds:
            frames, total, mw = rollout(agent, wind, rs, use_adaptive=use_adaptive)
            rolls.append((ms, total, mw, frames, rs))
    medR = sorted([r[1] for r in rolls])[len(rolls)//2]
    chosen = min(rolls, key=lambda r: abs(r[1] - medR))
    print(f"    median R={medR:.1f} mean width={np.mean([r[2] for r in rolls]):.2f}; "
          f"picked ms={chosen[0]} rs={chosen[4]} R={chosen[1]:.1f} width={chosen[2]:.2f}")
    return chosen


def save_gif(frames, path, fps=30):
    print(f"  saving {path.name} ({len(frames)} frames)")
    imageio.mimsave(path, frames, format="GIF", fps=fps, loop=0)


def main():
    seeds = [0, 1, 2, 3, 4]
    rs_pool = [1, 3, 5, 7, 11, 17, 23, 29, 31]

    # Use a c=0.5-trained model for adaptive-c demos (the standard adaptive-c agent).
    load_fn = lambda ms: load_credal("0.5", ms)

    progression = [
        ("P1_id_w5",       5.0,  "in-distribution (wind = 5, well inside [0,15])"),
        ("P2_boundary_w15", 15.0, "boundary of training (wind = 15)"),
        ("P3_severe_w25",  25.0, "severe OOD (wind = 25, well beyond)"),
    ]
    for name, wind, label in progression:
        print(f"\n[{name}] {label}:")
        pick = find_typical(load_fn, wind=wind, model_seeds=seeds, rollout_seeds=rs_pool)
        save_gif(pick[3], OUT_DIR / f"{name}.gif")

    print("\n=== Progression GIFs ===")
    for f in sorted(OUT_DIR.iterdir()):
        if f.name.startswith("P") and f.suffix == ".gif":
            print(f"  {f.name}: {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
