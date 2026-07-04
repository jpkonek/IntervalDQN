"""Collision visualiser — render aircraft trajectories on the 2-D position grid.

Decodes each PRISM state to its (h, h_y) physical position via stormpy's
state valuations, plays the trained policy under a chosen sl in the credal
range, and renders a frame-by-frame GIF showing the path.

The agent (own ship) appears in blue; the crash region in red; the goal
cells in green; the intruder's *observed* position in orange.
"""

from __future__ import annotations
import sys, math, warnings, pathlib, random, json
import numpy as np, torch, imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

sys.path.insert(0, "/Users/jk13942/Documents/GitHub/IP-ML/Credal DQN")
from rpomdp_loader import load_rpomdp, instantiate
from credal_interval_dqn import CredalIntervalQNet
from rollout import rollout_episode

import stormpy

PRISM_PATH = "/Users/jk13942/Documents/GitHub/IP-ML/Credal DQN/sourcecode/data/input/envs/prism/collision.prism"
OUT_DIR = pathlib.Path("/Users/jk13942/Documents/GitHub/IP-ML/talk/gifs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid extents (MINh = -5, MAXh = 5 from the .prism source).
H_MIN, H_MAX = -5, 5
GRID_PAD = 0.5

# Crash formula in collision.prism: (h <= 2 & h >= -2) & (h_y <= 2 & h_y >= -2).
CRASH_HALF = 2


def build_state_valuations():
    """Decode every state into its (h, h_y, h_obs, h_obs_y, hdot, hdot_y, c) values."""
    prog = stormpy.parse_prism_program(PRISM_PATH, simplify=True)
    props = stormpy.parse_properties_for_prism_program('R=? [F "goal"]', prog)
    opts = stormpy.BuilderOptions([p.raw_formula for p in props])
    opts.set_build_state_valuations()
    opts.set_build_all_labels()
    opts.set_build_all_reward_models()
    model = stormpy.build_sparse_parametric_model_with_options(prog, opts)
    sv = model.state_valuations
    decoded = []
    for s in range(model.nr_states):
        # stormpy returns a JsonContainerRational; str() gives a JSON-shaped repr.
        d = json.loads(str(sv.get_json(s)))
        decoded.append(d)
    goal_states = set(model.labeling.get_states("goal"))
    return decoded, goal_states


def crash_at(h, h_y):
    return (-CRASH_HALF <= h <= CRASH_HALF) and (-CRASH_HALF <= h_y <= CRASH_HALF)


def render_frame(decoded, traj_h, traj_y, intr_h, intr_y, t, goal_set, decoded_states, dpi=80):
    """Single frame: grid + own-ship trail + intruder marker."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    # Grid
    ax.set_xlim(H_MIN - GRID_PAD, H_MAX + GRID_PAD)
    ax.set_ylim(H_MIN - GRID_PAD, H_MAX + GRID_PAD)
    ax.set_aspect("equal")
    ax.set_xticks(range(H_MIN, H_MAX + 1))
    ax.set_yticks(range(H_MIN, H_MAX + 1))
    ax.grid(True, linewidth=0.5, color="#cccccc", linestyle="--")
    ax.set_axisbelow(True)

    # Crash region (centre)
    ax.add_patch(Rectangle((-CRASH_HALF - 0.5, -CRASH_HALF - 0.5),
                            2*CRASH_HALF + 1, 2*CRASH_HALF + 1,
                            facecolor="#fce7e7", edgecolor="#b22222",
                            linewidth=1.5, alpha=0.6, zorder=1))
    ax.text(0, 0, "crash\nregion", ha="center", va="center",
            fontsize=10, color="#b22222", fontweight="bold", zorder=2)

    # Goal cells — gather their (h, h_y) from decoded states with "goal" label
    goal_cells = set()
    for s in goal_set:
        d = decoded_states[s]
        goal_cells.add((d["h"], d["h_y"]))
    for (gh, gy) in goal_cells:
        ax.add_patch(Rectangle((gh - 0.5, gy - 0.5), 1, 1,
                                facecolor="#e3f2e7", edgecolor="#2c7a3f",
                                linewidth=1.2, alpha=0.7, zorder=1))
    if goal_cells:
        ax.text(list(goal_cells)[0][0], list(goal_cells)[0][1] + 0.05, "goal",
                ha="center", va="center", fontsize=9, color="#2c7a3f",
                fontweight="bold", zorder=2)

    # Trajectory of own ship (blue trail)
    if t > 0:
        ax.plot(traj_h[:t+1], traj_y[:t+1], color="#2a4d8f", linewidth=2,
                alpha=0.6, zorder=3)
    # Own ship as a little plane glyph (Unicode ✈, blue)
    ax.text(traj_h[t], traj_y[t], "✈", ha="center", va="center",
            fontsize=22, color="#2a4d8f", fontweight="bold", zorder=5)
    ax.text(traj_h[t], traj_y[t] + 0.55, "own", ha="center", va="bottom",
            fontsize=9, color="#2a4d8f", fontweight="bold", zorder=5)

    # Intruder as a skull-and-crossbones glyph (Unicode ☠, orange)
    if intr_h[t] is not None and intr_y[t] is not None:
        ax.text(intr_h[t], intr_y[t], "☠", ha="center", va="center",
                fontsize=20, color="#a04020", fontweight="bold", zorder=4)
        ax.text(intr_h[t], intr_y[t] + 0.55, "intruder",
                ha="center", va="bottom",
                fontsize=9, color="#a04020", fontweight="bold", zorder=4)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def trajectory_from_episode(ep, decoded_states):
    """Convert an Episode (list of state IDs) to per-step (h, h_y) and intruder
    coordinates."""
    traj_h, traj_y, intr_h, intr_y = [], [], [], []
    state_ids = list(ep.state) + [int(ep.next_state[-1])] if len(ep.next_state) else list(ep.state)
    for sid in state_ids:
        d = decoded_states[sid]
        traj_h.append(d["h"])
        traj_y.append(d["h_y"])
        # The intruder's true position isn't a variable; the OBSERVED position
        # is. Display it; out-of-range obs (sentinel = 50) → hide.
        ho, yo = d["h_obs"], d["h_obs_y"]
        if abs(ho) > H_MAX + 1 or abs(yo) > H_MAX + 1:
            intr_h.append(None); intr_y.append(None)
        else:
            intr_h.append(ho); intr_y.append(yo)
    return traj_h, traj_y, intr_h, intr_y


def render_episode_gif(ep, decoded_states, goal_set, out_path, fps=8):
    traj_h, traj_y, intr_h, intr_y = trajectory_from_episode(ep, decoded_states)
    T = len(traj_h)
    print(f"  rendering {T} frames -> {out_path.name}")
    frames = []
    for t in range(T):
        frames.append(render_frame(decoded_states, traj_h, traj_y, intr_h, intr_y, t, goal_set, decoded_states))
    imageio.mimsave(out_path, frames, format="GIF", fps=fps, loop=0)


def load_credal_collision_agent(c: str, model_seed: int):
    ckpt_path = f"/Users/jk13942/Documents/GitHub/IP-ML/Credal DQN/results/collision/per_c_dh64_trunc/c{c}_seed{model_seed}/ckpt_final.pt"
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    train_args = ck["args"]
    rpomdp = load_rpomdp(PRISM_PATH, {"sl": (0.6, 0.8)})
    net = CredalIntervalQNet(rpomdp.nZ, rpomdp.nA, d_h=train_args["d_h"])
    net.load_state_dict(ck["online_state_dict"])
    net.eval()
    return net, rpomdp, train_args


def main():
    print("Loading collision state valuations …")
    decoded_states, goal_set = build_state_valuations()
    print(f"  {len(decoded_states)} states decoded; {len(goal_set)} goal states")
    print(f"  example state 0: h={decoded_states[0]['h']}, h_y={decoded_states[0]['h_y']}")

    def find_goal_reaching(net, rpomdp, c_eval, sl_value, max_attempts=20, max_steps=180):
        """Try multiple seeds; return the first goal-reaching episode (which we
        define as one whose final next_state is a labelled goal state)."""
        T = instantiate(rpomdp, sl_value)
        for seed in range(max_attempts):
            rng = np.random.default_rng(seed * 13 + 7)
            ep = rollout_episode(net, rpomdp, rng, epsilon=0.0, c_train=c_eval,
                                  max_steps=max_steps, T_concrete=T, sl_value=sl_value)
            last = int(ep.next_state[-1]) if len(ep.next_state) else -1
            reached = last in goal_set
            print(f"    attempt {seed}: len={len(ep.action)} cost={float(ep.reward.sum()):.0f} "
                  f"final={last} reached_goal={reached}")
            if reached:
                return ep
        return ep  # last attempt anyway

    # Best-performing model (truncation-fix, d_h=64): c=0.8 seed=3 had MC cost 107.7
    # — within +5% of PIP's reported 102.95.
    print("\n=== Best-model trajectory at nominal sl=0.6 (c_train=0.8) ===")
    net, rpomdp, _ = load_credal_collision_agent("0.8", model_seed=3)
    ep = find_goal_reaching(net, rpomdp, c_eval=0.8, sl_value=0.6, max_steps=250)
    render_episode_gif(ep, decoded_states, goal_set,
                       OUT_DIR / "collision_best_nominal.gif")

    print("\n=== Best-model trajectory at worst sl=0.8 ===")
    ep = find_goal_reaching(net, rpomdp, c_eval=0.8, sl_value=0.8, max_steps=250)
    render_episode_gif(ep, decoded_states, goal_set,
                       OUT_DIR / "collision_best_worst_sl.gif")

    print("\n=== Cautious deployment of same model at c=0 ===")
    ep = find_goal_reaching(net, rpomdp, c_eval=0.0, sl_value=0.7, max_steps=250)
    render_episode_gif(ep, decoded_states, goal_set,
                       OUT_DIR / "collision_cautious_deploy.gif")

    print("\n=== Daring deployment of same model at c=1 ===")
    ep = find_goal_reaching(net, rpomdp, c_eval=1.0, sl_value=0.7, max_steps=250)
    render_episode_gif(ep, decoded_states, goal_set,
                       OUT_DIR / "collision_daring_deploy.gif")

    print("\nGIFs:")
    for f in sorted(OUT_DIR.glob("collision_*.gif")):
        print(f"  {f.name}: {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
