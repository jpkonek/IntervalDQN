"""
Render radar animations of a trained bluebird interval-DQN agent.

Runs one episode with the checkpoint's policy (adaptive-c by default) while
drawing the env's own Radar display each step, and writes an animated GIF.
The bluebird sibling of render_comparison.py (LunarLander) — same purpose:
see HOW the agent flies, produce talk-ready assets.

Usage:
    python render_bluebird_radar.py --ckpt checkpoints/bluebird/best_run6.pt \
        --seed 10043 --duration 600 --out radar_run8_seed10043.gif
    python render_bluebird_radar.py --noop --seed 10043   # baseline contrast
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import numpy as np

import bluebird_interval_dqn as bid


def fig_to_rgb(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bluebird/best_run6.pt")
    ap.add_argument("--seed", type=int, default=10043)
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--noop", action="store_true",
                    help="all-NOOP baseline instead of the agent")
    ap.add_argument("--c", type=float, default=None,
                    help="fixed Hurwicz c (default: adaptive)")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    agent = None
    if not args.noop:
        agent, ckpt = bid.load_agent(args.ckpt, device="cpu")
        env = bid.make_env(scenario_duration=args.duration,
                           k_nearest=ckpt.get("k", 3),
                           route_parallel=ckpt.get("route_parallel", True),
                           centreline_coeff=ckpt.get("centreline_coeff", 0.2),
                           encoder_cls=ckpt.get("encoder_cls", "relative"))
        label = (f"interval DQN ({os.path.basename(args.ckpt)}, "
                 f"{'adaptive c' if args.c is None else f'c={args.c}'})")
    else:
        env = bid.make_env(scenario_duration=args.duration)
        label = "NOOP baseline"

    out = args.out or (f"radar_{'noop' if args.noop else 'agent'}"
                       f"_seed{args.seed}.gif")

    obs, info = env.reset(seed=args.seed)
    env.set_radar()
    maxstep = int(getattr(env, "maxstep", args.duration // bid.SEC_PER_STEP))

    adaptive = (not args.noop) and args.c is None
    if adaptive and not hasattr(agent, "adaptive_w_mid"):
        # calibrate on the first observation batch (cheap, adequate here)
        agent.calibrate_w_mid([dict(obs)])

    frames = []
    violated_at = None
    for step in range(maxstep):
        if not obs:
            break
        if agent is not None:
            actions, _ = agent.generate_action(
                obs, c=args.c, force_epsilon=0.0, adaptive=adaptive)
        else:
            actions = {cs: 0 for cs in obs}

        obs, rew, done, trunc, info = env.step(actions)
        violated, kind, involved = bid.detect_violation(info)

        fig, ax = env.radar.draw(info["simulator_environment"])
        t = (step + 1) * bid.SEC_PER_STEP
        status = f"{label} | seed {args.seed} | t={t:d}s | aircraft={len(obs)}"
        if violated:
            violated_at = t
            status += f"  <<< VIOLATION: {kind} ({', '.join(involved)})"
            ax.set_title(status, fontsize=9, color="red")
        else:
            ax.set_title(status, fontsize=9)
        frames.append(fig_to_rgb(fig))

        if violated:
            # hold the violation frame so it is visible
            frames.extend([frames[-1]] * args.fps)
            break

    env.close()
    imageio.mimsave(out, frames, fps=args.fps, loop=0)
    print(f"wrote {out}  ({len(frames)} frames, "
          f"{'violation at ' + str(violated_at) + 's' if violated_at else 'clean'})")


if __name__ == "__main__":
    main()
