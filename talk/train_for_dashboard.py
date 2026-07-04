"""Re-instrumented credal LunarLander training, single seed.

Drops in the same training loop as `train_credal_lunarlander.py` but logs
per-episode:
  - mean loss across the episode's gradient steps
  - observed coverage rate at end of episode (from agent.coverage_hits)
  - mean predicted interval width on a fixed pool of 32 held-out states
  - t (adaptive coverage knob)
  - reward, wind, env_steps

Used to feed the training-dynamics dashboard on slide 25 of the deck.

~1-2 min wall on a laptop.
"""

from __future__ import annotations
import json, os, pathlib, random, sys, time, warnings, argparse
import numpy as np, torch

sys.path.insert(0, "/Users/jk13942/Documents/GitHub/IP-ML")
from lunarlander_interval_dqn import IntervalDQNAgent

import gymnasium as gym


def make_env(wind_power: float = 0.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gym.make("LunarLander-v3",
                        enable_wind=(wind_power > 0.0),
                        wind_power=wind_power, turbulence_power=0.0)


def collect_eval_states(n: int = 32, seed: int = 9999):
    """Fixed set of held-out states for measuring width over time.
    Sampled from env resets so the network sees diverse initial conditions."""
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n):
        w = float(rng.uniform(0.0, 15.0))
        env = make_env(w)
        s, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        states.append(s)
        env.close()
    return np.stack(states).astype(np.float32)


def mean_width(agent: IntervalDQNAgent, eval_states: np.ndarray) -> float:
    with torch.no_grad():
        L, U = agent.q_net(torch.from_numpy(eval_states))
        return float((U - L).mean().item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-episodes", type=int, default=800)
    p.add_argument("--c-train", type=float, default=0.5)
    p.add_argument("--wind-lo", type=float, default=0.0)
    p.add_argument("--wind-hi", type=float, default=15.0)
    p.add_argument("--log-every", type=int, default=5,
                    help="Log a row every N episodes (default 5 → 160 rows over 800 ep)")
    p.add_argument("--out", default="talk/training_dashboard_log.json")
    return p.parse_args()


def install_loss_term_logger(agent: IntervalDQNAgent):
    """Monkey-patch agent.interval_loss_sampled to also record the two loss
    summands separately, so we can log the miss-term vs width-term decomposition.

      miss_term  = t · 1[v ∉ [Q_l, Q_u]] · min((v-Q_l)^2, (v-Q_u)^2)
      width_term = (1 - t)                 · max((v-Q_l)^2, (v-Q_u)^2)

    Both averaged across N target samples and across the batch.
    """
    def wrapped(L, U, target_L, target_U):
        N = agent.N_TARGET_SAMPLES
        alphas = torch.linspace(0, 1, N)
        targets = target_L.unsqueeze(0) + alphas.unsqueeze(1) * (target_U - target_L).unsqueeze(0)
        # Track coverage against midpoint (replicates original behaviour)
        target_mid = (target_L + target_U) / 2
        inside_mid = (target_mid >= L) & (target_mid <= U)
        for hit in inside_mid.detach().cpu().numpy():
            agent.coverage_hits.append(float(hit))
        miss_acc  = torch.zeros(L.shape[0])
        width_acc = torch.zeros(L.shape[0])
        total_loss = torch.zeros(L.shape[0])
        for i in range(N):
            t_i = targets[i]
            inside = (t_i >= L) & (t_i <= U)
            d_l = (t_i - L) ** 2
            d_u = (t_i - U) ** 2
            min_d = torch.min(d_l, d_u)
            max_d = torch.max(d_l, d_u)
            outside_penalty = torch.where(inside, torch.zeros_like(min_d), min_d)
            miss  = agent.t * outside_penalty
            width = (1 - agent.t) * max_d
            miss_acc  = miss_acc  + miss.detach()
            width_acc = width_acc + width.detach()
            total_loss = total_loss + miss + width
        total_loss = total_loss / N
        miss_acc   = miss_acc   / N
        width_acc  = width_acc  / N
        # Width regularisation when coverage exceeds target (also in original)
        coverage = sum(agent.coverage_hits) / max(1, len(agent.coverage_hits))
        if coverage > agent.target_coverage:
            wreg = agent.width_reg * ((U - L) ** 2)
            total_loss = total_loss + wreg
        # Expose decomposition for the training loop to log
        agent.last_miss_term  = float(miss_acc.mean().item())
        agent.last_width_term = float(width_acc.mean().item())
        return total_loss.mean()
    agent.interval_loss_sampled = wrapped


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    eval_states = collect_eval_states(n=32)

    agent = IntervalDQNAgent(c_train=args.c_train,
                              warmup_steps=5000, warmup_epsilon=0.8,
                              width_reg=0.01, target_coverage=0.85)
    agent.last_miss_term  = 0.0
    agent.last_width_term = 0.0
    install_loss_term_logger(agent)

    rows = []
    t0 = time.time()
    print(f"[diagnose seed={args.seed}] {args.total_episodes} ep, wind ∈ [{args.wind_lo}, {args.wind_hi}]", flush=True)

    for ep in range(args.total_episodes):
        wind = float(rng.uniform(args.wind_lo, args.wind_hi))
        env = make_env(wind_power=wind)
        state, _ = env.reset()
        total_reward = 0.0
        losses_this_ep = []
        miss_terms_this_ep  = []
        width_terms_this_ep = []
        for _ in range(1000):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.total_env_steps += 1
            loss = agent.train_step()
            if loss > 0.0:
                losses_this_ep.append(loss)
                miss_terms_this_ep.append(agent.last_miss_term)
                width_terms_this_ep.append(agent.last_width_term)
            state = next_state
            total_reward += reward
            if done: break
        env.close()

        if (ep + 1) % args.log_every == 0:
            cov = (sum(agent.coverage_hits) / max(1, len(agent.coverage_hits)))
            w = mean_width(agent, eval_states)
            mean_loss = float(np.mean(losses_this_ep)) if losses_this_ep else 0.0
            mean_miss  = float(np.mean(miss_terms_this_ep))  if miss_terms_this_ep  else 0.0
            mean_width_term = float(np.mean(width_terms_this_ep)) if width_terms_this_ep else 0.0
            rows.append({
                "ep":         ep + 1,
                "reward":     float(total_reward),
                "wind":       wind,
                "loss":       mean_loss,
                "miss_term":  mean_miss,
                "width_term": mean_width_term,
                "coverage":   float(cov),
                "width":      w,
                "t":          float(agent.t),
                "env_steps":  agent.total_env_steps,
            })
            if (ep + 1) % 50 == 0:
                print(f"  ep {ep+1}/{args.total_episodes}  "
                      f"R={total_reward:7.1f}  loss={mean_loss:6.2f}  "
                      f"(miss {mean_miss:5.2f}, width {mean_width_term:5.2f})  "
                      f"cov={cov:.3f}  w={w:.2f}  t={agent.t:.3f}  "
                      f"wall={(time.time()-t0)/60:.1f} min",
                      flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f)

    print(f"\nWrote {args.out}  ({len(rows)} rows, wall {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
