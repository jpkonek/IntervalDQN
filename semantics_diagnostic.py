"""Semantics diagnostic: what does an interval's WIDTH actually mean?
=====================================================================

Standalone read-only diagnostic over a trained Bluebird interval-DQN
checkpoint. It separates two readings of the interval [Q_l, Q_u] at a
state-action pair, per risk stratum (nearest-neighbour distance from the
relative encoder: <10 NM / 10-30 NM / >30 NM-or-none, the same strata as
the --strat_t training flag in bluebird_interval_dqn.py):

  READING A — per-trajectory coverage (the existing training convention):
    does [Q_l, Q_u](s, a) contain the INDIVIDUAL realized discounted
    return G of the specific flight that passed through (s, a)?
    Computed on terminal-resolved streams only (flights that cleanly
    exited, vanished, or were terminated by a violation), with the same
    reward shaping the checkpoint was trained under (exit bonus,
    violation penalty, potential-based progress shaping).

  READING B — population coverage (shadow reading):
    does [Q_l, Q_u](s, a) contain the MEAN realized return of COMPARABLE
    states — i.e. the empirical average G of all samples in the same risk
    stratum? An interval can be honest under reading B ("I know the
    average outcome of situations like this") while failing reading A
    ("I cannot tell you how THIS flight ends") whenever outcome variance
    within a stratum is aleatoric.

  EPISTEMIC / ALEATORIC SPLIT — bootstrap-ensemble proxy:
    three checkpoints of the SAME training run at nearby episodes are
    loaded as a poor-man's ensemble. Per (s, a): the standard deviation of
    the three interval MIDPOINTS is an epistemic-uncertainty proxy
    ("the training process itself has not settled on a value here"), and
    the mean predicted WIDTH is the network's total uncertainty claim.
    ratio = midpoint_spread / mean_width per stratum: the fraction of the
    stated width attributable to 'ignorance' (still-moving estimates)
    rather than residual/aleatoric variability.

HONESTY CAVEATS (read before trusting the numbers):
  * The "ensemble" is NOT a bootstrap ensemble: the three members share
    initialization, replay data, and all but ~250 episodes of training.
    Their disagreement is a LOWER BOUND on epistemic uncertainty — small
    spread does not certify knowledge, it may just mean the run has
    converged (possibly to the wrong value). Consecutive checkpoints also
    make the spread partly a "training churn" measure.
  * Reading B uses the empirical stratum mean over the N eval episodes
    (finite-sample; strata with few samples give noisy means), and the
    stratum is assigned from the instantaneous obs at s — "comparable
    states" is a coarse notion.
  * Censored streams (aircraft still flying at episode end / at the first
    violation) are dropped, exactly as in training's realized-coverage
    tracker: long uneventful flights are under-represented.
  * Episodes end at the first violation (training convention), so
    post-violation traffic patterns are never observed.

Usage:
    .venv/bin/python semantics_diagnostic.py \
        [--ckpt checkpoints/bluebird/best_run6.pt] [--episodes 4] \
        [--duration 900] [--seed 20042] [--c 0.0]

Output: one table (strata x {readingA_cov, readingB_cov, mean_width,
epistemic_spread, ratio}) to stdout and JSON under
checkpoints/bluebird/diagnostics/.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bluebird_interval_dqn as bid  # single source of truth: env, agent,
                                     # violation detector, strata

STRATA_LABELS = ["<10nm", "10-30nm", ">30nm-or-none"]

DEFAULT_CKPT = os.path.join(bid.CHECKPOINT_DIR, "best_run6.pt")
# Poor-man's ensemble: SAME run (run 7, tag 20260704_212733), nearby episodes.
DEFAULT_ENSEMBLE = [
    os.path.join(bid.CHECKPOINT_DIR,
                 f"train_seed42_20260704_212733_ep{ep}.pt")
    for ep in (5350, 5475, 5600)
]


def collect_streams(env, agent, seed, violation_penalty, exit_bonus,
                    progress_coeff, c):
    """One eval episode; returns the terminal-resolved per-aircraft streams
    [(states, actions, rewards)] with training-convention reward shaping.

    Mirrors bluebird_interval_dqn.run_episode's train-branch bookkeeping
    (BEFORE_ENTRY skip, violation penalty, exit bonus, potential-based
    progress shaping, first-violation episode end) WITHOUT touching the
    agent's replay buffer, trackers, or weights.
    """
    obs, info = env.reset(seed=seed)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // bid.SEC_PER_STEP))
    streams = {}
    finished = []
    violated = False
    for _ in range(maxstep):
        actions, _w = agent.generate_action(obs, c=c, force_epsilon=0.0)
        phi_prev = ({cs: bid.exit_potential(env, cs) for cs in obs}
                    if progress_coeff else {})
        next_obs, rew, done, trunc, info = env.step(actions)
        violated, _kind, involved = bid.detect_violation(info)

        for cs, s in obs.items():
            d = info.get(cs)
            if isinstance(d, dict) and d.get("pos_status") == "BEFORE_ENTRY":
                continue
            a = actions[cs]
            r = float(rew.get(cs, 0.0))
            if cs in next_obs:
                terminal = (bool(done.get(cs, False))
                            and not bool(trunc.get(cs, False)))
            else:
                terminal = True
            if violated and cs in involved:
                r -= violation_penalty
                terminal = True
            elif terminal and not (isinstance(d, dict)
                                   and d.get("pos_status") == "OUT_SECTOR"):
                r += exit_bonus
            if progress_coeff and not terminal:
                phi = phi_prev.get(cs)
                phi_next = bid.exit_potential(env, cs)
                if phi is not None and phi_next is not None:
                    r += progress_coeff * (agent.gamma * phi_next - phi)
            st_s, st_a, st_r = streams.setdefault(cs, ([], [], []))
            st_s.append(np.asarray(s, dtype=np.float32))
            st_a.append(a)
            st_r.append(r)
            if terminal:
                finished.append(streams.pop(cs))

        obs = next_obs
        if violated:
            break
    n_censored = len(streams)  # unknown returns: dropped (as in training)
    return finished, n_censored, violated


def main():
    ap = argparse.ArgumentParser(
        description="Interval-width semantics diagnostic (reading A vs B + "
                    "epistemic/aleatoric ensemble proxy)")
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--ensemble", type=str, nargs=3, default=DEFAULT_ENSEMBLE,
                    help="3 checkpoints of the SAME run at nearby episodes")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--duration", type=int, default=900)
    ap.add_argument("--seed", type=int, default=20042,
                    help="base eval seed (episodes use seed, seed+1, ...)")
    ap.add_argument("--c", type=float, default=0.0,
                    help="Hurwicz c for the eval policy")
    args = ap.parse_args()

    print(f"Loading main checkpoint: {args.ckpt}")
    agent, ckpt = bid.load_agent(args.ckpt)
    gamma = agent.gamma
    print(f"  obs dim {ckpt['state_dim']}, {ckpt['n_actions']} actions, "
          f"episode {ckpt.get('episode', '?')}, gamma {gamma}, "
          f"t={ckpt.get('t', 0.5):.3f}")

    ens_agents = []
    for p in args.ensemble:
        a_i, c_i = bid.load_agent(p)
        ens_agents.append(a_i)
        print(f"  ensemble member: {p} (episode {c_i.get('episode', '?')})")

    k = ckpt.get("k", 3)
    encoder_cls = ckpt.get("encoder_cls", "extra_minimal")
    if encoder_cls != "relative":
        raise SystemExit("strata need the relative encoder; checkpoint used "
                         f"{encoder_cls!r}")
    env = bid.make_env(scenario_duration=args.duration, k_nearest=k,
                       route_parallel=ckpt.get("route_parallel", False),
                       centreline_coeff=ckpt.get("centreline_coeff", 1.0),
                       encoder_cls=encoder_cls)

    violation_penalty = ckpt.get("violation_penalty", 50.0)
    exit_bonus = ckpt.get("exit_bonus", 10.0)
    progress_coeff = ckpt.get("progress_coeff", 0.05)
    print(f"Env: duration {args.duration}s, k={k}, encoder={encoder_cls}, "
          f"route_parallel={ckpt.get('route_parallel', False)}; shaping: "
          f"violation_penalty={violation_penalty}, exit_bonus={exit_bonus}, "
          f"progress_coeff={progress_coeff}; policy c={args.c}")

    # ---- collect terminal-resolved (s, a, G) samples over N episodes ----
    samples = []          # (state, action, G)
    tot_censored = 0
    ep_stats = []
    for i in range(args.episodes):
        seed = args.seed + i
        finished, n_cens, violated = collect_streams(
            env, agent, seed, violation_penalty, exit_bonus,
            progress_coeff, args.c)
        tot_censored += n_cens
        n_sa = 0
        for states, acts, rews in finished:
            G = 0.0
            returns = [0.0] * len(rews)
            for j in range(len(rews) - 1, -1, -1):
                G = rews[j] + gamma * G
                returns[j] = G
            for s, a, g in zip(states, acts, returns):
                samples.append((s, a, g))
                n_sa += 1
        ep_stats.append({"seed": seed, "violated": bool(violated),
                         "finished_streams": len(finished),
                         "censored_streams": n_cens, "samples": n_sa})
        print(f"  ep {i+1}/{args.episodes} seed {seed}: "
              f"{len(finished)} finished streams, {n_cens} censored, "
              f"{n_sa} (s,a,G) samples, violated={violated}")
    env.close()
    if not samples:
        raise SystemExit("no terminal-resolved samples collected")

    states = np.stack([s for s, _a, _g in samples]).astype(np.float32)
    actions = np.array([a for _s, a, _g in samples], dtype=np.int64)
    gs = np.array([g for _s, _a, g in samples], dtype=np.float32)
    strata = np.array([bid.state_stratum(s) for s in states], dtype=np.int64)

    # ---- main-net intervals at the taken actions ----
    with torch.no_grad():
        st = torch.from_numpy(states)
        at = torch.from_numpy(actions).unsqueeze(1)
        lo, up = agent.q_net(st)
        l_a = lo.gather(1, at).squeeze(1).numpy()
        u_a = up.gather(1, at).squeeze(1).numpy()
        # ensemble midpoints and widths at the same (s, a)
        mids, widths_ens = [], []
        for a_i in ens_agents:
            lo_i, up_i = a_i.q_net(st)
            l_i = lo_i.gather(1, at).squeeze(1).numpy()
            u_i = up_i.gather(1, at).squeeze(1).numpy()
            mids.append((l_i + u_i) / 2.0)
            widths_ens.append(u_i - l_i)
    mids = np.stack(mids)                    # (3, N)
    widths_ens = np.stack(widths_ens)        # (3, N)
    mid_spread = mids.std(axis=0)            # epistemic proxy per sample
    mean_width_ens = widths_ens.mean(axis=0)  # total claim per sample
    width_main = u_a - l_a

    hit_A = (gs >= l_a) & (gs <= u_a)

    # ---- per-stratum table ----
    table = {}
    for s_id, label in enumerate(STRATA_LABELS):
        m = strata == s_id
        n = int(m.sum())
        if n == 0:
            table[label] = {"n": 0}
            continue
        mean_g = float(gs[m].mean())
        hit_B = (mean_g >= l_a[m]) & (mean_g <= u_a[m])
        mw = float(mean_width_ens[m].mean())
        sp = float(mid_spread[m].mean())
        table[label] = {
            "n": n,
            "readingA_cov": round(float(hit_A[m].mean()), 4),
            "readingB_cov": round(float(hit_B.mean()), 4),
            "stratum_mean_G": round(mean_g, 4),
            "stratum_std_G": round(float(gs[m].std()), 4),
            "mean_width": round(mw, 4),                 # ensemble-mean width
            "mean_width_main": round(float(width_main[m].mean()), 4),
            "epistemic_spread": round(sp, 4),           # midpoint std (3 ckpts)
            "ratio": round(sp / mw, 4) if mw > 0 else None,
        }

    # ---- report ----
    hdr = (f"{'stratum':>14} | {'n':>5} | {'readA':>6} | {'readB':>6} | "
           f"{'meanW':>7} | {'epiSpread':>9} | {'ratio':>6} | "
           f"{'meanG':>8} | {'stdG':>7}")
    print()
    print("SEMANTICS TABLE — reading A (individual G) vs reading B "
          "(stratum-mean G) vs epistemic proxy")
    print(hdr)
    print("-" * len(hdr))
    for label in STRATA_LABELS:
        row = table[label]
        if row["n"] == 0:
            print(f"{label:>14} | {0:>5} | (no samples)")
            continue
        print(f"{label:>14} | {row['n']:>5} | {row['readingA_cov']:>6.3f} | "
              f"{row['readingB_cov']:>6.3f} | {row['mean_width']:>7.3f} | "
              f"{row['epistemic_spread']:>9.3f} | {row['ratio']:>6.3f} | "
              f"{row['stratum_mean_G']:>8.3f} | {row['stratum_std_G']:>7.3f}")
    print()
    print("readA = P(G_individual in [l,u]); readB = P(stratum-mean G in "
          "[l,u]); meanW = ensemble-mean width;")
    print("epiSpread = std of the 3 checkpoints' interval midpoints "
          "(epistemic LOWER BOUND, see docstring); ratio = epiSpread/meanW.")

    out = {
        "ckpt": args.ckpt,
        "ensemble": list(args.ensemble),
        "episodes": args.episodes,
        "duration": args.duration,
        "base_seed": args.seed,
        "c": args.c,
        "gamma": gamma,
        "shaping": {"violation_penalty": violation_penalty,
                    "exit_bonus": exit_bonus,
                    "progress_coeff": progress_coeff},
        "n_samples": len(samples),
        "censored_streams_dropped": tot_censored,
        "episode_stats": ep_stats,
        "strata_bounds_nm": list(bid.STRAT_BOUNDS_NM),
        "table": table,
        "caveats": "see semantics_diagnostic.py header docstring: same-run "
                   "checkpoint ensemble is a lower-bound epistemic proxy; "
                   "reading B uses finite-sample stratum means; censored "
                   "streams dropped.",
    }
    out_dir = os.path.join(bid.CHECKPOINT_DIR, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                            f"semantics_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON written to {out_path}")


if __name__ == "__main__":
    main()
