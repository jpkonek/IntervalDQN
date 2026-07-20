"""
M2-XL hold-your-fire at scale (run-13 micro battery; A5, eval-only)
===================================================================

EVAL-ONLY scripted pass at 20+ concurrent aircraft (101+ candidates):
NOOP net-argmax and command-rate statistics vs candidate count for an
ARBITRARY checkpoint (--ckpt). NO training, NO GT rollouts (cheap by
design). Rationale (A5): P(argmax != NOOP) under ~10-wide intervals
rises toward 1 with candidate count, so an 11-candidate M2 pass cannot
certify hold-your-fire at scale.

Traffic: CustomInfiniteEnv at spawn 0.02/s (measured 19 Jul: max ~37
concurrent by step ~140; starters are capped at 4 by the spawn
machinery, so density comes from the spawn rate). The measurement pass
walks the ALL-NOOP trajectory and IGNORES violations (density-probe
convention, diagnose_controller B1-v2 notes) — the net is a passenger;
we read its would-be argmax at every state.

Outputs:
  - overall NOOP net-argmax fraction across visited states;
  - per-candidate-count bins: state count, NOOP-argmax fraction, mean
    Hurwicz margin of the best non-NOOP candidate over NOOP, mean width;
  - PRE-REGISTERED REGRESSION CHECK vs M2 (the A5 failure signature):
    P(argmax != NOOP) RISING with candidate count. Registered check:
    Spearman(candidate count, argmax != NOOP) <= 0.2 -> MEET, else MISS
    (reported; M2-XL is a report bar in the composite, not a blocker);
  - a greedy AGENT-DRIVEN episode (train=False) for the realized command
    rate and time-to-violation at scale (report).

Usage:
    .venv/bin/python micro_battery_m2xl.py                      # latest ckpt
    .venv/bin/python micro_battery_m2xl.py --ckpt path/to.pt
"""

import argparse
import glob
import os
import time

import numpy as np

import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import build_tokens, run_episode
from bluebird_interval_dqn import detect_violation, SEC_PER_STEP
from micro_battery_common import (
    bar, announce, write_summary, BATTERY_DIR,
)
from diagnose_controller import make_custom_density_env, spearman

OUT_DIR = os.path.join(BATTERY_DIR, "m2xl")

BINS = ((2, 51), (52, 101), (102, 151), (152, 10 ** 9))


def default_ckpt():
    pats = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "checkpoints", "bluebird_controller", "train_*.pt")
    pts = glob.glob(pats)
    finals = [p for p in pts if p.endswith("_final.pt")]
    pool = finals or pts
    if not pool:
        raise FileNotFoundError(f"no checkpoint matches {pats}")
    return max(pool, key=os.path.getmtime)


def state_read(agent, toks, n_cand, c):
    """The M2-XL per-state net read. C-INSTRUMENTS LOCK (ROUND3
    re-review): this reads candidate_q DIRECTLY and must NEVER route
    through select_candidate / generate_action — the NOOP tie-break
    tolerance and the count bonus live on the selection path, and this
    read is one of the four locked instruments the regression test
    (test_c_instruments_lock.py) asserts bit-identical under any
    tolerance value. Do not 'unify' it with selection.
    Returns (argmax_idx, margin_best_nonNOOP_minus_NOOP, mean_width)."""
    cl, cu = agent.candidate_q(toks)
    scores = (cl + c * (cu - cl))[:n_cand]
    best = scores.max()
    k = 0 if scores[0] == best else int(scores.argmax())
    margin = (float(scores[1:].max() - scores[0])
              if n_cand > 1 else float("-inf"))
    return k, margin, float((cu - cl)[:n_cand].mean())


def measurement_pass(env, agent, c, seed, max_steps):
    """All-NOOP walk, violations ignored (measurement only); per-state
    net argmax + margin + width. Returns list of state records."""
    obs, info = env.reset(seed=seed)
    recs = []
    n_viol = 0
    for step in range(max_steps):
        cs_list = sorted(obs.keys())
        if cs_list:
            toks = build_tokens(env, obs, info, cs_list)
            n_cand = 1 + agent.n_instr * len(cs_list)
            k, margin, mean_w = state_read(agent, toks, n_cand, c)
            recs.append({"step": step, "n_ac": len(cs_list),
                         "n_cand": n_cand, "argmax_noop": k == 0,
                         "margin": margin,
                         "mean_width": mean_w})
        obs, _r, _d, _t, info = env.step({cs: 0 for cs in obs})
        v, _k, _inv = detect_violation(info)
        n_viol += int(v)
    return recs, n_viol


def main():
    ap = argparse.ArgumentParser(description="M2-XL eval-only scale pass")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="checkpoint path (default: newest "
                         "checkpoints/bluebird_controller/train_*_final.pt)")
    ap.add_argument("--duration", type=int, default=1800)
    ap.add_argument("--spawn", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=10043)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--c", type=float, default=None,
                    help="Hurwicz c (default: the checkpoint's c_train)")
    ap.add_argument("--vertical_ramp", type=str, default="gate",
                    choices=["gate", "smooth"])
    ap.add_argument("--delta_conflict", type=float, default=None)
    ap.add_argument("--m2_summary", type=str, default=None,
                    help="optional M2 summary JSON for the cross-check "
                         "line (default: newest in the m2 out dir)")
    args = ap.parse_args()
    args.nstep, args.cf_replay = "-", False   # announce() fields
    announce("M2-XL HOLD-YOUR-FIRE AT SCALE (eval-only, 20+ aircraft)",
             args)

    ckpt_path = args.ckpt or default_ckpt()
    agent, ckpt = bcd.load_agent(ckpt_path)
    c = args.c if args.c is not None else ckpt.get("c_train", 0.5)
    print(f"  checkpoint: {ckpt_path}\n  episode {ckpt.get('episode')}, "
          f"c={c}, token_dim {ckpt['token_dim']}")

    env = make_custom_density_env(duration=args.duration,
                                  initial_spawn_rate=args.spawn,
                                  max_spawn_rate=args.spawn,
                                  num_starter_aircraft=4)
    t0 = time.time()
    recs, n_viol = measurement_pass(env, agent, c, args.seed,
                                    args.max_steps)
    n_cands = np.array([r["n_cand"] for r in recs])
    non_noop = np.array([not r["argmax_noop"] for r in recs], dtype=float)
    print(f"\n  measurement pass: {len(recs)} states in "
          f"{time.time() - t0:.0f}s; max concurrent "
          f"{max(r['n_ac'] for r in recs)} aircraft "
          f"({n_cands.max()} candidates); {n_viol} violation-steps "
          f"ignored (density-probe convention)")
    at_scale = int((n_cands >= 101).sum())
    print(f"  states with >= 101 candidates (20+ aircraft): {at_scale}")

    overall_noop = 1.0 - float(non_noop.mean())
    print(f"\n  NOOP net-argmax fraction (all states): "
          f"{overall_noop:.3f}")
    bin_rows = []
    for lo, hi in BINS:
        m = (n_cands >= lo) & (n_cands <= hi)
        if not m.any():
            print(f"    candidates {lo:3d}-{hi if hi < 10**9 else '...'}: "
                  f"no states")
            continue
        row = {"bin": (lo, hi), "n_states": int(m.sum()),
               "noop_frac": 1.0 - float(non_noop[m].mean()),
               "mean_margin": float(np.mean([recs[i]["margin"]
                                             for i in np.where(m)[0]])),
               "mean_width": float(np.mean([recs[i]["mean_width"]
                                            for i in np.where(m)[0]]))}
        bin_rows.append(row)
        print(f"    candidates {lo:3d}-{hi if hi < 10**9 else '...':>3}: "
              f"n={row['n_states']:3d} NOOP-argmax={row['noop_frac']:.3f} "
              f"margin(best_nonNOOP - NOOP)={row['mean_margin']:+7.3f} "
              f"width={row['mean_width']:6.2f}")

    rho_trend = spearman(n_cands.astype(float), non_noop)
    rho_trend = float("nan") if rho_trend is None else rho_trend
    print(f"\n  regression check (A5 failure signature: P(argmax != "
          f"NOOP) rising with candidate count):")
    meet = bar("M2XL trend Spearman(n_cand, argmax != NOOP)", rho_trend,
               "<=", 0.2, note="report bar — composite gate takes the "
                               "report, not a block")

    # cross-check line vs the 11-candidate M2 probe statistic
    import json
    m2s = args.m2_summary
    if m2s is None:
        cands = sorted(glob.glob(os.path.join(BATTERY_DIR, "m2",
                                              "m2_summary_*.json")))
        m2s = cands[-1] if cands else None
    if m2s and os.path.exists(m2s):
        with open(m2s) as f:
            m2 = json.load(f)
        print(f"  M2 cross-check: M2 NOOP-argmax "
              f"{m2.get('noop_argmax_frac'):.3f} (11 candidates, seed "
              f"{m2.get('scenario_seed')}) vs at-scale overall "
              f"{overall_noop:.3f}  [{os.path.basename(m2s)}]")
    else:
        print("  M2 cross-check: no M2 summary found yet — run "
              "micro_battery_m2.py --train first")

    # greedy agent-driven episode at scale (report)
    stats = run_episode(env, agent, seed=args.seed, train=False, c=c,
                        cbp=True, objective_v2=True)
    print(f"\n  (report) greedy agent-driven episode: "
          f"cmd={stats['commands']} steps={stats['steps']} "
          f"rate={stats['instruction_rate']:.3f}/step, "
          f"{'VIOLATION[' + str(stats['violation_kind']) + ']@' + str(stats['time_to_violation']) + 's' if stats['violated'] else 'clean'}"
          f", G={stats['ep_return']:.2f}")

    write_summary(OUT_DIR, "m2xl_report",
                  {"test": "M2-XL", "ckpt": ckpt_path, "c": c,
                   "seed": args.seed, "spawn": args.spawn,
                   "n_states": len(recs), "at_scale_states": at_scale,
                   "overall_noop_argmax": overall_noop,
                   "bins": bin_rows, "trend_spearman": rho_trend,
                   "trend_check_meet": meet,
                   "greedy_episode": {"commands": stats["commands"],
                                      "steps": stats["steps"],
                                      "rate": stats["instruction_rate"],
                                      "violated": stats["violated"],
                                      "kind": stats["violation_kind"],
                                      "G": stats["ep_return"]}})


if __name__ == "__main__":
    main()
