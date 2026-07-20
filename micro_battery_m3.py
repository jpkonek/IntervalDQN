"""
M3 lateral-only crossing (run-13 micro battery; RUN13_FIX_C_DESIGN.md S3)
=========================================================================

The M1 level-allocation scenario (same-FL crossing pair, NOOP ->
loss_of_separation; micro_level_allocation.py finds it) but with the
VERTICAL candidates MASKED at training time: the action set the learner
may select from — and bootstrap through — is {NOOP, L10, R10,
route_parallel}. The mask lives at CANDIDATE level in the harness agent
(micro_battery_common.HarnessAgent.masked_instr; selection, epsilon
sampling AND the Double-DQN target argmax), NOT in
bluebird_controller_dqn, which is unchanged.

Scripted gates (pre-registered, run before any training):
  (M3-i)  NOOP through bcd.run_episode -> loss_of_separation (the
          conflict is real; identical to M1 gate (i));
  (M3-ii) a single well-timed LATERAL clearance (L10 / R10 /
          route_parallel) -> clean episode; the sweep prints the lateral
          resolution window (the task is solvable INSIDE the masked
          action set, so a MISS on the bars is not a geometry artifact).

GT horizon (A10 CPA-matched): --h defaults to NOOP-LoS step + 2 (capped
80) so every probe state's conflict outcome is inside its GT window
(the micro default h=15 misses it from early states — the A10 finding).
GT windows are capped at the episode end (basis consistency; see
micro_battery_common.build_probe_set_capped).

PRE-REGISTERED BARS (design S3, M3 line):
  (B1) trailing-100 clean rate >= 0.80;
  (B2) rho > 0.4 on the LATERAL-ONLY candidate subset (the trained
       support: NOOP + L10/R10/route_parallel per aircraft — masked
       verticals never receive targets by construction, so ranking them
       tests untrained heads; full-11-candidate rho is REPORTED, not
       gated).
NOTE (blocked-on-CF): B2 grounds UNTAKEN candidates; without --cf_replay
it is expected unreachable (the 12d-era null) — run and report anyway.

Usage:
    .venv/bin/python micro_battery_m3.py --gates
    .venv/bin/python micro_battery_m3.py --gates --train --episodes 50
"""

import argparse
import os

import numpy as np

import bluebird_controller_dqn as bcd
from micro_battery_common import (
    bar, base_argparser, announce, write_summary, make_agent,
    run_training_arm, greedy_eval_episode, build_probe_set_capped,
    rho_eval_subset, BATTERY_DIR, LATERAL_ACTS, VERTICAL_TYPES,
    SEC_PER_STEP,
)
from micro_level_allocation import (
    make_micro_env, noop_trace, find_conflict_seed, SingleActionPolicy,
    rho_eval,
)
from diagnose_controller import NoopPolicy, run_scripted

OUT_DIR = os.path.join(BATTERY_DIR, "m3")

LATERAL_NAMES = {1: "L10", 2: "R10", 3: "route_parallel"}


def lateral_keep(k):
    """Candidate-index filter for the lateral-only trained support."""
    return k == 0 or (k - 1) % bcd.N_INSTR < 3


def run_gates(env, seed, rows, vstep, args):
    paired = [r for r in rows if "d_nm" in r]
    cpa = min(paired, key=lambda r: r["d_nm"])
    cs_list = list(paired[0]["pair"])
    print(f"\n  conflict pair {cs_list}, FLs {paired[0]['fls']}, NOOP LoS "
          f"at step {vstep} ({(vstep + 1) * SEC_PER_STEP} s), CPA proxy "
          f"step {cpa['step']} @ {cpa['d_nm']:.2f} nm")

    # ---- gate M3-i: NOOP -> LoS, priced by the system itself ---------
    g_noop = run_scripted(env, NoopPolicy(), seed, cbp=True)
    gate_i = g_noop["violated"] and \
        g_noop["violation_kind"] == "loss_of_separation"
    print(f"  GATE (M3-i) NOOP: G={g_noop['ep_return']:.2f} "
          f"{'VIOLATION[' + str(g_noop['violation_kind']) + ']' if g_noop['violated'] else 'clean'}"
          f" -> {'PASS' if gate_i else 'FAIL'}")

    # ---- gate M3-ii: single LATERAL clearance -> clean; sweep --------
    windows = {}
    for target in cs_list:
        for act in LATERAL_ACTS:
            clean_steps = []
            for s in range(0, vstep + 1):
                st = run_scripted(env, SingleActionPolicy(target, act, s),
                                  seed)
                if not st["violated"]:
                    clean_steps.append(s)
            windows[(target, LATERAL_NAMES[act])] = clean_steps
            if clean_steps:
                print(f"  GATE (M3-ii) {LATERAL_NAMES[act]:14s} on "
                      f"{target}: clean for issue steps "
                      f"[{clean_steps[0]}..{clean_steps[-1]}] "
                      f"({len(clean_steps)}/{vstep + 1})")
            else:
                print(f"  GATE (M3-ii) {LATERAL_NAMES[act]:14s} on "
                      f"{target}: no clean single issue")
    gate_ii = any(windows.values())
    print(f"  GATE (M3-ii) lateral-only resolvable -> "
          f"{'PASS' if gate_ii else 'FAIL'}")

    h_cpa = min(80, max(15, vstep + 2))
    print(f"  A10 CPA-matched GT horizon: h={h_cpa} (NOOP LoS step "
          f"{vstep})")
    return {"pair": cs_list, "vstep": vstep, "cpa_step": cpa["step"],
            "cpa_d_nm": cpa["d_nm"], "gate_i": gate_i, "gate_ii": gate_ii,
            "h_cpa": h_cpa, "noop_G": g_noop["ep_return"],
            "windows": {f"{t}:{n}": (min(v), max(v)) if v else None
                        for (t, n), v in windows.items()}}


def train_arm(env, seed, probe, gates_out, args):
    agent = make_agent(env, seed, args, masked_instr=VERTICAL_TYPES)

    def rho_fn(a):
        r_lat = rho_eval_subset(a, probe, args.c_train, lateral_keep)
        r_full = rho_eval(a, probe, args.c_train)
        r_lat["rho_full"] = r_full["rho"]
        return r_lat

    out = run_training_arm(env, seed, agent, args, label="m3",
                           rho_fn=rho_fn, out_dir=OUT_DIR)

    # ---- pre-registered verdict block --------------------------------
    print("\n" + "=" * 74)
    print("M3 PRE-REGISTERED VERDICT BLOCK (lateral-only crossing)")
    print(f"  [{bcd.conflict_pricing_str()}] episodes={args.episodes} "
          f"(SMOKE run if < 300)")
    trail = [h["clean"] for h in out["hist"][-100:]]
    clean100 = float(np.mean(trail))
    vert_taken = sum(sum(h["mix"][n] for n in ("climb10", "descend10"))
                     for h in out["hist"])
    print(f"  mask check: vertical clearances taken over training = "
          f"{vert_taken} (must be 0 — candidate mask)")
    b1 = bar("M3-1 clean rate (last 100)", clean100, ">=", 0.80)
    final = out["rho_curve"][-1] if out["rho_curve"] else \
        {"rho": float("nan"), "rho_full": float("nan")}
    b2 = bar("M3-2 rho (lateral-only subset)", final["rho"], ">", 0.4,
             note=("CF-grounded arm; " if getattr(args, "cf_replay", False)
                   else "EXPECTED BLOCKED without --cf_replay; ")
                  + "full-set rho "
                  f"{final.get('rho_full', float('nan')):+.3f} reported")
    g_eval = greedy_eval_episode(env, agent, seed, cbp=args.cbp)
    print(f"  (report) greedy eval: G={g_eval['ep_return']:.2f} "
          f"cmd={g_eval['commands']} "
          f"{'clean' if not g_eval['violated'] else 'VIOLATION[' + str(g_eval['violation_kind']) + ']'}")
    print(f"  M3 composite: {'MEET' if (b1 and b2) else 'MISS'} "
          f"(bars {int(b1)}{int(b2)}; mask held: {vert_taken == 0})")
    print("=" * 74)

    payload = {"test": "M3", "scenario_seed": seed,
               "episodes": args.episodes, "nstep": args.nstep,
               "cf_replay": args.cf_replay, "gates": gates_out,
               "clean100": clean100, "vert_taken": vert_taken,
               "rho_curve": out["rho_curve"],
               "bars": {"b1_clean": b1, "b2_rho_lat": b2},
               "wall_seconds": out["wall_seconds"]}
    write_summary(OUT_DIR, f"m3_summary_seed{seed}", payload)
    return payload


def main():
    ap = argparse.ArgumentParser(description="M3 lateral-only crossing "
                                             "micro test")
    base_argparser(ap)
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--scan_seeds", type=int, nargs="+",
                    default=[10043, 20042] + list(range(100, 160)))
    args = ap.parse_args()
    announce("M3 LATERAL-ONLY CROSSING (M1 scenario, verticals masked "
             "at candidate level)", args)
    env = make_micro_env(duration=args.duration)

    if args.scenario_seed is not None:
        seed = args.scenario_seed
        rows, violated, kind, vstep = noop_trace(env, seed)
        assert violated and kind == "loss_of_separation", \
            f"seed {seed} does not produce a NOOP LoS ({kind})"
    else:
        print("scanning for the M1-style crossing seed ...")
        seed, rows, vstep, _cs, _cd = find_conflict_seed(env,
                                                         args.scan_seeds)
        assert seed is not None, "no conflict seed found in the scan range"
    print(f"scenario seed: {seed} (NOOP LoS at step {vstep})")

    gates_out = run_gates(env, seed, rows, vstep, args)
    assert gates_out["gate_i"] and gates_out["gate_ii"], \
        "M3 scripted gates failed — do not train on this scenario"
    h = args.h if args.h is not None else gates_out["h_cpa"]

    if args.train:
        print(f"\n  building frozen probe set (n={args.probe_states}, "
              f"GT h={h}) ...")
        probe = build_probe_set_capped(env, seed, args.probe_states, h)
        train_arm(env, seed, probe, gates_out, args)
    else:
        write_summary(OUT_DIR, f"m3_gates_seed{seed}",
                      {"test": "M3", "scenario_seed": seed,
                       "gates": gates_out, "h": h})


if __name__ == "__main__":
    main()
