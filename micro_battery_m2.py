"""
M2 hold-your-fire (run-13 micro battery; RUN13_FIX_C_DESIGN.md S3, A6)
======================================================================

Scenario: a LEVEL-SEPARATED (|dFL| = 10, i.e. exactly one +-10 FL command
away from the LoS band) but LATERALLY-PROXIMATE (would-be CPA < 5 nm)
2-aircraft pair, nothing else in the sector. Under NOOP the episode is
clean for the full duration; a single RE-LEVELING climb/descend (the RA4
endangerment mechanism) produces loss_of_separation inside the GT
horizon. The correct policy is to DO NOTHING — the test prices the 12d
compulsion pathology (inventing interventions for traffic that needs
none; the SAFE-stratum finding in Section 0 of the design).

Scripted gates (pre-registered, run before any training):
  (M2-i)  NOOP through bcd.run_episode -> clean for the FULL duration;
  (M2-ii) a single re-leveling command (climb on the lower aircraft or
          descend on the upper) -> loss_of_separation; the full
          issue-step sweep prints the endangerment window (this is what
          gives the GT table its -50 tier).

Pre-training probe-set gate (A6.2): >= 5 probe states with gt_spread > 5
spanning >= 3 of 5 equal trajectory segments (distinct segments — the
probe states are autocorrelated along one NOOP trajectory; the gate
refuses a GT table whose ordering content is concentrated in one bend).

GT horizon (A10 CPA-matched): --h defaults to the measured worst-case
LoS delay of a re-leveling command + 2 (capped at 80), so every probe
state's endangerment event is inside its GT window.

PRE-REGISTERED BARS (A6-repaired; all four printed in the verdict block):
  (B1) command rate < 0.1/step        (greedy eval episode at the end);
  (B2) NOOP net-argmax >= 90% of frozen probe states (greedy, eps 0);
  (B3) rho > 0.4                      (net Hurwicz vs frozen GT);
  (B4) |Hurwicz(NOOP) - GT(NOOP)| < 2 at >= 80% of probe states
       (taken_noop bootstrap propagates Q(NOOP) into targets; a biased
       NOOP level passes B1-B3 silently — A6.3).
NOTE (blocked-on-CF): B3 grounds UNTAKEN candidates; without --cf_replay
(stage-1 machinery, separate work item) untaken candidates never receive
targets, so B3 is expected unreachable — run the bar anyway and report.

Usage:
    .venv/bin/python micro_battery_m2.py --gates
    .venv/bin/python micro_battery_m2.py --gates --train --episodes 50
"""

import argparse
import os
import time

import numpy as np

import bluebird_controller_dqn as bcd
from micro_battery_common import (
    bar, base_argparser, announce, write_summary, make_agent,
    noop_trace_multi, pair_series, segments_spanned,
    run_training_arm, greedy_eval_episode, build_probe_set_capped,
    probe_argmax, noop_argmax_fraction, noop_level_hits,
    BATTERY_DIR, CLIMB, DESCEND, SEC_PER_STEP,
)
from micro_level_allocation import SingleActionPolicy, rho_eval
from diagnose_controller import (
    make_custom_density_env, NoopPolicy, run_scripted,
)

OUT_DIR = os.path.join(BATTERY_DIR, "m2")

# scenario acceptance thresholds (pre-registered)
DFL_LO, DFL_HI = 10.0, 20.0      # level-separated band re-levelable by ±10
WOULD_BE_CPA_NM = 5.0            # lateral min dist < LoS radius


def make_m2_env(duration=600):
    return make_custom_density_env(duration=duration,
                                   initial_spawn_rate=0.0,
                                   max_spawn_rate=0.0,
                                   num_starter_aircraft=2)


def find_scenario(env, seeds):
    """First seed with: NOOP clean full duration; pair |dFL| in
    [10, 20) throughout; lateral min distance < 5 nm (would-be CPA).
    Prints every scanned seed; returns (seed, rows, series, cpa)."""
    closest = None
    for seed in seeds:
        rows, violated, kind, vstep, inv = noop_trace_multi(env, seed)
        paired = [r for r in rows if r["pairs"]]
        if not paired:
            print(f"  seed {seed}: no paired steps")
            continue
        pair = sorted(paired[0]["pairs"])[0]
        ser = pair_series(rows, pair)
        dmin_step, dmin, dfl_at = min(ser, key=lambda x: x[1])
        dfl_min = min(x[2] for x in ser)
        ok = (not violated and DFL_LO <= dfl_at < DFL_HI
              and dfl_min >= DFL_LO and dmin < WOULD_BE_CPA_NM)
        print(f"  seed {seed}: "
              f"{'VIOL[' + str(kind) + ']@' + str(vstep) if violated else 'clean':24s}"
              f" dmin={dmin:6.2f} nm @step {dmin_step:3d}, dFL@dmin="
              f"{dfl_at:4.0f}{'   <- M2 SCENARIO' if ok else ''}")
        if ok:
            return seed, rows, ser, (dmin_step, dmin)
        if not violated and dfl_min >= DFL_LO and (
                closest is None or dmin < closest[4]):
            closest = (seed, rows, ser, dmin_step, dmin, dfl_at)
    if closest is not None:
        print(f"  NO seed met the spec; closest: seed {closest[0]} "
              f"dmin={closest[4]:.2f} nm dFL={closest[5]:.0f}")
    return None, None, None, None


def run_gates(env, seed, rows, ser, cpa, args):
    """Scripted gates M2-i / M2-ii + endangerment window sweep."""
    cpa_step, cpa_d = cpa
    first_paired = next(r for r in rows if r["pairs"])
    fls = first_paired["fls"]
    pair = sorted(fls, key=lambda cs: fls[cs])
    lower, upper = pair[0], pair[-1]
    print(f"\n  pair {pair}, FLs {fls[lower]:.0f}/{fls[upper]:.0f}, "
          f"would-be CPA step {cpa_step} ({cpa_step * SEC_PER_STEP} s) @ "
          f"{cpa_d:.2f} nm")

    # ---- gate M2-i: NOOP clean through the SYSTEM's pricing ----------
    g_noop = run_scripted(env, NoopPolicy(), seed, cbp=True)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    gate_i = (not g_noop["violated"]) and g_noop["steps"] == maxstep
    print(f"  GATE (M2-i) NOOP: G={g_noop['ep_return']:.2f} "
          f"{'clean, full ' + str(g_noop['steps']) + ' steps' if not g_noop['violated'] else 'VIOLATION[' + str(g_noop['violation_kind']) + ']'}"
          f" -> {'PASS' if gate_i else 'FAIL'}")

    # ---- gate M2-ii: single re-leveling command -> LoS; window sweep --
    windows = {}
    los_delays = []
    for target, act, name in ((lower, CLIMB, "climb(lower)"),
                              (upper, DESCEND, "descend(upper)")):
        viol_steps = []
        for s in range(0, min(cpa_step + 12, maxstep - 1)):
            st = run_scripted(env, SingleActionPolicy(target, act, s), seed)
            if st["violated"] and \
                    st["violation_kind"] == "loss_of_separation":
                vstep = st["steps"] - 1
                viol_steps.append(s)
                los_delays.append(vstep - s)
        windows[name] = viol_steps
        if viol_steps:
            print(f"  GATE (M2-ii) {name} on {target}: LoS for issue steps"
                  f" [{viol_steps[0]}..{viol_steps[-1]}] "
                  f"({len(viol_steps)} steps); LoS delay after issue "
                  f"{min(los_delays)}-{max(los_delays)} steps")
        else:
            print(f"  GATE (M2-ii) {name} on {target}: no issue step "
                  f"produces LoS")
    gate_ii = any(windows.values())
    print(f"  GATE (M2-ii) -> {'PASS' if gate_ii else 'FAIL'}")

    h_cpa = (max(los_delays) + 2) if los_delays else 15
    h_cpa = min(80, max(15, h_cpa))
    print(f"  A10 CPA-matched GT horizon: h={h_cpa} "
          f"(worst LoS delay {max(los_delays) if los_delays else '--'} "
          f"steps)")
    return {"pair": pair, "lower": lower, "upper": upper,
            "cpa_step": cpa_step, "cpa_d_nm": cpa_d,
            "gate_i": gate_i, "gate_ii": gate_ii,
            "windows": windows, "h_cpa": h_cpa,
            "noop_G": g_noop["ep_return"]}


def probe_gate(env, seed, args, h):
    """A6.2 pre-training gate: >= 5 probe states with gt_spread > 5
    spanning >= 3 of 5 trajectory segments. Reuses build_probe_set (GT:
    CBP signal, NOOP continuation) by import."""
    print(f"\n  building frozen probe set (n={args.probe_states}, "
          f"GT h={h}) ...")
    probe = build_probe_set_capped(env, seed, args.probe_states, h)
    hot = [p for p in probe if p["gt_spread"] > 5.0]
    steps = [p["step"] for p in probe]
    lo, hi = min(steps), max(steps)
    n_seg = segments_spanned([p["step"] for p in hot], lo, hi, n_seg=5)
    for p in probe:
        print(f"    probe step {p['step']:3d}: gt_spread="
              f"{p['gt_spread']:7.2f} GT(NOOP)={p['gt'][0]:+7.2f}"
              f"{'  HOT' if p['gt_spread'] > 5.0 else ''}")
    ok_n = bar("M2-pre hot probe states", len(hot), ">=", 5,
               note="gt_spread > 5")
    ok_s = bar("M2-pre segments spanned", n_seg, ">=", 3,
               note="of 5 equal trajectory segments")
    return probe, (ok_n and ok_s), {"n_hot": len(hot), "segments": n_seg}


def train_arm(env, seed, probe, gates_out, args):
    agent = make_agent(env, seed, args)
    rho_fn = lambda a: rho_eval(a, probe, args.c_train)   # noqa: E731
    out = run_training_arm(env, seed, agent, args, label="m2",
                           rho_fn=rho_fn, out_dir=OUT_DIR)

    # ---- pre-registered verdict block (the four A6 bars) -------------
    print("\n" + "=" * 74)
    print("M2 PRE-REGISTERED VERDICT BLOCK (hold-your-fire)")
    print(f"  [{bcd.conflict_pricing_str()}] episodes={args.episodes} "
          f"(SMOKE run if < 300)")
    g_eval = greedy_eval_episode(env, agent, seed, cbp=args.cbp)
    cmd_rate = g_eval["commands"] / max(1, g_eval["steps"])
    print(f"  greedy eval: G={g_eval['ep_return']:.2f} "
          f"cmd={g_eval['commands']} steps={g_eval['steps']} "
          f"{'clean' if not g_eval['violated'] else 'VIOLATION[' + str(g_eval['violation_kind']) + ']'}"
          f" issues={g_eval['issue_list'][:10]}")
    b1 = bar("M2-1 command rate/step", cmd_rate, "<", 0.1,
             note="greedy eval episode")
    nf = noop_argmax_fraction(agent, probe, args.c_train)
    b2 = bar("M2-2 NOOP net-argmax fraction", nf, ">=", 0.90,
             note="frozen probe states, eps 0")
    final_rho = out["rho_curve"][-1] if out["rho_curve"] else {"rho":
                                                              float("nan")}
    b3 = bar("M2-3 rho", final_rho["rho"], ">", 0.4,
             note=("CF-grounded arm" if getattr(args, "cf_replay", False)
                   else "EXPECTED BLOCKED without --cf_replay (untaken "
                        "candidates get no targets)"))
    frac4, devs = noop_level_hits(agent, probe, args.c_train, tol=2.0)
    b4 = bar("M2-4 |Hurwicz(NOOP)-GT(NOOP)|<2 fraction", frac4, ">=", 0.80,
             note=f"median dev {np.median(devs):.2f}")
    print(f"  M2 composite: {'MEET' if (b1 and b2 and b3 and b4) else 'MISS'}"
          f" (bars {int(b1)}{int(b2)}{int(b3)}{int(b4)})")
    print("=" * 74)

    payload = {"test": "M2", "scenario_seed": seed,
               "episodes": args.episodes, "nstep": args.nstep,
               "cf_replay": args.cf_replay,
               "gates": {k: v for k, v in gates_out.items()
                         if k != "windows"},
               "cmd_rate_eval": cmd_rate, "noop_argmax_frac": nf,
               "rho_curve": out["rho_curve"],
               "noop_level_frac": frac4,
               "bars": {"b1_cmd_rate": b1, "b2_noop_argmax": b2,
                        "b3_rho": b3, "b4_noop_level": b4},
               "wall_seconds": out["wall_seconds"]}
    write_summary(OUT_DIR, f"m2_summary_seed{seed}", payload)
    return payload


def main():
    ap = argparse.ArgumentParser(description="M2 hold-your-fire micro test")
    base_argparser(ap)
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--scan_seeds", type=int, nargs="+",
                    default=[10043, 20042] + list(range(100, 260)))
    args = ap.parse_args()
    announce("M2 HOLD-YOUR-FIRE (2 starters, level-separated pair, "
             "spawn 0.0)", args)
    env = make_m2_env(duration=args.duration)

    if args.scenario_seed is not None:
        seeds = [args.scenario_seed]
    else:
        seeds = args.scan_seeds
        print("scanning for a level-separated laterally-proximate NOOP-"
              "clean seed ...")
    seed, rows, ser, cpa = find_scenario(env, seeds)
    assert seed is not None, ("no M2 scenario in the scan range — report "
                              "the scan, do not relax the spec")
    print(f"scenario seed: {seed}")

    gates_out = run_gates(env, seed, rows, ser, cpa, args)
    assert gates_out["gate_i"] and gates_out["gate_ii"], \
        "M2 scripted gates failed — do not train on this scenario"
    h = args.h if args.h is not None else gates_out["h_cpa"]

    probe, pg_ok, pg = probe_gate(env, seed, args, h)
    if not pg_ok:
        print("  PRE-TRAINING GATE FAILED — re-seed before training")
    if args.train:
        assert pg_ok, "A6.2 probe-set gate failed; refusing to train"
        train_arm(env, seed, probe, gates_out, args)
    elif not args.train:
        write_summary(OUT_DIR, f"m2_gates_seed{seed}",
                      {"test": "M2", "scenario_seed": seed,
                       "gates": {k: v for k, v in gates_out.items()
                                 if k != "windows"},
                       "windows": gates_out["windows"],
                       "probe_gate": pg, "probe_gate_pass": pg_ok,
                       "h": h})


if __name__ == "__main__":
    main()
