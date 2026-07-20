"""
M5 delivery (run-13 micro battery; RUN13_FIX_C_DESIGN.md S3, A7)
================================================================

Scenario: a SINGLE aircraft, empty sector, where doing nothing NEVER
delivers — a heading/route_parallel clearance is REQUIRED to reach the
exit. This makes delivery ACTION-CONTINGENT, the only way the delivery
bonus acquires candidate-ordering content (A7: on a clean-transit seed a
never-commanding degenerate net delivers ~1.0/ep and the positive-credit
pathway — 0 deliveries in 4000 12d episodes — is never exercised).

Scenario mechanics note (documented deviation from A7's ideal): the env
exposes NO off-route spawn knob (CustomInfiniteEnv spawns on routes;
verified 19 Jul), so the finder uses SEED SEARCH over single-aircraft
episodes under wind. Selected seeds have NOOP -> 0 deliveries
(sector_excursion; pre-registered as exactly 0) while a single EARLY
clearance delivers cleanly. The finder prefers seeds whose earliest
working clearance step is <= 5 — functionally an off-route spawn (the
correction is required from spawn time, not a late boundary-pressure
retrim); the achieved earliest step is printed and recorded.

Scripted gates (pre-registered, run before any training):
  (M5-i)  NOOP through bcd.run_episode -> deliveries == 0 (registered
          next to the >= 0.8 bar, per A7.2);
  (M5-ii) ONE scripted clearance -> >= 1 delivery, no violation.

PRE-REGISTERED BARS (A7-repaired):
  (B1) deliveries/ep >= 0.8 (trailing 100 episodes);
  (B2) command rate over OCCUPIED steps < 0.2 — denominator = steps with
       >= 1 aircraft in obs (the episode runs to maxstep after the
       delivery empties obs, so a raw per-step rate is diluted; A7.3).
       Occupancy is counted in HarnessAgent.generate_action (bcd's
       step_hook carries no obs count — documented in
       micro_battery_common).

Usage:
    .venv/bin/python micro_battery_m5.py --gates
    .venv/bin/python micro_battery_m5.py --gates --train --episodes 15
"""

import argparse
import os

import numpy as np

import bluebird_controller_dqn as bcd
from micro_battery_common import (
    bar, base_argparser, announce, write_summary, make_agent,
    run_training_arm, greedy_eval_episode, BATTERY_DIR,
    L10, R10, ROUTE_PARALLEL, SEC_PER_STEP,
)
from micro_level_allocation import SingleActionPolicy
from diagnose_controller import (
    make_custom_density_env, NoopPolicy, run_scripted,
)

OUT_DIR = os.path.join(BATTERY_DIR, "m5")

ACT_NAMES = {L10: "L10", R10: "R10", ROUTE_PARALLEL: "route_parallel"}
SPAWN_CONTINGENT_STEP = 5    # earliest-clearance threshold to call the
                             # seed off-route-spawn-like (see docstring)


def make_m5_env(duration=1800):
    return make_custom_density_env(duration=duration,
                                   initial_spawn_rate=0.0,
                                   max_spawn_rate=0.0,
                                   num_starter_aircraft=1)


def clearance_sweep(env, seed, target, args):
    """Sweep single clearances (route_parallel first, then headings) over
    issue steps; returns list of working (act, step, deliveries, G)."""
    working = []
    for act in (ROUTE_PARALLEL, L10, R10):
        for s in range(0, args.sweep_max + 1, args.sweep_stride):
            st = run_scripted(env, SingleActionPolicy(target, act, s),
                              seed, cbp=True)
            if st["deliveries"] >= 1 and not st["violated"]:
                working.append((act, s, st["deliveries"],
                                st["ep_return"]))
    return working


def find_scenario(env, seeds, args):
    """First seed with NOOP deliveries == 0 AND a working single
    clearance, preferring earliest-step clearances (spawn-contingent).
    Returns (seed, noop_stats, working list)."""
    fallback = None
    for seed in seeds:
        g = run_scripted(env, NoopPolicy(), seed, cbp=True)
        tag = (f"VIOL[{g['violation_kind']}]@{g['time_to_violation']}s"
               if g["violated"] else "clean")
        if g["deliveries"] != 0:
            print(f"  seed {seed}: NOOP delivers {g['deliveries']} "
                  f"({tag}) — not action-contingent, skip")
            continue
        # single aircraft callsign: probe one obs
        obs, _ = env.reset(seed=seed)
        step = 0
        while not obs and step < 50:
            obs = env.step({})[0]
            step += 1
        target = sorted(obs)[0]
        working = clearance_sweep(env, seed, target, args)
        if not working:
            print(f"  seed {seed}: NOOP 0 deliveries ({tag}) but NO "
                  f"single clearance delivers — skip")
            continue
        earliest = min(working, key=lambda w: w[1])
        print(f"  seed {seed}: NOOP 0 deliveries ({tag}); "
              f"{len(working)} working clearances, earliest "
              f"{ACT_NAMES[earliest[0]]}@step {earliest[1]}"
              + ("   <- M5 SCENARIO (spawn-contingent)"
                 if earliest[1] <= SPAWN_CONTINGENT_STEP else ""))
        if earliest[1] <= SPAWN_CONTINGENT_STEP:
            return seed, g, working
        # fallback: the seed with the EARLIEST working clearance across
        # the scan (closest to A7's off-route-spawn intent; a late-step
        # correction is drift-dominated boundary retrim)
        if fallback is None or earliest[1] < fallback[3]:
            fallback = (seed, g, working, earliest[1])
    if fallback is not None:
        seed, g, working, e_step = fallback
        print(f"  no spawn-contingent seed (none <= step "
              f"{SPAWN_CONTINGENT_STEP}); falling back to the BEST-"
              f"EARLIEST seed {seed} (earliest clearance step {e_step}; "
              f"A7 deviation recorded)")
        return seed, g, working
    return None, None, None


def run_gates(env, seed, noop_stats, working, args):
    g = noop_stats
    gate_i = (g["deliveries"] == 0)
    print(f"\n  GATE (M5-i) NOOP: deliveries={g['deliveries']} "
          f"G={g['ep_return']:.2f} "
          f"{'VIOLATION[' + str(g['violation_kind']) + ']@' + str(g['time_to_violation']) + 's' if g['violated'] else 'clean'}"
          f" -> {'PASS (pre-registered 0)' if gate_i else 'FAIL'}")
    act, s, ndel, G = min(working, key=lambda w: w[1])
    st = None
    for a_, s_, d_, G_ in sorted(working, key=lambda w: w[1]):
        print(f"    working: {ACT_NAMES[a_]:14s} @ step {s_:3d} -> "
              f"{d_} delivery, G={G_:+.2f}")
    gate_ii = ndel >= 1
    print(f"  GATE (M5-ii) single {ACT_NAMES[act]} @ step {s}: "
          f"{ndel} delivery, G={G:+.2f} -> "
          f"{'PASS' if gate_ii else 'FAIL'}")
    return {"gate_i": gate_i, "gate_ii": gate_ii,
            "clearance": (ACT_NAMES[act], s), "clearance_G": G,
            "noop_G": g["ep_return"],
            "noop_violation": g["violation_kind"],
            "working": [(ACT_NAMES[a_], s_, d_, G_)
                        for a_, s_, d_, G_ in working]}


def train_arm(env, seed, gates_out, args):
    agent = make_agent(env, seed, args)
    out = run_training_arm(env, seed, agent, args, label="m5",
                           rho_fn=None, out_dir=OUT_DIR)

    print("\n" + "=" * 74)
    print("M5 PRE-REGISTERED VERDICT BLOCK (delivery)")
    print(f"  [{bcd.conflict_pricing_str()}] episodes={args.episodes} "
          f"(SMOKE run if < 300)")
    trail = out["hist"][-100:]
    deliv = float(np.mean([h["deliveries"] for h in trail]))
    occ_rates = [h["commands"] / max(1, h["occupied_steps"])
                 for h in trail]
    occ_rate = float(np.mean(occ_rates))
    raw_rates = [h["commands"] / max(1, h["steps"]) for h in trail]
    print(f"  occupied-steps denominator: mean occupied "
          f"{np.mean([h['occupied_steps'] for h in trail]):.0f} of "
          f"{np.mean([h['steps'] for h in trail]):.0f} steps/ep "
          f"(raw cmd rate {np.mean(raw_rates):.3f} — diluted, "
          f"report-only)")
    b1 = bar("M5-1 deliveries/ep (trailing 100)", deliv, ">=", 0.8,
             note="NOOP baseline pre-registered at exactly 0")
    b2 = bar("M5-2 cmd rate over OCCUPIED steps", occ_rate, "<", 0.2)
    g_eval = greedy_eval_episode(env, agent, seed, cbp=args.cbp)
    print(f"  (report) greedy eval: deliveries={g_eval['deliveries']} "
          f"cmd={g_eval['commands']} occupied={g_eval['occupied_steps']} "
          f"G={g_eval['ep_return']:.2f}")
    print(f"  M5 composite: {'MEET' if (b1 and b2) else 'MISS'} "
          f"(bars {int(b1)}{int(b2)})")
    print("=" * 74)

    payload = {"test": "M5", "scenario_seed": seed,
               "episodes": args.episodes, "nstep": args.nstep,
               "cf_replay": args.cf_replay, "gates": gates_out,
               "deliveries_trail": deliv, "occ_cmd_rate": occ_rate,
               "greedy_eval": {"deliveries": g_eval["deliveries"],
                               "commands": g_eval["commands"],
                               "occupied": g_eval["occupied_steps"]},
               "bars": {"b1_deliveries": b1, "b2_occ_cmd_rate": b2},
               "wall_seconds": out["wall_seconds"]}
    write_summary(OUT_DIR, f"m5_summary_seed{seed}", payload)
    return payload


def main():
    ap = argparse.ArgumentParser(description="M5 delivery micro test")
    base_argparser(ap)
    ap.add_argument("--duration", type=int, default=1800,
                    help="single-aircraft transits need 700-1500 s "
                         "(B4 finding); 1800 s = 300 steps")
    ap.add_argument("--scan_seeds", type=int, nargs="+",
                    default=[20042] + list(range(100, 140)))
    ap.add_argument("--sweep_max", type=int, default=60)
    ap.add_argument("--sweep_stride", type=int, default=5)
    args = ap.parse_args()
    announce("M5 DELIVERY (1 starter, spawn 0.0, action-contingent "
             "delivery)", args)
    env = make_m5_env(duration=args.duration)

    if args.scenario_seed is not None:
        seeds = [args.scenario_seed]
    else:
        seeds = args.scan_seeds
        print("scanning single-aircraft seeds (NOOP must deliver 0; a "
              "single clearance must deliver) ...")
    seed, noop_stats, working = find_scenario(env, seeds, args)
    assert seed is not None, ("no M5 scenario in the scan range — report "
                              "the scan, do not relax the spec")
    print(f"scenario seed: {seed}")

    gates_out = run_gates(env, seed, noop_stats, working, args)
    assert gates_out["gate_i"] and gates_out["gate_ii"], \
        "M5 scripted gates failed — do not train on this scenario"
    if args.train:
        train_arm(env, seed, gates_out, args)
    else:
        write_summary(OUT_DIR, f"m5_gates_seed{seed}",
                      {"test": "M5", "scenario_seed": seed,
                       "gates": gates_out})


if __name__ == "__main__":
    main()
