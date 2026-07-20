"""
D5 scripted probes G1/G2 (D5_VERTICAL_POTENTIAL_DESIGN.md, review
amendments; implemented 2026-07-18). No training, report-only — but G2
is PROMOTED TO GATE per the adversarial review: if escaping a conflict
vertically (climbing out of the sector band) is priced materially BETTER
than resolving it, that is an open exploit and must be flagged
prominently before the smooth flag trains.

G1 — SPAWN PREMISE: distribution of |selected_fl - fl| at first
    appearance across ~20 scenario seeds (reset starters AND mid-episode
    spawns under all-NOOP). The D5 p1 arithmetic (design section 2.2)
    assumes co-level pairs are also co-COMMANDED (g_cmd(0) = 1 at the
    NOOP branch); if aircraft spawn mid-climb (sel != fl ambiently), the
    NOOP-branch commanded factor is not 1 and per-state p1 must be
    computed from actual sel values (the d2phi analytic already does).

G2 — OUT_SECTOR / BAND-EDGE (GATE): on the micro 2-aircraft co-level
    conflict, compare
      RESOLVE arm: one well-timed +10 climb (stays in band)
      ESCAPE arm : repeated climbs pushing the same aircraft through the
                   sector band top
    under the CBP training signal (bcd.run_episode(cbp=True) itself).
    Report: does the escaper exit the sector, which terminal fires, what
    the episode returns are, and the cumulative conflict-term CBP each
    arm collects. GATE: escape must NOT out-score resolve; if it does
    (or comes close), FLAG as an open exploit with a mitigation option
    (do not implement mitigation without flagging).

Usage:
    .venv/bin/python d5_vertical_probes.py --g1 --g2 \
        --vertical_ramp smooth --delta_conflict 0.5
"""

import argparse
import sys
import time

import numpy as np
import torch

torch.set_num_threads(2)

import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import make_controller_env, sector_snapshot
from bluebird_interval_dqn import detect_violation, SEC_PER_STEP
from diagnose_controller import (make_custom_density_env, ScriptedShim,
                                 tracked, pos_status_name)
from micro_level_allocation import (make_micro_env, find_conflict_seed,
                                    noop_trace, SingleActionPolicy, CLIMB)
from bluebird_controller_dqn import run_episode

sys.stdout.reconfigure(line_buffering=True)


# ===========================================================================
# G1 — spawn premise
# ===========================================================================

def probe_g1(args):
    print("\n" + "=" * 78)
    print(f"G1 — SPAWN PREMISE: |selected_fl - fl| at first appearance "
          f"({len(args.g1_seeds)} seeds, {args.g1_steps} all-NOOP steps "
          f"each) [{bcd.conflict_pricing_str()}]")
    print("=" * 78)
    env = make_controller_env(scenario_duration=args.g1_steps
                              * SEC_PER_STEP)
    at_reset, mid_spawn = [], []
    for seed in args.g1_seeds:
        obs, _info = env.reset(seed=seed)
        sim = env.get_simulator_env()
        seen = set()
        for cs, ac in sim.aircraft.items():
            if ac.fl is None or ac.selected_fl is None:
                continue
            at_reset.append(abs(float(ac.selected_fl) - float(ac.fl)))
            seen.add(cs)
        for step in range(args.g1_steps):
            obs, _r, _d, _t, info = env.step({cs: 0 for cs in obs})
            sim = env.get_simulator_env()
            for cs, ac in sim.aircraft.items():
                if cs in seen:
                    continue
                seen.add(cs)
                if ac.fl is None or ac.selected_fl is None:
                    continue
                mid_spawn.append(abs(float(ac.selected_fl)
                                     - float(ac.fl)))
            v, _k, _inv = detect_violation(info)
            if v:
                break
    env.close()

    def summarize(name, vals):
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            print(f"  {name}: none observed")
            return {}
        nz = vals[vals > 0.5]
        print(f"  {name}: n={len(vals)}, |sel-fl| > 0.5 FL in "
              f"{len(nz)}/{len(vals)} ({100 * len(nz) / len(vals):.1f}%)"
              + (f"; nonzero diffs: min {nz.min():.1f}, median "
                 f"{np.median(nz):.1f}, max {nz.max():.1f} FL"
                 if len(nz) else ""))
        return {"n": int(len(vals)), "n_mid_climb": int(len(nz)),
                "frac_mid_climb": float(len(nz) / len(vals)),
                "nonzero_diffs": sorted(float(v) for v in nz)}

    r_reset = summarize("at reset (starters)", at_reset)
    r_mid = summarize("mid-episode spawns", mid_spawn)
    frac = max(r_reset.get("frac_mid_climb", 0.0),
               r_mid.get("frac_mid_climb", 0.0))
    if frac > 0.0:
        print("  IMPLICATION for the p1 arithmetic: aircraft DO appear "
              "mid-climb (sel != fl ambiently), so the design's co-level "
              "=> co-commanded assumption (NOOP-branch g_cmd = 1) does "
              "not always hold; per-state analytic p1 must be computed "
              "from actual sel values — the d2phi analytic and the "
              "selftest formula recomputations already do this.")
    else:
        print("  IMPLICATION: every observed aircraft appears with "
              "sel == fl; the co-level => co-commanded p1 arithmetic of "
              "design section 2.2 holds ambiently (climb-in-progress "
              "states arise only from issued clearances).")
    return {"probe": "g1", "at_reset": r_reset, "mid_spawn": r_mid}


# ===========================================================================
# G2 — OUT_SECTOR band-edge gate
# ===========================================================================

class RepeatedClimbPolicy:
    """Climb `target` every `period` steps from `start` until `n_max`
    climbs are out (escape arm). Global NOOP otherwise."""
    name = "ESCAPER"

    def __init__(self, target_cs, start, period=4, n_max=12):
        self.target = target_cs
        self.start = start
        self.period = period
        self.n_max = n_max
        self.n = 0
        self.t = -1

    def decide(self, env, obs):
        self.t += 1
        if (self.target in obs and self.n < self.n_max
                and self.t >= self.start
                and (self.t - self.start) % self.period == 0):
            self.n += 1
            return self.target, CLIMB
        return None, 0


def run_arm(env, policy, seed, label):
    hooks = []
    stats = run_episode(env, ScriptedShim(policy), seed=seed, train=False,
                        cbp=True, step_hook=hooks.append)
    conf_paid = sum(h["paid_conflict"] for h in hooks)
    conf_paid_cmd = sum(h["paid_conflict"] for h in hooks if h["issued"])
    print(f"  {label:10s}: G={stats['ep_return']:8.3f} "
          f"cmd={stats['commands']:2d} "
          f"{'VIOLATION[' + str(stats['violation_kind']) + ']@' + str(stats['time_to_violation']) + 's' if stats['violated'] else 'clean'}"
          f" | conflict-CBP total {conf_paid:+.4f} "
          f"(at command steps {conf_paid_cmd:+.4f})")
    return stats, hooks, conf_paid


def probe_g2(args):
    print("\n" + "=" * 78)
    print(f"G2 — OUT_SECTOR / BAND-EDGE (PROMOTED TO GATE) "
          f"[{bcd.conflict_pricing_str()}]")
    print("=" * 78)
    env = make_micro_env(duration=600)
    print("  scanning for the micro conflict seed ...")
    seed, rows, vstep, cpa_step, cpa_d = find_conflict_seed(
        env, [args.g2_seed] if args.g2_seed else
        [10043, 20042] + list(range(100, 160)))
    assert seed is not None, "no micro conflict seed found"
    paired = [r for r in rows if "d_nm" in r]
    pair = list(paired[0]["pair"])
    fls = paired[0]["fls"]
    print(f"  scenario seed {seed}: pair {pair} at FLs {fls}, NOOP LoS "
          f"at step {vstep}")
    # climb target: the higher member (its climbs open the pair gap AND
    # walk it toward the band top). Band top from the env's own sector
    # volumes (X-Plus: FL200-300).
    hi_idx = int(np.argmax(fls))
    target = pair[hi_idx]
    try:
        sim = env.get_simulator_env()
        sec = sim.airspace.sectors[env.active_airspace_sector]
        band = (min(float(v.min_fl) for v in sec.volumes),
                max(float(v.max_fl) for v in sec.volumes))
    except Exception:
        band = (200.0, 300.0)
    fl0 = fls[hi_idx]
    n_to_top = int(np.ceil((band[1] - fl0) / 10.0)) + 1
    print(f"  target {target} at FL{fl0:.0f}; sector band "
          f"FL{band[0]:.0f}-{band[1]:.0f}; {n_to_top} climbs command it "
          f"past the top")

    # a climb step both arms share (from the micro gate-(ii) window
    # logic: early enough to resolve)
    start = max(0, vstep - 25)
    print(f"  issue schedule: RESOLVE = one +10 climb at step {start}; "
          f"ESCAPE = +10 climbs every 4 steps from step {start} "
          f"({n_to_top + 1} max)")
    s_res, h_res, c_res = run_arm(
        env, SingleActionPolicy(target, CLIMB, start), seed, "RESOLVE")
    s_esc, h_esc, c_esc = run_arm(
        env, RepeatedClimbPolicy(target, start, period=4,
                                 n_max=n_to_top + 1), seed, "ESCAPE")

    exited = s_esc["violated"] and \
        s_esc["violation_kind"] == "sector_excursion"
    if exited:
        esc_str = ("EXITED the sector (OUT_SECTOR -> sector_excursion "
                   "terminal, -50)")
    elif s_esc["violated"]:
        esc_str = ("did NOT exit the sector: "
                   f"VIOLATION[{s_esc['violation_kind']}]")
    else:
        esc_str = "did NOT exit the sector: clean episode"
    print(f"\n  escape arm {esc_str}")
    gap = s_res["ep_return"] - s_esc["ep_return"]
    gate_ok = gap > 1.0   # resolve must beat escape by a clear margin
    print(f"  GATE: resolve G={s_res['ep_return']:.3f} vs escape "
          f"G={s_esc['ep_return']:.3f} (gap {gap:+.3f}) -> "
          f"{'PASS (escape priced clearly worse)' if gate_ok else 'FAIL'}")
    if not gate_ok:
        print("  !! OPEN EXPLOIT FLAG !!  Escaping a conflict vertically "
              "is NOT priced materially worse than resolving it. "
              "Mitigation option (NOT implemented, per amendment G2): "
              "price commanded-out-of-band selected FLs in the conflict "
              "potential (extend g_cmd's support to the band edge) or "
              "add an explicit out-of-band command penalty; either needs "
              "its own review before landing.")
    return {"probe": "g2", "seed": seed, "target": target,
            "band": band, "resolve": s_res, "escape": s_esc,
            "resolve_conflict_cbp": c_res, "escape_conflict_cbp": c_esc,
            "escape_exited": bool(exited), "gap": gap,
            "gate_ok": bool(gate_ok)}


def main():
    ap = argparse.ArgumentParser(description="D5 scripted probes G1/G2")
    ap.add_argument("--g1", action="store_true")
    ap.add_argument("--g2", action="store_true")
    ap.add_argument("--g1_seeds", type=int, nargs="+",
                    default=[10043, 20042] + list(range(70042, 70060)))
    ap.add_argument("--g1_steps", type=int, default=150)
    ap.add_argument("--g2_seed", type=int, default=None)
    ap.add_argument("--vertical_ramp", type=str, default="gate",
                    choices=["gate", "smooth"])
    ap.add_argument("--delta_conflict", type=float, default=None)
    args = ap.parse_args()
    bcd.set_conflict_pricing(args.vertical_ramp, args.delta_conflict)
    print(f"[d5_vertical_probes] {bcd.conflict_pricing_str()}")
    if not (args.g1 or args.g2):
        ap.error("select --g1 and/or --g2")
    t0 = time.time()
    if args.g1:
        probe_g1(args)
    if args.g2:
        probe_g2(args)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
