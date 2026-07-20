"""T2 + T3 overnight audit measurement (20 July 2026).

T2: cf-target decomposition trajectory. For each round-1 M2 checkpoint
(ep200/400/600/800 of the 20260719_205235 run), replay the 20 frozen
NOOP probe states of the M2 scenario (seed 167) through the PRODUCTION
CF path — bcd.cf_branch_rollout with the checkpoint agent as the frozen
continuation policy, first_idx = NOOP, H=36, cbp=True,
cbp_lag=bcd.CBP_LAG, objective_v2=True. Record per state: G_cf, disc,
the bootstrap tail term disc * midpoint(Q at s_H, stored next_a) via
agent.candidate_q on the returned ns tokens, the full target y, the
violation step if any, and whether the branch simulated past the
100-step episode end.

T3: the disc=0 (pure Monte Carlo) target is G_cf alone — recorded per
state per checkpoint as the contamination floor the running pure-MC arm
(m2b) can at best converge to.

Read-only with respect to project sources; writes JSON to
audit_measurements/. Pricing: smooth/0.5 set before anything prices.
"""
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import sector_snapshot
from diagnose_controller import collect_probe_states, noop_action
from micro_level_allocation import make_micro_env

OUT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join("checkpoints", "micro_battery", "m2")
CKPTS = [(ep, os.path.join(CKPT_DIR, f"m2_seed167_20260719_205235_ep{ep}.pt"))
         for ep in (200, 400, 600, 800)]
SEED = 167
H = 36


def main():
    bcd.set_conflict_pricing("smooth", 0.5)
    print(f"[t2t3] {bcd.conflict_pricing_str()}")
    env = make_micro_env(duration=600)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // bcd.SEC_PER_STEP)) \
        if hasattr(bcd, "SEC_PER_STEP") else 100
    # SEC_PER_STEP lives in bluebird_interval_dqn; compute robustly:
    from bluebird_interval_dqn import SEC_PER_STEP
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    print(f"[t2t3] episode maxstep = {maxstep}")

    print("[t2t3] collecting 20 frozen NOOP probe states (seed 167) ...")
    t0 = time.time()
    states = collect_probe_states(
        env, SEED, lambda e, o, i: (noop_action(o), False), 20,
        min_aircraft=2)
    print(f"[t2t3] {len(states)} states at steps "
          f"{[s[3] for s in states]} in {time.time() - t0:.0f}s")

    results = {"pricing": bcd.conflict_pricing_str(), "H": H,
               "cbp_lag": bcd.CBP_LAG, "scenario_seed": SEED,
               "maxstep": maxstep, "first_idx": 0,
               "note_tail": ("tail uses agent.candidate_q (q_net); "
                             "load_agent copies q_net into target_net so "
                             "q_net midpoint == target-net midpoint here"),
               "checkpoints": {}}

    for ep, path in CKPTS:
        assert os.path.exists(path), path
        agent, ckpt = bcd.load_agent(path)
        rows = []
        t1 = time.time()
        for env_s, obs_s, stratum, step in states:
            snap_s = sector_snapshot(env_s, obs_s.keys())
            rec = bcd.cf_branch_rollout(
                env_s, copy.deepcopy(obs_s), snap_s, {}, agent,
                0, H, True, bcd.CBP_LAG, True, budget=None)
            g_cf = float(rec["g_cf"])
            disc = float(rec["disc"])
            steps_run = int(rec["steps"])
            violated = bool(rec["violated"])
            # branch step kk simulates absolute episode step (step + kk)
            viol_abs_step = (step + steps_run - 1) if violated else None
            past_end = (step + steps_run) > maxstep
            tail = 0.0
            q_mid_next = None
            if not violated and rec["ns"][1]:
                toks = rec["ns"][0]
                cl, cu = agent.candidate_q(toks)
                na = int(rec["next_a"])
                q_mid_next = float((cl[na] + cu[na]) / 2.0)
                tail = disc * q_mid_next
            rows.append({
                "probe_step": int(step), "stratum": int(stratum),
                "g_cf": g_cf, "disc": disc,
                "q_mid_at_next_a": q_mid_next, "next_a": int(rec["next_a"]),
                "tail_term": float(tail), "y_full": float(g_cf + tail),
                "violated": violated,
                "violation_abs_step": viol_abs_step,
                "branch_steps_run": steps_run,
                "ran_past_episode_end": bool(past_end)})
            print(f"  ep{ep} s{step:3d}: G_cf={g_cf:8.2f} disc={disc:.3f} "
                  f"tail={tail:8.2f} y={g_cf + tail:8.2f} "
                  f"viol={violated} past_end={past_end}")
        nv = sum(r["violated"] for r in rows)
        npe = sum(r["ran_past_episode_end"] for r in rows)
        results["checkpoints"][str(ep)] = {
            "path": path, "rows": rows,
            "n_violated": nv, "n_past_end": npe,
            "gcf_mean": float(np.mean([r["g_cf"] for r in rows])),
            "y_mean": float(np.mean([r["y_full"] for r in rows])),
            "wall_s": round(time.time() - t1, 1)}
        print(f"[t2t3] ep{ep}: {nv}/20 branches violated, {npe}/20 past "
              f"episode end, mean G_cf "
              f"{results['checkpoints'][str(ep)]['gcf_mean']:.2f}, "
              f"{time.time() - t1:.0f}s")

    with open(os.path.join(OUT, "cf_decomposition_by_ckpt.json"), "w") as f:
        json.dump(results, f, indent=1)

    # T3: pure-MC floor = the G_cf column alone
    floor = {"note": ("disc=0 target profile (G_cf alone) per state per "
                      "checkpoint — the profile the pure-MC arm m2b can "
                      "at best converge to; NOT zero"),
             "pricing": results["pricing"], "H": H,
             "scenario_seed": SEED,
             "checkpoints": {
                 ep: {"probe_steps": [r["probe_step"]
                                      for r in results["checkpoints"][ep]["rows"]],
                      "g_cf": [r["g_cf"]
                               for r in results["checkpoints"][ep]["rows"]],
                      "violated": [r["violated"]
                                   for r in results["checkpoints"][ep]["rows"]],
                      "frac_violating": float(np.mean(
                          [r["violated"]
                           for r in results["checkpoints"][ep]["rows"]]))}
                 for ep in results["checkpoints"]}}
    with open(os.path.join(OUT, "puremc_floor.json"), "w") as f:
        json.dump(floor, f, indent=1)
    print("[t2t3] wrote cf_decomposition_by_ckpt.json and puremc_floor.json")


if __name__ == "__main__":
    main()
