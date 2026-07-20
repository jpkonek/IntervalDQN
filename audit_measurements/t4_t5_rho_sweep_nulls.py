"""T4 + T5 overnight audit measurements (20 July 2026).

T4 width-collapse control rescue: for every round-1 M1 checkpoint
(micro_seed134_20260719_205216_ep*.pt) and every round-1 M3 checkpoint
(m3_seed134_20260719_205320_ep*.pt), re-run the module's own rho
instrument at c=0 (pure lower bound — width-free ordering), c=0.5 (the
training c, where Hurwicz == interval midpoint, the audit's vacuous
identity point) and c=1 (pure upper bound), plus interval-width
quantiles (25/50/75) over the probe candidates.
  M1 instrument: micro_level_allocation.rho_eval on
    micro_level_allocation.build_probe_set(env, 134, 20, 69) — exactly
    the round-1 run's probe (its log: "GT H=69", 18 hot).
  M3 instrument: the m3 harness rho —
    micro_battery_common.rho_eval_subset(agent, probe_capped, c,
    micro_battery_m3.lateral_keep) on
    build_probe_set_capped(env, 134, 20, 69).

T5 untrained null bands: 30 fresh random-init ControllerAgent nets
(token_dim 60, n_instr 5 — matched to the checkpoints; torch seed
varied) scored by the SAME two rho instruments at c in {0, 0.5, 1};
mean/sd/2.5-97.5 percentile bands archived.

Pricing smooth/0.5 set before the probe GT is built (GT rollouts price
through bcd). Writes rho_c_sweep.json + null_bands.json.
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import ControllerAgent, N_INSTR
from micro_level_allocation import make_micro_env, build_probe_set, rho_eval
from micro_battery_common import build_probe_set_capped, rho_eval_subset
from micro_battery_m3 import lateral_keep

OUT = os.path.dirname(os.path.abspath(__file__))
M1_GLOB = "checkpoints/micro_level_allocation/micro_seed134_20260719_205216_ep*.pt"
M3_GLOB = "checkpoints/micro_battery/m3/m3_seed134_20260719_205320_ep*.pt"
CS = (0.0, 0.5, 1.0)


def ep_of(p):
    return int(re.search(r"_ep(\d+)\.pt$", p).group(1))


def widths_quantiles(agent, probe, keep_fn=None):
    ws = []
    for p in probe:
        cl, cu = agent.candidate_q(p["tokens"])
        w = (cu - cl)[:p["n_cand"]]
        if keep_fn is not None:
            k = np.array([keep_fn(i) for i in range(p["n_cand"])])
            w = w[k]
        ws.extend(w.tolist())
    q = np.percentile(ws, [25, 50, 75])
    return {"q25": float(q[0]), "q50": float(q[1]), "q75": float(q[2]),
            "n": len(ws)}


def sweep_ckpt(agent, probe, rho_fn, keep_fn=None):
    out = {}
    for c in CS:
        r = rho_fn(agent, probe, c)
        out[f"c{c}"] = {k: (None if isinstance(v, float) and v != v else v)
                        for k, v in r.items()}
    out["widths_full"] = widths_quantiles(agent, probe)
    if keep_fn is not None:
        out["widths_lateral"] = widths_quantiles(agent, probe, keep_fn)
    return out


def main():
    bcd.set_conflict_pricing("smooth", 0.5)
    print(f"[t4t5] {bcd.conflict_pricing_str()}")
    env = make_micro_env(duration=600)

    print("[t4t5] building M1 probe set (build_probe_set, h=69) ...")
    probe_m1 = build_probe_set(env, 134, 20, 69)
    print("[t4t5] building M3 probe set (build_probe_set_capped, h=69) ...")
    probe_m3 = build_probe_set_capped(env, 134, 20, 69)

    def rho_m3(agent, probe, c):
        return rho_eval_subset(agent, probe, c, lateral_keep)

    res = {"pricing": bcd.conflict_pricing_str(),
           "note": ("c=0.5 Hurwicz score IS the interval midpoint "
                    "(l + 0.5(u-l) = (l+u)/2) — the audit's vacuous "
                    "rho-vs-rho_mid identity; c=0 (lower bound) and c=1 "
                    "(upper bound) are the width-sensitive controls"),
           "m1": {}, "m3": {}}

    for label, gpat, probe, rho_fn, keep in (
            ("m1", M1_GLOB, probe_m1,
             lambda a, p, c: rho_eval(a, p, c), None),
            ("m3", M3_GLOB, probe_m3, rho_m3, lateral_keep)):
        paths = sorted(glob.glob(gpat), key=ep_of)
        print(f"[t4t5] {label}: {len(paths)} checkpoints")
        for path in paths:
            ep = ep_of(path)
            agent, _ck = bcd.load_agent(path)
            res[label][str(ep)] = sweep_ckpt(agent, probe, rho_fn, keep)
            r = res[label][str(ep)]

            def _f(x):
                return "nan" if x is None else f"{x:+.3f}"
            print(f"  {label} ep{ep:4d}: "
                  f"rho c0={_f(r['c0.0']['rho'])} "
                  f"c0.5={_f(r['c0.5']['rho'])} "
                  f"c1={_f(r['c1.0']['rho'])} "
                  f"| w50={r['widths_full']['q50']:.3f}")

    with open(os.path.join(OUT, "rho_c_sweep.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("[t4t5] wrote rho_c_sweep.json")

    # ---- T5 null bands -------------------------------------------------
    token_dim = 60  # matched: round-1 logs "agent: token_dim 60, n_instr 5"
    nulls = {"m1": {f"c{c}": {"rho": [], "rho_hot": []} for c in CS},
             "m3": {f"c{c}": {"rho": [], "rho_hot": []} for c in CS}}
    t0 = time.time()
    for i in range(30):
        torch.manual_seed(7000 + i)
        agent = ControllerAgent(token_dim=token_dim, n_instr=N_INSTR,
                                device="cpu")
        agent.q_net.eval()
        for c in CS:
            r1 = rho_eval(agent, probe_m1, c)
            r3 = rho_eval_subset(agent, probe_m3, c, lateral_keep)
            nulls["m1"][f"c{c}"]["rho"].append(r1["rho"])
            nulls["m1"][f"c{c}"]["rho_hot"].append(r1["rho_hot"])
            nulls["m3"][f"c{c}"]["rho"].append(r3["rho"])
            nulls["m3"][f"c{c}"]["rho_hot"].append(r3["rho_hot"])
    print(f"[t4t5] 30 null nets scored in {time.time() - t0:.0f}s")

    def band(v):
        v = np.asarray(v, float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "p2.5": float(np.percentile(v, 2.5)),
                "p97.5": float(np.percentile(v, 97.5)),
                "min": float(v.min()), "max": float(v.max()),
                "values": [float(x) for x in v]}

    out5 = {"n_inits": 30, "token_dim": token_dim, "n_instr": N_INSTR,
            "torch_seeds": "7000..7029",
            "pricing": bcd.conflict_pricing_str(),
            "instruments": {
                "m1": "micro_level_allocation.rho_eval, probe h=69",
                "m3": ("micro_battery_common.rho_eval_subset + "
                       "micro_battery_m3.lateral_keep, capped probe h=69")},
            "bands": {m: {ck: {stat: band(vals)
                               for stat, vals in d.items()}
                          for ck, d in nulls[m].items()}
                      for m in nulls}}
    with open(os.path.join(OUT, "null_bands.json"), "w") as f:
        json.dump(out5, f, indent=1)
    for m in ("m1", "m3"):
        b = out5["bands"][m]["c0.5"]["rho"]
        print(f"[t4t5] null {m} rho@c0.5: mean {b['mean']:+.3f} "
              f"sd {b['sd']:.3f} band [{b['p2.5']:+.3f}, {b['p97.5']:+.3f}]")
    print("[t4t5] wrote null_bands.json")


if __name__ == "__main__":
    main()
