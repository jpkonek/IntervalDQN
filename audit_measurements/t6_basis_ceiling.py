"""T6 overnight audit measurement (20 July 2026).

rho ceiling under the Branch-A (frozen-policy) basis, M2 scenario
(seed 167). For each of the 20 frozen NOOP probe states and EVERY
candidate (NOOP + 5 instr x 2 aircraft = 11), two CBP-priced GT
rollouts with H = min(80, episode steps remaining):

  GT_noop:   candidate then all-NOOP continuation
             (exactly build_probe_set_capped's table — the basis the
             pre-registered bars judge against)
  GT_frozen: candidate then the round-1 ep800 M2 agent as continuation
             (greedy epsilon-0, re-issue-mask-aware — the basis
             CF-replay training actually prices, D1 Branch A)

Both via diagnose_controller.rollout_return (use_cbp=True), the C2
machinery the task names. Then:
  (a) Spearman(GT_noop, GT_frozen) per state — how far the two bases
      agree at all;
  (b) the ep800 net's rho (Hurwicz c=0.5 ordering) against BOTH tables
      — is the 0.4 bar reachable in principle under the training basis?

Pricing smooth/0.5 set first. Writes basis_ceiling.json.
"""
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import sector_snapshot, build_tokens, N_INSTR
from bluebird_interval_dqn import SEC_PER_STEP
from diagnose_controller import (collect_probe_states, noop_action,
                                 rollout_return, spearman)
from micro_level_allocation import make_micro_env

OUT = os.path.dirname(os.path.abspath(__file__))
CKPT = "checkpoints/micro_battery/m2/m2_seed167_20260719_205235_ep800.pt"
SEED = 167
HMAX = 80


def frozen_policy(agent, li):
    """Greedy mask-aware continuation closure for rollout_return.
    li: last_issued dict, pre-seeded with the branch's first command."""
    def pol(env, obs):
        acts, aux = agent.generate_action(env, obs, None,
                                          force_epsilon=0.0,
                                          last_issued=li,
                                          rng=agent.cf_rng)
        idx = aux["cand_idx"]
        if agent.mask_reissue and idx > 0:
            csl = sorted(obs)
            i, j = divmod(idx - 1, agent.n_instr)
            li[csl[i]] = j
        return acts, idx != 0
    return pol


def main():
    bcd.set_conflict_pricing("smooth", 0.5)
    print(f"[t6] {bcd.conflict_pricing_str()}")
    env = make_micro_env(duration=600)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    agent, _ck = bcd.load_agent(CKPT)
    print(f"[t6] loaded {CKPT}")

    states = collect_probe_states(
        env, SEED, lambda e, o, i: (noop_action(o), False), 20,
        min_aircraft=2)
    print(f"[t6] {len(states)} probe states, steps {[s[3] for s in states]}")

    per_state = []
    t0 = time.time()
    for env_s, obs_s, stratum, step in states:
        cs_list = sorted(obs_s.keys())
        snap_s = sector_snapshot(env_s, obs_s.keys())
        toks = build_tokens(env_s, obs_s, None, cs_list)
        n_cand = 1 + N_INSTR * len(cs_list)
        h_s = max(1, min(HMAX, maxstep - step))
        gt_noop, gt_frozen, viol_frozen = [], [], []
        for k in range(n_cand):
            acts = noop_action(obs_s)
            issued = k != 0
            li = {}
            if issued:
                i, j = divmod(k - 1, N_INSTR)
                acts[cs_list[i]] = j + 1
                if agent.mask_reissue:
                    li[cs_list[i]] = j
            g_n, _s1, _v1, _d1 = rollout_return(
                env_s, copy.deepcopy(obs_s), snap_s, dict(acts), issued,
                h_s, continue_policy=None, use_cbp=True)
            g_f, _s2, v2, _d2 = rollout_return(
                env_s, copy.deepcopy(obs_s), snap_s, dict(acts), issued,
                h_s, continue_policy=frozen_policy(agent, li),
                use_cbp=True)
            gt_noop.append(float(g_n))
            gt_frozen.append(float(g_f))
            viol_frozen.append(bool(v2))
        rho_bases = spearman(gt_noop, gt_frozen)
        cl, cu = agent.candidate_q(toks)
        scores = (cl + 0.5 * (cu - cl))[:n_cand]
        rho_net_noop = spearman(gt_noop, scores)
        rho_net_frozen = spearman(gt_frozen, scores)
        per_state.append({
            "probe_step": int(step), "h": h_s, "n_cand": n_cand,
            "gt_noop": gt_noop, "gt_frozen": gt_frozen,
            "frozen_branch_violated": viol_frozen,
            "spread_noop": float(max(gt_noop) - min(gt_noop)),
            "spread_frozen": float(max(gt_frozen) - min(gt_frozen)),
            "rho_noop_vs_frozen": rho_bases,
            "rho_net_vs_noop": rho_net_noop,
            "rho_net_vs_frozen": rho_net_frozen})
        print(f"  s{step:3d} h={h_s:2d}: bases rho="
              f"{'nan' if rho_bases is None else f'{rho_bases:+.3f}'} "
              f"net-vs-noop="
              f"{'nan' if rho_net_noop is None else f'{rho_net_noop:+.3f}'} "
              f"net-vs-frozen="
              f"{'nan' if rho_net_frozen is None else f'{rho_net_frozen:+.3f}'}"
              f" | spreads {per_state[-1]['spread_noop']:.1f}/"
              f"{per_state[-1]['spread_frozen']:.1f} "
              f"| {time.time() - t0:.0f}s")

    def mean_of(key, hot_key=None, thr=5.0):
        vs = [r[key] for r in per_state if r[key] is not None]
        out = {"mean": float(np.mean(vs)) if vs else None, "n": len(vs)}
        if hot_key:
            hv = [r[key] for r in per_state
                  if r[key] is not None and r[hot_key] > thr]
            out["mean_hot"] = float(np.mean(hv)) if hv else None
            out["n_hot"] = len(hv)
        return out

    summary = {
        "rho_noop_vs_frozen": mean_of("rho_noop_vs_frozen", "spread_noop"),
        "rho_net_vs_noop": mean_of("rho_net_vs_noop", "spread_noop"),
        "rho_net_vs_frozen": mean_of("rho_net_vs_frozen", "spread_frozen"),
        "n_states_frozen_hot": sum(1 for r in per_state
                                   if r["spread_frozen"] > 5.0),
        "n_states_noop_hot": sum(1 for r in per_state
                                 if r["spread_noop"] > 5.0)}
    res = {"pricing": bcd.conflict_pricing_str(), "ckpt": CKPT,
           "scenario_seed": SEED, "h_max": HMAX, "maxstep": maxstep,
           "hurwicz_c": 0.5, "summary": summary, "per_state": per_state,
           "wall_seconds": round(time.time() - t0)}
    with open(os.path.join(OUT, "basis_ceiling.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[t6] summary: {json.dumps(summary)}")
    print("[t6] wrote basis_ceiling.json")


if __name__ == "__main__":
    main()
