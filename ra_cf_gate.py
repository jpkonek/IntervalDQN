"""RA-CF gate (RUN13_FIX_C_DESIGN.md Section 2 step 2; audit A3).

Re-proves the reward-structure identities THROUGH the production CF
branch path (bcd.cf_branch_rollout) before any training run consumes CF
targets. The 18-July RA battery proved the identities on LIVE episodes;
the audit requires them re-proven on the branch, because a branch-side
pricing bug would poison every CF target while the live battery stays
green.

Design (M1 scenario, seed 134, the same probe geometry as the live RA
battery; all suffixes SCRIPTED — candidate then all-NOOP — so both
layers price the identical action sequence and no net weights enter):

  RA-CF0 cross-layer identity: for each probe candidate, the branch's
         discounted return G_cf equals the live layer's discounted
         suffix return (run_episode via run_scripted + step_hook) from
         the branch step, to <= 1e-6. THE core check: branch pricing ==
         live pricing on identical futures.
  RA-CF1 resolution pays on the branch: G_cf(resolving climb) >
         G_cf(NOOP) at s*.
  RA-CF4 endangerment costs on the branch: at s2 (post-resolution,
         separated), G_cf(re-leveling climb on the other aircraft) <
         G_cf(NOOP) - MARGIN.
  RA-CF5 no pump on the branch: at s2, G_cf(descend back) <
         G_cf(NOOP) - MARGIN (descend-back re-creates the conflict; a
         branch that priced it positive would teach pumping).
  RA-CF6 internal identity + determinism: g_cf == sum(gamma^k r_k) to
         1e-9, and an identical second call reproduces g_cf bitwise.

Run: .venv/bin/python ra_cf_gate.py [--scenario_seed 134
     --vertical_ramp smooth --delta_conflict 0.5]
"""

import argparse
import copy
import sys

import numpy as np

import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import GAMMA, N_INSTR, ControllerAgent
import diagnose_controller as dc
from diagnose_controller import run_scripted, ScriptedShim
from micro_level_allocation import (
    make_micro_env, noop_trace, run_gates, ScriptPolicy, CLIMB, DESCEND,
)

MARGIN = 5.0     # RA-CF4/5: re-endangering must lose by at least this
                 # (the in-window LoS is worth ~-50 discounted; 5 is a
                 # conservative floor far above fee/shaping noise)


def walk_to(env, seed, script, stop_step):
    """Walk the env to stop_step applying scripted (step -> (cs, act))
    via the validated sector_step mirror; returns (env_state==live obs,
    snap, last_issued) AT stop_step. Mirror == run_episode is asserted
    by dc.validate_reward_mirror at import-run time."""
    obs, info = env.reset(seed=seed)
    bcd.reset_delivery_clock(env)
    snap = dc.sector_snapshot(env, obs.keys())
    li = {}
    for step in range(stop_step):
        actions = {cs: 0 for cs in obs}
        issued = False
        if step in script and script[step][0] in obs:
            cs, act = script[step]
            actions[cs] = act
            issued = True
            li[cs] = act - 1
        out = dc.sector_step(env, obs, snap, actions, issued)
        obs, snap = out["next_obs"], out["snap_next"]
        assert not out["violated"], \
            f"walk hit a violation at step {step} before stop {stop_step}"
    return obs, snap, li


def cand_idx(cs_list, target, env_action):
    i = cs_list.index(target)
    return 1 + N_INSTR * i + (env_action - 1)


def branch(env, obs, snap, li, agent, cs_list, target, env_action, H,
           budget=None):
    """Scripted CF branch: candidate then all-NOOP, through the
    PRODUCTION cf_branch_rollout."""
    first = {cs: 0 for cs in obs}
    idx = 0
    if target is not None:
        first = dict(first, **{target: env_action})
        idx = cand_idx(cs_list, target, env_action)
    script = [(first, idx)] + \
        [({cs: 0 for cs in obs}, 0)] * (H - 1)   # obs keys only advisory;
    # cf_branch_rollout rebuilds NOOP dicts per live branch obs when the
    # scripted dict misses aircraft (defensive-copy fix, see bcd)
    return bcd.cf_branch_rollout(env, obs, snap, li, agent, idx, H,
                                 cbp=True, cbp_lag=bcd.CBP_LAG,
                                 objective_v2=True, budget=budget,
                                 script=script)


def live_suffix_return(env, seed, pre_script, branch_step, target,
                       env_action, H):
    """The live layer's discounted return over [branch_step,
    branch_step+H) for the same action sequence, via run_episode
    (run_scripted + step_hook)."""
    entries = [(s, cs, act) for s, (cs, act) in pre_script.items()]
    if target is not None:
        entries.append((branch_step, target, env_action))
    rs = {}

    def hook(d):
        rs[d["step"]] = d["r"]
    run_scripted(env, ScriptPolicy(entries), seed, cbp=True,
                 step_hook=hook)
    return sum(GAMMA ** k * rs[branch_step + k]
               for k in range(H) if (branch_step + k) in rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario_seed", type=int, default=134)
    ap.add_argument("--h", type=int, default=36)
    ap.add_argument("--vertical_ramp", type=str, default="smooth",
                    choices=["gate", "smooth"])
    ap.add_argument("--delta_conflict", type=float, default=0.5)
    args = ap.parse_args()
    bcd.set_conflict_pricing(args.vertical_ramp, args.delta_conflict)
    print("=" * 74)
    print(f"RA-CF GATE (branch-path pricing identities; "
          f"{bcd.conflict_pricing_str()}, H={args.h})")
    print("=" * 74)
    dc.validate_cbp_mirror()

    env = make_micro_env(duration=600)
    seed = args.scenario_seed
    rows, violated, kind, vstep = noop_trace(env, seed)
    assert violated and kind == "loss_of_separation", kind
    gates = run_gates(env, seed, rows, vstep,
                      argparse.Namespace(cbp=True))
    d_nm = {r["step"]: r["d_nm"] for r in rows if "d_nm" in r}
    cand = [(s, t) for t, win in gates["windows"].items() for s in win
            if d_nm.get(s + 2) is not None and d_nm[s + 2] < 15.0]
    assert cand, "no in-support climb step"
    s_star, tgt = max(cand)
    other = [c for c in gates["pair"] if c != tgt][0]
    s2 = s_star + 6
    H = args.h
    print(f"\n  probe: s*={s_star} climb {tgt}; s2={s2} (post-resolution)"
          f"; H={H}")

    agent0 = ControllerAgent(token_dim=60, n_instr=N_INSTR, gamma=GAMMA,
                             cf_replay=True, cf_seed=777)

    checks = []

    # ---- state at s* (all-NOOP prefix) --------------------------------
    # Branch ONLY from a frozen deepcopy: env is reset in place by every
    # run_scripted/walk_to call, and the first two gate runs each caught
    # a variant of branching from a stale live env.
    obs, snap, li = walk_to(env, seed, {}, s_star)
    env_star = copy.deepcopy(env)
    cs_list = sorted(obs)
    b_noop = branch(env_star, obs, snap, li, agent0, cs_list, None, 0, H)
    b_climb = branch(env_star, obs, snap, li, agent0, cs_list, tgt,
                     CLIMB, H)

    # RA-CF0 identity at s*
    for name, tg, act, rec in (("noop@s*", None, 0, b_noop),
                               ("climb@s*", tgt, CLIMB, b_climb)):
        live = live_suffix_return(env, seed, {}, s_star, tg, act, H)
        err = abs(rec["g_cf"] - live)
        ok = err <= 1e-6
        checks.append((f"RA-CF0 identity {name}", ok,
                       f"branch {rec['g_cf']:+.4f} vs live {live:+.4f} "
                       f"(err {err:.1e})"))

    # RA-CF1 resolution pays on the branch
    ok1 = b_climb["g_cf"] > b_noop["g_cf"]
    checks.append(("RA-CF1 resolution pays", ok1,
                   f"G_cf(climb) {b_climb['g_cf']:+.3f} > G_cf(NOOP) "
                   f"{b_noop['g_cf']:+.3f}"))

    b_again = branch(env_star, obs, snap, li, agent0, cs_list, tgt,
                     CLIMB, H)

    # ---- state at s2 (climb-at-s* prefix, separated) ------------------
    obs2, snap2, li2 = walk_to(env, seed, {s_star: (tgt, CLIMB)}, s2)
    env_s2 = copy.deepcopy(env)
    cs2 = sorted(obs2)
    b2_noop = branch(env_s2, obs2, snap2, li2, agent0, cs2, None, 0, H)
    b2_reconf = branch(env_s2, obs2, snap2, li2, agent0, cs2, other,
                       CLIMB, H)
    b2_pump = branch(env_s2, obs2, snap2, li2, agent0, cs2, tgt,
                     DESCEND, H)

    live2 = live_suffix_return(env, seed, {s_star: (tgt, CLIMB)}, s2,
                               None, 0, H)
    err2 = abs(b2_noop["g_cf"] - live2)
    checks.append((f"RA-CF0 identity noop@s2", err2 <= 1e-6,
                   f"branch {b2_noop['g_cf']:+.4f} vs live {live2:+.4f} "
                   f"(err {err2:.1e})"))

    ok4 = b2_reconf["g_cf"] < b2_noop["g_cf"] - MARGIN
    checks.append(("RA-CF4 endangerment costs", ok4,
                   f"G_cf(re-level {other}) {b2_reconf['g_cf']:+.3f} < "
                   f"G_cf(NOOP) {b2_noop['g_cf']:+.3f} - {MARGIN}"))

    ok5 = b2_pump["g_cf"] < b2_noop["g_cf"] - MARGIN
    checks.append(("RA-CF5 no pump", ok5,
                   f"G_cf(descend-back) {b2_pump['g_cf']:+.3f} < "
                   f"G_cf(NOOP) {b2_noop['g_cf']:+.3f} - {MARGIN}"))

    # RA-CF6 internal identity + determinism
    ok6 = True
    for name, rec in (("climb@s*", b_climb), ("noop@s2", b2_noop)):
        g = sum(GAMMA ** k * r for k, r in enumerate(rec["rewards"]))
        ok6 &= abs(g - rec["g_cf"]) <= 1e-9
    ok6 &= (b_again["g_cf"] == b_climb["g_cf"])
    checks.append(("RA-CF6 sum-identity + determinism", bool(ok6),
                   f"repeat err {abs(b_again['g_cf'] - b_climb['g_cf']):.1e}"))

    print()
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"  {name}: {detail} -> {'PASS' if ok else 'FAIL'}")
    print(f"\n  RA-CF GATE: {'PASS' if all_ok else 'FAIL'} "
          f"[{bcd.conflict_pricing_str()}]")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
