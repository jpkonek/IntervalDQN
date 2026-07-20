"""
Micro level-allocation problem (2026-07-13; run-13 prep, Phase B)
=================================================================

TWO aircraft on a collision course at the SAME flight level, nothing
else in the sector (CustomInfiniteEnv: 2 starters, spawn rate 0.0,
600 s). Tests whether the controller-frame interval-DQN machinery can
learn WHEN and HOW to reroute to a safe altitude, now that the action
space has vertical clearances (bluebird_controller_dqn N_INSTR = 5:
L10, R10, route_parallel, climb +10 FL, descend -10 FL).

This doubles as the ARCHITECTURE-vs-CREDIT-STARVATION split for the 12b
C2 finding (Spearman ~ 0 between net Hurwicz scores and ground-truth
rollout returns all run, suspected credit starvation: per-candidate
credit ~2e-3 vs -50 terminals). Here ONE climb swings the outcome by
+-50, so candidate ordering MUST emerge if the architecture is capable.

Faithfulness: the scenario env goes through diagnose_controller.
make_custom_density_env (mirrors bcd.make_controller_env's config incl.
the vertical actions); scripted episodes are priced by bcd.run_episode
itself (ScriptedShim); training IS bcd.run_episode(train=True) with the
objective-v2 + CBP training signal (12b's pricing) — nothing is
reimplemented. Ground-truth rollouts reuse diagnose_controller.
rollout_return (sector_step_cbp, mirror-validated on 2026-07-09).

PRE-REGISTERED EXPECTATIONS (written 2026-07-13 BEFORE the micro-
training was first run; budget <= 2000 episodes, wall <= ~2 h):
  (1) CLEAN RATE: the trailing-100-episode clean-episode rate should
      exceed 80% within the budget if the architecture can learn level
      allocation (scripted gate (ii) proves a single well-timed climb
      suffices mechanically, so the task is learnable by construction).
  (2) ORDERING: C2-style rank correlation (Spearman rho of net Hurwicz
      scores at c_train vs ground-truth H=15-step rollout returns, ~20
      frozen probe states on the NOOP conflict trajectory) should
      clearly exceed the 12b null: rho > 0.4, and on the outcome-
      swinging subset (GT spread > 5) it should be higher still.
  INTERPRETATION GRID (fixed in advance):
      clean-rate learns AND rho > 0.4  -> 12b's rho ~ 0 was CREDIT
                                          STARVATION, not architecture;
      clean-rate learns BUT rho ~ 0    -> ordering is ARCHITECTURE-
                                          LIMITED (rank information does
                                          not reach the Hurwicz scores
                                          even with dense credit);
      neither learns                   -> dig into what is broken and
                                          report (no verdict on the
                                          split).
  MEASUREMENT NOTES (fixed in advance): (i) ground truth uses NOOP
  continuation after the first candidate action (deviation from C2's
  frozen-policy continuation, documented: it makes GT checkpoint-
  independent, so ONE GT table serves every rho evaluation and the rho
  learning curve is comparable across checkpoints; in a 2-aircraft
  problem the first clearance is the decision that matters). (ii) GT is
  priced with the CBP training signal (use_cbp=True), the quantity the
  net is trained to estimate. (iii) rho is evaluated every --rho_every
  episodes on the SAME frozen states.

Scripted sanity gates (run BEFORE any training; --gates):
  (i)   NOOP -> loss_of_separation (the conflict is real);
  (ii)  a single well-timed climb (env action 4) on one aircraft ->
        clean episode (the vertical action works mechanically); the
        full issue-step sweep prints the WINDOW in which the climb
        must go out;
  (iii) time-to-CPA and per-step pair geometry are printed, plus the
        climb window relative to CPA;
  (iv)  bonus: the upgraded vertical-first DELIVERER oracle must also
        keep the episode clean.

Usage:
    .venv/bin/python micro_level_allocation.py --gates
    .venv/bin/python micro_level_allocation.py --train --episodes 800
    .venv/bin/python micro_level_allocation.py --gates --train
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import (
    GAMMA, N_INSTR, INSTR_NAMES, KIN_FEATS,
    ControllerAgent, run_episode, sector_snapshot, save_checkpoint,
    build_tokens,
)
from bluebird_interval_dqn import detect_violation, haversine_nm, \
    SEC_PER_STEP
from diagnose_controller import (
    make_custom_density_env, ScriptedShim, NoopPolicy, Deliverer,
    run_scripted, collect_probe_states, rollout_return, spearman,
    noop_action, tracked, pos_status_name,
)

sys.stdout.reconfigure(line_buffering=True)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "checkpoints", "micro_level_allocation")

CLIMB = 4      # env action int: simple_fl_climb +10 FL
DESCEND = 5    # env action int: simple_fl_descent -10 FL


def make_micro_env(duration=600):
    """Two starters, zero spawns — the level-allocation field."""
    return make_custom_density_env(duration=duration,
                                   initial_spawn_rate=0.0,
                                   max_spawn_rate=0.0,
                                   num_starter_aircraft=2)


class SingleActionPolicy:
    """Issue exactly one (target, env_action) at a fixed step; global
    NOOP otherwise. Runs through bcd.run_episode via ScriptedShim so the
    episode is priced by the system's own reward layer."""
    name = "SINGLE"

    def __init__(self, target_cs, action, step):
        self.target = target_cs
        self.action = action
        self.step = step
        self.t = 0

    def decide(self, env, obs):
        t = self.t
        self.t += 1
        if t == self.step and self.target in obs:
            return self.target, self.action
        return None, 0


class ScriptPolicy:
    """Issue a fixed list of (step, target_cs, env_action) clearances;
    global NOOP otherwise. Same contract as SingleActionPolicy — runs
    through bcd.run_episode via ScriptedShim so every episode is priced
    by the system's own reward layer."""
    name = "SCRIPT"

    def __init__(self, script):
        self.script = {s: (t, a) for s, t, a in script}
        self.t = 0

    def decide(self, env, obs):
        t = self.t
        self.t += 1
        entry = self.script.get(t)
        if entry is not None and entry[0] in obs:
            return entry
        return None, 0


def noop_trace(env, seed):
    """Manual all-NOOP rollout recording per-step pair geometry (read-
    only measurement; pricing not needed here). Returns a list of dicts
    and the violation info."""
    obs, info = env.reset(seed=seed)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    rows = []
    violated, kind, vstep = False, None, None
    for step in range(maxstep):
        snap = sector_snapshot(env, obs.keys())
        row = {"step": step, "n_obs": len(obs), "n_in_sector": len(snap)}
        if len(snap) >= 2:
            a, b = sorted(snap)[:2]
            row["pair"] = (a, b)
            row["d_nm"] = haversine_nm(snap[a].lat, snap[a].lon,
                                       snap[b].lat, snap[b].lon)
            row["dfl"] = abs(snap[a].fl - snap[b].fl)
            row["fls"] = (snap[a].fl, snap[b].fl)
        rows.append(row)
        obs, _r, _d, _t, info = env.step({cs: 0 for cs in obs})
        v, k, involved = detect_violation(info)
        if v:
            violated, kind, vstep = True, k, step
            break
    return rows, violated, kind, vstep


def find_conflict_seed(env, seeds):
    """First seed whose 2-starter NOOP episode ends in loss_of_separation
    (not a wind-drift sector excursion). Returns (seed, trace, vstep,
    cpa_step, cpa_d)."""
    for seed in seeds:
        rows, violated, kind, vstep = noop_trace(env, seed)
        paired = [r for r in rows if "d_nm" in r]
        d_min = min((r["d_nm"] for r in paired), default=float("nan"))
        cpa = min(paired, key=lambda r: r["d_nm"]) if paired else None
        dfl0 = paired[0]["dfl"] if paired else float("nan")
        print(f"  seed {seed}: {'VIOLATION[' + str(kind) + ']@' + str((vstep + 1) * SEC_PER_STEP) + 's' if violated else 'clean'}"
              f" | min pair dist {d_min:.2f} nm | dFL(first paired step) "
              f"{dfl0:.0f}")
        if violated and kind == "loss_of_separation":
            return seed, rows, vstep, cpa["step"], cpa["d_nm"]
    return None, None, None, None, None


def run_gates(env, seed, rows, vstep, args):
    """Scripted sanity gates (i)-(iv). Returns dict incl. the climb
    window (list of clean issue steps) used to report timing later."""
    paired = [r for r in rows if "d_nm" in r]
    cpa = min(paired, key=lambda r: r["d_nm"])
    cs_list = list(paired[0]["pair"])
    print(f"\n  conflict pair {cs_list}, FLs {paired[0]['fls']}, first "
          f"paired dist {paired[0]['d_nm']:.1f} nm")
    print(f"  NOOP CPA proxy: step {cpa['step']} ({cpa['step'] * SEC_PER_STEP}"
          f" s), min dist {cpa['d_nm']:.2f} nm; LoS at step {vstep} "
          f"({(vstep + 1) * SEC_PER_STEP} s)")
    closure = [paired[i]["d_nm"] - paired[i + 1]["d_nm"]
               for i in range(len(paired) - 1)]
    print(f"  mean closure rate {np.mean(closure):.2f} nm/step over the "
          f"NOOP approach")

    # ---- gate (i): NOOP -> LoS, priced by the system itself ----------
    g_noop = run_scripted(env, NoopPolicy(), seed, cbp=True)
    gate_i = g_noop["violated"] and \
        g_noop["violation_kind"] == "loss_of_separation"
    print(f"\n  GATE (i) NOOP: G={g_noop['ep_return']:.2f} "
          f"{'VIOLATION[' + str(g_noop['violation_kind']) + ']@' + str(g_noop['time_to_violation']) + 's' if g_noop['violated'] else 'clean'}"
          f" -> {'PASS' if gate_i else 'FAIL'}")

    # ---- gate (ii): single well-timed climb -> clean; full sweep -----
    windows = {}
    for target in cs_list:
        clean_steps = []
        for s in range(0, vstep + 1):
            pol = SingleActionPolicy(target, CLIMB, s)
            st = run_scripted(env, pol, seed)
            if not st["violated"]:
                clean_steps.append(s)
        windows[target] = clean_steps
        if clean_steps:
            print(f"  GATE (ii) climb sweep on {target}: clean for issue "
                  f"steps [{clean_steps[0]}..{clean_steps[-1]}] "
                  f"({len(clean_steps)}/{vstep + 1} steps); latest "
                  f"workable climb {(cpa['step'] - clean_steps[-1])} steps "
                  f"({(cpa['step'] - clean_steps[-1]) * SEC_PER_STEP} s) "
                  f"before CPA, {(vstep - clean_steps[-1])} steps before "
                  f"the NOOP LoS")
        else:
            print(f"  GATE (ii) climb sweep on {target}: NO single climb "
                  f"keeps the episode clean")
    gate_ii = any(windows.values())
    print(f"  GATE (ii) -> {'PASS' if gate_ii else 'FAIL'}")

    # descend also works? (context, not gated)
    tgt0 = cs_list[0] if windows[cs_list[0]] else cs_list[1]
    s0 = (windows[tgt0] or [0])[0]
    st_desc = run_scripted(env, SingleActionPolicy(tgt0, DESCEND, s0), seed)
    print(f"  (context) single DESCEND on {tgt0} at step {s0}: "
          f"{'clean' if not st_desc['violated'] else 'VIOLATION[' + str(st_desc['violation_kind']) + ']'}")

    # ---- gate (iii) is the CPA/window print above ---------------------
    # ---- gate (iv): upgraded vertical-first DELIVERER oracle ----------
    g_del = run_scripted(env, Deliverer(), seed, cbp=True)
    gate_iv = not g_del["violated"]
    print(f"  GATE (iv) DELIVERER(vertical-first): "
          f"G={g_del['ep_return']:.2f} cmd={g_del['commands']} "
          f"{'clean' if gate_iv else 'VIOLATION[' + str(g_del['violation_kind']) + ']@' + str(g_del['time_to_violation']) + 's'}"
          f" -> {'PASS' if gate_iv else 'FAIL'}")

    return {"pair": cs_list, "cpa_step": cpa["step"],
            "cpa_d_nm": cpa["d_nm"], "vstep": vstep,
            "gate_i": gate_i, "gate_ii": gate_ii, "gate_iv": gate_iv,
            "windows": windows,
            "noop_G": g_noop["ep_return"], "deliverer_G": g_del["ep_return"],
            "deliverer_clean": gate_iv}


# ===========================================================================
# 12d reward-structure audit (RA battery)
# ===========================================================================

def reward_audit(env, seed, rows, vstep, gates_out, args):
    """Empirical verification of the 12d composite reward structure
    through bcd.run_episode ITSELF (real env, real CBP lag-1 clone path,
    real sel_fl plumbing). The synthetic --selftest proves the formulas
    on hand-built snapshots; THIS proves the running system pays them.

    Scope: pricing only. nstep / bootstrap_support / explore_bonus are
    learning-side (selection and targets) with no reward surface — the
    scripted path never touches selection — so they are NOT tested here.
    All probe episodes price with cbp=True (the 12d training signal),
    regardless of --cbp.

    Pre-registered checks:
      RA1 gate-vs-smooth contrast: the SAME well-timed resolving climb,
          priced in both modes at the same delta. Gate mode pays
          |pc| <= 1e-3 (12c's 'verticals pay exactly zero'; tolerance =
          lateral CAS leakage, measured 7.2e-06 in --selftest); smooth
          mode pays pc > 0. This is D5's core promise, end to end.
      RA2 fee-vs-benefit richness at the run delta: pc_smooth/CMD_COST
          in the JK-approved band [0.3, 2] (the delta=0.5 approval
          arithmetic). PASS in band / PARTIAL positive outside /
          FAIL nonpositive.
      RA3 composite ordering: G(resolving climb) > G(NOOP) under
          cbp + objective v2 (resolution must beat the -50).
      RA4 endangerment: after separation is established, a climb on the
          OTHER aircraft (re-levels the pair) pays pc < 0 through the
          real path — the commanded-factor blend, empirically.
      RA5 anti-pump: climb-then-descend-back on one aircraft:
          |pc_up + pc_down| < 2*CMD_COST and the round trip scores
          strictly below the single climb (pumping never pays). The
          first payment must also reproduce RA1's smooth payment to
          1e-9 (pricing determinism).
      RA6 budget identities on every probe episode (external re-check of
          the returned budget columns, 4dp-rounding tolerance):
          cmd_cost == -CMD_COST*commands; sum(budget) == ep_return;
          NOOP pays cbp_shaping == 0.0 and zero fees exactly.
    Deliveries/bonus of the clean climb episode are reported (v2 decay
    sanity: bonus/delivery in [3, ~10+]) but not gated — B2 owns that.
    """
    print("\n" + "=" * 74)
    print("REWARD-STRUCTURE AUDIT (12d composite, empirical RA battery)")
    d_nm = {r["step"]: r["d_nm"] for r in rows if "d_nm" in r}
    windows = gates_out["windows"]

    # lag-1 CBP measures the branches 2 steps after issuance, and the
    # conflict term only pays inside f_lat's 15 nm support (the 12c-era
    # micro metric was diluted by exactly this — out-of-range climbs pay
    # zero by design). Pick the LATEST clean climb step whose measured
    # state sits well inside the support, leaving >= 8 steps of headroom
    # before the NOOP LoS for the second-issuance probes (RA4/RA5).
    cand = [(s, tgt) for tgt, win in windows.items() for s in win
            if d_nm.get(s + 2) is not None and d_nm[s + 2] <= 12.0
            and s + 8 <= vstep]
    if not cand:
        cand = [(s, tgt) for tgt, win in windows.items() for s in win
                if d_nm.get(s + 2) is not None and d_nm[s + 2] < 15.0]
    assert cand, (
        "RA: no clean climb step has its lag-measured state inside the "
        "15 nm conflict support — this scenario cannot exercise the "
        "conflict term; audit another seed")
    s_star, tgt = max(cand)
    other = [c for c in gates_out["pair"] if c != tgt][0]
    s2 = s_star + 6          # second-issuance step (RA4/RA5)
    print(f"  probe: climb {tgt} at step {s_star} (pair dist at measured "
          f"state {d_nm.get(s_star + 2, float('nan')):.1f} nm; NOOP LoS "
          f"step {vstep}); second issuance at step {s2}")

    def priced(policy):
        paid = {}

        def hook(d):
            if d["issued"]:
                paid[d["step"]] = d["paid_conflict"]
        st = run_scripted(env, policy, seed, cbp=True, step_hook=hook)
        return st, paid

    mode0, delta0 = bcd.VERTICAL_RAMP, bcd.DELTA_CONFLICT
    try:
        bcd.set_conflict_pricing("smooth", delta0)
        st_noop, _ = priced(NoopPolicy())
        st_sm, paid_sm = priced(ScriptPolicy([(s_star, tgt, CLIMB)]))
        st_re, paid_re = priced(ScriptPolicy([(s_star, tgt, CLIMB),
                                              (s2, other, CLIMB)]))
        st_rt, paid_rt = priced(ScriptPolicy([(s_star, tgt, CLIMB),
                                              (s2, tgt, DESCEND)]))
        bcd.set_conflict_pricing("gate", delta0)
        _st_gt, paid_gt = priced(ScriptPolicy([(s_star, tgt, CLIMB)]))
    finally:
        bcd.set_conflict_pricing(mode0, delta0)

    pc_sm = paid_sm.get(s_star)
    pc_gt = paid_gt.get(s_star)
    pc_re2 = paid_re.get(s2)
    pc_rt1, pc_rt2 = paid_rt.get(s_star), paid_rt.get(s2)
    assert None not in (pc_sm, pc_gt, pc_re2, pc_rt1, pc_rt2), (
        f"RA: a scripted issuance did not fire (paid keys: sm="
        f"{sorted(paid_sm)}, gt={sorted(paid_gt)}, re={sorted(paid_re)}, "
        f"rt={sorted(paid_rt)}) — target left obs early?")

    results = {}

    # ---- RA1 gate-vs-smooth ------------------------------------------
    ra1 = abs(pc_gt) <= 1e-3 and pc_sm > 0.0
    results["RA1"] = {"pc_gate": pc_gt, "pc_smooth": pc_sm, "pass": ra1}
    print(f"  RA1 gate-vs-smooth: gate pc={pc_gt:+.6f} (|.|<=1e-3), "
          f"smooth pc={pc_sm:+.6f} (>0) -> {'PASS' if ra1 else 'FAIL'}")

    # ---- RA2 richness band -------------------------------------------
    ratio = pc_sm / bcd.CMD_COST
    ra2 = ("PASS" if 0.3 <= ratio <= 2.0
           else "PARTIAL" if ratio > 0 else "FAIL")
    results["RA2"] = {"ratio": ratio, "delta": delta0, "verdict": ra2}
    print(f"  RA2 richness: pc/fee = {ratio:.2f} at delta={delta0} "
          f"(band [0.3, 2]) -> {ra2}")

    # ---- RA3 composite ordering --------------------------------------
    ra3 = (st_sm["ep_return"] > st_noop["ep_return"]
           and not st_sm["violated"] and st_noop["violated"])
    results["RA3"] = {"G_climb": st_sm["ep_return"],
                      "G_noop": st_noop["ep_return"], "pass": ra3}
    print(f"  RA3 ordering: G(climb)={st_sm['ep_return']:.2f} "
          f"({'clean' if not st_sm['violated'] else 'VIOL'}) > "
          f"G(NOOP)={st_noop['ep_return']:.2f} "
          f"({'VIOL' if st_noop['violated'] else 'clean'}) "
          f"-> {'PASS' if ra3 else 'FAIL'}")

    # ---- RA4 endangerment pays negative ------------------------------
    ra4 = pc_re2 < 0.0
    results["RA4"] = {"pc_reconflict": pc_re2, "pass": ra4}
    print(f"  RA4 endangerment: re-conflicting climb on {other} pays "
          f"pc={pc_re2:+.6f} (<0) -> {'PASS' if ra4 else 'FAIL'}")

    # ---- RA5 anti-pump -----------------------------------------------
    net = pc_rt1 + pc_rt2
    ra5 = (abs(pc_rt1 - pc_sm) <= 1e-9
           and abs(net) < 2 * bcd.CMD_COST
           and st_rt["ep_return"] < st_sm["ep_return"])
    results["RA5"] = {"pc_up": pc_rt1, "pc_down": pc_rt2, "net": net,
                      "G_roundtrip": st_rt["ep_return"],
                      "G_climb": st_sm["ep_return"], "pass": ra5}
    print(f"  RA5 anti-pump: up {pc_rt1:+.6f} + down {pc_rt2:+.6f} = "
          f"net {net:+.6f} (|.|<{2 * bcd.CMD_COST}), "
          f"G(roundtrip)={st_rt['ep_return']:.2f} < "
          f"G(climb)={st_sm['ep_return']:.2f}, determinism err "
          f"{abs(pc_rt1 - pc_sm):.1e} -> {'PASS' if ra5 else 'FAIL'}")

    # ---- RA6 budget identities ---------------------------------------
    ra6, details = True, []
    for name, st in (("noop", st_noop), ("climb", st_sm),
                     ("reconflict", st_re), ("roundtrip", st_rt)):
        recon = (st["cbp_shaping"] + st["fuel"] + st["cmd_cost"]
                 + st["delivery_bonus"] + st["violation_term"])
        e_rec = abs(recon - st["ep_return"])
        e_fee = abs(st["cmd_cost"] + bcd.CMD_COST * st["commands"])
        ok = e_rec <= 1e-3 and e_fee <= 1e-4
        ra6 &= ok
        details.append({"ep": name, "recon_err": e_rec, "fee_err": e_fee,
                        "ok": ok})
    e_noop = abs(st_noop["cbp_shaping"])
    ra6 &= e_noop <= 5e-5 and st_noop["commands"] == 0
    results["RA6"] = {"episodes": details, "noop_cbp_abs": e_noop,
                      "pass": bool(ra6)}
    print(f"  RA6 identities: max recon err "
          f"{max(d['recon_err'] for d in details):.1e}, max fee err "
          f"{max(d['fee_err'] for d in details):.1e}, NOOP cbp_shaping "
          f"{st_noop['cbp_shaping']:+.5f} -> {'PASS' if ra6 else 'FAIL'}")

    # ---- delivery report (not gated) ---------------------------------
    n_del, bonus = st_sm["deliveries"], st_sm["delivery_bonus"]
    results["deliveries"] = {"n": n_del, "bonus": bonus}
    print(f"  (report) climb episode delivers {n_del}, bonus "
          f"{bonus:+.2f}" + (f" ({bonus / n_del:.2f}/delivery; v2 floor "
                             f"3.0)" if n_del else ""))

    audit_pass = (ra1 and ra3 and ra4 and ra5 and ra6 and ra2 != "FAIL")
    print(f"  AUDIT VERDICT [{bcd.conflict_pricing_str()}]: "
          f"{'PASS' if audit_pass else 'FAIL'}"
          + (" (RA2 PARTIAL — richness outside approved band, "
             "positive)" if ra2 == "PARTIAL" and audit_pass else ""))
    print("=" * 74)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR,
                       f"micro_reward_audit_seed{seed}_"
                       f"{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump({"seed": seed, "s_star": s_star, "target": tgt,
                   "s2": s2, "vertical_ramp": mode0,
                   "delta_conflict": delta0, "results": results,
                   "audit_pass": audit_pass}, f, indent=1)
    print(f"  audit written: {out}")
    assert audit_pass, "reward-structure audit FAILED — see RA lines"
    return results


# ===========================================================================
# Ground truth + rho instrument (C2-style, frozen states, NOOP continuation)
# ===========================================================================

def build_probe_set(env, seed, n_states, h):
    """Frozen probe states along the NOOP conflict trajectory + the
    checkpoint-independent ground-truth table (see MEASUREMENT NOTES in
    the module docstring). Returns list of dicts."""
    states = collect_probe_states(
        env, seed, lambda e, o, i: (noop_action(o), False),
        n_states, min_aircraft=2)
    probe = []
    t0 = time.time()
    for env_s, obs_s, stratum, step in states:
        cs_list = sorted(obs_s.keys())
        snap_s = sector_snapshot(env_s, obs_s.keys())
        toks = build_tokens(env_s, obs_s, None, cs_list)
        n_cand = 1 + N_INSTR * len(cs_list)
        gts = []
        for k in range(n_cand):
            acts = noop_action(obs_s)
            issued = k != 0
            if issued:
                i, j = divmod(k - 1, N_INSTR)
                acts[cs_list[i]] = j + 1
            G, _st, _vio, _dl = rollout_return(
                env_s, obs_s, snap_s, acts, issued, h,
                continue_policy=None, use_cbp=True)
            gts.append(G)
        gts = np.asarray(gts)
        probe.append({"step": step, "tokens": toks, "cs_list": cs_list,
                      "n_cand": n_cand, "gt": gts,
                      "gt_spread": float(gts.max() - gts.min())})
    hot = sum(1 for p in probe if p["gt_spread"] > 5.0)
    print(f"  probe set: {len(probe)} frozen states (GT H={h}, CBP "
          f"signal, NOOP continuation), {hot} outcome-swinging "
          f"(spread > 5), built in {time.time() - t0:.0f}s")
    return probe


def rho_eval(agent, probe, c):
    """Mean Spearman rho of net scores vs the frozen GT, overall and on
    the outcome-swinging subset; plus GT-in-interval coverage. Reported
    in BOTH ranking variants (D5 amendment I(d) / the D7 measurement
    decision): 'rho' ranks by the Hurwicz score l + c*(u - l) (what
    selection uses); 'rho_mid' ranks by the interval MIDPOINT (l + u)/2
    (the width-independent readout)."""
    rhos, rhos_hot, inside, total = [], [], 0, 0
    rhos_mid, rhos_mid_hot = [], []
    for p in probe:
        cl, cu = agent.candidate_q(p["tokens"])
        scores = cl + c * (cu - cl)
        mids = (cl + cu) / 2.0
        rho = spearman(p["gt"], scores[:p["n_cand"]])
        rho_m = spearman(p["gt"], mids[:p["n_cand"]])
        if rho is not None:
            rhos.append(rho)
            if p["gt_spread"] > 5.0:
                rhos_hot.append(rho)
        if rho_m is not None:
            rhos_mid.append(rho_m)
            if p["gt_spread"] > 5.0:
                rhos_mid_hot.append(rho_m)
        hit = (p["gt"] >= cl[:p["n_cand"]]) & (p["gt"] <= cu[:p["n_cand"]])
        inside += int(hit.sum())
        total += p["n_cand"]
    return {"rho": float(np.mean(rhos)) if rhos else float("nan"),
            "rho_hot": float(np.mean(rhos_hot)) if rhos_hot
            else float("nan"),
            "rho_mid": float(np.mean(rhos_mid)) if rhos_mid
            else float("nan"),
            "rho_mid_hot": float(np.mean(rhos_mid_hot)) if rhos_mid_hot
            else float("nan"),
            "n_states": len(rhos), "n_hot": len(rhos_hot),
            "gt_coverage": inside / max(1, total)}


# ===========================================================================
# Micro-training
# ===========================================================================

def train_micro(env, scenario_seed, gates_out, args):
    torch.manual_seed(args.agent_seed)
    np.random.seed(args.agent_seed)
    import random as _random
    _random.seed(args.agent_seed)

    obs, info = env.reset(seed=scenario_seed)
    obs_dim = int(next(iter(obs.values())).shape[0])
    token_dim = obs_dim + KIN_FEATS
    n_env_actions = int(env.get_action_parser().get_total_num_actions())
    assert n_env_actions == 1 + N_INSTR
    # CF-replay (run-13 M1 arm): agent-level switch, mirrors
    # micro_battery_common.cf_agent_kwargs (inlined — the battery
    # imports FROM this module, so no reverse import)
    cf_kw = {}
    if getattr(args, "cf_replay", False):
        cf_kw = {"cf_replay": True, "cf_seed": 10_000 + args.agent_seed}
        if getattr(args, "cf_budget_rate", None) is not None:
            cf_kw["cf_budget_rate"] = args.cf_budget_rate
    agent = ControllerAgent(
        token_dim=token_dim, n_instr=N_INSTR, lr=args.lr, gamma=GAMMA,
        c_train=args.c_train, warmup_steps=args.warmup_steps,
        warmup_epsilon=0.5, buffer_size=20000, batch_size=64,
        device="cpu", bootstrap_support=args.bootstrap_support,
        explore_bonus=args.explore_bonus,
        explore_bonus_scale=args.explore_bonus_scale,
        explore_bonus_halflife=args.explore_bonus_halflife,
        tracker_tripwire=args.tracker_tripwire, **cf_kw)
    print(f"  agent: token_dim {token_dim}, n_instr {N_INSTR}, "
          f"{agent.param_count()} params, c_train {args.c_train}, "
          f"warmup {args.warmup_steps} steps, cbp="
          f"{'ON' if args.cbp else 'OFF'}, bootstrap_support="
          f"{args.bootstrap_support}, explore_bonus={args.explore_bonus}"
          + (f" (scale={args.explore_bonus_scale}, "
             f"halflife={args.explore_bonus_halflife})"
             if args.explore_bonus != "off" else "")
          + f", tripwire={'ON' if args.tracker_tripwire else 'OFF'}"
          + f", nstep={args.nstep}, {bcd.conflict_pricing_str()}")

    print("  building frozen probe set + ground truth ...")
    probe = build_probe_set(env, scenario_seed, args.probe_states, args.h)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(OUT_DIR, f"micro_seed{scenario_seed}_{tag}.jsonl")
    log = open(log_path, "w")
    print(f"  logging to {log_path}")

    r0 = rho_eval(agent, probe, args.c_train)
    rho_curve = [{"episode": 0, **r0}]
    print(f"  rho@ep0 (untrained): {r0['rho']:.3f} "
          f"(hot {r0['rho_hot']:.3f}, n={r0['n_states']}/{r0['n_hot']}), "
          f"GT coverage {r0['gt_coverage']:.3f}")

    cpa_step = gates_out["cpa_step"] if gates_out else None
    clean_hist = []
    climb_pc_hist = []     # per-episode list of paid_conflict on climbs
    descend_pc_hist = []   # ... and on descends (context)
    t_start = time.time()
    for ep in range(1, args.episodes + 1):
        issues = []

        def hook(d, issues=issues):
            if d["issued"]:
                i, j = divmod(d["cand_idx"] - 1, agent.n_instr)
                # paid_conflict: D5 per-term CBP component (amendment D)
                issues.append((d["step"], i, j,
                               d.get("paid_conflict", 0.0)))

        stats = run_episode(env, agent, seed=scenario_seed, train=True,
                            nstep=args.nstep, cbp=args.cbp,
                            objective_v2=True, step_hook=hook)
        clean = not stats["violated"]
        clean_hist.append(int(clean))
        mix = {name: 0 for name in INSTR_NAMES}
        for _s, _i, j, _pc in issues:
            mix[INSTR_NAMES[j]] += 1
        first_vert = next((s for s, _i, j, _pc in issues
                           if j in (3, 4)), None)
        climb_pcs = [pc for _s, _i, j, pc in issues if j == 3]
        descend_pcs = [pc for _s, _i, j, pc in issues if j == 4]
        climb_pc_hist.append(climb_pcs)
        descend_pc_hist.append(descend_pcs)
        rec = {"episode": ep, "clean": clean,
               "violated": stats["violated"],
               "violation_kind": stats["violation_kind"],
               "steps": stats["steps"], "ep_return": stats["ep_return"],
               "commands": stats["commands"], "mix": mix,
               "first_vertical_step": first_vert,
               "first_vertical_before_cpa":
                   (cpa_step - first_vert) if (first_vert is not None
                                               and cpa_step is not None)
                   else None,
               # D5 micro-gate observability: mean conflict-term CBP on
               # this episode's taken climbs/descends (None if none)
               "climb_paid_conflict_mean":
                   (float(np.mean(climb_pcs)) if climb_pcs else None),
               "descend_paid_conflict_mean":
                   (float(np.mean(descend_pcs)) if descend_pcs else None),
               "issues": issues[:50],
               "mean_width": stats["mean_width"],
               "epsilon": round(agent._epsilon(), 4),
               "coverage": round(agent.coverage, 4),
               "t": round(agent.t, 4),
               # fix 2 observability: cumulative (stratum x instruction-
               # type) counts — logged in EVERY run (counts are tracked
               # regardless of the bonus flag) so bonus vs no-bonus
               # candidate coverage is comparable across runs
               "explore_counts": agent.explore_counts.tolist(),
               **({"cf": {"pool": len(agent.buffer.cf_buf),
                          "states_selected": agent.cf_states_selected,
                          "skipped_budget": agent.cf_skipped_budget,
                          "bank": round(agent.cf_budget.bank, 1),
                          "spent_steps": agent.cf_budget.spent_steps,
                          "spent_clones": agent.cf_budget.spent_clones}}
                  if getattr(agent, "cf_replay", False) else {})}
        if agent.bootstrap_support == "taken_noop":
            # fix 1 observability: restricted-argmax winner share
            rec["winner_share_taken"] = round(agent.winner_share_taken, 4)
            rec["noop_fallback_share"] = round(agent.noop_fallback_share, 4)
        if agent.tracker_tripwire:
            # fix 3: per-episode tripwire (pure instrumentation).
            # PRE-REGISTERED TRIGGER: if accuracy_proxy improves >= 30%
            # from its ep500 level while t stays pinned at cap for 1000+
            # episodes, the tracker-feed change (12D doc fix E+F) gets
            # revived.
            rec["tripwire"] = {
                "strat_t": [round(tr.t, 4) for tr in agent.trackers],
                "accuracy_proxy": agent.pop_tripwire_proxy(),
                "strat_realized_cov": [round(tr.coverage, 4)
                                       for tr in agent.trackers]}
        log.write(json.dumps(rec) + "\n")
        log.flush()

        if ep % args.rho_every == 0 or ep == args.episodes:
            r = rho_eval(agent, probe, args.c_train)
            rho_curve.append({"episode": ep, **r})
            trail = clean_hist[-100:]
            ws = (f" wsT={agent.winner_share_taken:.2f}"
                  if agent.bootstrap_support == "taken_noop" else "")
            cov5 = int((agent.explore_counts >= 5).sum())
            recent_climb_pc = [pc for pcs in climb_pc_hist[-100:]
                               for pc in pcs]
            cpc = (f"{np.mean(recent_climb_pc):+.4f}"
                   if recent_climb_pc else "  --  ")
            print(f"  ep {ep:4d} | clean(last100)="
                  f"{np.mean(trail):.2f} | rho={r['rho']:+.3f} "
                  f"hot={r['rho_hot']:+.3f} mid={r['rho_mid']:+.3f} "
                  f"| GTcov={r['gt_coverage']:.2f} "
                  f"| eps={agent._epsilon():.2f} W={stats['mean_width']:.2f}"
                  f" | mix={mix} first_vert={first_vert}{ws} "
                  f"climbPC(100)={cpc} "
                  f"cells>=5:{cov5}/{agent.explore_counts.size} "
                  f"| {time.time() - t_start:.0f}s")
            ck = os.path.join(OUT_DIR,
                              f"micro_seed{scenario_seed}_{tag}_ep{ep}.pt")
            save_checkpoint(agent, ck, ep,
                            extra={"micro_scenario_seed": scenario_seed})
        elif ep % 10 == 0:
            trail = clean_hist[-100:]
            print(f"  ep {ep:4d} | {'clean' if clean else 'VIOL'} "
                  f"| clean(last100)={np.mean(trail):.2f} "
                  f"| G={stats['ep_return']:7.2f} cmd={stats['commands']:2d}"
                  f" | eps={agent._epsilon():.2f}")

    log.close()

    # ---- pre-registered verdicts --------------------------------------
    trail100 = float(np.mean(clean_hist[-100:]))
    final_rho = rho_curve[-1]
    p1 = trail100 > 0.80
    p2 = final_rho["rho"] > 0.4
    print("\n" + "=" * 74)
    print("PRE-REGISTERED VERDICTS (micro level allocation)")
    print(f"  [{bcd.conflict_pricing_str()}]")
    print(f"  (1) clean rate last 100 eps = {trail100:.2f} "
          f"(need > 0.80): {'MEET' if p1 else 'MISS'}")
    print(f"  (2) rho = {final_rho['rho']:.3f} "
          f"(hot {final_rho['rho_hot']:.3f}; need > 0.4): "
          f"{'MEET' if p2 else 'MISS'}")
    print(f"      rho_mid = {final_rho['rho_mid']:.3f} "
          f"(hot {final_rho['rho_mid_hot']:.3f}) "
          f"[MIDPOINT-rank variant — D7 measurement decision; reported, "
          f"not gated]")
    # ---- D5 micro-gate observables (pre-registered deltas a/b/c) ------
    all_climb_pc = [pc for pcs in climb_pc_hist for pc in pcs]
    all_desc_pc = [pc for pcs in descend_pc_hist for pc in pcs]
    bins = {}
    for k in range(0, len(clean_hist), 100):
        n_cl = sum(len(pcs) for pcs in climb_pc_hist[k:k + 100])
        bins[f"{k + 1}-{min(k + 100, len(clean_hist))}"] = n_cl
    print(f"  D5 (a) conflict-term CBP on taken climbs: n="
          f"{len(all_climb_pc)}, mean="
          f"{np.mean(all_climb_pc) if all_climb_pc else float('nan'):+.5f}"
          f", frac>0="
          f"{np.mean([p > 0 for p in all_climb_pc]) if all_climb_pc else float('nan'):.2f}"
          f" (descends: n={len(all_desc_pc)}, mean="
          f"{np.mean(all_desc_pc) if all_desc_pc else float('nan'):+.5f})")
    print(f"  D5 (b) climb usage per 100-ep bin: {bins}")
    print(f"  D5 (c) clean rate last 100 = {trail100:.2f} "
          f"(composite gate needs >= 0.85)")
    if p1 and p2:
        verdict = ("BOTH LEARN -> 12b's rho~0 was CREDIT STARVATION, "
                   "not an architecture limit")
    elif p1 and not p2:
        verdict = ("clean-rate learns but ordering does not -> "
                   "ARCHITECTURE-LIMITED ordering")
    elif not p1 and p2:
        verdict = ("ordering emerges but the policy does not exploit it "
                   "-> selection/exploration problem (outside the grid; "
                   "report)")
    else:
        verdict = "NEITHER learns -> dig in (no verdict on the split)"
    print(f"  VERDICT: {verdict}")

    summary = {"scenario_seed": scenario_seed, "episodes": args.episodes,
               "vertical_ramp": bcd.VERTICAL_RAMP,
               "delta_conflict": bcd.DELTA_CONFLICT,
               "nstep": args.nstep,
               "climb_paid_conflict": {
                   "n": len(all_climb_pc),
                   "mean": (float(np.mean(all_climb_pc))
                            if all_climb_pc else None),
                   "frac_pos": (float(np.mean([p > 0
                                               for p in all_climb_pc]))
                                if all_climb_pc else None)},
               "descend_paid_conflict": {
                   "n": len(all_desc_pc),
                   "mean": (float(np.mean(all_desc_pc))
                            if all_desc_pc else None)},
               "climbs_per_100ep": bins,
               "bootstrap_support": args.bootstrap_support,
               "explore_bonus": args.explore_bonus,
               "explore_counts_final": agent.explore_counts.tolist(),
               "winner_share_taken_final":
                   (round(agent.winner_share_taken, 4)
                    if agent.bootstrap_support == "taken_noop" else None),
               "clean_hist": clean_hist, "rho_curve": rho_curve,
               "trail100_clean": trail100, "p1_clean80": p1,
               "p2_rho04": p2, "verdict": verdict,
               "gates": {k: v for k, v in (gates_out or {}).items()
                         if k != "windows"},
               "windows": {k: [int(min(v)), int(max(v))] if v else None
                           for k, v in (gates_out or {}).get(
                               "windows", {}).items()},
               "wall_seconds": round(time.time() - t_start)}
    out = os.path.join(OUT_DIR, f"micro_summary_seed{scenario_seed}_{tag}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"  summary written: {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Two-aircraft level-"
                                             "allocation micro-problem")
    ap.add_argument("--gates", action="store_true",
                    help="run the scripted sanity gates")
    ap.add_argument("--reward_audit", action="store_true",
                    help="empirical 12d reward-structure battery "
                         "(RA1-RA6) through the real CBP path; runs the "
                         "scripted gates first for the climb window")
    ap.add_argument("--train", action="store_true",
                    help="run the micro-training (small; minutes-scale)")
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--scenario_seed", type=int, default=None,
                    help="skip the scan and use this seed")
    ap.add_argument("--scan_seeds", type=int, nargs="+",
                    default=[10043, 20042] + list(range(100, 160)))
    ap.add_argument("--agent_seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--c_train", type=float, default=0.5)
    ap.add_argument("--warmup_steps", type=int, default=1500)
    ap.add_argument("--cbp", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--bootstrap_support", type=str, default="full",
                    choices=["full", "taken_noop"],
                    help="Double-DQN target argmax support (fix D; "
                         "see bluebird_controller_dqn.py)")
    ap.add_argument("--explore_bonus", type=str, default="off",
                    choices=["off", "count"],
                    help="count-based selection bonus (fix 2)")
    ap.add_argument("--explore_bonus_scale", type=float, default=0.5)
    ap.add_argument("--explore_bonus_halflife", type=float, default=2000.0)
    ap.add_argument("--tracker_tripwire",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="per-episode tripwire JSONL block (fix 3; pure "
                         "instrumentation)")
    ap.add_argument("--rho_every", type=int, default=50)
    ap.add_argument("--probe_states", type=int, default=20)
    ap.add_argument("--h", type=int, default=15)
    ap.add_argument("--nstep", type=int, default=bcd.NSTEP,
                    help="n-step window for the micro training "
                         "(threaded into bcd.run_episode; the 12d "
                         "composite uses 36 per the knee-rerun decision)")
    ap.add_argument("--vertical_ramp", type=str, default="gate",
                    choices=["gate", "smooth"],
                    help="D5 conflict-pricing mode (pass-through to "
                         "bcd.set_conflict_pricing; prices training, "
                         "scripted gates AND the GT probe table alike)")
    ap.add_argument("--delta_conflict", type=float, default=None,
                    help="D5 DELTA_CONFLICT override (default: bcd's "
                         "0.2; the D5 micro gate runs both arms at 0.5)")
    ap.add_argument("--cf_replay", action="store_true", default=False,
                    help="run-13 fix C: CF-replay on (agent-level "
                         "switch; RA-CF gate must be green first)")
    ap.add_argument("--cf_budget_rate", type=float, default=None)
    args = ap.parse_args()

    # D5 amendment B: set pricing before ANYTHING prices a step and
    # announce it (the micro gate must not silently run gate mode).
    bcd.set_conflict_pricing(args.vertical_ramp, args.delta_conflict)
    print("=" * 74)
    print(f"MICRO LEVEL ALLOCATION (duration {args.duration}s, "
          f"2 starters, spawn 0.0)")
    print(f"[micro] {bcd.conflict_pricing_str()}, nstep={args.nstep}")
    print("=" * 74)
    env = make_micro_env(duration=args.duration)

    if args.scenario_seed is not None:
        seed = args.scenario_seed
        rows, violated, kind, vstep = noop_trace(env, seed)
        assert violated and kind == "loss_of_separation", \
            f"seed {seed} does not produce a NOOP LoS ({kind})"
        paired = [r for r in rows if "d_nm" in r]
        cpa = min(paired, key=lambda r: r["d_nm"])
        cpa_step, cpa_d = cpa["step"], cpa["d_nm"]
    else:
        print("scanning for a 2-starter NOOP loss-of-separation seed ...")
        seed, rows, vstep, cpa_step, cpa_d = find_conflict_seed(
            env, args.scan_seeds)
        assert seed is not None, "no conflict seed found in the scan range"
    print(f"scenario seed: {seed} (NOOP LoS at step {vstep}, CPA proxy "
          f"step {cpa_step} @ {cpa_d:.2f} nm)")

    gates_out = None
    if args.gates or args.reward_audit:
        gates_out = run_gates(env, seed, rows, vstep, args)
        assert gates_out["gate_i"] and gates_out["gate_ii"], \
            "sanity gates failed — do not train on this scenario"
    if args.reward_audit:
        reward_audit(env, seed, rows, vstep, gates_out, args)
    if args.train:
        if gates_out is None:
            paired = [r for r in rows if "d_nm" in r]
            cpa = min(paired, key=lambda r: r["d_nm"])
            gates_out = {"cpa_step": cpa["step"], "windows": {}}
        train_micro(env, seed, gates_out, args)


if __name__ == "__main__":
    main()
