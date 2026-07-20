"""
Credit-pipeline diagnostics D1-D3 (CREDIT_DIAGNOSTICS.md, pre-registered)
=========================================================================

Where does candidate-ordering information die?  Every diagnostic runs on
BOTH the 12c ep2000 checkpoint (the patient) and an LL per_c checkpoint
(the positive control).  Pre-registered predictions are printed next to
every result; a miss is a FINDING, never a tuning cue.

Reuse policy: ALL bluebird ground-truth machinery is imported from
diagnose_controller (sector_step_cbp mirror, rollout deepcopies, probe
state collection, candidate enumeration, spearman) — nothing env-side is
reimplemented.  The C2 probe is the template for GT-vs-net comparisons
and its documented caveats carry over verbatim:
  (i)  GT is a TRUNCATED discounted return (biases coverage downward for
       far-from-terminal states; rank correlations stay meaningful);
  (ii) GT is priced with the CBP TRAINING signal (the quantity a
       CBP-trained net estimates), via dc.sector_step_cbp, mirror-
       validated against bcd.run_episode(cbp=True) before any oracle;
  (iii) the frozen continuation policy applies re-issue masking with a
       per-rollout last_issued dict seeded only with the rolled-out
       first candidate (pre-rollout mask history is unknowable from a
       state snapshot).

DEFINITIONS HELD FIXED (written before any run)
-----------------------------------------------
* "best conflict-resolving candidate" (D2): the ISSUED (non-NOOP)
  candidate with the highest GT return at the LONGEST horizon (ATC
  H=96, LL H=24), computed ONCE from the sweep rollout and reused at
  every shorter horizon (shorter-horizon gaps are PREFIX returns of the
  same determinized trajectory — the env is deterministic and the
  continuation policy frozen, so gap(H) is exactly the H-truncation).
* "target noise": std of GT returns across 3 repeated rollouts of the
  SAME (state, candidate) — bluebird deepcopy rollouts are determinized
  so ~0 is expected but it is MEASURED (any nonzero = env nondeterminism
  worth knowing); for LL, 3 seeded branch replays.
* D1 "TAKEN" candidate: first-step candidate index chosen at least once
  in N=30 epsilon-0.1 rollouts of 5 steps from the probe state;
  "UNTAKEN": never chosen in the N rollouts.
* D1 pooled rank correlations: within each state, GT and net scores are
  centred by that state's ALL-candidate means; the centred values of
  group members (taken / untaken) are pooled across states and Spearman
  is computed on the pool (per-state group sizes of 1-2 make per-state
  rho undefined; per-state rho is also reported where group size >= 3).
* D3 V(s_t): max over candidates of the net Hurwicz score at c_train
  (UNMASKED — the pure net readout; the re-issue-masked max is recorded
  as a side column).  G_t = realized discounted return-to-go of the CBP
  training-signal stream (ATC, gamma=0.97) / the native reward stream
  (LL, gamma=0.99 — that IS the LL training signal).

LL COUNTERFACTUAL BRANCHING — ADAPTATION, FLAGGED PROMINENTLY
-------------------------------------------------------------
copy.deepcopy of gymnasium LunarLander-v3 FAILS on this setup: the
Box2D bodies do not survive deepcopy (env.unwrapped.lander is None on
the copy; step() raises AssertionError — verified 2026-07-14, and
re-verified at runtime by ll_branching_check()).  Instead of the doc's
H=1-only fallback we use EXACT SEEDED ACTION REPLAY: env.reset(seed) +
the recorded action prefix reproduces the probe state bit-exactly
(runtime determinism check: two replays of the same (seed, actions)
yield identical observation and reward streams), and branching is done
by taking the candidate action at t and continuing with the frozen
greedy policy.  This preserves the full pre-registered design at ALL
horizons; it is exact, not approximate.  Cost: each branch pays the
prefix length in sim steps (LL steps are ~microseconds).

PRE-REGISTERED PREDICTIONS AND FIXED OPERATIONALIZATIONS
--------------------------------------------------------
D1 H-coverage: 12c rho_taken >> rho_untaken ~ 0; LL both healthy.
  Fixed rule: 12c AS-PREDICTED iff pooled rho_untaken < 0.15 AND
  (rho_taken - rho_untaken) > 0.2; LL healthy iff both pooled rhos
  > 0.3 with |difference| < 0.3.
D1 width-exposure check: untaken candidates "systematically wider" iff
  mean_width_untaken / mean_width_taken > 1.1 AND untaken mean is wider
  in > 60% of states.  NOT wider => width does not track training
  exposure at the action level (state-OOD narrowing pathology and the
  ordering failure are one mechanism).
D2 H-mismatch: 12c gap ~ noise for H <= 12, emerging at H >= 24-48; LL
  gap significant already at H = 1-3.  Fixed rule: 12c AS-PREDICTED iff
  median |gap(12)| < 0.25 * median |gap(96)| AND median |gap(96)| >
  max(5 * noise_p95, 1e-6); LL AS-PREDICTED iff median |gap(1)| >
  max(5 * noise_p95, 1e-6).
D3 H-chain: 12c Spearman(V, G) collapses beyond ~20 steps from
  terminal; LL stays high at all depths.  Fixed rule (bins with n >=
  10): 12c AS-PREDICTED iff mean rho over bins >= 20 steps < 0.2 AND
  mean rho over bins < 20 steps > 0.4; LL AS-PREDICTED iff every bin
  rho > 0.5.

Operational constraints honoured: a 12c training run is LIVE on this
machine — torch is pinned to 2 threads (as diagnose_controller does),
nothing is trained, no training files are touched.

Usage:
    .venv/bin/python credit_diagnostics.py --all --side both
    .venv/bin/python credit_diagnostics.py --d2 --side atc
    .venv/bin/python credit_diagnostics.py --all --side both --smoke
"""

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch

torch.set_num_threads(2)   # live 12c training run on this machine

import diagnose_controller as dc
import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import (
    GAMMA as ATC_GAMMA, CONFLICT_FL, CHECKPOINT_DIR, N_INSTR,
    build_tokens, sector_snapshot, load_agent, build_reissue_mask,
)
from bluebird_interval_dqn import detect_violation, haversine_nm, \
    SEC_PER_STEP

import gymnasium as gym
from lunarlander_interval_dqn import IntervalQNetwork

sys.stdout.reconfigure(line_buffering=True)

DIAG_DIR = os.path.join(CHECKPOINT_DIR, "diagnostics")
RUN12C_LOG = os.path.join(CHECKPOINT_DIR, "run12c_console.log")
LL_CKPT_DEFAULT = "checkpoints/per_c/c0.5_seed0/ckpt_final.pt"

LL_GAMMA = 0.99
ATC_HORIZONS = (6, 12, 24, 48, 96)
LL_HORIZONS = (1, 3, 6, 12, 24)
# knee rerun (12D_CREDIT_DESIGN.md, fix B / decision D1): denser grid in
# the 12-48 band; 96 stays as the ground-truth reference horizon (knees
# are defined against gap(96); prefix sums of the SAME rollout give all
# horizons for free).
KNEE_HORIZONS = (12, 18, 24, 30, 36, 48, 96)
KNEE_STATES = 30          # double the D2 sample
KNEE_SEEDS = tuple(range(70042, 70102))   # mixed seeds: wide band ...
KNEE_PER_SEED_CAP = 4     # ... and no seed may dominate (prior run: 9/15
                          # states came from seed 70046 alone)
KNEE_FRAC = 0.8           # "personal knee": smallest H with
                          # gap(H) >= KNEE_FRAC * gap(96)

# ---------------------------------------------------------------------------
# D2-Phi rerun (D5_VERTICAL_POTENTIAL_DESIGN.md section 6a, repaired
# two-tier per the adversarial reviews; implemented 2026-07-18).
# PRE-REGISTERED before any run — this header IS the registration:
#
#   Setup: same 30 knee states (same 12c ep2000 agent, same KNEE_SEEDS +
#   per-seed cap, same KNEE_HORIZONS grid), pricing smooth + the CLI
#   delta (JK-approved 0.5), baseline = knee_rerun_20260714_201334.json
#   (gate pricing). Phi does not move the dynamics, so per-arm physical
#   trajectories are IDENTICAL to the baseline; gaps move only by the
#   re-priced shaping. Sanity gate: noop_violation_step per state must
#   equal the baseline exactly.
#
#   TIER 1 (integration test, MUST PASS): for every state whose
#   baseline GT-best arm is VERTICAL (env j in {4, 5}), the same-arm
#   gap at H=12 moves off the baseline value by that state's analytic
#   p1 (summed over all pairs of the climbed aircraft, planning rocd
#   2.75 FL/step, lateral geometry assumed common-mode): sign-correct
#   and |measured shift| within [0.5x, 2x] of analytic (tolerance for
#   lateral drift inside the horizon). States whose baseline best arm
#   is LATERAL: same-arm gap(12) unchanged within 1e-3 as registered in
#   the design (registered at DELTA=0.2; at DELTA=0.5 the ambient
#   lateral conflict differentials rescale 2.5x, so a marginal breach
#   of 1e-3 is reported with that caveat, not silently excused).
#
#   TIER 2 (the commissioned prediction, honestly scoped): computed
#   FIRST, before any rollout — the explicit list of states predicted
#   to flip gap(12) sign, namely baseline gap(12) <= 0 AND
#   baseline gap(12) + max-over-(aircraft, direction) analytic p1 > 0.
#   If the list is empty the pre-registered statement is: "Tier 2
#   predicts a null; efficacy rides on the micro gate." After the run,
#   measured flips (smooth gap_best(12) > 0 at states with baseline
#   gap_best(12) <= 0) are compared against the list; misses both ways
#   are findings.
# ---------------------------------------------------------------------------
KNEE_BASELINE_JSON = os.path.join(
    CHECKPOINT_DIR, "diagnostics", "knee_rerun_20260714_201334.json")
D5_ROCD_PLAN = 2.75       # planning FL/step (percentile-sampled live)
D1_H = 15                 # GT horizon for D1 (as C2)
D1_N_ROLLOUTS = 30        # visitation rollouts per probe state
D1_ROLLOUT_LEN = 5
D1_EPS = 0.1
D2_NOISE_REPEATS = 3      # total rollouts per (state, arm) incl. sweep
D2_ATC_SEP_NM = 12.0      # conflict-state gate: min pair sep < 12 nm
D3_BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60),
           (60, 70), (70, 80), (80, 10 ** 9)]

# soft sim-step budgets in CLONE-WEIGHTED units (an issued CBP step
# counts 4 — two clone deepcopies + three clone steps ride along with
# it; the 12c greedy policy issues at ~0.68/step). Sized so the
# pre-registered design (20/15/20 states/episodes) completes; these are
# safety rails against runaway rollouts, not the primary budget control.
BUDGETS = {"d1_atc": 40000, "d2_atc": 100000, "d3_atc": 30000,
           "knee_atc": 250000}

EXPECT = {
    "d1": ("H-coverage: 12c rho_taken >> rho_untaken ~ 0; LL both "
           "healthy (all 4 actions frequently sampled -> little "
           "difference).  WIDTH-EXPOSURE: if untaken candidates are NOT "
           "systematically wider than taken ones, width does not track "
           "training exposure at the action level — the state-OOD "
           "narrowing pathology and the ordering failure are one "
           "mechanism."),
    "d2": ("H-mismatch: 12c gap ~ noise for H <= 12 (the n-step window "
           "sees nothing), emerging only at H >= 24-48.  LL: gap "
           "significant already at H = 1-3 (dense shaping "
           "differentiates immediately)."),
    "d3": ("H-chain: 12c correlation collapses beyond ~20 steps from "
           "terminal (the chain cannot carry conflict credit to "
           "decision time 50-80 steps out).  LL: correlation stays "
           "high at all depths."),
}


# ===========================================================================
# Shared helpers
# ===========================================================================

class StepBudget:
    """Priced-step counter with a soft cap (C2's budget trick)."""

    def __init__(self, cap):
        self.cap = cap
        self.n = 0

    def add(self, k):
        self.n += k

    def exhausted(self):
        return self.n >= self.cap


def pooled_centered_spearman(pairs):
    """pairs: list of (state_id, gt, score, gt_center, score_center).
    Centres by the supplied per-state all-candidate means, pools, and
    computes Spearman.  Returns (rho, n)."""
    if not pairs:
        return None, 0
    g = [p[1] - p[3] for p in pairs]
    s = [p[2] - p[4] for p in pairs]
    return dc.spearman(g, s), len(pairs)


def bin_index(steps_to_terminal):
    for k, (lo, hi) in enumerate(D3_BINS):
        if lo <= steps_to_terminal < hi:
            return k
    return len(D3_BINS) - 1


def bin_label(k):
    lo, hi = D3_BINS[k]
    return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"


def d3_bin_table(records):
    """records: list of (steps_to_terminal, V, G).  Returns per-bin rows."""
    rows = []
    for k in range(len(D3_BINS)):
        sub = [(v, g) for stt, v, g in records if bin_index(stt) == k]
        if not sub:
            rows.append({"bin": bin_label(k), "n": 0, "rho": None,
                         "mean_abs_err": None, "mean_V": None,
                         "mean_G": None})
            continue
        V = [v for v, _ in sub]
        G = [g for _, g in sub]
        rows.append({"bin": bin_label(k), "n": len(sub),
                     "rho": dc.spearman(V, G),
                     "mean_abs_err": float(np.mean(np.abs(
                         np.array(V) - np.array(G)))),
                     "mean_V": float(np.mean(V)),
                     "mean_G": float(np.mean(G))})
    return rows


def print_d3_table(rows, title):
    print(f"  {title}")
    print(f"    {'steps-to-term':<14}{'n':>6}{'rho(V,G)':>10}"
          f"{'|V-G|':>10}{'mean V':>10}{'mean G':>10}")
    for r in rows:
        rho = f"{r['rho']:.3f}" if r["rho"] is not None else "  -- "
        if r["n"] == 0:
            print(f"    {r['bin']:<14}{0:>6}")
            continue
        print(f"    {r['bin']:<14}{r['n']:>6}{rho:>10}"
              f"{r['mean_abs_err']:>10.2f}{r['mean_V']:>10.2f}"
              f"{r['mean_G']:>10.2f}")


def _bin_lo(label):
    return int(label.split("-")[0].rstrip("+"))


def d3_verdict_atc(rows):
    near = [r["rho"] for r in rows if r["n"] >= 10
            and r["rho"] is not None and _bin_lo(r["bin"]) < 20]
    far = [r["rho"] for r in rows if r["n"] >= 10
           and r["rho"] is not None and _bin_lo(r["bin"]) >= 20]
    if not near or not far:
        return "INSUFFICIENT-DATA", None, None
    m_near, m_far = float(np.mean(near)), float(np.mean(far))
    ok = m_far < 0.2 and m_near > 0.4
    return ("AS-PREDICTED" if ok else "MISS"), m_near, m_far


def d3_verdict_ll(rows):
    rhos = [r["rho"] for r in rows if r["n"] >= 10 and r["rho"] is not None]
    if not rhos:
        return "INSUFFICIENT-DATA", None
    ok = all(r > 0.5 for r in rhos)
    return ("AS-PREDICTED" if ok else "MISS"), float(np.min(rhos))


# ===========================================================================
# ATC side — everything env-facing is dc.* machinery
# ===========================================================================

def find_12c_ckpt(args):
    if args.atc_ckpt:
        return args.atc_ckpt
    with open(RUN12C_LOG) as f:
        for line in f:
            m = re.search(r"Logging to (\S+\.jsonl)", line)
            if m:
                stem = os.path.basename(m.group(1))[:-len(".jsonl")]
                path = os.path.join(CHECKPOINT_DIR, stem + "_ep2000.pt")
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                return path
    raise RuntimeError("no 'Logging to ...jsonl' line in " + RUN12C_LOG)


_ATC = {}


def atc_agent(args):
    if "agent" not in _ATC:
        path = find_12c_ckpt(args)
        agent, ckpt = load_agent(path, device="cpu")
        print(f"  [ckpt/ATC] {os.path.basename(path)} (episode "
              f"{ckpt.get('episode', '?')}, n_instr={agent.n_instr}, "
              f"c_train={agent.c_train}, mask_reissue="
              f"{agent.mask_reissue})")
        _ATC.update(agent=agent, ckpt=ckpt, path=path)
    return _ATC["agent"], _ATC["ckpt"], _ATC["path"]


def make_continue(agent, c, li0):
    """C2's frozen continuation policy: greedy at c with re-issue
    masking, last_issued seeded with the rolled-out first candidate
    (caveat iii)."""
    li = dict(li0)

    def continue_policy(e, o):
        if not o:
            return {}, False
        acts, aux = agent.generate_action(e, o, None, c=c,
                                          force_epsilon=0.0,
                                          last_issued=li)
        idx = aux["cand_idx"]
        if idx != 0 and getattr(agent, "mask_reissue", False):
            i, j = divmod(idx - 1, agent.n_instr)
            li[aux["callsigns"][i]] = j
        return acts, idx != 0
    return continue_policy


def atc_prefix_rollout(env_s, obs_s, snap_s, first_actions, first_issued,
                       horizons, continue_policy, budget):
    """Deepcopy GT rollout priced with the CBP training signal
    (dc.sector_step_cbp — the C2 pricing), recording the discounted
    PREFIX return at every horizon in `horizons`.  A violation ends the
    trajectory (terminal; the return stays frozen at longer horizons,
    exactly as dc.rollout_return truncates).  Returns ({H: G_H}, steps,
    violated, violation_step)."""
    env = copy.deepcopy(env_s)
    hmax = max(horizons)
    G, disc = 0.0, 1.0
    out_g = {}
    o, s = obs_s, snap_s
    acts, issued = first_actions, first_issued
    violated, vstep, steps = False, None, 0
    for h in range(1, hmax + 1):
        out = dc.sector_step_cbp(env, o, s, acts, issued)
        steps += 1
        budget.add(4 if issued else 1)
        G += disc * out["r"]
        disc *= ATC_GAMMA
        o, s = out["next_obs"], out["snap_next"]
        if h in horizons:
            out_g[h] = G
        if out["violated"]:
            violated, vstep = True, h
            break
        acts, issued = continue_policy(env, o)
    for h in horizons:
        out_g.setdefault(h, G)
    return out_g, steps, violated, vstep


def cand_to_actions(k, cs_list, obs_s, n_instr):
    """Candidate index k (net layout: 0=NOOP, then n_instr per aircraft)
    -> (actions dict, issued, li0)."""
    acts = dc.noop_action(obs_s)
    li0 = {}
    if k != 0:
        i, j = divmod(k - 1, n_instr)
        acts[cs_list[i]] = j + 1
        li0[cs_list[i]] = j
    return acts, k != 0, li0


def min_pair_sep_nm(snap):
    """Min lateral separation among vertically-proximate (<CONFLICT_FL)
    pairs; inf if no such pair (matches snapshot_stratum's geometry)."""
    acs = list(snap.values())
    best = float("inf")
    for i in range(len(acs)):
        for j in range(i + 1, len(acs)):
            if abs(acs[i].fl - acs[j].fl) >= CONFLICT_FL:
                continue
            best = min(best, haversine_nm(acs[i].lat, acs[i].lon,
                                          acs[j].lat, acs[j].lon))
    return best


# ---------------------------------------------------------------------------
# D1 — ATC
# ---------------------------------------------------------------------------

def _scripted_act_fn(policy):
    """Adapt a decide(env, obs) scripted policy to the collect_probe_states
    act_fn contract (env, obs, info) -> (actions, issued)."""
    def act(env, obs, info):
        tgt, a = policy.decide(env, obs)
        acts = {cs: 0 for cs in obs}
        issued = tgt is not None and tgt in obs
        if issued:
            acts[tgt] = a
        return acts, issued
    return act


def _frozen_act_fn(name):
    """The two canonical checkpoint-independent generating policies."""
    if name == "noop":
        return lambda e, o, i: ({cs: 0 for cs in o}, False)
    assert name == "deliverer", name
    return _scripted_act_fn(dc.Deliverer())


def _collect_at_steps(env, seed, act_fn, wanted):
    """Replay the act_fn trajectory from seed and snapshot (deepcopy env,
    obs, stratum, step) at exactly the given step indices. Mirrors
    dc.collect_probe_states's loop so regenerated states are bit-faithful
    to the ones its selection pass produced (deterministic env)."""
    wanted = set(wanted)
    obs, info = env.reset(seed=seed)
    bcd.reset_delivery_clock(env)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // dc.SEC_PER_STEP))
    out_states = []
    snap = dc.sector_snapshot(env, obs.keys())
    for step in range(maxstep):
        if step in wanted:
            out_states.append((copy.deepcopy(env), copy.deepcopy(obs),
                               dc.snapshot_stratum(snap), step))
        if len(out_states) == len(wanted):
            break
        actions, issued = act_fn(env, obs, info)
        o = dc.sector_step(env, obs, snap, actions, issued)
        obs, snap, info = o["next_obs"], o["snap_next"], o["info"]
        if o["violated"]:
            break
    return out_states


def load_or_build_frozen_d1(env, args):
    """Frozen shared D1 state set (JK-approved 19 July 2026): the same
    states for EVERY checkpoint, so conflict-state NOOP levels are
    comparable across training runs — the on-policy collection each net
    was previously probed on confounded the level half of the D7 gate
    (each net gets tested on its own trouble).

    The env object is not picklable (internal lambda), so the file
    freezes the RECIPE, not the objects: (policy, seed, step indices)
    per leg, regenerated exactly on load (deterministic env + fixed
    scripted policies), with a per-state stratum fingerprint asserted so
    any env/policy drift fails loudly instead of silently changing the
    measurement basis. Legs: NOOP (unmanaged traffic, the baseline every
    run faces) and the DELIVERER oracle (managed traffic), three fixed
    seeds each."""
    import pickle
    path = args.d1_frozen
    if os.path.exists(path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        states = []
        for leg in payload["legs"]:
            got = _collect_at_steps(env, leg["seed"],
                                    _frozen_act_fn(leg["policy"]),
                                    leg["steps"])
            assert [s[3] for s in got] == leg["steps"] and \
                [s[2] for s in got] == leg["strata"], (
                f"frozen D1 leg {leg['policy']}@{leg['seed']} did not "
                f"regenerate (env or policy drift) — rebuild the set "
                f"and treat older frozen readings as a separate series")
            states.extend(got)
        print(f"  [frozen] regenerated {len(states)} shared states from "
              f"{path} (built {payload['built']}; "
              f"{', '.join(l['policy'] + '@' + str(l['seed']) for l in payload['legs'])})")
        return states
    states, legs = [], []
    per_leg = max(3, args.d1_states // 6)
    for seed in (90042, 90043, 90044):
        for name in ("noop", "deliverer"):
            got = dc.collect_probe_states(env, seed, _frozen_act_fn(name),
                                          per_leg, min_aircraft=2)
            states.extend(got)
            legs.append({"policy": name, "seed": seed,
                         "steps": [s[3] for s in got],
                         "strata": [s[2] for s in got]})
    payload = {"legs": legs,
               "built": time.strftime("%Y-%m-%d %H:%M:%S"),
               "d1_states_arg": args.d1_states}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # atomic write: a crash mid-dump must not leave a corrupt file at the
    # canonical path (it would poison every later load)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)
    print(f"  [frozen] built {len(states)} shared states, recipe saved -> "
          f"{path}")
    return states


def d1_atc(args):
    print("\n" + "=" * 78)
    print(f"D1/ATC — TAKEN vs UNTAKEN candidate split "
          f"({args.d1_states} probe states, GT H={D1_H}, visitation "
          f"{args.d1_rollouts}x{D1_ROLLOUT_LEN}-step eps={D1_EPS} rollouts)")
    print("EXPECTATION:", EXPECT["d1"])
    print("=" * 78)
    dc.validate_cbp_mirror()
    agent, ckpt, path = atc_agent(args)
    env = dc.get_env(1200)
    c_pol = agent.c_train
    budget = StepBudget(BUDGETS["d1_atc"])
    if getattr(args, "d1_frozen", None):
        states = load_or_build_frozen_d1(env, args)
        print(f"  probe states: {len(states)} (FROZEN shared set — "
              f"cross-checkpoint comparable)")
    else:
        states = dc.collect_probe_states(env, 90042,
                                         dc._agent_act_fn(agent, c_pol),
                                         args.d1_states, min_aircraft=2)
        print(f"  probe states: {len(states)} (frozen policy c={c_pol}; "
              f"on-policy set — NOT cross-checkpoint comparable)")
    per_state = []
    pooled = {"taken": [], "untaken": []}
    t0 = time.time()
    for s_i, (env_s, obs_s, stratum, step) in enumerate(states):
        if budget.exhausted():
            print(f"  [budget] stopping at state {s_i} "
                  f"({budget.n} priced steps)")
            break
        cs_list = sorted(obs_s.keys())
        snap_s = sector_snapshot(env_s, obs_s.keys())
        toks = build_tokens(env_s, obs_s, None, cs_list)
        scores, cl, cu = dc._hurwicz_scores(agent, toks, c_pol)
        n_cand = 1 + agent.n_instr * len(cs_list)
        # ---- visitation: N short eps-greedy rollouts, first-step counts
        first_counts = np.zeros(n_cand, dtype=int)
        all_counts = np.zeros(n_cand, dtype=int)
        random.seed(1000 + s_i)
        for r_i in range(args.d1_rollouts):
            env_r = copy.deepcopy(env_s)
            o = copy.deepcopy(obs_s)
            li = {}
            for t in range(D1_ROLLOUT_LEN):
                if not o:
                    break
                acts, aux = agent.generate_action(
                    env_r, o, None, c=c_pol, force_epsilon=D1_EPS,
                    last_issued=li)
                idx = aux["cand_idx"]
                if idx < n_cand:   # aircraft set can grow mid-rollout
                    all_counts[idx] += 1
                    if t == 0:
                        first_counts[idx] += 1
                if idx != 0 and agent.mask_reissue:
                    i, j = divmod(idx - 1, agent.n_instr)
                    li[aux["callsigns"][i]] = j
                o = env_r.step(acts)[0]
        # ---- GT for EVERY candidate at H=D1_H (C2 machinery)
        gts = np.zeros(n_cand)
        for k in range(n_cand):
            acts, issued, li0 = cand_to_actions(k, cs_list, obs_s,
                                                agent.n_instr)
            g, st, vio, _ = atc_prefix_rollout(
                env_s, obs_s, snap_s, acts, issued, (D1_H,),
                make_continue(agent, c_pol, li0), budget)
            gts[k] = g[D1_H]
        taken = first_counts > 0
        g_mean, s_mean = float(gts.mean()), float(scores[:n_cand].mean())
        widths = (cu - cl)[:n_cand]
        hit = (gts >= cl[:n_cand]) & (gts <= cu[:n_cand])
        for k in range(n_cand):
            grp = "taken" if taken[k] else "untaken"
            pooled[grp].append((s_i, float(gts[k]), float(scores[k]),
                                g_mean, s_mean, float(widths[k]),
                                bool(hit[k])))
        rho_all = dc.spearman(gts, scores[:n_cand])
        rho_tk = dc.spearman(gts[taken], scores[:n_cand][taken]) \
            if taken.sum() >= 3 else None
        rho_un = dc.spearman(gts[~taken], scores[:n_cand][~taken]) \
            if (~taken).sum() >= 3 else None
        per_state.append({
            "step": step, "stratum": stratum, "n_cands": n_cand,
            "n_taken": int(taken.sum()),
            "first_counts": first_counts.tolist(),
            # D7 gate, level half: the net's absolute valuation of NOOP
            # at this state vs ground truth, and whether the net still
            # rates NOOP its best candidate (12c pathology: yes at every
            # conflict state). Recorded per state so conflict-stratum
            # levels are comparable across checkpoints.
            "noop_score": float(scores[0]), "noop_gt": float(gts[0]),
            "noop_is_net_best": bool(int(np.argmax(scores[:n_cand])) == 0),
            "score_mean": s_mean, "gt_mean": g_mean,
            "rho_all": rho_all, "rho_taken": rho_tk, "rho_untaken": rho_un,
            "mean_w_taken": float(widths[taken].mean()),
            "mean_w_untaken": float(widths[~taken].mean())
            if (~taken).any() else None,
            "cov_taken": float(hit[taken].mean()),
            "cov_untaken": float(hit[~taken].mean())
            if (~taken).any() else None,
        })
        print(f"  state {s_i:2d} (step {step:3d}, strat {stratum}): "
              f"{int(taken.sum())}/{n_cand} taken | rho_all="
              f"{rho_all if rho_all is not None else float('nan'):.3f} | "
              f"w_taken={widths[taken].mean():.2f} "
              f"w_untaken="
              f"{widths[~taken].mean() if (~taken).any() else float('nan'):.2f}")
    return _d1_report("ATC/12c", per_state, pooled, budget.n,
                      time.time() - t0, extra={
                          "ckpt": os.path.basename(path),
                          "episode": ckpt.get("episode")})


def _d1_report(side, per_state, pooled, sim_steps, wall, extra=None):
    out = {"side": side, "per_state": per_state,
           "sim_steps": sim_steps, "wall_s": round(wall)}
    if extra:
        out.update(extra)
    for grp in ("taken", "untaken"):
        rows = pooled[grp]
        rho, n = pooled_centered_spearman([r[:5] for r in rows])
        ws = [r[5] for r in rows]
        cov = [r[6] for r in rows]
        out[f"rho_{grp}_pooled"] = rho
        out[f"n_{grp}"] = n
        out[f"mean_width_{grp}"] = float(np.mean(ws)) if ws else None
        out[f"coverage_{grp}"] = float(np.mean(cov)) if cov else None
    ps_tk = [p["rho_taken"] for p in per_state if p["rho_taken"] is not None]
    ps_un = [p["rho_untaken"] for p in per_state
             if p["rho_untaken"] is not None]
    out["rho_taken_perstate_mean"] = float(np.mean(ps_tk)) if ps_tk else None
    out["rho_untaken_perstate_mean"] = float(np.mean(ps_un)) if ps_un \
        else None
    out["rho_all_mean"] = float(np.mean(
        [p["rho_all"] for p in per_state if p["rho_all"] is not None]))
    # width-exposure check
    both = [p for p in per_state if p["mean_w_untaken"] is not None]
    if both and out["mean_width_taken"]:
        frac_wider = float(np.mean([p["mean_w_untaken"] > p["mean_w_taken"]
                                    for p in both]))
        ratio = out["mean_width_untaken"] / out["mean_width_taken"]
        out["width_untaken_over_taken"] = ratio
        out["frac_states_untaken_wider"] = frac_wider
        out["width_tracks_exposure"] = bool(ratio > 1.1 and
                                            frac_wider > 0.6)
    # D7 gate, level half (ATC states carry the fields; LL path skips).
    # CONFLICT stratum is 0 (min proximate pair < 10 nm) per
    # snapshot_stratum — NOT 2 (2 = >30 nm or no proximate pair = SAFE).
    # ⚠ 19-Jul audit correction (A5): readings taken before this fix
    # summarized the SAFE stratum under a "conflict" label; treat the
    # 18-19 July "conflict-state NOOP level" numbers as safe-stratum
    # measurements pending JK-ordered re-derivation (decision D4).
    conf = [p for p in per_state
            if p.get("stratum") == 0 and "noop_score" in p]
    if conf:
        out["conflict_noop_score_mean"] = float(
            np.mean([p["noop_score"] for p in conf]))
        out["conflict_noop_gt_mean"] = float(
            np.mean([p["noop_gt"] for p in conf]))
        out["conflict_frac_noop_net_best"] = float(
            np.mean([p["noop_is_net_best"] for p in conf]))
        out["n_conflict_states"] = len(conf)

    print(f"\n  [{side}] D1 SUMMARY over {len(per_state)} states "
          f"({sim_steps} priced steps, {wall:.0f}s wall)")
    print(f"    {'group':<9}{'n':>6}{'rho pooled':>12}"
          f"{'rho/state':>11}{'coverage':>10}{'mean width':>12}")
    def _f(v):
        return float("nan") if v is None else v
    for grp in ("taken", "untaken"):
        print(f"    {grp:<9}{out[f'n_{grp}']:>6}"
              f"{_f(out[f'rho_{grp}_pooled']):>12.3f}"
              f"{_f(out[f'rho_{grp}_perstate_mean']):>11.3f}"
              f"{_f(out[f'coverage_{grp}']):>10.3f}"
              f"{_f(out[f'mean_width_{grp}']):>12.3f}")
    if "width_untaken_over_taken" in out:
        print(f"    width untaken/taken = "
              f"{out['width_untaken_over_taken']:.3f}; untaken wider in "
              f"{out['frac_states_untaken_wider'] * 100:.0f}% of states "
              f"-> width {'TRACKS' if out['width_tracks_exposure'] else 'does NOT track'} "
              f"action-level exposure")
    if conf:
        print(f"    conflict-state NOOP level ({len(conf)} stratum-0 "
              f"<10nm states): net {out['conflict_noop_score_mean']:+.2f}"
              f" vs GT {out['conflict_noop_gt_mean']:+.2f}; net rates "
              f"NOOP best in "
              f"{out['conflict_frac_noop_net_best'] * 100:.0f}% of them")
    return out


def d1_verdict(atc, ll):
    lines = []
    v_atc = v_ll = "INSUFFICIENT-DATA"
    if atc and atc.get("rho_taken_pooled") is not None and \
            atc.get("rho_untaken_pooled") is not None:
        rt, ru = atc["rho_taken_pooled"], atc["rho_untaken_pooled"]
        v_atc = "AS-PREDICTED" if (ru < 0.15 and rt - ru > 0.2) else "MISS"
        lines.append(f"12c: rho_taken={rt:.3f} rho_untaken={ru:.3f} "
                     f"-> {v_atc}")
    if ll and ll.get("rho_taken_pooled") is not None:
        rt = ll["rho_taken_pooled"]
        ru = ll.get("rho_untaken_pooled")
        if ru is None:
            v_ll = "AS-PREDICTED" if rt > 0.3 else "MISS"
            lines.append(f"LL: rho_taken={rt:.3f}, no untaken candidates "
                         f"(all actions sampled) -> {v_ll}")
        else:
            v_ll = "AS-PREDICTED" if (rt > 0.3 and ru > 0.3 and
                                      abs(rt - ru) < 0.3) else "MISS"
            lines.append(f"LL: rho_taken={rt:.3f} rho_untaken={ru:.3f} "
                         f"-> {v_ll}")
    return {"verdict_atc": v_atc, "verdict_ll": v_ll, "lines": lines}


# ---------------------------------------------------------------------------
# D2 — ATC
# ---------------------------------------------------------------------------

def collect_conflict_states_atc(agent, env, n_states, hmax, budget,
                                seeds=None, per_seed_cap=None):
    """On-policy (frozen greedy) states with min vertically-proximate
    pair separation < D2_ATC_SEP_NM AND a LoS ahead within hmax steps
    under all-NOOP (checked on a raw deepcopy).  Min 8 steps between
    accepted states.  `seeds`/`per_seed_cap` default to the original D2
    behaviour (seeds 70042-70053, no cap); the knee rerun passes a wide
    band with a cap so no single episode dominates the sample."""
    if seeds is None:
        seeds = range(70042, 70054)
    found = []
    for seed in seeds:
        if len(found) >= n_states or budget.exhausted():
            break
        obs, info = env.reset(seed=seed)
        bcd.reset_delivery_clock(env)
        li = {}
        maxstep = int(getattr(env, "maxstep",
                              env.config.scenario_duration // SEC_PER_STEP))
        snap = sector_snapshot(env, obs.keys())
        last_acc = -100
        n_this_seed = 0
        for step in range(maxstep):
            if per_seed_cap is not None and n_this_seed >= per_seed_cap:
                break
            sep = min_pair_sep_nm(snap)
            if sep < D2_ATC_SEP_NM and step - last_acc >= 8 and \
                    len(found) < n_states:
                env_c = copy.deepcopy(env)
                o = copy.deepcopy(obs)
                los_at = None
                for h in range(1, hmax + 1):
                    o, _, _, _, inf = env_c.step({cs: 0 for cs in o})
                    budget.add(1)
                    v, _, _ = detect_violation(inf)
                    if v:
                        los_at = h
                        break
                if los_at is not None:
                    found.append((copy.deepcopy(env), copy.deepcopy(obs),
                                  seed, step, sep, los_at))
                    last_acc = step
                    n_this_seed += 1
                    print(f"    conflict state {len(found):2d}: seed={seed}"
                          f" step={step:3d} sep={sep:.1f}nm "
                          f"NOOP-LoS at +{los_at}")
            actions, aux = agent.generate_action(env, obs, info,
                                                 c=agent.c_train,
                                                 force_epsilon=0.0,
                                                 last_issued=li)
            if aux["cand_idx"] != 0 and agent.mask_reissue:
                i, j = divmod(aux["cand_idx"] - 1, agent.n_instr)
                li[aux["callsigns"][i]] = j
            obs, _, _, _, info = env.step(actions)
            budget.add(1)
            snap = sector_snapshot(env, obs.keys())
            v, _, _ = detect_violation(info)
            if v:
                break
    return found


def d2_atc(args):
    horizons = args.d2_horizons_atc
    hmax = max(horizons)
    print("\n" + "=" * 78)
    print(f"D2/ATC — CONSEQUENCE vs CREDIT HORIZON ({args.d2_states} "
          f"conflict states x {{best, NOOP, random-other}} x H={horizons}; "
          f"gaps are prefix returns of ONE determinized H={hmax} rollout "
          f"per candidate; noise from {D2_NOISE_REPEATS} repeats)")
    print("EXPECTATION:", EXPECT["d2"])
    print("=" * 78)
    dc.validate_cbp_mirror()
    agent, ckpt, path = atc_agent(args)
    env = dc.get_env(1200)
    budget = StepBudget(BUDGETS["knee_atc" if getattr(args, "knee", False)
                                else "d2_atc"])
    t0 = time.time()
    states = collect_conflict_states_atc(
        agent, env, args.d2_states, hmax, budget,
        seeds=getattr(args, "d2_seeds", None),
        per_seed_cap=getattr(args, "d2_per_seed_cap", None))
    print(f"  conflict states found: {len(states)} "
          f"({budget.n} scan steps)")
    if len(states) < args.d2_states:
        print(f"  [FLAG] only {len(states)}/{args.d2_states} conflict "
              f"states found in the scan budget — proceeding with these")
    # D2-Phi rerun: pre-registration hook runs BEFORE any sweep rollout
    if getattr(args, "pre_sweep_hook", None) is not None:
        args.pre_sweep_hook(states, agent)
    per_state = []
    rng = np.random.default_rng(4242)
    for s_i, (env_s, obs_s, seed, step, sep, los_at) in enumerate(states):
        if budget.exhausted():
            print(f"  [budget] stopping at state {s_i} "
                  f"({budget.n} priced steps)")
            break
        cs_list = sorted(obs_s.keys())
        snap_s = sector_snapshot(env_s, obs_s.keys())
        cands = dc.conflict_ranked_candidates(snap_s, max_aircraft=2,
                                              n_instr=agent.n_instr)
        toks = build_tokens(env_s, obs_s, None, cs_list)
        scores, cl, cu = dc._hurwicz_scores(agent, toks, agent.c_train)

        def net_view(cs, j):
            if cs is None:
                k = 0
            else:
                k = 1 + cs_list.index(cs) * agent.n_instr + (j - 1)
            return {"cand_idx": k, "score": float(scores[k]),
                    "lower": float(cl[k]), "upper": float(cu[k]),
                    "width": float(cu[k] - cl[k])}

        # sweep: one prefix rollout per candidate (NOOP is cands[0])
        sweep = []
        for cs, j in cands:
            acts = dc.noop_action(obs_s)
            li0 = {}
            issued = cs is not None
            if issued:
                acts[cs] = j
                li0[cs] = j - 1
            g, st, vio, vstep = atc_prefix_rollout(
                env_s, obs_s, snap_s, acts, issued, horizons,
                make_continue(agent, agent.c_train, li0), budget)
            sweep.append({"cs": cs, "j": j, "G": g, "violated": vio,
                          "violation_step": vstep})
        noop = sweep[0]
        issued_arms = sweep[1:]
        best = max(issued_arms, key=lambda a: a["G"][hmax])
        others = [a for a in issued_arms if a is not best]
        rand = others[int(rng.integers(len(others)))] if others else None
        noop_best_at_hmax = noop["G"][hmax] >= best["G"][hmax]

        # noise: repeats of the SAME (state, candidate) for the 3 arms
        noise = {}
        for name, arm in (("best", best), ("noop", noop),
                          ("rand", rand)):
            if arm is None:
                continue
            gs = {h: [arm["G"][h]] for h in horizons}
            for _ in range(D2_NOISE_REPEATS - 1):
                acts = dc.noop_action(obs_s)
                li0 = {}
                issued = arm["cs"] is not None
                if issued:
                    acts[arm["cs"]] = arm["j"]
                    li0[arm["cs"]] = arm["j"] - 1
                g, st, _, _ = atc_prefix_rollout(
                    env_s, obs_s, snap_s, acts, issued, horizons,
                    make_continue(agent, agent.c_train, li0), budget)
                for h in horizons:
                    gs[h].append(g[h])
            noise[name] = {h: float(np.std(gs[h])) for h in horizons}
        rec = {
            "seed": seed, "step": step, "min_sep_nm": sep,
            "noop_los_at": los_at,
            "noop_G": noop["G"], "noop_violated": noop["violated"],
            "noop_violation_step": noop["violation_step"],
            "best": {"cs": best["cs"], "j": best["j"], "G": best["G"],
                     "violated": best["violated"],
                     "violation_step": best["violation_step"],
                     "net": net_view(best["cs"], best["j"])},
            "rand": None if rand is None else
                    {"cs": rand["cs"], "j": rand["j"], "G": rand["G"],
                     "violated": rand["violated"],
                     "violation_step": rand["violation_step"],
                     "net": net_view(rand["cs"], rand["j"])},
            "noop_net": net_view(None, 0),
            "noop_best_at_hmax": bool(noop_best_at_hmax),
            "gap_best": {h: best["G"][h] - noop["G"][h] for h in horizons},
            "gap_rand": None if rand is None else
                        {h: rand["G"][h] - noop["G"][h] for h in horizons},
            "noise": noise,
        }
        if getattr(args, "d2_full_sweep", False):
            # D2-Phi rerun: keep EVERY arm's prefix returns so tier-1
            # can be evaluated on the BASELINE's best arm even when the
            # re-priced best arm differs
            rec["sweep"] = [{"cs": a["cs"], "j": a["j"], "G": a["G"],
                             "violated": a["violated"],
                             "violation_step": a["violation_step"]}
                            for a in sweep]
        per_state.append(rec)
        gline = " ".join(f"H{h}:{rec['gap_best'][h]:+7.2f}"
                         for h in horizons)
        print(f"  state {s_i:2d} (sep {sep:4.1f}nm, NOOP-LoS +{los_at:2d})"
              f" best=({best['cs']},{dc.INSTR_NAMES[best['j'] - 1] if best['j'] <= len(dc.INSTR_NAMES) else best['j']})"
              f"{' [NOOP>=best@96]' if noop_best_at_hmax else ''}")
        print(f"           gap_best {gline}")
    return _d2_report("ATC/12c", horizons, per_state, budget.n,
                      time.time() - t0,
                      extra={"ckpt": os.path.basename(path),
                             "episode": ckpt.get("episode")})


def _d2_report(side, horizons, per_state, sim_steps, wall, extra=None):
    out = {"side": side, "horizons": list(horizons),
           "per_state": per_state, "sim_steps": sim_steps,
           "wall_s": round(wall)}
    if extra:
        out.update(extra)
    if not per_state:
        return out
    med_gap, mean_gap, noise_p95 = {}, {}, {}
    for h in horizons:
        g = [abs(p["gap_best"][h]) for p in per_state]
        med_gap[h] = float(np.median(g))
        mean_gap[h] = float(np.mean(g))
        ns = [p["noise"][a][h] for p in per_state
              for a in p["noise"]]
        noise_p95[h] = float(np.percentile(ns, 95)) if ns else 0.0
    out["median_abs_gap_best"] = med_gap
    out["mean_abs_gap_best"] = mean_gap
    out["noise_p95"] = noise_p95
    hmax = max(horizons)
    out["emergence"] = {h: (med_gap[h] / med_gap[hmax])
                        if med_gap[hmax] > 0 else None for h in horizons}
    out["frac_noop_best_at_hmax"] = float(np.mean(
        [p["noop_best_at_hmax"] for p in per_state]))
    print(f"\n  [{side}] D2 SUMMARY over {len(per_state)} states "
          f"({sim_steps} priced steps, {wall:.0f}s wall)")
    print(f"    {'H':>5}{'median|gap|':>13}{'mean|gap|':>11}"
          f"{'noise p95':>11}{'frac of gap(Hmax)':>19}")
    for h in horizons:
        em = out["emergence"][h]
        print(f"    {h:>5}{med_gap[h]:>13.3f}{mean_gap[h]:>11.3f}"
              f"{noise_p95[h]:>11.2e}"
              f"{em if em is not None else float('nan'):>19.3f}")
    print(f"    NOOP >= best issued candidate at H={hmax} in "
          f"{out['frac_noop_best_at_hmax'] * 100:.0f}% of states")
    return out


def d2_verdict(atc, ll):
    lines = []
    v_atc = v_ll = "INSUFFICIENT-DATA"
    if atc and atc.get("median_abs_gap_best"):
        mg, np95 = atc["median_abs_gap_best"], atc["noise_p95"]
        hmax = max(atc["horizons"])
        floor = max(5 * np95[hmax], 1e-6)
        ok = mg[12] < 0.25 * mg[hmax] and mg[hmax] > floor
        v_atc = "AS-PREDICTED" if ok else "MISS"
        lines.append(f"12c: |gap|(12)={mg[12]:.3f} vs |gap|({hmax})="
                     f"{mg[hmax]:.3f} (noise floor {floor:.2e}) -> {v_atc}")
    if ll and ll.get("median_abs_gap_best"):
        mg, np95 = ll["median_abs_gap_best"], ll["noise_p95"]
        floor = max(5 * np95[1], 1e-6)
        v_ll = "AS-PREDICTED" if mg[1] > floor else "MISS"
        lines.append(f"LL: |gap|(1)={mg[1]:.3f} (noise floor {floor:.2e}) "
                     f"-> {v_ll}")
    return {"verdict_atc": v_atc, "verdict_ll": v_ll, "lines": lines}


# ---------------------------------------------------------------------------
# Knee rerun analysis (12D_CREDIT_DESIGN.md fix B / decision D1)
# ---------------------------------------------------------------------------

def knee_report(atc):
    """Post-process a d2_atc result run on KNEE_HORIZONS.

    Per state (SIGNED gaps throughout — the decision is about sign-correct
    in-window signal, not magnitude):
      * action_helps: gap_best(96) > 0 (NOOP-optimal states have no knee;
        fee-only ordering is CORRECT there — 12D amendment).
      * personal_knee: smallest grid H with gap(H) >= KNEE_FRAC*gap(96);
        96 if only the reference horizon reaches it.
      * delay_not_avoid: best arm still violates by H=96 (risk 7,
        delay-credited-as-avoidance); delay_steps vs the NOOP arm.
    """
    horizons = list(atc["horizons"])
    hmax = max(horizons)
    grid = [h for h in horizons if h < hmax]
    per, knees = [], []
    for i, p in enumerate(atc["per_state"]):
        g = p["gap_best"]
        g96 = g[hmax]
        helps = g96 > 0
        knee = None
        if helps:
            knee = hmax
            for h in grid:
                if g[h] >= KNEE_FRAC * g96:
                    knee = h
                    break
            knees.append(knee)
        bvs = p["best"].get("violation_step")
        nvs = p["noop_violation_step"]
        per.append({
            "state": i, "seed": p["seed"], "step": p["step"],
            "min_sep_nm": p["min_sep_nm"],
            "noop_violation_step": nvs,
            "gap_best": {h: g[h] for h in horizons},
            "action_helps": bool(helps),
            "personal_knee": knee,
            "best_violated_by_96": bool(p["best"]["violated"]),
            "best_violation_step": bvs,
            "delay_steps": (bvs - nvs) if (bvs is not None and
                                           nvs is not None) else None,
        })
    helps_states = [q for q in per if q["action_helps"]]
    n_h = len(helps_states)

    def frac_knee_le(n):
        return (sum(1 for k in knees if k <= n) / n_h) if n_h else None

    def frac_sign_correct(n):
        if not n_h or n not in horizons:
            return None
        return sum(1 for q in helps_states if q["gap_best"][n] > 0) / n_h

    def frac_delay_past(n):
        """Delay-credited-as-avoidance at window n: best arm survives the
        window but still violates by 96."""
        if not n_h:
            return None
        return sum(1 for q in helps_states if q["best_violated_by_96"] and
                   (q["best_violation_step"] is None or
                    q["best_violation_step"] > n)) / n_h

    gap_table = {}
    for h in horizons:
        allg = [p["gap_best"][h] for p in atc["per_state"]]
        hg = [q["gap_best"][h] for q in helps_states]
        gap_table[h] = {
            "median": float(np.median(allg)),
            "q25": float(np.percentile(allg, 25)),
            "q75": float(np.percentile(allg, 75)),
            "median_action_helps": float(np.median(hg)) if hg else None,
            "q25_action_helps": float(np.percentile(hg, 25)) if hg else None,
            "q75_action_helps": float(np.percentile(hg, 75)) if hg else None,
        }
    nvs_list = [q["noop_violation_step"] for q in per
                if q["noop_violation_step"] is not None]
    delayers = [q for q in helps_states if q["best_violated_by_96"]]
    avoiders = [q for q in helps_states if not q["best_violated_by_96"]]
    delay_steps = [q["delay_steps"] for q in delayers
                   if q["delay_steps"] is not None]
    out = {
        "doc": "12D_CREDIT_DESIGN.md fix B / D1 knee rerun",
        "knee_frac": KNEE_FRAC,
        "horizon_grid": grid, "reference_h": hmax,
        "n_states": len(per), "n_action_helps": n_h,
        "n_noop_optimal": len(per) - n_h,
        "gap_vs_h": gap_table,
        "per_state": per,
        "knees": knees,
        "knee_median": float(np.median(knees)) if knees else None,
        "knee_distribution": {str(h): knees.count(h)
                              for h in grid + [hmax]},
        "frac_knee_le_24": frac_knee_le(24),
        "frac_knee_le_36": frac_knee_le(36),
        "frac_sign_correct_at_24": frac_sign_correct(24),
        "frac_sign_correct_at_36": frac_sign_correct(36),
        "noop_violation_step": {
            "median": float(np.median(nvs_list)) if nvs_list else None,
            "n_le_24": sum(1 for v in nvs_list if v <= 24),
            "n_le_36": sum(1 for v in nvs_list if v <= 36),
            "n": len(nvs_list)},
        "delay_check": {
            "n_delay": len(delayers), "n_avoid": len(avoiders),
            "delay_share_action_helps": (len(delayers) / n_h) if n_h
                                        else None,
            "median_delay_steps": float(np.median(delay_steps))
                                  if delay_steps else None,
            "frac_delay_credited_as_avoidance_at_24": frac_delay_past(24),
            "frac_delay_credited_as_avoidance_at_36": frac_delay_past(36)},
    }
    # decision rule pre-registered in 12D section 6 (D1): 24 unless the
    # knee rerun shows < 50% action-helps sign-correct in-window at 24.
    sc24 = out["frac_sign_correct_at_24"]
    out["rule_recommendation"] = (
        None if sc24 is None else ("nstep=24" if sc24 >= 0.5 else
                                   "nstep=36 (24 fails the 50% gate)"))

    print("\n" + "=" * 78)
    print("KNEE RERUN ANALYSIS (fix B / D1: nstep 24 vs 36)")
    print("=" * 78)
    print(f"  states: {len(per)} ({n_h} action-helps, "
          f"{len(per) - n_h} NOOP-optimal)")
    print(f"  {'H':>5}{'median gap':>12}{'IQR':>22}"
          f"{'median (helps)':>16}")
    for h in horizons:
        t = gap_table[h]
        mh = t["median_action_helps"]
        print(f"  {h:>5}{t['median']:>12.3f}"
              f"  [{t['q25']:>8.3f},{t['q75']:>8.3f}]"
              f"{mh if mh is not None else float('nan'):>16.3f}")
    print(f"  personal knees (action-helps only): {sorted(knees)}")
    print(f"    median={out['knee_median']}, "
          f"<=24: {out['frac_knee_le_24']}, <=36: {out['frac_knee_le_36']}")
    print(f"  sign-correct in-window (action-helps): "
          f"at 24: {out['frac_sign_correct_at_24']}, "
          f"at 36: {out['frac_sign_correct_at_36']}")
    print(f"  noop_violation_step: median="
          f"{out['noop_violation_step']['median']}, "
          f"<=24: {out['noop_violation_step']['n_le_24']}"
          f"/{out['noop_violation_step']['n']}, "
          f"<=36: {out['noop_violation_step']['n_le_36']}"
          f"/{out['noop_violation_step']['n']}")
    dchk = out["delay_check"]
    print(f"  delay-vs-avoid (action-helps): delay={dchk['n_delay']} "
          f"avoid={dchk['n_avoid']} "
          f"(median delay {dchk['median_delay_steps']} steps); "
          f"delay-credited-as-avoidance at 24: "
          f"{dchk['frac_delay_credited_as_avoidance_at_24']}, "
          f"at 36: {dchk['frac_delay_credited_as_avoidance_at_36']}")
    print(f"  pre-registered rule -> {out['rule_recommendation']}")
    return out


# ---------------------------------------------------------------------------
# D2-Phi rerun machinery (pre-registration is the header block above)
# ---------------------------------------------------------------------------

def _d5_g_vert(dcur, dcmd):
    gc = max(0.0, (20.0 - dcur) / 20.0) ** 2
    gm = max(0.0, (10.0 - dcmd) / 10.0) ** 2
    return 0.5 * gc + 0.5 * gm


def d5_analytic_p1(snap, cs, j_env):
    """Analytic one-time CBP conflict payment for issuing env action
    j_env (4 = climb +10, 5 = descend -10) to `cs`, summed over ALL its
    laterally-priced pairs, from the state geometry alone (no rollout):
    measured state = 1 acting step at the planning rocd (2.75 FL/step,
    both for the target's new selected FL and for any partner already
    mid-climb), lateral geometry assumed common-mode and static."""
    me = snap[cs]
    sel_me = me.sel_fl if me.sel_fl is not None else me.fl
    d_sel = 10.0 if j_env == 4 else -10.0
    new_sel = sel_me + d_sel

    def flown(fl, sel):
        step = np.clip(sel - fl, -D5_ROCD_PLAN, D5_ROCD_PLAN)
        return fl + float(step)

    tot = 0.0
    for o_cs, o in snap.items():
        if o_cs == cs:
            continue
        d = haversine_nm(me.lat, me.lon, o.lat, o.lon)
        m = max(0.0, (15.0 - d) / 15.0)
        f_lat = m * m
        if f_lat == 0.0:
            continue
        o_sel = o.sel_fl if o.sel_fl is not None else o.fl
        o_fl1 = flown(o.fl, o_sel)
        g_noop = _d5_g_vert(abs(flown(me.fl, sel_me) - o_fl1),
                            abs(sel_me - o_sel))
        g_act = _d5_g_vert(abs(flown(me.fl, new_sel) - o_fl1),
                           abs(new_sel - o_sel))
        tot += ATC_GAMMA * bcd.DELTA_CONFLICT * f_lat * (g_noop - g_act)
    return tot


def load_knee_baseline():
    with open(KNEE_BASELINE_JSON) as f:
        base = json.load(f)
    per = base["results"]["d2"]["atc"]["per_state"]
    return {(p["seed"], p["step"]): p for p in per}


def d2phi_prereg_hook(baseline, store):
    """Builds the pre-registration (analytic p1 table + tier-2 flip
    list) from the collected states BEFORE any sweep rollout runs.
    Returns a hook for d2_atc."""

    def hook(states, agent):
        print("\n  " + "-" * 74)
        print(f"  D2-PHI PRE-REGISTRATION ({bcd.conflict_pricing_str()}; "
              f"computed BEFORE any rollout)")
        print(f"  {'st':>4}{'seed':>7}{'step':>6}{'base best':>16}"
              f"{'base g12':>10}{'p1(best)':>10}{'p1(max)':>10}"
              f"{'pred g12':>10}{'flip?':>6}")
        for s_i, (env_s, obs_s, seed, step, sep, los_at) in \
                enumerate(states):
            snap_s = sector_snapshot(env_s, obs_s.keys())
            b = baseline.get((seed, step))
            cands = dc.conflict_ranked_candidates(snap_s, max_aircraft=2,
                                                  n_instr=agent.n_instr)
            vert = [(cs, j) for cs, j in cands
                    if cs is not None and j in (4, 5)]
            p1s = {(cs, j): d5_analytic_p1(snap_s, cs, j)
                   for cs, j in vert}
            p1_max = max(p1s.values()) if p1s else 0.0
            rec = {"state": s_i, "seed": seed, "step": step,
                   "baseline_found": b is not None,
                   "p1_by_arm": {f"{cs}|{j}": v
                                 for (cs, j), v in p1s.items()},
                   "p1_max": p1_max}
            if b is not None:
                bb = b["best"]
                rec["baseline_best"] = {"cs": bb["cs"], "j": bb["j"]}
                rec["baseline_best_vertical"] = bb["j"] in (4, 5)
                rec["baseline_gap12"] = b["gap_best"]["12"]
                rec["baseline_noop_violation_step"] = \
                    b["noop_violation_step"]
                rec["p1_baseline_best"] = p1s.get((bb["cs"], bb["j"]))
                rec["pred_gap12"] = rec["baseline_gap12"] + p1_max
                rec["pred_flip"] = (rec["baseline_gap12"] <= 0.0
                                    and rec["pred_gap12"] > 0.0)
                bstr = f"{bb['cs']}|{dc.INSTR_NAMES[bb['j'] - 1]}" \
                    if bb["cs"] is not None else "NOOP"
                p1b = rec["p1_baseline_best"]
                print(f"  {s_i:>4}{seed:>7}{step:>6}{bstr:>16}"
                      f"{rec['baseline_gap12']:>10.3f}"
                      f"{p1b if p1b is not None else float('nan'):>10.3f}"
                      f"{p1_max:>10.3f}{rec['pred_gap12']:>10.3f}"
                      f"{'YES' if rec['pred_flip'] else '':>6}")
            else:
                print(f"  {s_i:>4}{seed:>7}{step:>6}"
                      f"{'<NO BASELINE MATCH>':>16}")
            store.append(rec)
        flips = [r["state"] for r in store if r.get("pred_flip")]
        if flips:
            print(f"  TIER-2 PRE-REGISTERED FLIP LIST (gap12 <= 0 -> "
                  f"> 0): states {flips}")
        else:
            print("  TIER-2 PRE-REGISTERED: the flip list is EMPTY — "
                  "Tier 2 predicts a null; efficacy rides on the micro "
                  "gate.")
        print("  " + "-" * 74)
    return hook


def d2phi_report(atc, prereg, baseline):
    """Two-tier evaluation of the smooth-pricing D2 rerun against the
    gate-pricing knee baseline (pre-registration in the module header)."""
    print("\n" + "=" * 78)
    print(f"D2-PHI RERUN REPORT ({bcd.conflict_pricing_str()}; baseline "
          f"{os.path.basename(KNEE_BASELINE_JSON)})")
    print("=" * 78)
    pre = {r["state"]: r for r in prereg}
    tier1_rows, sanity_bad = [], []
    for s_i, p in enumerate(atc["per_state"]):
        r = pre.get(s_i, {})
        b = baseline.get((p["seed"], p["step"]))
        if b is None:
            continue
        # environment-property sanity: pricing must not move dynamics
        if p["noop_violation_step"] != b["noop_violation_step"]:
            sanity_bad.append((s_i, b["noop_violation_step"],
                               p["noop_violation_step"]))
        bb = b["best"]
        arm = None
        for a in p.get("sweep", []):
            if a["cs"] == bb["cs"] and a["j"] == bb["j"]:
                arm = a
                break
        if arm is None:
            continue
        gap12_same_arm = arm["G"][12] - p["noop_G"][12]
        base_gap12 = b["gap_best"]["12"]
        shift = gap12_same_arm - base_gap12
        p1 = r.get("p1_baseline_best")
        vertical = bb["j"] in (4, 5)
        if vertical and p1 is not None:
            ok = (np.sign(shift) == np.sign(p1)
                  and 0.5 * abs(p1) <= abs(shift) <= 2.0 * abs(p1))
        else:
            ok = abs(shift) <= 1e-3
        tier1_rows.append({
            "state": s_i, "seed": p["seed"], "step": p["step"],
            "baseline_best": {"cs": bb["cs"], "j": bb["j"]},
            "vertical": vertical, "base_gap12": base_gap12,
            "smooth_gap12_same_arm": gap12_same_arm, "shift": shift,
            "analytic_p1": p1, "tier1_ok": bool(ok)})
    vrows = [t for t in tier1_rows if t["vertical"]]
    lrows = [t for t in tier1_rows if not t["vertical"]]
    print(f"\n  TIER 1 — vertical-best states ({len(vrows)}): same-arm "
          f"gap(12) shift vs analytic p1 (PASS: sign-correct, within "
          f"[0.5x, 2x])")
    print(f"  {'st':>4}{'arm':>18}{'base g12':>10}{'smooth g12':>12}"
          f"{'shift':>10}{'p1':>10}{'ok':>5}")
    for t in vrows:
        arm = f"{t['baseline_best']['cs']}|" \
              f"{dc.INSTR_NAMES[t['baseline_best']['j'] - 1]}"
        print(f"  {t['state']:>4}{arm:>18}{t['base_gap12']:>10.3f}"
              f"{t['smooth_gap12_same_arm']:>12.3f}{t['shift']:>10.3f}"
              f"{t['analytic_p1']:>10.3f}"
              f"{'OK' if t['tier1_ok'] else 'MISS':>5}")
    n_ok_v = sum(t["tier1_ok"] for t in vrows)
    print(f"  TIER 1 — lateral-best states ({len(lrows)}): same-arm "
          f"gap(12) unchanged within 1e-3 (registered at DELTA=0.2; "
          f"DELTA={bcd.DELTA_CONFLICT} rescales ambient lateral "
          f"differentials 2.5x — breaches reported, not excused)")
    n_ok_l = sum(t["tier1_ok"] for t in lrows)
    for t in lrows:
        if not t["tier1_ok"]:
            print(f"    state {t['state']}: |shift| "
                  f"{abs(t['shift']):.4f} > 1e-3 (lateral arm)")
    tier1_pass = (n_ok_v == len(vrows))
    print(f"  TIER 1 verdict: vertical {n_ok_v}/{len(vrows)} OK, "
          f"lateral {n_ok_l}/{len(lrows)} within 1e-3 -> "
          f"{'PASS' if tier1_pass else 'MISS'} (gate is the vertical "
          f"set; lateral breaches are findings)")

    # tier 2: measured flips vs the pre-registered list
    pred = sorted(r["state"] for r in prereg if r.get("pred_flip"))
    meas = []
    for s_i, p in enumerate(atc["per_state"]):
        b = baseline.get((p["seed"], p["step"]))
        if b is None:
            continue
        if b["gap_best"]["12"] <= 0.0 and p["gap_best"][12] > 0.0:
            meas.append(s_i)
    hits = sorted(set(pred) & set(meas))
    print(f"\n  TIER 2 — pre-registered flip list: {pred if pred else 'EMPTY (null predicted)'}")
    print(f"  TIER 2 — measured flips (own-best gap(12) <=0 -> >0): "
          f"{meas if meas else 'none'}")
    print(f"  TIER 2 — hits {hits}; predicted-but-not-flipped "
          f"{sorted(set(pred) - set(meas))}; flipped-but-not-predicted "
          f"{sorted(set(meas) - set(pred))}")
    if sanity_bad:
        print(f"\n  [SANITY FAIL] noop_violation_step moved vs baseline "
              f"at states {sanity_bad} — pricing is NOT supposed to "
              f"move dynamics; investigate before trusting this rerun")
    else:
        print("\n  sanity: noop_violation_step identical to the "
              "baseline at every matched state (Phi does not move "
              "dynamics)  OK")
    return {"doc": "D5 D2-Phi rerun (two-tier, pre-registered in "
                   "credit_diagnostics.py header)",
            "pricing": bcd.conflict_pricing_str(),
            "baseline": os.path.basename(KNEE_BASELINE_JSON),
            "prereg": prereg, "tier1": tier1_rows,
            "tier1_vertical_ok": n_ok_v, "tier1_vertical_n": len(vrows),
            "tier1_lateral_ok": n_ok_l, "tier1_lateral_n": len(lrows),
            "tier1_pass": bool(tier1_pass),
            "tier2_predicted": pred, "tier2_measured": meas,
            "sanity_noop_violation_mismatches": sanity_bad}


# ---------------------------------------------------------------------------
# D3 — ATC
# ---------------------------------------------------------------------------

def d3_atc(args):
    print("\n" + "=" * 78)
    print(f"D3/ATC — BOOTSTRAP-CHAIN FIDELITY ({args.d3_episodes} fresh "
          f"greedy episodes; V(s_t)=max Hurwicz@c_train UNMASKED vs "
          f"realized CBP-signal G_t, gamma={ATC_GAMMA})")
    print("EXPECTATION:", EXPECT["d3"])
    print("=" * 78)
    dc.validate_cbp_mirror()
    agent, ckpt, path = atc_agent(args)
    env = dc.get_env(1200)
    c_pol = agent.c_train
    budget = StepBudget(BUDGETS["d3_atc"])
    records, records_masked = [], []   # (steps_to_terminal, V, G)
    ep_rows = []
    t0 = time.time()
    n_violated = 0
    for e_i in range(args.d3_episodes):
        if budget.exhausted():
            print(f"  [budget] stopping at episode {e_i}")
            break
        seed = 80042 + e_i
        obs, info = env.reset(seed=seed)
        bcd.reset_delivery_clock(env)
        li = {}
        maxstep = int(getattr(env, "maxstep",
                              env.config.scenario_duration // SEC_PER_STEP))
        snap = sector_snapshot(env, obs.keys())
        rs, Vs, Vms = [], [], []
        violated = False
        for step in range(maxstep):
            cs_list = sorted(obs.keys())
            if cs_list:
                toks = build_tokens(env, obs, info, cs_list)
                scores, _, _ = dc._hurwicz_scores(agent, toks, c_pol)
                rmask = build_reissue_mask(cs_list, li, agent.n_instr) \
                    if agent.mask_reissue else np.zeros(len(scores),
                                                        dtype=bool)
                V = float(scores.max())
                Vm = float(np.where(rmask, -np.inf, scores).max())
            else:
                V = Vm = 0.0
            if cs_list:
                actions, aux = agent.generate_action(
                    env, obs, info, c=c_pol, force_epsilon=0.0,
                    last_issued=li)
                issued = aux["cand_idx"] != 0
                if issued and agent.mask_reissue:
                    i, j = divmod(aux["cand_idx"] - 1, agent.n_instr)
                    li[aux["callsigns"][i]] = j
            else:
                actions, issued = {}, False
            out = dc.sector_step_cbp(env, obs, snap, actions, issued)
            budget.add(4 if issued else 1)
            rs.append(out["r"])
            Vs.append(V)
            Vms.append(Vm)
            obs, snap, info = out["next_obs"], out["snap_next"], out["info"]
            if out["violated"]:
                violated = True
                break
        G = dc.returns_to_go(rs, gamma=ATC_GAMMA)
        T = len(rs)
        n_violated += int(violated)
        for t in range(T):
            stt = (T - 1) - t
            records.append((stt, Vs[t], G[t], violated))
            records_masked.append((stt, Vms[t], G[t], violated))
        ep_rows.append({"seed": seed, "steps": T, "violated": violated,
                        "return": float(sum(rs))})
        print(f"  ep {e_i:2d} seed={seed}: {T:3d} steps "
              f"{'VIOLATED' if violated else 'censored'} "
              f"return={sum(rs):8.2f}")
    # terminal (violation-ended) episodes carry unbiased G_t; censored
    # episodes have truncated G_t (biased) — reported separately
    rec_term = [(s, v, g) for s, v, g, vio in records if vio]
    rec_all = [(s, v, g) for s, v, g, vio in records]
    rec_term_m = [(s, v, g) for s, v, g, vio in records_masked if vio]
    rows_term = d3_bin_table(rec_term)
    rows_all = d3_bin_table(rec_all)
    rows_term_m = d3_bin_table(rec_term_m)
    print(f"\n  [ATC/12c] D3 SUMMARY: {len(ep_rows)} episodes "
          f"({n_violated} violated, {len(ep_rows) - n_violated} censored), "
          f"{budget.n} priced steps, {time.time() - t0:.0f}s wall")
    print_d3_table(rows_term, "TERMINAL (violation-ended) episodes — "
                              "unbiased G_t [PRIMARY]")
    print_d3_table(rows_all, "ALL episodes (censored G_t truncated — "
                             "biased, context only)")
    verdict, m_near, m_far = d3_verdict_atc(rows_term)
    print(f"  H-chain (12c): rho near(<20)="
          f"{m_near if m_near is not None else float('nan'):.3f} "
          f"far(>=20)={m_far if m_far is not None else float('nan'):.3f}"
          f" -> {verdict}")
    return {"side": "ATC/12c", "ckpt": os.path.basename(path),
            "episode": ckpt.get("episode"), "episodes": ep_rows,
            "n_violated": n_violated,
            "bins_terminal": rows_term, "bins_all": rows_all,
            "bins_terminal_masked": rows_term_m,
            "verdict": verdict, "rho_near": m_near, "rho_far": m_far,
            "sim_steps": budget.n, "wall_s": round(time.time() - t0)}


# ===========================================================================
# LL side — positive control
# ===========================================================================

def ll_env():
    return gym.make("LunarLander-v3", gravity=-10.0)


def ll_load(ckpt_path):
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = IntervalQNetwork()
    net.load_state_dict(d["q_net_state_dict"])
    net.eval()
    return net, float(d["args"]["c_train"])


def ll_intervals(net, state):
    with torch.no_grad():
        lo, up = net(torch.FloatTensor(np.asarray(state)).unsqueeze(0))
    return lo[0].numpy(), up[0].numpy()


def ll_greedy(net, state, c):
    lo, up = ll_intervals(net, state)
    return int(np.argmax(lo + c * (up - lo)))


def ll_branching_check():
    """Prominently re-verify BOTH halves of the adaptation at runtime:
    (a) deepcopy of the LL env FAILS (so the doc's primary mechanism is
    unavailable); (b) seeded action replay is exact."""
    env = ll_env()
    env.reset(seed=123)
    for a in (2, 2, 1, 0, 3) * 4:
        env.step(a)
    deepcopy_ok = False
    try:
        e2 = copy.deepcopy(env)
        e2.step(0)
        deepcopy_ok = True
    except Exception as e:
        print(f"  [LL-branching] copy.deepcopy(env) FAILS as expected "
              f"({type(e).__name__}: Box2D bodies do not survive "
              f"deepcopy — unwrapped.lander is None on the copy)")
    env.close()
    if deepcopy_ok:
        print("  [LL-branching] NOTE: deepcopy unexpectedly worked; "
              "still using seeded replay (pre-registered adaptation)")

    def run(seed, acts):
        e = ll_env()
        s, _ = e.reset(seed=seed)
        obs, rews = [s], []
        for a in acts:
            s, r, term, trunc, _ = e.step(a)
            obs.append(s)
            rews.append(r)
            if term or trunc:
                break
        e.close()
        return np.array(obs), np.array(rews)
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 4, 250).tolist()
    o1, r1 = run(777, acts)
    o2, r2 = run(777, acts)
    assert np.array_equal(o1, o2) and np.array_equal(r1, r2), \
        "LL seeded replay is NOT deterministic — abort LL diagnostics"
    print(f"  [LL-branching] seeded action replay IS exact "
          f"({len(o1)} steps bit-identical) -> counterfactual branching "
          f"via replay at ALL horizons (ADAPTATION: deepcopy "
          f"unavailable; full pre-registered design preserved)")


def ll_replay_to(seed, prefix):
    env = ll_env()
    s, _ = env.reset(seed=seed)
    for a in prefix:
        s, _, term, trunc, _ = env.step(a)
        assert not (term or trunc), "prefix crosses a terminal"
    return env, s


def ll_branch_return(seed, prefix, first_action, net, c, horizons):
    """GT branch return via exact seeded replay: replay prefix, take
    first_action, then frozen greedy; native reward stream discounted at
    LL_GAMMA; prefix returns recorded at every horizon; terminal ends
    the trajectory (return frozen at longer horizons)."""
    env, s = ll_replay_to(seed, prefix)
    hmax = max(horizons)
    G, disc = 0.0, 1.0
    out_g = {}
    a = first_action
    done = False
    for h in range(1, hmax + 1):
        s, r, term, trunc, _ = env.step(a)
        G += disc * r
        disc *= LL_GAMMA
        if h in horizons:
            out_g[h] = G
        if term or trunc:
            done = True
            break
        a = ll_greedy(net, s, c)
    env.close()
    for h in horizons:
        out_g.setdefault(h, G)
    return out_g, done


# ---------------------------------------------------------------------------
# D1 — LL
# ---------------------------------------------------------------------------

def d1_ll(args):
    print("\n" + "=" * 78)
    print(f"D1/LL — TAKEN vs UNTAKEN action split ({args.d1_states} "
          f"on-policy probe states, GT H={D1_H} native-reward branch "
          f"returns, visitation {args.d1_rollouts}x{D1_ROLLOUT_LEN}-step "
          f"eps={D1_EPS} rollouts)")
    print("EXPECTATION:", EXPECT["d1"])
    print("=" * 78)
    ll_branching_check()
    net, c = ll_load(args.ll_ckpt)
    print(f"  [ckpt/LL] {args.ll_ckpt} (c_train={c})")
    # on-policy probe states: greedy (eps 0.02) episodes, recorded
    # action prefixes for replay; evenly spaced picks per episode
    n_eps = max(4, args.d1_states // 5)
    per_ep = int(math.ceil(args.d1_states / n_eps))
    picks = []
    rng = np.random.default_rng(2026)
    for e_i in range(n_eps):
        seed = 50000 + e_i
        env = ll_env()
        s, _ = env.reset(seed=seed)
        acts, states = [], [s]
        for t in range(1000):
            a = int(rng.integers(0, 4)) if rng.random() < 0.02 \
                else ll_greedy(net, s, c)
            s, r, term, trunc, _ = env.step(a)
            acts.append(a)
            states.append(s)
            if term or trunc:
                break
        env.close()
        T = len(acts)
        idx = np.linspace(3, max(4, T - 5), per_ep).round().astype(int)
        for t in sorted(set(idx.tolist())):
            picks.append({"seed": seed, "t": int(t), "prefix": acts[:t],
                          "state": states[t], "ep_len": T})
    picks = picks[:args.d1_states]
    print(f"  probe states: {len(picks)} from {n_eps} greedy episodes")
    per_state = []
    pooled = {"taken": [], "untaken": []}
    t0 = time.time()
    for s_i, p in enumerate(picks):
        lo, up = ll_intervals(net, p["state"])
        scores = lo + c * (up - lo)
        widths = up - lo
        n_cand = 4
        first_counts = np.zeros(n_cand, dtype=int)
        all_counts = np.zeros(n_cand, dtype=int)
        vrng = np.random.default_rng(3000 + s_i)
        for r_i in range(args.d1_rollouts):
            env, s = ll_replay_to(p["seed"], p["prefix"])
            for t in range(D1_ROLLOUT_LEN):
                a = int(vrng.integers(0, 4)) if vrng.random() < D1_EPS \
                    else ll_greedy(net, s, c)
                all_counts[a] += 1
                if t == 0:
                    first_counts[a] += 1
                s, r, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
            env.close()
        gts = np.zeros(n_cand)
        for a in range(n_cand):
            g, _ = ll_branch_return(p["seed"], p["prefix"], a, net, c,
                                    (D1_H,))
            gts[a] = g[D1_H]
        taken = first_counts > 0
        g_mean, s_mean = float(gts.mean()), float(scores.mean())
        hit = (gts >= lo) & (gts <= up)
        for k in range(n_cand):
            grp = "taken" if taken[k] else "untaken"
            pooled[grp].append((s_i, float(gts[k]), float(scores[k]),
                                g_mean, s_mean, float(widths[k]),
                                bool(hit[k])))
        per_state.append({
            "seed": p["seed"], "t": p["t"], "n_cands": n_cand,
            "n_taken": int(taken.sum()),
            "first_counts": first_counts.tolist(),
            "rho_all": dc.spearman(gts, scores),
            "rho_taken": dc.spearman(gts[taken], scores[taken])
            if taken.sum() >= 3 else None,
            "rho_untaken": dc.spearman(gts[~taken], scores[~taken])
            if (~taken).sum() >= 3 else None,
            "mean_w_taken": float(widths[taken].mean()),
            "mean_w_untaken": float(widths[~taken].mean())
            if (~taken).any() else None,
            "cov_taken": float(hit[taken].mean()),
            "cov_untaken": float(hit[~taken].mean())
            if (~taken).any() else None,
        })
    return _d1_report("LL/per_c", per_state, pooled, 0, time.time() - t0,
                      extra={"ckpt": args.ll_ckpt, "c_train": c,
                             "gt_note": "native reward stream, "
                                        "gamma=0.99, seeded replay"})


# ---------------------------------------------------------------------------
# D2 — LL
# ---------------------------------------------------------------------------

def collect_crash_states_ll(net, c, n_states, eps=0.3, seed0=61000):
    """Pre-terminal states: roll a DEGRADED policy (epsilon 0.3) until a
    crash; take states 30-80 steps before the crash (up to 2 per crash
    episode).  Action prefixes recorded for exact replay."""
    out = []
    seed = seed0
    n_eps = 0
    while len(out) < n_states and seed < seed0 + 400:
        env = ll_env()
        rng = np.random.default_rng(seed)
        s, _ = env.reset(seed=seed)
        acts, states = [], [s]
        crashed = False
        for t in range(1000):
            a = int(rng.integers(0, 4)) if rng.random() < eps \
                else ll_greedy(net, s, c)
            s, r, term, trunc, _ = env.step(a)
            acts.append(a)
            states.append(s)
            if term or trunc:
                crashed = bool(term and env.unwrapped.game_over)
                break
        env.close()
        n_eps += 1
        T = len(acts)
        if crashed and T > 35:
            lo_t, hi_t = max(1, T - 80), T - 30
            if hi_t >= lo_t:
                k = min(2, hi_t - lo_t + 1, n_states - len(out))
                ts = sorted(rng.choice(np.arange(lo_t, hi_t + 1), size=k,
                                       replace=False).tolist())
                for t in ts:
                    out.append({"seed": seed, "t": int(t),
                                "prefix": acts[:t], "state": states[t],
                                "steps_to_crash": T - t})
        seed += 1
    print(f"    crash-state scan: {len(out)} states from {n_eps} "
          f"degraded episodes")
    return out[:n_states]


def d2_ll(args):
    horizons = args.d2_horizons_ll
    hmax = max(horizons)
    print("\n" + "=" * 78)
    print(f"D2/LL — CONSEQUENCE vs CREDIT HORIZON ({args.d2_states} "
          f"pre-crash states (30-80 steps out, degraded eps=0.3 policy) "
          f"x {{best, NOOP(a=0), random-other}} x H={horizons}; frozen "
          f"greedy after the first action; {D2_NOISE_REPEATS} branch "
          f"replays for noise)")
    print("EXPECTATION:", EXPECT["d2"])
    print("=" * 78)
    ll_branching_check()
    net, c = ll_load(args.ll_ckpt)
    print(f"  [ckpt/LL] {args.ll_ckpt} (c_train={c})")
    t0 = time.time()
    states = collect_crash_states_ll(net, c, args.d2_states)
    if len(states) < args.d2_states:
        print(f"  [FLAG] only {len(states)}/{args.d2_states} pre-crash "
              f"states found — proceeding with these")
    per_state = []
    rng = np.random.default_rng(4243)
    for s_i, p in enumerate(states):
        lo, up = ll_intervals(net, p["state"])
        scores = lo + c * (up - lo)
        sweep = {}
        for a in range(4):
            g, done = ll_branch_return(p["seed"], p["prefix"], a, net, c,
                                       horizons)
            sweep[a] = g
        best_a = max((1, 2, 3), key=lambda a: sweep[a][hmax])
        others = [a for a in (1, 2, 3) if a != best_a]
        rand_a = int(others[int(rng.integers(len(others)))])
        noise = {}
        for name, a in (("best", best_a), ("noop", 0), ("rand", rand_a)):
            gs = {h: [sweep[a][h]] for h in horizons}
            for _ in range(D2_NOISE_REPEATS - 1):
                g, _ = ll_branch_return(p["seed"], p["prefix"], a, net,
                                        c, horizons)
                for h in horizons:
                    gs[h].append(g[h])
            noise[name] = {h: float(np.std(gs[h])) for h in horizons}
        rec = {
            "seed": p["seed"], "t": p["t"],
            "steps_to_crash": p["steps_to_crash"],
            "noop_G": sweep[0],
            "best": {"a": best_a, "G": sweep[best_a],
                     "net": {"score": float(scores[best_a]),
                             "width": float(up[best_a] - lo[best_a])}},
            "rand": {"a": rand_a, "G": sweep[rand_a],
                     "net": {"score": float(scores[rand_a]),
                             "width": float(up[rand_a] - lo[rand_a])}},
            "noop_net": {"score": float(scores[0]),
                         "width": float(up[0] - lo[0])},
            "noop_best_at_hmax": bool(sweep[0][hmax] >=
                                      sweep[best_a][hmax]),
            "gap_best": {h: sweep[best_a][h] - sweep[0][h]
                         for h in horizons},
            "gap_rand": {h: sweep[rand_a][h] - sweep[0][h]
                         for h in horizons},
            "noise": noise,
        }
        per_state.append(rec)
    return _d2_report("LL/per_c", horizons, per_state, 0,
                      time.time() - t0,
                      extra={"ckpt": args.ll_ckpt, "c_train": c})


# ---------------------------------------------------------------------------
# D3 — LL
# ---------------------------------------------------------------------------

def d3_ll(args):
    print("\n" + "=" * 78)
    print(f"D3/LL — BOOTSTRAP-CHAIN FIDELITY ({args.d3_episodes} fresh "
          f"greedy episodes; V(s_t)=max Hurwicz@c_train vs realized "
          f"native-reward G_t, gamma={LL_GAMMA})")
    print("EXPECTATION:", EXPECT["d3"])
    print("=" * 78)
    net, c = ll_load(args.ll_ckpt)
    print(f"  [ckpt/LL] {args.ll_ckpt} (c_train={c})")
    records = []
    ep_rows = []
    t0 = time.time()
    n_term = 0
    for e_i in range(args.d3_episodes):
        seed = 60000 + e_i
        env = ll_env()
        s, _ = env.reset(seed=seed)
        rs, Vs = [], []
        terminal = False
        for t in range(1000):
            lo, up = ll_intervals(net, s)
            V = float((lo + c * (up - lo)).max())
            a = int(np.argmax(lo + c * (up - lo)))
            s, r, term, trunc, _ = env.step(a)
            rs.append(r)
            Vs.append(V)
            if term or trunc:
                terminal = bool(term)
                break
        env.close()
        G = dc.returns_to_go(rs, gamma=LL_GAMMA)
        T = len(rs)
        n_term += int(terminal)
        for t in range(T):
            records.append(((T - 1) - t, Vs[t], G[t], terminal))
        ep_rows.append({"seed": seed, "steps": T, "terminal": terminal,
                        "return": float(sum(rs))})
        print(f"  ep {e_i:2d} seed={seed}: {T:4d} steps "
              f"{'terminal' if terminal else 'TRUNCATED'} "
              f"return={sum(rs):8.2f}")
    rec_term = [(s, v, g) for s, v, g, tm in records if tm]
    rec_all = [(s, v, g) for s, v, g, tm in records]
    rows_term = d3_bin_table(rec_term)
    rows_all = d3_bin_table(rec_all)
    print(f"\n  [LL/per_c] D3 SUMMARY: {len(ep_rows)} episodes "
          f"({n_term} terminal), {time.time() - t0:.0f}s wall")
    print_d3_table(rows_term, "TERMINAL episodes — unbiased G_t [PRIMARY]")
    if n_term < len(ep_rows):
        print_d3_table(rows_all, "ALL episodes (truncated G_t biased)")
    verdict, min_rho = d3_verdict_ll(rows_term)
    print(f"  H-chain (LL): min bin rho="
          f"{min_rho if min_rho is not None else float('nan'):.3f} "
          f"-> {verdict}")
    return {"side": "LL/per_c", "ckpt": args.ll_ckpt,
            "episodes": ep_rows, "n_terminal": n_term,
            "bins_terminal": rows_term, "bins_all": rows_all,
            "verdict": verdict, "min_bin_rho": min_rho,
            "wall_s": round(time.time() - t0)}


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Credit-pipeline diagnostics D1-D3 "
                    "(CREDIT_DIAGNOSTICS.md)")
    ap.add_argument("--d1", action="store_true")
    ap.add_argument("--d2", action="store_true")
    ap.add_argument("--d3", action="store_true")
    ap.add_argument("--knee", action="store_true",
                    help="knee rerun (12D fix B / D1): D2 on ATC with "
                         f"H={KNEE_HORIZONS}, {KNEE_STATES} mixed-seed "
                         "conflict states, knee/delay analysis; writes "
                         "knee_rerun_<ts>.json")
    ap.add_argument("--d2phi", action="store_true",
                    help="D5 D2-Phi rerun (pre-registered in the module "
                         "header): D2 on the SAME 30 knee states / seeds "
                         "/ horizons as the knee rerun, priced with the "
                         "CLI --vertical_ramp/--delta_conflict, two-tier "
                         "evaluation against the gate-pricing knee "
                         "baseline; writes d2phi_rerun_<ts>.json")
    ap.add_argument("--vertical_ramp", type=str, default="gate",
                    choices=["gate", "smooth"],
                    help="D5 conflict-pricing mode for every ATC "
                         "diagnostic (bcd.set_conflict_pricing pass-"
                         "through; default gate = 12c pricing)")
    ap.add_argument("--delta_conflict", type=float, default=None,
                    help="D5 DELTA_CONFLICT override (default: bcd's "
                         "0.2; JK approved 0.5 for the composite)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--side", choices=["atc", "ll", "both"],
                    default="both")
    ap.add_argument("--atc_ckpt", default=None,
                    help="default: 12c run tag from run12c_console.log, "
                         "_ep2000.pt")
    ap.add_argument("--ll_ckpt", default=LL_CKPT_DEFAULT)
    ap.add_argument("--d1_states", type=int, default=20)
    ap.add_argument("--d1_rollouts", type=int, default=D1_N_ROLLOUTS)
    ap.add_argument("--d1_frozen", type=str, default=None,
                    help="path to the pickled frozen shared D1 state set "
                         "(built from NOOP + DELIVERER trajectories on "
                         "first use; loaded verbatim after). Makes the "
                         "conflict-state NOOP-level half of the D7 gate "
                         "comparable across checkpoints.")
    ap.add_argument("--d2_states", type=int, default=15)
    ap.add_argument("--d3_episodes", type=int, default=20)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny budgets to validate plumbing")
    args = ap.parse_args()
    # D5 amendment B: set the pricing BEFORE anything prices a step, and
    # announce it so the battery cannot silently run the wrong mode.
    bcd.set_conflict_pricing(args.vertical_ramp, args.delta_conflict)
    print(f"[credit_diagnostics] {bcd.conflict_pricing_str()}")
    if args.all:
        args.d1 = args.d2 = args.d3 = True
    if not (args.d1 or args.d2 or args.d3 or args.knee or args.d2phi):
        ap.error("select at least one of --d1 --d2 --d3 --all --knee "
                 "--d2phi")
    args.d2_horizons_atc = ATC_HORIZONS
    args.d2_horizons_ll = LL_HORIZONS
    args.d2_seeds = None
    args.d2_per_seed_cap = None
    args.pre_sweep_hook = None
    args.d2_full_sweep = False
    if args.knee or args.d2phi:
        args.d2 = True
        args.side = "atc"
        args.d2_states = KNEE_STATES
        args.d2_horizons_atc = KNEE_HORIZONS
        args.d2_seeds = list(KNEE_SEEDS)
        args.d2_per_seed_cap = KNEE_PER_SEED_CAP
    d2phi_baseline, d2phi_store = None, []
    if args.d2phi:
        d2phi_baseline = load_knee_baseline()
        args.d2_full_sweep = True
        args.pre_sweep_hook = d2phi_prereg_hook(d2phi_baseline,
                                                d2phi_store)
    if args.smoke:
        args.d1_states = 3
        args.d1_rollouts = 6
        args.d2_states = 2
        args.d3_episodes = 2
        args.d2_horizons_atc = (12, 24, 96) if (args.knee or args.d2phi) \
            else (6, 12)
        args.d2_horizons_ll = (1, 3, 6)

    torch.manual_seed(0)
    do_atc = args.side in ("atc", "both")
    do_ll = args.side in ("ll", "both")
    results = {}
    t_start = time.time()

    def run(tag, fn):
        try:
            return fn(args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"{type(e).__name__}: {e}", "tag": tag}

    if args.d1:
        results["d1"] = {}
        if do_atc:
            results["d1"]["atc"] = run("d1_atc", d1_atc)
        if do_ll:
            results["d1"]["ll"] = run("d1_ll", d1_ll)
        results["d1"]["verdict"] = d1_verdict(
            results["d1"].get("atc"), results["d1"].get("ll"))
    if args.d2:
        results["d2"] = {}
        if do_atc:
            results["d2"]["atc"] = run("d2_atc", d2_atc)
        if do_ll:
            results["d2"]["ll"] = run("d2_ll", d2_ll)
        results["d2"]["verdict"] = d2_verdict(
            results["d2"].get("atc"), results["d2"].get("ll"))
        if args.knee and results["d2"].get("atc") and \
                "per_state" in results["d2"]["atc"]:
            results["knee"] = knee_report(results["d2"]["atc"])
        if args.d2phi and results["d2"].get("atc") and \
                "per_state" in results["d2"]["atc"]:
            results["d2phi"] = d2phi_report(results["d2"]["atc"],
                                            d2phi_store, d2phi_baseline)
    if args.d3:
        results["d3"] = {}
        if do_atc:
            results["d3"]["atc"] = run("d3_atc", d3_atc)
        if do_ll:
            results["d3"]["ll"] = run("d3_ll", d3_ll)

    print("\n" + "=" * 78)
    print("CREDIT-DIAGNOSTICS SUMMARY (misses are findings)")
    print("=" * 78)
    if "d1" in results:
        print("\n[D1 H-coverage + width-exposure]")
        for line in results["d1"]["verdict"]["lines"]:
            print("  " + line)
        for side in ("atc", "ll"):
            r = results["d1"].get(side)
            if r and "width_untaken_over_taken" in r:
                print(f"  {r['side']}: width untaken/taken="
                      f"{r['width_untaken_over_taken']:.3f} -> width "
                      f"{'TRACKS' if r['width_tracks_exposure'] else 'does NOT track'}"
                      f" action-level exposure")
    if "d2" in results:
        print("\n[D2 H-mismatch]")
        for line in results["d2"]["verdict"]["lines"]:
            print("  " + line)
    if "d3" in results:
        print("\n[D3 H-chain]")
        for side in ("atc", "ll"):
            r = results["d3"].get(side)
            if r and "verdict" in r:
                print(f"  {r['side']}: {r['verdict']}")

    os.makedirs(DIAG_DIR, exist_ok=True)
    stem = ("d2phi_rerun" if args.d2phi
            else ("knee_rerun" if args.knee else "credit_diag"))
    out_path = os.path.join(
        DIAG_DIR, f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    # ROUND3 D2: full flag echo in every summary JSON (the battery's
    # canonical builder; vertical_ramp/delta_conflict kept as top-level
    # fields for older readers).
    from micro_battery_common import bcd_config_echo
    payload = {
        "timestamp": time.strftime("%F %T"),
        "doc": "CREDIT_DIAGNOSTICS.md",
        "smoke": bool(args.smoke),
        "vertical_ramp": bcd.VERTICAL_RAMP,
        "delta_conflict": bcd.DELTA_CONFLICT,
        "config_echo": bcd_config_echo(args),
        "side": args.side,
        "expectations": EXPECT,
        "wall_seconds": round(time.time() - t_start),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(dc.to_jsonable(payload), f, indent=1)
    print(f"\nJSON written: {out_path}")


if __name__ == "__main__":
    main()
