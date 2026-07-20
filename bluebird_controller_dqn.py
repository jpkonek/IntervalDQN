"""
Controller-Frame Interval DQN on BluebirdATC (run 12+)
======================================================

Implements CONTROLLER_ARCHITECTURE.md: ONE agent with a global view plays
the air traffic controller, issuing AT MOST ONE clearance per 6 s sweep:
(aircraft, instruction) or a global NOOP. The pilot-frame implementation
(bluebird_interval_dqn.py) is untouched and remains the baseline; selected
pieces (CoverageTracker, detect_violation, haversine, adaptive-c rule,
interval-loss form, n-step window semantics) are reused here.

Environment role
----------------
The DECENTRALIZED env is kept exactly as bluebird_interval_dqn.make_env
builds it (relative encoder k=3, route_parallel) but ONLY as a token source
and action conduit:
  - per-aircraft obs vectors (54-dim relative encoding) become per-aircraft
    TOKENS, augmented with 6 sector-frame kinematics features (normalized
    lat/lon against the X-Plus sector bounding box read once from
    airspace.get_bounds(), sin/cos heading, fl/400, tas/600) -> 60-dim.
  - the env's per-aircraft reward dict is IGNORED entirely (env is built
    with reward_config = position_status_const @ coeff 0.0 — verified live:
    a single zero-coeff fn is accepted and pos_status info stays populated).
  - the reward the agent trains on is computed in OUR reward layer from
    simulator state (Section 4 of the spec; weights JK-approved).

Action encoding (canonical candidate ordering — documented contract)
--------------------------------------------------------------------
The candidate set at a state with aircraft callsigns CS (SORTED
lexicographically) is indexed:

    idx 0                  : global NOOP  (env action 0 for every aircraft)
    idx 1 + N_INSTR*i + j  : aircraft CS[i], instruction j, where
                       j=0 -> env action 1 (simple_heading_left  10 = L10)
                       j=1 -> env action 2 (simple_heading_right 10 = R10)
                       j=2 -> env action 3 (simple_heading_route_parallel)
                       j=3 -> env action 4 (simple_fl_climb   10 = +10 FL)
                       j=4 -> env action 5 (simple_fl_descent 10 = -10 FL)

VERTICAL ACTIONS (2026-07-13, run 13 prep): 12b showed lateral-only
control caps time-to-violation at ~600 s at ~25-aircraft density; LoS
requires BOTH <5 nm lateral AND <10 FL vertical, so altitude clearances
are the missing primary tool. simple_fl_climb/descent are RELATIVE
commands (bluebird_gymnasium actions/simple/climb_descent.py:
change_flight_level_to = selected_fl -/+ value, clipped to [FL0, FL500]);
value 10 = the package's DEFAULT_RELATIVE_CLIMB_DESCENT and its minimum
granularity (values must be multiples of DEFAULT_INTERVAL_FL = 10). One
step (+/-10 FL) is exactly the LoS vertical threshold: a single climb
takes a co-level pair out of the <10 FL violation band once flown.
Like L10/R10, repeats accumulate (+10 FL each). The action parser
assigns env ints in action_config insertion order, so the heading
actions KEEP ints 1..3 and old candidate indices keep their meaning.
Old 3-instruction checkpoints load and evaluate unchanged: n_instr is
persisted in the checkpoint and every encode/decode site uses the
agent's n_instr, never the module constant (--selftest covers this).

The chosen aircraft receives env action j+1; every other aircraft receives
NOOP (0). Replay stores the token array AND the sorted callsign tuple with
each transition, so targets recompute the candidate set without the env.
Aircraft still BEFORE_ENTRY appear in obs and are kept as tokens and as
candidates (a clearance to them is ignored by the env but still costs the
-0.1 instruction fee — the agent can learn not to waste calls on them).

Sector objective (spec Section 4 + OBJECTIVE v2, gamma = 0.97)
---------------------------------------------------------------
  Phi_sector = -0.05 * SUM_i alongtrack_dist_to_exit(i)          [nm]
               -0.02 * SUM_i centreline_offset(i)                 [nm]
               -0.20 * SUM_pairs f(pair),
                   f = max(0, (15 - d_nm)/15)^2 for IN_SECTOR pairs with
                   |delta FL| < 20
  r_t = gamma*Phi(s_{t+1}) - Phi(s_t)
        - 0.1   * (1 if a clearance was issued)
        + SUM over deliveries of 10 * max(0.3, nominal_T / actual_T)
        - 50 at the first violation (episode ends there)

  OBJECTIVE v2 (2026-07-09, default ON; --no-objective_v2 restores v1):
  the global fuel term (-0.005 per IN_SECTOR aircraft per step) is
  REMOVED — B1 measured it taxing survival itself (HOLDER beat NOOP on
  seed 20042 by dying at 144 s) — and the flat +10 delivery bonus decays
  with transit time (see the OBJECTIVE v2 note at DeliveryClock for the
  nominal_T source). Delay pressure now lives in the delivery bonus,
  not in a survival tax.

  Phi sums run over aircraft IN_SECTOR; the potential difference is taken
  over the MATCHED set (aircraft/pairs in-sector at both t and t+1), the
  same convention the pilot-frame progress shaping used: entries/exits do
  not create phantom potential kicks, and each aircraft's payments
  telescope exactly over its in-sector lifetime (see --selftest).

Learning machinery (ported, simplified to ONE stream per episode)
-----------------------------------------------------------------
n-step windows (n=12; see NSTEP note) with terminal collapse at violation and censored
bootstrap at the time limit; terminal-boost replay (3x); Double-DQN
targets (argmax over the s' candidate set at c_train, target net
evaluates, per-sample disc column); stratified-t with strata by SECTOR
risk = min pairwise lateral separation among vertically-proximate
(<20 FL) IN_SECTOR pairs at the decision step (<10 nm / 10-30 nm /
>30-or-none; risky-stratum t-cap 0.99, others 0.95); realized-coverage
tracker fed at episode end with the realized returns-to-go of every step
(violation-terminated episodes only; time-limit episodes are counted but
not fed).

Training-signal upgrades (2026-07-08, default ON)
-------------------------------------------------
--cbp: counterfactual-baselined potential. The battery (B1/B3/C2) showed
the potential term is ambient-dominated (policy-independent flow from
self-flying aircraft drowns the controller's marginal signal). Training
shaping becomes gamma*(Phi(s_action) - Phi(s_noop)) with the NOOP
branch stepped on a bit-faithful deepcopy; fuel/fees/delivery/violation
stay ABSOLUTE. When the issued action is global NOOP the term is exactly
0.0 (selftested). Eval-time ep_return stays on the absolute objective.
IMPORTANT (--cbp_lag, default 1): commands act with ONE SWEEP of latency,
so the spec-literal one-step counterfactual is provably identically zero
(measured; see the CBP_LAG note) — both branches are measured 1+lag
steps after s so the new command's first acting step is included.

--mask_reissue: battery A1 showed 74% of clearances re-issue the
aircraft's already-active instruction (conceptual no-op, fee waste).
Candidates repeating last_issued[callsign] are excluded from selection
(argmax + epsilon) and from the Double-DQN target argmax (mask bits are
stored in replay per transition, s' layout). Global NOOP never masked.

Usage:
    python bluebird_controller_dqn.py --selftest
    python bluebird_controller_dqn.py --smoke                    # cbp+mask on
    python bluebird_controller_dqn.py --train --episodes 1500 --duration 1200
    python bluebird_controller_dqn.py --train --no-cbp --no-mask_reissue ...
    python bluebird_controller_dqn.py --eval --ckpt checkpoints/bluebird_controller/... --c 0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
import json
import math
import sys
import os
import time
from collections import deque

import bluebird_interval_dqn as bid
from bluebird_interval_dqn import (CoverageTracker, detect_violation,
                                   haversine_nm, SEC_PER_STEP)

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAMMA = 0.97                 # spec Section 4
LR = 1e-3
BATCH_SIZE = 128
BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 100
GRAD_CLIP = 1.0
# NSTEP 6 -> 12 (2026-07-09): B3 measured the marginal 1-step CBP signal
# at ~1e-3 — event terms (delivery bonus, -50 terminal) must carry the
# credit, so the window must reach them more often. Interaction with the
# terminal-boost pools: terminal collapse pushes ALL pending windows into
# the boosted terminal buffer, so one violation now contributes up to 12
# (not 6) boosted items; the boost TARGET min(0.5, boost*share)
# self-limits, so the sampled terminal fraction rises but stays capped.
# Windower behaviour at n=12 is brute-force verified by --selftest.
NSTEP = 12
TERMINAL_BOOST = 3.0

# ROUND 3 core (20 July 2026; ROUND3_DESIGN.md rulings + ROUND3_REVIEW.md
# build-order steps 2-4 + ROUND3_REREVIEW.md binding amendments). All
# round-3 mechanisms are flag-gated and DEFAULT OFF; the flag-off paths
# are bitwise the pre-round-3 code (no-contamination selftests).
#
# B2 buffer/windower contract: every replay row carries a 9th field
# terminal_kind, stamped at the WINDOWER COLLAPSE SITE (the R_n reward
# sign is NOT a class proxy — shaping, bonuses and discounting break
# it; re-review amendment). armed_recovery/unarmed_recovery are enum
# values only this round: the machinery accepts them but nothing emits
# them until A2 lands.
TERMINAL_KINDS = ("live", "censored", "crash", "completion",
                  "armed_recovery", "unarmed_recovery")
# --class_replay per-class policy (pre-registered constants):
CRASH_AGE_CAP = 400        # evict crash rows older than this many
                           # episodes (mirrors CF_AGE_CAP)
CRASH_BATCH_FLOOR = 0.10   # violation-row batch floor (draft 10%,
                           # pre-registered in ROUND3_CORE_STATUS.md)

# RUN 13 FIX C — counterfactual replay (--cf_replay, default OFF).
# RUN13_FIX_C_DESIGN.md Section 1 with audit amendments A1-A5, A12-A14.
# At selected live states s, K untaken candidates a' (mandatory NOOP
# probe) are rolled out on env deepcopies under the FROZEN CURRENT
# POLICY (greedy, epsilon 0, dedicated CF RNG) for H_cf = nstep steps
# (A4 matched depth), priced through the SAME CBP code path as live
# episodes (priced_env_step), and stored in a boosted cf replay pool
# with y = G_cf + gamma^H * Q_target(s_H, a_cont(s_H)).
# NOTE (D5 ruling): run 13 launches with --nstep 36; the module default
# stays 12 — H_cf always follows the run's nstep, never this constant.
CF_STATE_P1 = 0.3      # stratum-1 (10-30 nm) state-selection probability
CF_STATE_FLOOR = 0.05  # state-level random floor across ALL strata (A5)
CF_CAND_FLOOR = 0.1    # candidate-level random floor within a state
CF_K_FLOOR = 3         # min candidates per selected state incl. NOOP
CF_AGE_CAP = 400       # evict cf items older than this many episodes
CF_BOOST = 3.0         # share-proportional cf-pool sampling boost (A14)
CF_MAX_FRAC = 0.25     # 1:3 cf:real — never-expected-to-bind upper bound
CF_BUDGET_RATE = 4.0   # step-equivalents banked per live training step
                       # (D2: cap rises, coverage-first; banked across
                       # episodes per A2.3 — tune via --cf_budget_rate)
CF_CLONE_COST = 1.5    # one deepcopy charged at 1.5 step-equivalents
                       # (A2.2: StepBudget extended to price clones)

# Sector-objective weights (spec Section 4, JK-approved)
ALPHA_PROGRESS = 0.05        # per nm along-track distance to exit
BETA_CENTRE = 0.02           # per nm centreline offset
DELTA_CONFLICT = 0.2         # pair-margin potential weight (see docstring;
                             # mutable via --delta_conflict /
                             # set_conflict_pricing — D5 escalation dial,
                             # JK-approved 0.5 for the smooth-ramp battery)
CONFLICT_RANGE_NM = 15.0     # f ramps up inside this lateral range
CONFLICT_FL = 20.0           # pairs closer than this vertically count
                             # (gate mode); reused as V_cur (smooth mode)

# D5 smooth vertical conflict potential (D5_VERTICAL_POTENTIAL_DESIGN.md,
# 2026-07-14; implemented 2026-07-18 with the adversarial-review
# amendments). VERTICAL_RAMP selects the vertical response of
# pair_conflict_f:
#   "gate"   (default): f = 1[|dFL| < 20] * f_lat  — bit-identical 12c
#            reproduction; the flag is the ablation path.
#   "smooth": f = f_lat * g_vert with
#            g_vert = 0.5*ramp^2((V_cur - |dFL_cur|)/V_cur)
#                   + 0.5*ramp^2((V_cmd - |dFL_cmd|)/V_cmd),
#            V_cur = CONFLICT_FL = 20 (current flight levels),
#            V_cmd = CONFLICT_FL_CMD = 10 (selected/commanded FLs).
# All constants ABSOLUTE (E1 attempt-1 lesson: no live statistic enters
# the potential). snapshot_stratum deliberately KEEPS the binary <20 gate
# (strata are a bucketing instrument; design section 4.6).
VERTICAL_RAMP = "gate"
CONFLICT_FL_CMD = 10.0       # V_cmd: one atomic +-10 FL clearance
                             # commands a pair exactly out of the LoS band
VERT_CMD_WEIGHT = 0.5        # w: blend of commanded vs current factor


def set_conflict_pricing(vertical_ramp=None, delta_conflict=None):
    """Set the module-level conflict-pricing knobs (D5). Every consumer
    (diagnose_controller mirrors, credit_diagnostics, micro gate) prices
    through THIS module's pair_conflict_f/shaping_terms, so setting the
    globals here is the single pass-through point. Returns the effective
    (vertical_ramp, delta_conflict)."""
    global VERTICAL_RAMP, DELTA_CONFLICT
    if vertical_ramp is not None:
        assert vertical_ramp in ("gate", "smooth"), vertical_ramp
        VERTICAL_RAMP = vertical_ramp
    if delta_conflict is not None:
        assert delta_conflict >= 0.0, delta_conflict
        DELTA_CONFLICT = float(delta_conflict)
    return VERTICAL_RAMP, DELTA_CONFLICT


def conflict_pricing_str():
    """One-line pricing banner used by the validation battery so no probe
    can silently run the wrong mode."""
    return f"pricing: {VERTICAL_RAMP}, delta={DELTA_CONFLICT}"
EPS_FUEL = 0.005             # per IN_SECTOR aircraft per step (v1 ONLY;
                             # REMOVED from objective v2 — survival tax)
CMD_COST = 0.1               # per issued clearance
DELIVERY_BONUS = 10.0        # per aircraft delivered (EXIT_REACHED);
                             # v2 scales it by max(0.3, nominal_T/actual_T)
DELIVERY_FLOOR = 0.3         # v2 bonus floor: late delivery >> none
VIOLATION_PENALTY = 50.0     # terminal, episode ends

# Sector-risk strata (min pairwise separation buckets)
N_STRATA = 3
STRAT_BOUNDS_NM = (10.0, 30.0)

# Token layout: base relative-encoder obs (dim discovered at runtime, 54
# with k=3) + 6 sector-frame kinematics features appended in this order.
KIN_FEATS = 6  # [norm_lat, norm_lon, sin(hdg), cos(hdg), fl/400, tas/600]

N_INSTR = 5    # L10, R10, route_parallel, climb +10 FL, descend -10 FL
               # (env actions 1..5; see the VERTICAL ACTIONS note above).
               # Checkpoints persist their own n_instr; old 3-instr nets
               # keep working (all encode/decode sites use agent.n_instr).
INSTR_NAMES = ("L10", "R10", "route_parallel", "climb10", "descend10")

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "checkpoints", "bluebird_controller")


# ===========================================================================
# ENVIRONMENT — token source and action conduit only
# ===========================================================================

def make_controller_env(scenario_duration=1200, k_nearest=3):
    """Decentralized InfiniteEnv exactly as the pilot frame builds it
    (relative encoder, k=3, route_parallel) but with the env reward
    neutered: a single zero-coeff fn (position_status_const @ 0.0), since
    the reward layer lives on our side. Verified live (2026-07-07 probe):
    the config is accepted, rewards come back 0.0, and info per-callsign
    pos_status stays populated (it is set by the base tracker, not by the
    reward fns)."""
    from bluebird_gymnasium.envs import InfiniteEnv
    from bluebird_gymnasium.envs.infinite import ScenarioName

    cfg = InfiniteEnv.get_default_env_config()
    cfg.state_repr_config = {"encoder_cls": "relative",
                             "k_nearest_aircraft": k_nearest}
    # dict INSERTION ORDER fixes the env action ints (ActionParser
    # iterates action_config.items()): heading actions stay 1..3, the
    # vertical actions append as 4 (climb +10 FL) and 5 (descent -10 FL)
    # — see the VERTICAL ACTIONS note in the module docstring.
    cfg.action_config = {"simple_heading_left": [10],
                         "simple_heading_right": [10],
                         "simple_heading_route_parallel": True,
                         "simple_fl_climb": [10],
                         "simple_fl_descent": [10]}
    cfg.reward_config = {"fns": ["position_status_const"], "coeffs": [0.0]}
    cfg.scenario_config["scenario_name"] = ScenarioName.sector_xplus
    cfg.view_config["type"] = "decentralized"
    cfg.view_config["decentralized_params"] = {}
    cfg.scenario_duration = scenario_duration
    return InfiniteEnv(config=cfg)


def sector_bounds(env):
    """(lat_min, lat_max, lon_min, lon_max) of the sector airspace, read
    once from airspace.get_bounds() (returns lower/upper [lon, lat, fl])
    and cached on the env instance. X-Plus probe values: lon [-4.650,
    -2.230], lat [50.245, 51.542]."""
    b = getattr(env, "_ctrl_sector_bounds", None)
    if b is None:
        lo, hi = env.get_simulator_env().airspace.get_bounds()
        b = (float(lo[1]), float(hi[1]), float(lo[0]), float(hi[0]))
        env._ctrl_sector_bounds = b
    return b


def build_tokens(env, obs_dict, info_dict, cs_list):
    """Per-aircraft token array [N, obs_dim + 6] for cs_list (caller fixes
    the order; generate_action always uses sorted callsigns).

    Kinematics come from info["simulator_environment"].aircraft (falling
    back to env.get_simulator_env(), which is the same object — the
    fallback covers callers without an info dict). Aircraft missing from
    the simulator (should not happen for obs keys) get zero kinematics.
    """
    if not cs_list:
        obs_dim = KIN_FEATS  # unused; shape (0, D) with a nominal width
        first = None
    else:
        first = obs_dict[cs_list[0]]
        obs_dim = int(np.asarray(first).shape[0])
    D = obs_dim + KIN_FEATS
    toks = np.zeros((len(cs_list), D), dtype=np.float32)
    if not cs_list:
        return toks
    lat0, lat1, lon0, lon1 = sector_bounds(env)
    sim = None
    if isinstance(info_dict, dict):
        sim = info_dict.get("simulator_environment")
    if sim is None:
        sim = env.get_simulator_env()
    for i, cs in enumerate(cs_list):
        toks[i, :obs_dim] = np.asarray(obs_dict[cs], dtype=np.float32)
        ac = sim.aircraft.get(cs)
        if ac is None or ac.lat is None:
            continue
        toks[i, obs_dim + 0] = (ac.lat - lat0) / (lat1 - lat0)
        toks[i, obs_dim + 1] = (ac.lon - lon0) / (lon1 - lon0)
        h = math.radians(ac.heading if ac.heading is not None else 0.0)
        toks[i, obs_dim + 2] = math.sin(h)
        toks[i, obs_dim + 3] = math.cos(h)
        toks[i, obs_dim + 4] = (ac.fl or 0.0) / 400.0
        toks[i, obs_dim + 5] = (ac.speed_tas or 0.0) / 600.0
    return toks


# ===========================================================================
# SECTOR SNAPSHOT + REWARD LAYER (spec Section 4)
# ===========================================================================

class AcSnap:
    """Per-aircraft slice of simulator state used by the reward layer.

    sel_fl (D5): the aircraft's SELECTED (commanded) flight level — the
    exact field the climb/descend actions mutate (bluebird_gymnasium
    actions/simple/climb_descent.py: change_flight_level_to =
    selected_fl +- value). Defaulted so the two positional construction
    sites in the selftests (and any legacy caller) keep working; when
    None the smooth pricing degrades to current-FL-only for that
    aircraft (risk R5 fallback — never crashes)."""
    __slots__ = ("cs", "lat", "lon", "fl", "dist_exit", "centre_off",
                 "sel_fl")

    def __init__(self, cs, lat, lon, fl, dist_exit, centre_off,
                 sel_fl=None):
        self.cs = cs
        self.lat = lat
        self.lon = lon
        self.fl = fl
        self.dist_exit = dist_exit      # along-track nm to exit (>= 0)
        self.centre_off = centre_off    # nm off the current-route centreline
        self.sel_fl = sel_fl            # selected/commanded FL (D5)


def sector_snapshot(env, callsigns):
    """{callsign: AcSnap} for the aircraft among `callsigns` that are
    IN_SECTOR right now, read from the env's tracked-aircraft data (same
    source detect_violation uses via info; tracked data works at reset
    too, before any step info exists). sel_fl (D5) is read from the
    simulator aircraft object (selected_fl — the field climb/descend
    mutate); fallback: current fl (degrades smooth pricing to
    current-only locally, never crashes — risk R5)."""
    snap = {}
    try:
        sim_aircraft = env.get_simulator_env().aircraft
    except Exception:
        sim_aircraft = {}
    for cs in callsigns:
        try:
            td = env.get_tracked_aircraft_data(cs)
        except Exception:
            continue
        if td is None or td.pos_status is None:
            continue
        if getattr(td.pos_status, "name", str(td.pos_status)) != "IN_SECTOR":
            continue
        pos = td.position
        if pos is None or td.flight_level is None:
            continue
        d_exit = td.track_dist_to_exit_cr
        c_info = td.centreline_info_cr or td.centreline_info_fr
        centre = float(c_info[0]) if c_info is not None else 0.0
        fl = float(td.flight_level)
        ac = sim_aircraft.get(cs)
        sel = getattr(ac, "selected_fl", None) if ac is not None else None
        sel_fl = float(sel) if sel is not None else fl
        snap[cs] = AcSnap(cs, float(pos.lat), float(pos.lon), fl,
                          float(d_exit) if d_exit is not None else None,
                          centre, sel_fl)
    return snap


def pair_conflict_f(a, b):
    """Pair conflict hazard (D5 flag-gated; see the VERTICAL_RAMP note).

    gate (default, bit-identical 12c):
        f = 1[|dFL| < 20] * max(0, (15 - d_nm)/15)^2
    smooth (D5):
        f = f_lat(d) * [ (1-w) * ramp^2((V_cur - |dFL_cur|)/V_cur)
                       +    w  * ramp^2((V_cmd - |dFL_cmd|)/V_cmd) ]
        with dFL_cur from current fl, dFL_cmd from selected fl (sel_fl;
        falls back to current fl when unavailable — risk R5), w = 0.5,
        V_cur = 20, V_cmd = 10. Product form: hazard is a conjunction
        (LoS needs BOTH d < 5 nm AND |dFL| < 10), so the potential must
        vanish when EITHER separation is large; zero-set:
        f = 0 iff d >= 15 OR (|dFL_cur| >= 20 AND |dFL_cmd| >= 10)."""
    if VERTICAL_RAMP == "gate":
        if abs(a.fl - b.fl) >= CONFLICT_FL:
            return 0.0
        d = haversine_nm(a.lat, a.lon, b.lat, b.lon)
        m = max(0.0, (CONFLICT_RANGE_NM - d) / CONFLICT_RANGE_NM)
        return m * m
    # smooth mode
    d_cur = abs(a.fl - b.fl)
    sa = a.sel_fl if a.sel_fl is not None else a.fl
    sb = b.sel_fl if b.sel_fl is not None else b.fl
    d_cmd = abs(sa - sb)
    if d_cur >= CONFLICT_FL and d_cmd >= CONFLICT_FL_CMD:
        return 0.0
    d = haversine_nm(a.lat, a.lon, b.lat, b.lon)
    m = max(0.0, (CONFLICT_RANGE_NM - d) / CONFLICT_RANGE_NM)
    if m == 0.0:
        return 0.0
    g_cur = max(0.0, (CONFLICT_FL - d_cur) / CONFLICT_FL) ** 2
    g_cmd = max(0.0, (CONFLICT_FL_CMD - d_cmd) / CONFLICT_FL_CMD) ** 2
    g_vert = (1.0 - VERT_CMD_WEIGHT) * g_cur + VERT_CMD_WEIGHT * g_cmd
    return m * m * g_vert


def shaping_terms(snap_t, snap_t1, gamma):
    """Per-term potential-difference payments between two snapshots.

    Returns (phi_progress, phi_centre, phi_conflict): each term is
    sum over MATCHED aircraft/pairs of  gamma*phi_i(t+1) - phi_i(t)  with
      phi_progress_i = -ALPHA * dist_exit_i
      phi_centre_i   = -BETA  * centre_off_i
      phi_conflict_p = -DELTA * f(pair p)
    Matched = present (IN_SECTOR) in BOTH snapshots (pairs: both aircraft),
    so entries/exits never create phantom potential kicks and each
    aircraft's payments telescope exactly over its in-sector lifetime.
    """
    prog = 0.0
    centre = 0.0
    matched = [cs for cs in snap_t if cs in snap_t1]
    for cs in matched:
        a0, a1 = snap_t[cs], snap_t1[cs]
        if a0.dist_exit is not None and a1.dist_exit is not None:
            prog += gamma * (-ALPHA_PROGRESS * a1.dist_exit) \
                    - (-ALPHA_PROGRESS * a0.dist_exit)
        centre += gamma * (-BETA_CENTRE * a1.centre_off) \
                  - (-BETA_CENTRE * a0.centre_off)
    conflict = 0.0
    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            ci, cj = matched[i], matched[j]
            f0 = pair_conflict_f(snap_t[ci], snap_t[cj])
            f1 = pair_conflict_f(snap_t1[ci], snap_t1[cj])
            conflict += gamma * (-DELTA_CONFLICT * f1) \
                        - (-DELTA_CONFLICT * f0)
    return prog, centre, conflict


# OBJECTIVE v2 (2026-07-09, default ON). The 2026-07-08 battery's B1 FAIL
# was structural: the global fuel charge (-EPS_FUEL x ~25 IN_SECTOR
# aircraft per step) taxed SURVIVAL itself — scripted HOLDER beat NOOP on
# seed 20042 simply by dying at 144 s, and a genuinely competent DELIVERER
# (TTV 462 -> 768 s plus a delivery on 10043) still scored below NOOP on
# undiscounted sums. v2:
#   (a) REMOVES the global fuel term entirely;
#   (b) replaces the flat +10 delivery bonus with a transit-time-decaying
#       bonus  10 * max(DELIVERY_FLOOR, nominal_T / actual_T),  where
#       actual_T  = the delivered aircraft's steps from its first
#                   IN_SECTOR snapshot to the step EXIT_REACHED is
#                   observed, and
#       nominal_T = its direct transit estimate captured AT ENTRY from
#                   the tracker's own route data:
#                   track_dist_to_exit_cr(entry)          [along-track nm
#                   to exit on the current == filed route at entry — the
#                   same tracker field Phi_progress uses]
#                   / (speed_tas(entry) * SEC_PER_STEP / 3600)  [nm/step
#                   at the aircraft's entry TAS from the simulator].
#       TAS ignores wind, so nominal_T is a no-wind straight-transit
#       estimate; tailwind can push the ratio slightly above 1 and the
#       bonus is deliberately NOT capped there. The 0.3 floor keeps any
#       delivery clearly better than none. If route distance or speed is
#       unavailable at entry it is captured at the first step it appears
#       (elapsed steps are added to the estimate); if it never appears
#       the factor falls back to 1.0 (flat +10, logged in the clock).
# CBP shaping (lag 1), the -0.1 instruction fee, the -50 any-violation
# terminal and the matched-set convention are UNCHANGED. The budget
# reconciliation identity keeps the same columns (the fuel column is
# identically 0.0 under v2), so it still holds exactly in both modes.
class DeliveryClock:
    """Per-episode tracker of aircraft entry steps and nominal transit
    estimates for the v2 delivery bonus (see the OBJECTIVE v2 note).
    Stored ON the env instance (delivery_clock / reset_delivery_clock) so
    env deepcopies — CBP branches, probe rollouts from mid-episode
    snapshots — carry the entry history along with the state.
    observe() must be called EXACTLY once per env step, BEFORE the step,
    with the pre-step sector snapshot; bonus_factor() is read after the
    step, when EXIT_REACHED is observed."""

    def __init__(self):
        self.step = 0       # steps observed so far
        self.entry = {}     # cs -> step index of first IN_SECTOR snapshot
        self.nominal = {}   # cs -> nominal_T (steps, >= 1)

    def observe(self, env, snap):
        sim = env.get_simulator_env()
        for cs, s in snap.items():
            if cs not in self.entry:
                self.entry[cs] = self.step
            if cs not in self.nominal and s.dist_exit is not None:
                ac = sim.aircraft.get(cs)
                kt = getattr(ac, "speed_tas", None) \
                    if ac is not None else None
                if kt:
                    nm_per_step = kt * SEC_PER_STEP / 3600.0
                    self.nominal[cs] = max(
                        1.0, (self.step - self.entry[cs])
                        + s.dist_exit / nm_per_step)
        self.step += 1

    def bonus_factor(self, cs):
        """max(DELIVERY_FLOOR, nominal_T / actual_T) for a delivery
        observed right after the current step's observe(); 1.0 (flat
        bonus) when no nominal estimate was ever available."""
        nom = self.nominal.get(cs)
        if nom is None or cs not in self.entry:
            return 1.0
        actual = max(1.0, float(self.step - self.entry[cs]))
        return max(DELIVERY_FLOOR, nom / actual)


def delivery_clock(env):
    ck = getattr(env, "_ctrl_delivery_clock", None)
    if ck is None:
        ck = DeliveryClock()
        env._ctrl_delivery_clock = ck
    return ck


def reset_delivery_clock(env):
    """Fresh clock; call right after env.reset() on any path that prices
    steps with the v2 objective (run_episode does this itself)."""
    env._ctrl_delivery_clock = DeliveryClock()
    return env._ctrl_delivery_clock


# CBP LATENCY FINDING (2026-07-08, measured; AMENDED 2026-07-18 for D5):
# BlueBird clearances act with EXACTLY one sweep of latency — issuing
# L10/R10/route_parallel leaves the simulator KINEMATIC state
# (lat/lon/heading/fl/tas) BIT-IDENTICAL to all-NOOP after one step and
# divergent only from the second step (verified directly with
# env_fingerprint on seed 10043; consistent with the old battery's B3,
# where every 1-step candidate delta equalled the -0.1 fee exactly, and
# with a 40-episode tiny train whose per-episode CBP term was 0.0
# bitwise). Because global NOOP does NOT cancel an aircraft's active
# command, the two branches of the lag-0 counterfactual share all
# previously-issued commands and their KINEMATIC states are therefore
# identical FOREVER.
#
# D5 AMENDMENT (2026-07-18): the old claim "the spec-literal lag-0 CBP
# term is provably identically zero" is NO LONGER TRUE in smooth mode.
# selected_fl updates SYNCHRONOUSLY with the issuing step (the default
# Pilot queues actions with process_time == receipt_time, so
# change_flight_level_to lands in selected_instructions.fl during the
# very step that issued it), and smooth-mode Phi reads selected_fl
# through AcSnap.sel_fl — so at lag 0 the commanded factor g_cmd already
# differs between the branches even though the kinematic state is
# bit-identical. In GATE mode (fl-only pricing) the lag-0 term remains
# provably zero as before. NOTE (amendment E): the synchronous-
# selected_fl property depends on the DEFAULT pilot (process_time ==
# receipt_time); a non-default pilot with a processing delay would
# reintroduce a sel-lag and the lag-0 smooth term would go back toward
# zero — re-verify if the pilot model ever changes.
#
# CBP_LAG extends BOTH branches by `lag` extra all-NOOP steps so the
# potential difference is measured at the first state the new command can
# have acted on (lag 1 = minimal latency-matched counterfactual; lag 0 =
# the spec-literal construct — provably zero for the kinematic terms,
# nonzero only through the D5 commanded factor). The NOOP-zero invariant
# is unaffected in BOTH modes: for a global-NOOP action both branches are
# the same determinized computation at ANY lag (asserted by --selftest).
CBP_LAG = 1


def cbp_noop_snapshot(env, obs_dict, lag=CBP_LAG):
    """Counterfactual NOOP branch for CBP (counterfactual-baselined
    potential): deepcopy the env AT state s, step the CLONE with all-NOOP
    (1 + lag) times, and return the clone's sector snapshot. The live env
    is untouched (asserted by --selftest via env_fingerprint).
    Determinized clones are bit-faithful (verified 2026-07-07), so when
    the issued action IS global NOOP the two branches are identical and
    the CBP shaping term is EXACTLY 0.0."""
    env_cf = copy.deepcopy(env)
    o = env_cf.step({cs: 0 for cs in obs_dict})[0]
    for _ in range(lag):
        o = env_cf.step({cs: 0 for cs in o})[0]
    return sector_snapshot(env_cf, o.keys())


def env_fingerprint(env):
    """Bitwise fingerprint of the live simulator state (per-aircraft
    kinematics + selected_fl), used by --selftest to prove the CBP
    counterfactual step is side-effect-free on the live env. selected_fl
    added 2026-07-18 (D5 amendment C): smooth-mode pricing reads it, so
    clone side-effect-freedom must cover the newly priced state too."""
    sim = env.get_simulator_env()
    fp = []
    for cs in sorted(sim.aircraft.keys()):
        ac = sim.aircraft[cs]
        fp.append((cs, ac.lat, ac.lon, ac.heading, ac.fl, ac.speed_tas,
                   ac.selected_fl))
    return tuple(fp)


def build_reissue_mask(cs_list, last_issued, n_instr=N_INSTR):
    """Candidate-layout bool mask (True = MASKED) for re-issue suppression:
    candidate (i, j) is masked when j == last_issued[cs_i], i.e. the
    clearance would repeat the aircraft's still-active instruction (a
    conceptual no-op that only pays the fee). last_issued[cs] is
    overwritten whenever ANY instruction is issued to cs, so a different
    intervening instruction unmasks the old one. Global NOOP (index 0) is
    NEVER masked. Layout identical to the candidate matrix:
    len == 1 + n_instr * len(cs_list)."""
    m = np.zeros(1 + n_instr * len(cs_list), dtype=bool)
    if last_issued:
        for i, cs in enumerate(cs_list):
            j = last_issued.get(cs)
            if j is not None:
                m[1 + n_instr * i + j] = True
    return m


def snapshot_stratum(snap):
    """Sector-risk stratum of a decision step: min pairwise lateral
    separation among vertically-proximate (<20 FL) IN_SECTOR pairs.
    0: <10 nm, 1: 10-30 nm, 2: >30 nm or no such pair."""
    acs = list(snap.values())
    nearest = None
    for i in range(len(acs)):
        for j in range(i + 1, len(acs)):
            if abs(acs[i].fl - acs[j].fl) >= CONFLICT_FL:
                continue
            d = haversine_nm(acs[i].lat, acs[i].lon, acs[j].lat, acs[j].lon)
            if nearest is None or d < nearest:
                nearest = d
    if nearest is None or nearest > STRAT_BOUNDS_NM[1]:
        return 2
    if nearest < STRAT_BOUNDS_NM[0]:
        return 0
    return 1


# ===========================================================================
# REPLAY — opaque variable-N states: (tokens [N, D], callsign tuple)
# ===========================================================================

def empty_state(dim):
    return (np.zeros((0, dim), dtype=np.float32), ())


class ControllerReplayBuffer:
    """Terminal-boost split-pool replay (ported from the pilot frame) over
    opaque controller states. Items (9 fields; 8 fields 2026-07-14 fix D,
    12D_CREDIT_DESIGN.md Section 2.4; terminal_kind appended 2026-07-20,
    ROUND 3 B2 schema — re-review 'windower/buffer contract'):
    (state, a_idx, R_n, next_state, ns_mask, disc, stratum, next_a_idx,
    terminal_kind)
    where ns_mask is the re-issue candidate mask AT next_state (same
    layout as the s' candidate matrix, True = masked), applied to the
    Double-DQN argmax on the target side, and next_a_idx is the candidate
    index of the action ACTUALLY TAKEN at next_state (0 = NOOP fallback
    for censored tails; unused at terminals where disc == 0). Consumed by
    --bootstrap_support taken_noop; carried but ignored under 'full'.
    terminal_kind (TERMINAL_KINDS) is stamped at the windower collapse
    site (R_n sign is NOT a proxy); production rows always arrive tagged
    — the disc-based default below exists ONLY for legacy synthetic
    callers (selftests) and is never the tag source in training.

    --class_replay (ROUND 3 B2, default OFF; flag-off routing/sampling
    is bitwise legacy): per-class boost/eviction policy —
      - crash rows keep the current terminal boost (term_buf) and are
        age-capped at crash_age_cap episodes (mirrors the cf age cap;
        gen_episode stamps ride in the aligned term_gen deque);
      - completion/win rows (and the future armed_recovery/
        unarmed_recovery rows) go to the REGULAR pool, never term_buf —
        wins must not capture the 3x boost built for sparse -50 anchors
        (~16x violation dilution otherwise, ROUND3_REVIEW B2);
      - sample() enforces a violation-row batch floor (crash_floor,
        draft 10%) and records per-class batch composition in
        last_kind_counts for the JSONL telemetry."""

    N_FIELDS = 9

    def __init__(self, capacity=BUFFER_SIZE, terminal_boost=TERMINAL_BOOST,
                 cf_boost=CF_BOOST, cf_max_frac=CF_MAX_FRAC,
                 class_replay=False, crash_age_cap=CRASH_AGE_CAP,
                 crash_floor=CRASH_BATCH_FLOOR):
        self.terminal_boost = float(terminal_boost)
        # ROUND 3 B2 per-class policy (default OFF = bitwise legacy)
        self.class_replay = bool(class_replay)
        self.crash_age_cap = int(crash_age_cap)
        self.crash_floor = float(crash_floor)
        self.gen_episode = 0            # advanced once per ENV episode
        self.last_kind_counts = None    # per-class batch composition
        if self.terminal_boost > 1.0:
            # cap the floor at capacity//2 so tiny buffers keep a regular pool
            term_cap = max(min(1000, capacity // 2), capacity // 5)
            self.term_buf = deque(maxlen=term_cap)
            # gen_episode stamps aligned with term_buf (same maxlen, so
            # a silent left-drop on append keeps them in lockstep) —
            # the crash-row age cap pops both from the left.
            self.term_gen = deque(maxlen=term_cap)
            self.reg_buf = deque(maxlen=capacity - term_cap)
            self.buffer = None
        else:
            self.buffer = deque(maxlen=capacity)
        # RUN 13 FIX C: cf pool — a SEPARATE additive deque, never carved
        # out of the regular pool (the term/reg carve has the documented
        # tiny-capacity footgun, and cf items must not evict live
        # experience). Entries are [8-tuple, gen_episode, replay_count]
        # lists so the A14 per-item replay-count log increments in place.
        # Empty (and RNG-silent in sample()) unless --cf_replay pushes.
        self.cf_boost = float(cf_boost)
        self.cf_max_frac = float(cf_max_frac)
        self.cf_buf = deque(maxlen=min(20000, max(1000, capacity // 5)))
        self.cf_sample_enabled = True   # severed by the A12.3 selftest
        self.last_cf_mask = None        # per-sample cf row flags (or None)
        self.last_cf_ages = []          # gen_episode of sampled cf rows

    def push(self, state, a_idx, reward, next_state, ns_mask, disc, stratum,
             next_a_idx, terminal_kind=None):
        if terminal_kind is None:
            # LEGACY-CALLER FALLBACK ONLY (synthetic selftest items /
            # old harnesses): production tags always come from the
            # windower collapse site — disc conflates censored with
            # live and completion with crash, so this default is never
            # the tag source on the training path.
            terminal_kind = "crash" if disc == 0.0 else "live"
        assert terminal_kind in TERMINAL_KINDS, terminal_kind
        item = (state, a_idx, reward, next_state, ns_mask, disc, stratum,
                next_a_idx, terminal_kind)
        assert len(item) == self.N_FIELDS  # loud failure at missed sites
        if self.buffer is not None:
            self.buffer.append(item)
        elif (terminal_kind == "crash" if self.class_replay
              else disc == 0.0):
            # class-aware routing (--class_replay): ONLY crash rows keep
            # the boosted pool; completion/win rows go regular. Flag-off:
            # the legacy disc==0.0 route, bitwise.
            self.term_buf.append(item)
            self.term_gen.append(self.gen_episode)
        else:
            self.reg_buf.append(item)

    def push_cf(self, state, a_idx, g_cf, next_state, ns_mask, disc,
                stratum, next_a_idx, gen_episode, terminal_kind=None):
        """CF-replay item (--cf_replay): same 9-field layout as live
        items plus the generation-episode stamp (A1 staleness) and an
        in-place replay counter (A14 observability). terminal_kind for
        cf rows comes from the branch collapse in cf_branch_rollout
        ('crash' when the branch violated, else 'live')."""
        if terminal_kind is None:
            terminal_kind = "crash" if disc == 0.0 else "live"
        assert terminal_kind in TERMINAL_KINDS, terminal_kind
        item = (state, a_idx, g_cf, next_state, ns_mask, disc, stratum,
                next_a_idx, terminal_kind)
        assert len(item) == self.N_FIELDS
        self.cf_buf.append([item, int(gen_episode), 0])

    def evict_crash_older_than(self, min_gen_episode):
        """ROUND 3 B2 crash-row age cap (mirrors evict_cf_older_than;
        --class_replay path only — under legacy routing term_buf also
        holds completion rows and is never age-evicted). term_buf is
        appended in episode order, so eviction pops from the left."""
        if self.buffer is not None:      # single-pool: no boosted pool
            return 0
        n = 0
        while self.term_buf and self.term_gen \
                and self.term_gen[0] < min_gen_episode:
            self.term_buf.popleft()
            self.term_gen.popleft()
            n += 1
        return n

    def evict_cf_older_than(self, min_gen_episode):
        """Drop cf items with gen_episode < min_gen_episode (the pool is
        appended in episode order, so eviction pops from the left)."""
        n = 0
        while self.cf_buf and self.cf_buf[0][1] < min_gen_episode:
            self.cf_buf.popleft()
            n += 1
        return n

    def cf_replay_counts(self):
        return [e[2] for e in self.cf_buf]

    def sample(self, batch_size):
        self.last_cf_mask = None
        self.last_cf_ages = []
        self.last_kind_counts = None
        if self.buffer is not None:
            batch = random.sample(self.buffer, batch_size)
            assert all(len(it) == self.N_FIELDS for it in batch)
            return batch
        # A14 cf sampler: share-proportional boosted cf pool (the
        # terminal-boost pattern); the 1:3 cf:real cap is a never-
        # expected-to-bind upper bound. With the cf pool empty (flag
        # OFF) or severed, this path is bitwise the legacy two-pool
        # sampler — no extra RNG draws (no-contamination selftest).
        n_cf = len(self.cf_buf) if self.cf_sample_enabled else 0
        n_term, n_reg = len(self.term_buf), len(self.reg_buf)
        k_c = 0
        if n_cf:
            share_cf = n_cf / max(1, n_cf + n_term + n_reg)
            k_c = min(n_cf, int(round(batch_size *
                                      min(self.cf_max_frac,
                                          self.cf_boost * share_cf))))
        rest = batch_size - k_c
        share = n_term / max(1, n_term + n_reg)
        target = min(0.5, self.terminal_boost * share)
        k_t = min(n_term, int(round(rest * target)))
        if self.class_replay:
            # ROUND 3 B2 violation-row batch floor (pre-registered
            # draft 10%): under --class_replay term_buf holds ONLY
            # crash rows, so flooring k_t floors the violation share.
            k_t = max(k_t, min(n_term, int(round(rest * self.crash_floor))))
        k_r = rest - k_t
        if k_r > n_reg:
            k_r = n_reg
            k_t = min(n_term, rest - k_r)
        if k_t + k_r < rest and n_cf:      # tiny-live-pool top-up
            k_c = min(n_cf, k_c + (rest - k_t - k_r))
        batch = (random.sample(self.term_buf, k_t)
                 + random.sample(self.reg_buf, k_r))
        if k_c:
            picks = random.sample(range(n_cf), k_c)
            for p in picks:
                entry = self.cf_buf[p]
                entry[2] += 1              # A14 per-item replay count
                batch.append(entry[0])
                self.last_cf_ages.append(entry[1])
            self.last_cf_mask = [False] * (k_t + k_r) + [True] * k_c
        if self.class_replay:
            # ROUND 3 B2 per-class batch-composition telemetry (JSONL);
            # flag-off leaves last_kind_counts None (bitwise legacy).
            kc = {}
            for it in batch:
                kc[it[8]] = kc.get(it[8], 0) + 1
            self.last_kind_counts = kc
        assert all(len(it) == self.N_FIELDS for it in batch)
        return batch

    def __len__(self):
        if self.buffer is not None:
            return len(self.buffer)
        return len(self.term_buf) + len(self.reg_buf)


class ControllerWindower:
    """n-step window builder over the SINGLE controller trajectory
    (same semantics as the pilot NStepWindower, states opaque, stratum
    passed in explicitly — it comes from sim state, not from the obs):

      - full window: R_n = sum_{k<n} gamma^k r, disc = gamma^n;
      - terminal (violation) inside the window: exact realized tail,
        next = empty state, disc = 0 (the bootstrap VANISHES);
      - censored end (time limit): bootstrap at the last available next
        state with disc = gamma^m.
    ns_mask (the re-issue mask AT the bootstrap next-state) rides along
    with ns so the target side can mask the Double-DQN argmax; terminal
    windows carry the empty-state mask (NOOP only, nothing masked).

    NEXT-ACTION THREADING (2026-07-14, fix D 'trained-support bootstrap'):
    a completed bootstrap window's ns is the state s_{i+n}; the action
    taken AT s_{i+n} is only known at the NEXT add() call. So a completed
    window is HELD for exactly one add() and stamped UNCONDITIONALLY with
    that add's a_idx (next_a_idx); the held ns and the stamping add's
    state must carry the same callsign tuple (asserted — per the design
    amendments we do NOT use a fragile ns-array-equality trigger, since
    tokens are rebuilt at t+1). Terminal collapse stamps the held window
    first (the terminal add's own a_idx IS the action taken at its ns),
    then pushes the collapsed items with next_a_idx = 0 (unused, disc 0).
    flush_censored PUSHES the held window (design amendment: it must not
    be silently dropped) and all pending windows with the NOOP fallback
    next_a_idx = 0 — no action was ever taken at the censored bootstrap
    state; noop_fallbacks counts those items for the run-time monitor.

    TERMINAL-KIND THREADING (2026-07-20, ROUND 3 B2 — the windower
    collapse site is the tag ORIGIN; re-review 'windower/buffer
    contract'): completed windows push kind 'live'; terminal collapse
    pushes the caller-supplied terminal_kind ('crash' default; A1
    passes 'completion'; A2 will pass armed_recovery/unarmed_recovery);
    flush_censored pushes kind 'censored' — including the held window
    it stamps with the NOOP fallback, keeping B3's censored identifier
    (kind == 'censored') exact where next_a_idx == 0 alone is not.
    """

    def __init__(self, buffer, gamma, n):
        self.buffer = buffer
        self.gamma = gamma
        self.n = n
        self.pending = []   # [(state, a_idx, r, stratum)]
        self.last_ns = None
        self.last_ns_mask = None
        self.held = None    # completed window awaiting its next_a_idx
        self.noop_fallbacks = 0   # censored items pushed with next_a = 0

    def _window_return(self, i):
        return sum(self.gamma ** k * r
                   for k, (_s, _a, r, _st) in enumerate(self.pending[i:]))

    def _stamp_held(self, next_a_idx, s_check=None, kind="live"):
        """Push the held window with its now-known next action. kind
        (ROUND 3 B2): 'live' for the normal stamping add(); the censored
        flush passes 'censored' so B3's identifier stays exact."""
        if self.held is None:
            return
        s_0, a_0, R, ns, ns_mask, disc, st_0 = self.held
        if s_check is not None:
            assert ns[1] == s_check[1], (
                "held window's bootstrap next-state callsigns "
                f"{ns[1]} != stamping add()'s state callsigns {s_check[1]}")
        self.buffer.push(s_0, a_0, R, ns, ns_mask, disc, st_0, next_a_idx,
                         kind)
        self.held = None

    def add(self, s, a_idx, r, ns, ns_mask, terminal, stratum,
            terminal_kind=None):
        # a_idx is the action taken AT s == the held window's ns
        self._stamp_held(a_idx, s_check=s)
        self.pending.append((s, a_idx, r, stratum))
        self.last_ns = ns
        self.last_ns_mask = ns_mask
        if terminal:
            # ROUND 3 B2: the collapse site stamps the class. Default
            # 'crash' keeps every legacy caller (violation collapse)
            # exact; A1 passes 'completion'; A2 will pass the recovery
            # kinds (accepted by the machinery, nothing emits them yet).
            kind = terminal_kind if terminal_kind is not None else "crash"
            assert kind in ("crash", "completion", "armed_recovery",
                            "unarmed_recovery"), kind
            dim = s[0].shape[1]
            zeros = empty_state(dim)
            zeros_mask = np.zeros(1, dtype=bool)   # NOOP-only candidate set
            for i in range(len(self.pending)):
                s_i, a_i, _, st_i = self.pending[i]
                self.buffer.push(s_i, a_i, self._window_return(i),
                                 zeros, zeros_mask, 0.0, st_i, 0, kind)
            self.pending.clear()
        elif len(self.pending) == self.n:
            s_0, a_0, _, st_0 = self.pending[0]
            self.held = (s_0, a_0, self._window_return(0), ns, ns_mask,
                         self.gamma ** self.n, st_0)
            self.pending.pop(0)

    def flush_censored(self):
        if self.held is not None:
            self.noop_fallbacks += 1
            self._stamp_held(0, kind="censored")    # NOOP fallback
        for i in range(len(self.pending)):
            s_i, a_i, _, st_i = self.pending[i]
            m = len(self.pending) - i
            self.buffer.push(s_i, a_i, self._window_return(i),
                             self.last_ns, self.last_ns_mask,
                             self.gamma ** m, st_i, 0,   # NOOP fallback
                             "censored")
            self.noop_fallbacks += 1
        self.pending.clear()


# ===========================================================================
# NETWORK — token MLP -> 2x self-attention -> pointer-style interval heads
# ===========================================================================

class ControllerQNet(nn.Module):
    """Permutation-equivariant sector encoder with interval Q heads.

    Token MLP (D_tok -> 64, 2 layers) -> 2 blocks of 2-head self-attention
    (d=64, key-padding masks) with residual+LayerNorm and a small FFN ->
    per-aircraft contextual embeddings + masked-mean-pooled sector summary.
    Heads: per-aircraft instruction head 64 -> (lower, delta_raw) x 3;
    global NOOP head from the summary -> (lower, delta_raw).
    Interval = [l, l + softplus(delta_raw) + 1e-6] (pilot-frame convention).
    ~59k parameters (target < 100k).
    """

    def __init__(self, token_dim, d=64, n_heads=2, n_layers=2,
                 n_instr=N_INSTR, width_scalars=False):
        super().__init__()
        self.token_dim = token_dim
        self.n_instr = n_instr
        # width_scalars: UNNORMALIZED sector scalars (aircraft count and
        # count^2, /10-scaled) added to delta_raw pre-softplus via tiny
        # zero-init linear layers. The LN trunk pins activation magnitude,
        # so beyond-training-range density can only reach the width heads
        # through a deliberate channel like this one (JK directive, 8 July;
        # WIDTH_MECHANISM_PROBES.md). Zero-init => exactly inert at init;
        # in-range slope is learned, beyond-range behavior extrapolates it.
        self.width_scalars = width_scalars
        if width_scalars:
            self.wscalar_instr = nn.Linear(2, n_instr)
            self.wscalar_noop = nn.Linear(2, 1)
            for lin in (self.wscalar_instr, self.wscalar_noop):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
        )
        self.attn = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.ln2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn.append(nn.MultiheadAttention(d, n_heads,
                                                   batch_first=True))
            self.ln1.append(nn.LayerNorm(d))
            self.ffn.append(nn.Sequential(nn.Linear(d, d), nn.ReLU()))
            self.ln2.append(nn.LayerNorm(d))
        self.instr_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_instr * 2))
        self.noop_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))
        self.d = d

    def forward(self, tokens, mask=None):
        """tokens [B, N, D_tok] float32; mask [B, N] bool, True = real
        aircraft (None -> all real). Returns (ac_l, ac_u, noop_l, noop_u):
        ac_* [B, N, n_instr] (rows at padded positions are meaningless and
        must be masked by the caller), noop_* [B]."""
        B, N, _ = tokens.shape
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=tokens.device)
        if N == 0:
            summary = tokens.new_zeros(B, self.d)
            ac_l = tokens.new_zeros(B, 0, self.n_instr)
            ac_u = tokens.new_zeros(B, 0, self.n_instr)
        else:
            h = self.token_mlp(tokens)
            kpm = ~mask
            # rows with zero real tokens (terminal next-states) would make
            # every key masked -> NaN; unmask their (zero) pad token 0 for
            # the attention pass only. Their outputs are never read: the
            # candidate mask is empty and the pooled summary uses `mask`.
            empty_rows = ~mask.any(dim=1)
            if bool(empty_rows.any()):
                kpm = kpm.clone()
                kpm[empty_rows, 0] = False
            for attn, ln1, ffn, ln2 in zip(self.attn, self.ln1,
                                           self.ffn, self.ln2):
                a, _ = attn(h, h, h, key_padding_mask=kpm,
                            need_weights=False)
                h = ln1(h + a)
                h = ln2(h + ffn(h))
            out = self.instr_head(h).view(B, N, self.n_instr, 2)
            ac_l = out[..., 0]
            ac_raw = out[..., 1]
            if self.width_scalars:
                ac_raw = ac_raw + self.wscalar_instr(
                    self._count_feats(mask)).unsqueeze(1)
            ac_u = ac_l + F.softplus(ac_raw) + 1e-6
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            summary = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        no = self.noop_head(summary)
        noop_l = no[:, 0]
        noop_raw = no[:, 1]
        if self.width_scalars:
            noop_raw = noop_raw + self.wscalar_noop(
                self._count_feats(mask))[:, 0]
        noop_u = noop_l + F.softplus(noop_raw) + 1e-6
        return ac_l, ac_u, noop_l, noop_u

    @staticmethod
    def _count_feats(mask):
        n = mask.sum(dim=1).float() / 10.0
        return torch.stack([n, n * n], dim=-1)


def candidate_intervals(ac_l, ac_u, noop_l, noop_u, mask):
    """Flatten per-aircraft interval heads into the canonical candidate
    ordering. Returns (cand_l, cand_u, valid) each [B, 1 + n_instr*N]:
    column 0 = NOOP, column 1 + 3*i + j = (aircraft i, instruction j)."""
    B, N, n_instr = ac_l.shape
    cand_l = torch.cat([noop_l.unsqueeze(1), ac_l.reshape(B, N * n_instr)],
                       dim=1)
    cand_u = torch.cat([noop_u.unsqueeze(1), ac_u.reshape(B, N * n_instr)],
                       dim=1)
    valid = torch.cat(
        [torch.ones(B, 1, dtype=torch.bool, device=ac_l.device),
         mask.repeat_interleave(n_instr, dim=1)], dim=1)
    return cand_l, cand_u, valid


def pad_state_batch(states, device):
    """Pad a list of (tokens [N_i, D], callsigns) to [B, Nmax, D] + mask."""
    B = len(states)
    D = states[0][0].shape[1]
    Nmax = max(s[0].shape[0] for s in states)
    tok = np.zeros((B, Nmax, D), dtype=np.float32)
    mask = np.zeros((B, Nmax), dtype=bool)
    for b, (t, _cs) in enumerate(states):
        n = t.shape[0]
        if n:
            tok[b, :n] = t
            mask[b, :n] = True
    return (torch.from_numpy(tok).to(device),
            torch.from_numpy(mask).to(device))


def build_e1_negatives(tok, mask):
    """E1-ATC negative builder (WIDTH_MECHANISM_PROBES.md, ATC transfer
    spec). Off-manifold negative states from the current training batch,
    TWO generators applied 50/50 within each batch:

      (a) AIRCRAFT-SWAP (first half): each negative's token set is
          assembled from individual real aircraft rows sampled (without
          replacement) from DIFFERENT states of the batch — plausible
          aircraft, impossible joint traffic picture (the 12b condition
          that shows NO width elevation; the operational target). Counts
          are sampled from the batch's own real per-state counts; when a
          negative has >=2 rows and the batch spans >=2 source states,
          >=2 distinct source states are enforced by construction.
      (b) FIELD-SHUFFLE (second half): per-(token,feature) recombination —
          an independent permutation of each feature column across the
          pooled real rows of the second half (the certain-basin garbage
          condition). Per-feature marginals are exactly preserved; counts
          are the originals.

    Padding rows are never sampled (the pool is mask-selected). Returns
    (neg_tok [Bn, Nn, D], neg_mask [Bn, Nn] bool, info) or None when the
    batch has no real aircraft rows. info carries provenance for
    --selftest: {"n_swap", "swap_src" (per-negative source-state index
    lists), "shuf_counts"}. Consumes torch RNG only."""
    B, Nmax, D = tok.shape
    counts = mask.sum(dim=1)
    nz = torch.nonzero(counts > 0).flatten()
    if len(nz) == 0:
        return None
    # global real-row pool with source-state provenance
    src_idx, row_idx = torch.nonzero(mask, as_tuple=True)
    pool = tok[src_idx, row_idx]                     # [P, D]
    P = pool.shape[0]
    n_states = len(torch.unique(src_idx))
    real_counts = counts[nz]

    negs, swap_src = [], []
    n_swap = B // 2
    for _ in range(n_swap):                          # (a) aircraft-swap
        n_i = int(real_counts[torch.randint(len(nz), (1,))].item())
        n_i = min(n_i, P)
        sel = torch.randperm(P, device=tok.device)[:n_i]
        srcs = src_idx[sel]
        if n_i >= 2 and n_states >= 2 and len(torch.unique(srcs)) < 2:
            other = torch.nonzero(src_idx != srcs[0]).flatten()
            sel = sel.clone()
            sel[-1] = other[torch.randint(len(other), (1,))]
            srcs = src_idx[sel]
        negs.append(pool[sel])
        swap_src.append(srcs.tolist())

    shuf_states = [b for b in range(n_swap, B) if counts[b] > 0]
    shuf_counts = [int(counts[b].item()) for b in shuf_states]
    if shuf_states:                                  # (b) field-shuffle
        pool2 = torch.cat([tok[b][mask[b]] for b in shuf_states], dim=0)
        P2 = pool2.shape[0]
        shuf = torch.stack(
            [pool2[torch.randperm(P2, device=tok.device), d]
             for d in range(D)], dim=1)
        off = 0
        for n_i in shuf_counts:
            negs.append(shuf[off:off + n_i])
            off += n_i

    if not negs:
        return None
    Nn = max(t.shape[0] for t in negs)
    neg_tok = tok.new_zeros(len(negs), Nn, D)
    neg_mask = torch.zeros(len(negs), Nn, dtype=torch.bool,
                           device=tok.device)
    for j, t in enumerate(negs):
        neg_tok[j, :t.shape[0]] = t
        neg_mask[j, :t.shape[0]] = True
    return neg_tok, neg_mask, {"n_swap": n_swap, "swap_src": swap_src,
                               "shuf_counts": shuf_counts}


# ===========================================================================
# CONTROLLER AGENT
# ===========================================================================

class ControllerAgent:
    N_TARGET_SAMPLES = 5

    def __init__(self, token_dim, n_instr=N_INSTR, lr=LR, gamma=GAMMA,
                 c_train=0.5, target_coverage=0.85, width_reg=0.01,
                 warmup_steps=1000, warmup_epsilon=0.5,
                 buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 device="cpu", t_cap_risky=0.99,
                 terminal_boost=TERMINAL_BOOST, mask_reissue=True,
                 width_scalars=False, e1_width=False, e1_lambda=0.5,
                 e1_floor=30.0, bootstrap_support="full",
                 explore_bonus="off", explore_bonus_scale=0.5,
                 explore_bonus_halflife=2000.0, tracker_tripwire=True,
                 cf_replay=False, cf_k=3, cf_boost=CF_BOOST,
                 cf_max_frac=CF_MAX_FRAC, cf_age_cap=CF_AGE_CAP,
                 cf_budget_rate=CF_BUDGET_RATE, cf_seed=None,
                 cf_pure_mc=False, cf_exclude_limit=False,
                 class_replay=False, crash_age_cap=CRASH_AGE_CAP,
                 crash_floor=CRASH_BATCH_FLOOR, noop_tolerance=None):
        self.token_dim = token_dim
        self.n_instr = n_instr
        self.gamma = gamma
        self.c_train = c_train
        self.width_reg = width_reg
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.mask_reissue = mask_reissue
        self.width_scalars = width_scalars
        # --- fix D: trained-support bootstrap (12D doc 2.4, default OFF).
        # 'taken_noop' restricts the Double-DQN target argmax at s' to
        # {NOOP (col 0), next_a_idx (the action actually taken at s')} —
        # honest framing per the audit: this is n-step SARSA with a NOOP
        # floor, not an optimality backup; recorded as an algorithm change.
        assert bootstrap_support in ("full", "taken_noop"), bootstrap_support
        self.bootstrap_support = bootstrap_support
        # winner-share observability (design amendment): among bootstrap
        # rows (disc > 0) with a real next action, did the taken action or
        # NOOP win the restricted argmax? Plus the NOOP-fallback fraction
        # (censored tails with next_a == 0).
        self.bsupport_taken_wins = deque(maxlen=2000)   # 1.0 taken / 0.0 NOOP
        self.bsupport_fallbacks = deque(maxlen=2000)    # 1.0 next_a == 0
        # --- fix 2: count-based exploration bonus at ACTION SELECTION
        # ONLY (never in targets, never in stored rewards). Counts are per
        # (sector-risk stratum, instruction type incl. NOOP) and are
        # ALWAYS tracked during training (cheap bookkeeping, enables
        # coverage comparison against no-bonus runs); the bonus term is
        # added to selection scores only when explore_bonus == 'count'.
        # Reference anchored to counts only — no self-reference to net
        # outputs (E1/LL runaway lesson).
        assert explore_bonus in ("off", "count"), explore_bonus
        self.explore_bonus = explore_bonus
        self.explore_bonus_scale = float(explore_bonus_scale)
        self.explore_bonus_halflife = float(explore_bonus_halflife)
        self.explore_counts = np.zeros((N_STRATA, 1 + n_instr),
                                       dtype=np.int64)
        # --- fix 3: tracker tripwire (default ON, pure instrumentation —
        # no behavior change whatsoever; see train_step and the loggers).
        self.tracker_tripwire = tracker_tripwire
        self._tripwire_devs = []   # per-batch mean |target_mid - pred_mid|
        # E1-ATC contrastive off-manifold width term (default OFF): push
        # candidate widths on build_e1_negatives() states up to an
        # ABSOLUTE floor via the bounded hinge lambda*relu(1 - w/floor)^2.
        # Floor default 30.0 = 3x the measured median on-dist candidate
        # width (10.1) of the 12b ep1500 net on the a4v2 frozen probe
        # states. NEVER key the floor to live batch widths — the
        # self-referential floor feeds back through the shared trunk and
        # runs away (LL attempt 1: on-dist width 4 -> 2033;
        # WIDTH_MECHANISM_PROBES.md). Adds NO parameters.
        self.e1_width = e1_width
        self.e1_lambda = e1_lambda
        self.e1_floor = e1_floor
        self.step_count = 0

        self.q_net = ControllerQNet(token_dim, n_instr=n_instr,
                                    width_scalars=width_scalars
                                    ).to(self.device)
        self.target_net = ControllerQNet(token_dim, n_instr=n_instr,
                                         width_scalars=width_scalars
                                         ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        # --- ROUND 3 B2 (--class_replay, default OFF): per-class
        # boost/eviction policy lives on the buffer; the agent mirrors
        # the flag for run_episode's per-episode aging call.
        self.class_replay = bool(class_replay)
        if self.class_replay:
            assert terminal_boost > 1.0, (
                "--class_replay requires the split-pool buffer "
                "(terminal_boost > 1)")
        self.buffer = ControllerReplayBuffer(buffer_size,
                                             terminal_boost=terminal_boost,
                                             cf_boost=cf_boost,
                                             cf_max_frac=cf_max_frac,
                                             class_replay=class_replay,
                                             crash_age_cap=crash_age_cap,
                                             crash_floor=crash_floor)
        self._kind_counts = {}          # per-episode batch composition
        # --- ROUND 3 A1 (--completion_terminal): run_episode counts
        # completion-terminated episodes here (realized, no bootstrap).
        self.completed_episodes = 0
        # --- ROUND 3 B1 (--cf_exclude_limit, default OFF): CF branch
        # windows whose H-step horizon crosses the ENV time limit are
        # EXCLUDED (not capped); cf_excluded_limit counts them.
        self.cf_exclude_limit = bool(cf_exclude_limit)
        self.cf_excluded_limit = 0
        # --- ROUND 3 C2 scaffold (--noop_tolerance FLOAT, default
        # None=OFF): NOOP-preference tolerance in select_candidate ONLY,
        # evaluated on a RAW PRE-BONUS copy of the Hurwicz scores with a
        # stratum guard (never on stratum-0 / in-band states). The
        # THRESHOLD is a measured quantity (ROUND3_REREVIEW 'C2
        # threshold derivation') and stays unset this round.
        self.noop_tolerance = (float(noop_tolerance)
                               if noop_tolerance is not None else None)
        self.tol_invoked_ep = 0         # tolerance flipped the pick to NOOP
        self.tol_bonus_override_ep = 0  # bonus-driven pick stood (logged
                                        # exploration; see select_candidate)

        # --- RUN 13 FIX C: counterfactual replay (--cf_replay, default
        # OFF). The attributes below exist unconditionally (cheap) but
        # nothing runs unless cf_replay is True — the flag-off path is
        # bitwise unchanged (no-contamination selftest).
        self.cf_replay = bool(cf_replay)
        # cf_pure_mc (19 July, M2 diagnosis): cf rows train on the pure
        # H_cf-window return with NO bootstrap tail (per-row disc zeroed
        # in train_step). Rationale: the A12.2 NOOP-head calibration
        # check effectively FIRED on the M2 800-ep arm — CF NOOP targets
        # of ~0 were recorded as ~-10 because the gamma^36*Q(s_H) tail
        # re-injects the doom prior the probe was meant to correct; the
        # audit pre-registered this variant as the fallback (truncation
        # bias ~0.69*V applies where V is genuinely nonzero — use on
        # clean-scenario arms, not conflict arms). Default OFF;
        # exploratory arms only until JK blesses wider use.
        self.cf_pure_mc = bool(cf_pure_mc)
        if self.cf_replay:
            assert terminal_boost > 1.0, (
                "--cf_replay requires the split-pool buffer "
                "(terminal_boost > 1)")
        self.cf_k = int(cf_k)
        self.cf_age_cap = int(cf_age_cap)
        self.cf_budget_rate = float(cf_budget_rate)
        # dedicated RNG stream (A12.3): EVERY CF-side draw — state
        # selection, candidate floor, continuation-policy epsilon
        # plumbing — comes from here; live RNG is never touched from
        # CF code.
        self.cf_rng = random.Random(
            0xCF0000 + (cf_seed if cf_seed is not None else 0))
        # SEPARATE CF-side count table (A12.3): explore_counts layout,
        # incremented per CF-probed candidate; feeds candidate rule 2b
        # without ever writing the live table.
        self.cf_counts = np.zeros((N_STRATA, 1 + n_instr),
                                  dtype=np.int64)
        self.cf_budget = CFBudget()
        self.cf_gen_ep = 0
        # A12.1 parallel cf_* series (the live trackers are quarantined
        # from cf rows; the tripwire revival trigger and its ep500
        # baseline read the LIVE-ONLY series)
        self.cf_bootstrap_hits = deque(maxlen=2000)
        self._cf_tripwire_devs = []
        self.cf_batch_frac = deque(maxlen=2000)
        self.cf_sampled_ages = deque(maxlen=2000)
        # A3 pricing instrumentation
        self.cf_event_flags = deque(maxlen=2000)
        self.cf_gcf_spreads = deque(maxlen=2000)
        self.cf_states_selected = 0
        self.cf_rollouts = 0
        self.cf_items = 0
        self.cf_analytic_items = 0
        self.cf_skipped_budget = 0

        # Stratified-t is ALWAYS on in the controller frame: one tracker
        # per sector-risk stratum; the risky stratum (min pair sep < 10 nm)
        # runs a raised t cap. Driven by realized returns only.
        self.trackers = [
            CoverageTracker(target=target_coverage,
                            t_max=(t_cap_risky if s == 0 else 0.95))
            for s in range(N_STRATA)]
        self.bootstrap_hits = deque(maxlen=2000)
        self.realized_episodes = 0
        self.censored_episodes = 0

        self.warmup_steps = warmup_steps
        self.warmup_epsilon = warmup_epsilon
        self.total_env_steps = 0

    # ---- diagnostics passthroughs -----------------------------------
    @property
    def t(self):
        return float(np.mean([tr.t for tr in self.trackers]))

    @property
    def coverage(self):
        n = sum(len(tr.hits) for tr in self.trackers)
        return sum(sum(tr.hits) for tr in self.trackers) / max(1, n)

    @property
    def bootstrap_coverage(self):
        return sum(self.bootstrap_hits) / max(1, len(self.bootstrap_hits))

    @property
    def winner_share_taken(self):
        """Fraction of recent restricted-bootstrap argmaxes (disc > 0,
        real next action) won by the taken action rather than NOOP.
        nan until the first taken_noop train_step contributes."""
        if not self.bsupport_taken_wins:
            return float("nan")
        return sum(self.bsupport_taken_wins) / len(self.bsupport_taken_wins)

    @property
    def noop_fallback_share(self):
        """Fraction of recent bootstrap rows carrying the censored-tail
        NOOP fallback (next_a_idx == 0 with disc > 0)."""
        if not self.bsupport_fallbacks:
            return float("nan")
        return sum(self.bsupport_fallbacks) / len(self.bsupport_fallbacks)

    # ---- fix 2: count-based exploration bonus ------------------------
    def explore_bonus_value(self, count):
        """Bonus for a cell with `count` visits (scalar or ndarray):
        scale * sqrt(1 / (1 + 3*count/halflife)) — the UCB-style
        sqrt(1/(1+count)) form reparameterized so the bonus is `scale` at
        count 0 and exactly scale/2 at count == halflife (the literal
        sqrt(1/(1+count)) has a fixed half-life of 3 counts, which would
        make --explore_bonus_halflife dead)."""
        c = np.asarray(count, dtype=np.float64)
        return self.explore_bonus_scale * np.sqrt(
            1.0 / (1.0 + 3.0 * c / self.explore_bonus_halflife))

    def _explore_bonus_vec(self, stratum, n_cand):
        """Candidate-layout bonus vector for a decision at `stratum`:
        col 0 <- type 0 (NOOP), col 1 + n_instr*i + j <- type 1 + j."""
        b = self.explore_bonus_value(self.explore_counts[stratum])
        vec = np.empty(n_cand, dtype=np.float64)
        vec[0] = b[0]
        if n_cand > 1:
            vec[1:] = np.tile(b[1:], (n_cand - 1) // self.n_instr)
        return vec

    def note_taken_action(self, stratum, cand_idx):
        """Count the TAKEN action (post-epsilon) in the per-(stratum,
        instruction-type) table. Pure bookkeeping — always safe."""
        jt = 0 if cand_idx == 0 else 1 + (cand_idx - 1) % self.n_instr
        self.explore_counts[stratum, jt] += 1

    # ---- fix 3: tripwire proxy ----------------------------------------
    def pop_tripwire_proxy(self):
        """Mean per-batch |bootstrap-target midpoint - predicted midpoint|
        accumulated since the last call (the episode's training batches);
        None when no batch ran. Clears the accumulator."""
        if not self._tripwire_devs:
            return None
        v = float(np.mean(self._tripwire_devs))
        self._tripwire_devs = []
        return v

    # ---- ROUND 3 C2: per-episode tolerance counters -------------------
    def pop_tolerance_counts(self):
        """{'invoked', 'bonus_override'} accumulated since the last call
        (one episode's selections); clears the counters. Logged per
        episode by run_episode when --noop_tolerance is set."""
        out = {"invoked": self.tol_invoked_ep,
               "bonus_override": self.tol_bonus_override_ep}
        self.tol_invoked_ep = 0
        self.tol_bonus_override_ep = 0
        return out

    # ---- ROUND 3 B2: per-episode batch-composition telemetry ----------
    def pop_kind_counts(self):
        """Per-terminal_kind row counts over the episode's training
        batches (accumulated in train_step from the sampler's
        last_kind_counts); clears the accumulator. Only populated under
        --class_replay."""
        out = dict(self._kind_counts)
        self._kind_counts = {}
        return out

    # ---- RUN 13 FIX C: cf_* parallel series (A12.1) -------------------
    def pop_cf_tripwire_proxy(self):
        """cf-row analogue of pop_tripwire_proxy: mean |target mid -
        predicted mid| over recent cf batch rows; None when no cf row
        trained. NEVER feeds the live tripwire trigger."""
        if not self._cf_tripwire_devs:
            return None
        v = float(np.mean(self._cf_tripwire_devs))
        self._cf_tripwire_devs = []
        return v

    def cf_episode_stats(self):
        """Cumulative CF-replay observability snapshot logged per
        episode (A2.4 coverage, A14 sampler stats, A3 pricing
        instrumentation). Counters are cumulative; the JSONL consumer
        differences them."""
        ages = [self.cf_gen_ep - a for a in self.cf_sampled_ages]
        counts = self.buffer.cf_replay_counts()
        return {
            "pool": len(self.buffer.cf_buf),
            "states_selected": self.cf_states_selected,
            "rollouts": self.cf_rollouts,
            "items": self.cf_items,
            "analytic_items": self.cf_analytic_items,
            "skipped_budget": self.cf_skipped_budget,
            # ROUND 3 B1 telemetry: states whose branch horizon crossed
            # the ENV time limit and were EXCLUDED (0 unless
            # --cf_exclude_limit)
            "excluded_limit": self.cf_excluded_limit,
            "bank": round(self.cf_budget.bank, 1),
            "spent_steps": self.cf_budget.spent_steps,
            "spent_clones": self.cf_budget.spent_clones,
            "batch_frac": (round(float(np.mean(self.cf_batch_frac)), 4)
                           if self.cf_batch_frac else 0.0),
            "median_sampled_age": (int(np.median(ages)) if ages
                                   else None),
            "max_replay_count": (max(counts) if counts else 0),
            "event_frac": (round(float(np.mean(self.cf_event_flags)), 4)
                           if self.cf_event_flags else None),
            "gcf_spread_med": (round(float(
                np.median(self.cf_gcf_spreads)), 4)
                if self.cf_gcf_spreads else None),
            "bootstrap_cov": (round(float(
                np.mean(self.cf_bootstrap_hits)), 4)
                if self.cf_bootstrap_hits else None),
            "accuracy_proxy": self.pop_cf_tripwire_proxy(),
        }

    def param_count(self):
        return sum(p.numel() for p in self.q_net.parameters())

    def _epsilon(self, force_epsilon=None):
        if force_epsilon is not None:
            return force_epsilon
        if self.total_env_steps < self.warmup_steps:
            progress = self.total_env_steps / self.warmup_steps
            return self.warmup_epsilon * (1 - progress)
        return 0.0

    # ---- action selection --------------------------------------------
    def candidate_q(self, tokens_np):
        """Candidate intervals for ONE state. Returns (cand_l, cand_u)
        as 1-D numpy arrays of length 1 + n_instr*N."""
        with torch.no_grad():
            tok = torch.from_numpy(
                tokens_np[None].astype(np.float32)).to(self.device)
            mask = torch.ones(1, tokens_np.shape[0], dtype=torch.bool,
                              device=self.device)
            ac_l, ac_u, nl, nu = self.q_net(tok, mask)
            cl, cu, _ = candidate_intervals(ac_l, ac_u, nl, nu, mask)
        return cl[0].cpu().numpy(), cu[0].cpu().numpy()

    def select_candidate(self, tokens_np, c=None, adaptive=False,
                         reissue_mask=None, bonus_stratum=None):
        """Hurwicz l + c*(u - l) over ALL candidates; adaptive c is a
        SINGLE width-conditioned c from the mean candidate width (pilot
        sigmoid rule); exact ties involving NOOP resolve to NOOP.
        reissue_mask (candidate layout, True = masked) excludes re-issue
        candidates from the argmax (score forced to -inf); NOOP is never
        masked so the argmax always has a candidate.
        bonus_stratum (fix 2, SELECTION ONLY): when given and
        explore_bonus == 'count', add the count-based bonus vector for
        that stratum to the scores BEFORE the re-issue mask is applied —
        the bonus can never resurrect a masked candidate, and it never
        touches targets or stored rewards (train_step never calls here).

        --noop_tolerance (ROUND 3 C2 scaffold, default None=OFF): prefer
        NOOP unless the best candidate beats it by MORE than the
        tolerance, evaluated on a RAW PRE-BONUS copy of the Hurwicz
        scores (the count bonus above, scale 0.5 vs tolerance ~0.1, is
        added before the argmax and would override a post-bonus
        tolerance ~5x during training then hand back control on bonus
        decay — re-review 'C2 tolerance placement'). The re-issue mask
        still applies to the raw copy (masked candidates are not
        selectable); NOOP is never masked. STRATUM GUARD: the tolerance
        never applies on stratum-0 (in-band) states, and only where the
        caller supplies a stratum (the bonus_stratum argument — the
        run_episode training path; probe/eval callers pass none and are
        unaffected). When the tolerance verdict is NOOP but the
        bonus-inclusive argmax was CHANGED by the bonus, the bonus pick
        stands as LOGGED exploration (tol_bonus_override_ep); otherwise
        the pick flips to NOOP (tol_invoked_ep). Both counters are
        popped per episode. The threshold itself is a measured quantity
        and ships unset this round.
        Returns (cand_idx, mean_width, c_used)."""
        cl, cu = self.candidate_q(tokens_np)
        widths = cu - cl
        mean_width = float(widths.mean())
        use_c = c if c is not None else self.c_train
        if adaptive:
            w_mid = getattr(self, "adaptive_w_mid", 4.0)
            use_c = float(bid.IntervalDQNAgent.adaptive_c(
                torch.tensor(mean_width), w_mid=w_mid))
        scores = cl + use_c * (cu - cl)
        tol = self.noop_tolerance
        raw_scores = scores.copy() if tol is not None else None
        if bonus_stratum is not None and self.explore_bonus == "count":
            scores = scores + self._explore_bonus_vec(bonus_stratum,
                                                      len(scores))
        if reissue_mask is not None:
            scores = np.where(reissue_mask, -np.inf, scores)
        best = scores.max()
        idx = 0 if scores[0] == best else int(scores.argmax())
        if tol is not None and idx != 0 and bonus_stratum is not None \
                and bonus_stratum != 0:
            raw = (np.where(reissue_mask, -np.inf, raw_scores)
                   if reissue_mask is not None else raw_scores)
            if raw.max() - raw[0] <= tol:      # NOOP verdict (raw margin)
                raw_best = raw.max()
                raw_idx = 0 if raw[0] == raw_best else int(raw.argmax())
                if idx == raw_idx:
                    idx = 0
                    self.tol_invoked_ep += 1
                else:
                    # bonus changed the argmax: exploration stands, logged
                    self.tol_bonus_override_ep += 1
        return idx, mean_width, float(use_c)

    def calibrate_w_mid(self, token_arrays):
        """Set the adaptive-c midpoint to the median mean-candidate-width
        over a sample of token arrays (e.g. one probe episode)."""
        widths = []
        for t in token_arrays:
            if t.shape[0] == 0:
                continue
            cl, cu = self.candidate_q(t)
            widths.append(float((cu - cl).mean()))
        self.adaptive_w_mid = float(np.median(widths)) if widths else 4.0
        return self.adaptive_w_mid

    def generate_action(self, env, obs_dict, info_dict, c=None,
                        force_epsilon=None, adaptive=False,
                        last_issued=None, stratum=None, rng=random):
        """Compute ONE clearance (aircraft, instruction) or global NOOP.

        Returns (action_dict {callsign: int}, aux) where aux carries the
        token array, sorted callsign tuple, chosen candidate index, mean
        candidate width and the c used — everything replay needs, so the
        buffer never touches the env. Epsilon warmup: uniform over the
        candidate set (NOOP + n_instr per aircraft), minus masked re-issues.
        last_issued ({callsign: instr j}) enables re-issue masking when
        self.mask_reissue is on: candidates repeating an aircraft's
        still-active instruction are excluded from BOTH the argmax and
        epsilon-sampling.
        stratum (fix 2): the decision step's sector-risk stratum, passed
        by run_episode on the TRAINING path only. When given, the taken
        action is counted in explore_counts (always), and the count-based
        selection bonus is applied when explore_bonus == 'count'.
        rng (RUN 13 fix C, A12.3): the random source for the epsilon
        draw/sample. Defaults to the module-level `random` stream —
        bitwise identical to the pre-cf code; CF-side callers pass the
        agent's dedicated cf_rng so CF continuations never advance the
        live stream."""
        cs_list = sorted(obs_dict.keys())
        tokens = build_tokens(env, obs_dict, info_dict, cs_list)
        rmask = None
        if self.mask_reissue and last_issued:
            rmask = build_reissue_mask(cs_list, last_issued, self.n_instr)
        idx, mean_width, c_used = self.select_candidate(
            tokens, c=c, adaptive=adaptive, reissue_mask=rmask,
            bonus_stratum=stratum)
        eps = self._epsilon(force_epsilon)
        if rng.random() < eps:
            n_cand = 1 + self.n_instr * len(cs_list)
            if rmask is not None:
                idx = rng.choice(
                    [k for k in range(n_cand) if not rmask[k]])
            else:
                idx = rng.randrange(n_cand)
        if stratum is not None:
            self.note_taken_action(stratum, idx)
        actions = {cs: 0 for cs in cs_list}
        if idx > 0:
            i, j = divmod(idx - 1, self.n_instr)
            actions[cs_list[i]] = j + 1     # env actions 1..n_instr
        aux = {"cand_idx": idx, "tokens": tokens,
               "callsigns": tuple(cs_list), "mean_width": mean_width,
               "c_used": c_used}
        return actions, aux

    # ---- realized coverage (drives t) ---------------------------------
    def record_realized_episode(self, states, a_idxs, strata, rewards):
        """Feed the realized returns-to-go of EVERY step of a
        violation-terminated episode into the stratum trackers. The last
        reward already carries the -50 terminal."""
        if not states:
            return
        G, returns = 0.0, [0.0] * len(rewards)
        for k in range(len(rewards) - 1, -1, -1):
            G = rewards[k] + self.gamma * G
            returns[k] = G
        with torch.no_grad():
            tok, mask = pad_state_batch(states, self.device)
            ac_l, ac_u, nl, nu = self.q_net(tok, mask)
            cl, cu, _ = candidate_intervals(ac_l, ac_u, nl, nu, mask)
            a = torch.LongTensor(a_idxs).to(self.device)
            l_a = cl.gather(1, a.unsqueeze(1)).squeeze(1).cpu().numpy()
            u_a = cu.gather(1, a.unsqueeze(1)).squeeze(1).cpu().numpy()
        g = np.asarray(returns, dtype=np.float32)
        hits = ((g >= l_a) & (g <= u_a)).astype(np.float32)
        for st, hit in zip(strata, hits):
            self.trackers[st].record([float(hit)])
        self.realized_episodes += 1

    # ---- interval loss (pilot form, per-sample t + width gate) --------
    def interval_loss_sampled(self, lower, upper, target_lower, target_upper,
                              t_vec, width_mask, cf_mask=None):
        N = self.N_TARGET_SAMPLES
        batch_size = lower.shape[0]
        alphas = torch.linspace(0, 1, N, device=lower.device)
        targets = target_lower.unsqueeze(0) + alphas.unsqueeze(1) * (
            target_upper - target_lower).unsqueeze(0)

        # bootstrap-target coverage: LOGGING ONLY (t runs on realized).
        # cf_mask (RUN 13 fix C, A12.1): cf rows are QUARANTINED from
        # the live bootstrap-coverage log and feed the parallel cf_*
        # series instead; None (the flag-off default) is the exact
        # pre-cf code path.
        target_mid = (target_lower + target_upper) / 2
        inside_mid = (target_mid >= lower) & (target_mid <= upper)
        if cf_mask is None:
            self.bootstrap_hits.extend(
                inside_mid.detach().cpu().numpy().astype(
                    np.float32).tolist())
        else:
            hits = inside_mid.detach().cpu().numpy().astype(np.float32)
            m = cf_mask.detach().cpu().numpy()
            self.bootstrap_hits.extend(hits[~m].tolist())
            self.cf_bootstrap_hits.extend(hits[m].tolist())

        total_loss = torch.zeros(batch_size, device=lower.device)
        for i in range(N):
            t_i = targets[i]
            inside = (t_i >= lower) & (t_i <= upper)
            dist_to_lower = (t_i - lower) ** 2
            dist_to_upper = (t_i - upper) ** 2
            min_dist_sq = torch.min(dist_to_lower, dist_to_upper)
            max_dist_sq = torch.max(dist_to_lower, dist_to_upper)
            outside_penalty = torch.where(
                inside, torch.zeros_like(min_dist_sq), min_dist_sq)
            total_loss += t_vec * outside_penalty + (1 - t_vec) * max_dist_sq
        total_loss /= N

        width = upper - lower
        total_loss = total_loss + self.width_reg * width ** 2 * width_mask
        return total_loss.mean()

    # ---- E1 hinge (factored for selftest) ------------------------------
    def e1_hinge(self, neg_tok, neg_mask):
        """Bounded hinge on the online net's candidate widths at negative
        states: mean over VALID candidates (NOOP head included) of
        relu(1 - w/floor)^2. In [0, 1]; exactly 0 (dead gradient) once
        every valid width reaches the floor — runaway impossible by
        construction. Pure extra loss term: touches no tracker, buffer,
        re-issue mask, or target computation."""
        ac_l, ac_u, nl, nu = self.q_net(neg_tok, neg_mask)
        cl, cu, valid = candidate_intervals(ac_l, ac_u, nl, nu, neg_mask)
        h = F.relu(1.0 - (cu - cl) / self.e1_floor).pow(2)
        v = valid.float()
        return (h * v).sum() / v.sum().clamp(min=1.0)

    # ---- target-side Double-DQN argmax (factored for selftest) --------
    @staticmethod
    def double_dqn_argmax(ocl, ocu, nvalid, reissue_mask, c):
        """Argmax of the online net's Hurwicz score over s' candidates,
        excluding padded candidates (~nvalid) AND re-issue-masked
        candidates (reissue_mask True). Used by train_step; exercised
        directly by --selftest with a hand-built batch."""
        nscore = ocl + c * (ocu - ocl)
        nscore = nscore.masked_fill(~nvalid, -1e9)
        if reissue_mask is not None:
            nscore = nscore.masked_fill(reissue_mask, -1e9)
        return nscore.argmax(dim=1)

    @staticmethod
    def restricted_support_argmax(ocl, ocu, nvalid, reissue_mask, c,
                                  next_as):
        """--bootstrap_support taken_noop (fix D, 12D doc 2.4): the
        Double-DQN target argmax at s' ranges over {NOOP (column 0), the
        action ACTUALLY TAKEN at s' (next_as)} instead of all 1 + 5N
        mostly-never-trained candidates. Padding validity and the stored
        re-issue mask still apply (the taken action was selected under
        that mask, so it is never masked in practice; NOOP never is).
        Exact ties resolve to NOOP (argmax picks the lower index).
        Exercised directly by --selftest with a hand-built batch."""
        nscore = ocl + c * (ocu - ocl)
        nscore = nscore.masked_fill(~nvalid, -1e9)
        if reissue_mask is not None:
            nscore = nscore.masked_fill(reissue_mask, -1e9)
        allowed = torch.zeros_like(nscore, dtype=torch.bool)
        allowed[:, 0] = True                       # NOOP floor
        idx = next_as.clamp(min=0, max=nscore.shape[1] - 1)
        allowed.scatter_(1, idx.unsqueeze(1), True)
        nscore = nscore.masked_fill(~allowed, -1e9)
        return nscore.argmax(dim=1)

    def _pad_reissue_masks(self, ns_masks, n_cand_max):
        """[B, n_cand_max] bool tensor from per-sample candidate-layout
        masks; padded candidate slots stay False (already excluded via
        the validity mask)."""
        out = torch.zeros(len(ns_masks), n_cand_max, dtype=torch.bool,
                          device=self.device)
        for b, m in enumerate(ns_masks):
            if m is None:
                continue
            L = min(len(m), n_cand_max)
            out[b, :L] = torch.from_numpy(
                np.ascontiguousarray(m[:L])).to(self.device)
        return out

    # ---- one gradient step --------------------------------------------
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch = self.buffer.sample(self.batch_size)
        # RUN 13 FIX C: cf row flags for this batch (None whenever no cf
        # item was drawn — the flag-off/severed path, bitwise legacy)
        cf_flags = getattr(self.buffer, "last_cf_mask", None)
        has_cf = cf_flags is not None
        states = [b[0] for b in batch]
        a_idxs = torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_states = [b[3] for b in batch]
        ns_masks = [b[4] for b in batch]
        discs = torch.FloatTensor([b[5] for b in batch]).to(self.device)
        strata = [b[6] for b in batch]
        next_as = torch.LongTensor([b[7] for b in batch]).to(self.device)
        cf_t = None
        if has_cf:
            cf_t = torch.tensor(cf_flags, dtype=torch.bool,
                                device=self.device)
            # A14 observability: realized cf batch fraction + sampled
            # cf generation stamps (median-age tripwire input)
            self.cf_batch_frac.append(float(np.mean(cf_flags)))
            self.cf_sampled_ages.extend(self.buffer.last_cf_ages)
        # ROUND 3 B2: per-class batch composition (terminal_kind rides in
        # every row, batch[i][8]); the sampler tallies it only under
        # --class_replay, so the flag-off path skips this bitwise.
        kind_counts = getattr(self.buffer, "last_kind_counts", None)
        if kind_counts:
            for kk, vv in kind_counts.items():
                self._kind_counts[kk] = self._kind_counts.get(kk, 0) + vv

        tok, mask = pad_state_batch(states, self.device)
        ac_l, ac_u, nl, nu = self.q_net(tok, mask)
        cl, cu, _ = candidate_intervals(ac_l, ac_u, nl, nu, mask)
        lower_a = cl.gather(1, a_idxs.unsqueeze(1)).squeeze(1)
        upper_a = cu.gather(1, a_idxs.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            ntok, nmask = pad_state_batch(next_states, self.device)
            # Double-DQN: online net picks the s' candidate at c_train
            # (padded AND re-issue-masked candidates excluded)...
            o_ac_l, o_ac_u, o_nl, o_nu = self.q_net(ntok, nmask)
            ocl, ocu, nvalid = candidate_intervals(o_ac_l, o_ac_u,
                                                   o_nl, o_nu, nmask)
            rmask = self._pad_reissue_masks(ns_masks, ocl.shape[1])
            if self.bootstrap_support == "taken_noop":
                na = self.restricted_support_argmax(ocl, ocu, nvalid,
                                                    rmask, self.c_train,
                                                    next_as)
                # winner-share + fallback observability (design
                # amendment). cf rows are QUARANTINED from bsupport_*
                # (A12.1) — their next_a is the FORCED continuation
                # action, not a restricted-argmax outcome.
                boot = discs > 0
                if has_cf:
                    boot = boot & ~cf_t
                live = boot & (next_as != 0)
                if bool(live.any()):
                    taken_won = (na[live] == next_as[live]).float()
                    self.bsupport_taken_wins.extend(
                        taken_won.cpu().numpy().tolist())
                if bool(boot.any()):
                    fb = (next_as[boot] == 0).float()
                    self.bsupport_fallbacks.extend(
                        fb.cpu().numpy().tolist())
            else:
                na = self.double_dqn_argmax(ocl, ocu, nvalid, rmask,
                                            self.c_train)
            if has_cf:
                # RUN 13 FIX C bootstrap: cf rows bootstrap on the
                # CONTINUATION policy's stored action at s_H —
                # y = G_cf + gamma^H * Q_target(s_H, a_cont(s_H))
                # (A1 Branch A) — never the argmax machinery.
                na = torch.where(
                    cf_t, next_as.clamp(min=0, max=ocl.shape[1] - 1),
                    na)
                if self.cf_pure_mc:
                    # pure-MC cf targets: y = G_cf, no tail (see the
                    # cf_pure_mc note in __init__)
                    discs = torch.where(cf_t, torch.zeros_like(discs),
                                        discs)
            # ...the target net evaluates it
            t_ac_l, t_ac_u, t_nl, t_nu = self.target_net(ntok, nmask)
            tcl, tcu, _ = candidate_intervals(t_ac_l, t_ac_u,
                                              t_nl, t_nu, nmask)
            tgt_lower = tcl.gather(1, na.unsqueeze(1)).squeeze(1)
            tgt_upper = tcu.gather(1, na.unsqueeze(1)).squeeze(1)
            # per-sample disc column: gamma^m, or 0 at terminals (the
            # target IS the realized return)
            target_l = rewards + discs * tgt_lower
            target_u = rewards + discs * tgt_upper

        # --- fix 3: tracker tripwire (PURE instrumentation; default ON;
        # no RNG, no gradient, no tracker/buffer contact — flag-off path
        # is bitwise identical). Accumulates the per-batch prediction-
        # accuracy proxy mean |bootstrap-target midpoint - predicted
        # midpoint|; per-episode JSONL logging happens in the training
        # loops. PRE-REGISTERED TRIGGER: if the accuracy proxy improves
        # >= 30% from its ep500 level while t stays pinned at cap for
        # 1000+ episodes, the tracker-feed change (12D doc fix E+F) gets
        # revived.
        if self.tracker_tripwire:
            with torch.no_grad():
                pred_mid = (lower_a.detach() + upper_a.detach()) / 2
                tgt_mid = (target_l + target_u) / 2
                if has_cf:
                    # A12.1: cf rows feed the PARALLEL cf series; the
                    # revival trigger and its ep500 baseline read the
                    # live-only series (cf errors start huge and fall
                    # fast — they must never fire or mask the trigger)
                    devs = (tgt_mid - pred_mid).abs()
                    if bool((~cf_t).any()):
                        self._tripwire_devs.append(
                            float(devs[~cf_t].mean()))
                    if bool(cf_t.any()):
                        self._cf_tripwire_devs.append(
                            float(devs[cf_t].mean()))
                else:
                    self._tripwire_devs.append(
                        float((tgt_mid - pred_mid).abs().mean()))

        t_vec = torch.FloatTensor(
            [self.trackers[si].t for si in strata]).to(self.device)
        width_mask = torch.FloatTensor(
            [1.0 if (self.trackers[si].coverage > self.trackers[si].target)
             else 0.0 for si in strata]).to(self.device)
        loss = self.interval_loss_sampled(lower_a, upper_a,
                                          target_l, target_u,
                                          t_vec, width_mask,
                                          cf_mask=cf_t)
        if self.e1_width and self.e1_lambda > 0.0:
            neg = build_e1_negatives(tok, mask)
            if neg is not None:
                loss = loss + self.e1_lambda * self.e1_hinge(neg[0], neg[1])
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return 0.0

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        for tr in self.trackers:
            tr.update_t()
        return loss.item()


# ===========================================================================
# EPISODE RUNNER
# ===========================================================================

def priced_env_step(env, obs, snap, actions, issued, gamma, cbp, cbp_lag,
                    clock, objective_v2):
    """One env step priced by the Section-4 reward layer — the EXACT
    computation run_episode performed inline before run 13 factored it
    out (identical operations in identical order; the flag-off selftests
    and the reconciliation identity cover the refactor). The CF-replay
    branch (--cf_replay) prices its suffix steps through THIS function,
    so live and counterfactual returns share one code path (design
    Section 1: 'priced via the live CBP reward layer'; audit A3).

    env must be AT state s; snap = sector snapshot at s; clock = the
    env-attached DeliveryClock (already observed AT s by the caller —
    observe discipline is the caller's job) or None under objective v1.
    Returns a dict with the stepped-env observations and every pricing
    component run_episode needs for its budget columns."""
    # CBP counterfactual branch: clone AT s, step the clone all-NOOP
    # 1+lag times (side-effect-free on the live env; --selftest)
    snap_cf = cbp_noop_snapshot(env, obs, lag=cbp_lag) if cbp else None

    next_obs, _rew_ignored, done, trunc, info = env.step(actions)
    violated, violation_kind, involved = detect_violation(info)

    delivered_cs = []
    for cs in obs:
        d_cs = info.get(cs)
        if isinstance(d_cs, dict) and \
                d_cs.get("pos_status") == "EXIT_REACHED":
            delivered_cs.append(cs)
    deliveries = len(delivered_cs)
    snap_next = sector_snapshot(env, next_obs.keys())

    # ---- OUR reward layer (spec Section 4 + CBP) ----
    prog, centre, conflict = shaping_terms(snap, snap_next, gamma)
    if cbp:
        if cbp_lag > 0:
            # action branch measured cbp_lag NOOP steps past s'
            # (command latency; see CBP_LAG note)
            env_a = copy.deepcopy(env)      # live env is AT s' here
            o_a = next_obs
            for _ in range(cbp_lag):
                o_a = env_a.step({cs: 0 for cs in o_a})[0]
            snap_act = sector_snapshot(env_a, o_a.keys())
            prog_a, centre_a, conf_a = shaping_terms(snap, snap_act,
                                                     gamma)
        else:
            prog_a, centre_a, conf_a = prog, centre, conflict
        prog_n, centre_n, conf_n = shaping_terms(snap, snap_cf, gamma)
        # per-term CBP components (D5 amendment D: exposed via
        # step_hook — the micro efficacy gate and the gate-mode
        # conflict-invariant selftest read them)
        paid_prog = prog_a - prog_n
        paid_centre = centre_a - centre_n
        paid_conf = conf_a - conf_n
    else:
        paid_prog, paid_centre, paid_conf = prog, centre, conflict
    shaping_paid = paid_prog + paid_centre + paid_conf
    fuel = 0.0 if objective_v2 else -EPS_FUEL * len(snap)
    cmd = -CMD_COST if issued else 0.0
    if objective_v2:
        deliv = sum(DELIVERY_BONUS * clock.bonus_factor(cs)
                    for cs in delivered_cs)
    else:
        deliv = DELIVERY_BONUS * deliveries
    r = shaping_paid + fuel + cmd + deliv
    if violated:
        r -= VIOLATION_PENALTY
    return {"next_obs": next_obs, "info": info, "snap_next": snap_next,
            "violated": violated, "violation_kind": violation_kind,
            "deliveries": deliveries, "delivered_cs": delivered_cs,
            "r": r, "shaping_paid": shaping_paid,
            "paid_prog": paid_prog, "paid_centre": paid_centre,
            "paid_conf": paid_conf,
            "prog": prog, "centre": centre, "conflict": conflict,
            "fuel": fuel, "cmd": cmd, "deliv": deliv}


def run_episode(env, agent, seed, train=True, c=None, adaptive=False,
                nstep=NSTEP, cbp=False, cbp_lag=CBP_LAG, step_hook=None,
                objective_v2=True, completion_terminal=False):
    """One controller episode; ends at the first violation (terminal, -50),
    the roster completion (--completion_terminal, ROUND 3 A1) or the
    time limit (censored). Returns a stats dict with the per-term
    additive reward budget (credit tracing per spec Section 4).

    completion_terminal (ROUND 3 A1, default OFF; spawn-0 micros only):
    the episode ends as a TRUE COMPLETION when delivered_count equals
    the scenario roster (num_starter_aircraft from the env config). The
    check runs strictly AFTER the violation branch — the corner is the
    OUT_SECTOR step, where an undelivered aircraft leaves obs on
    exactly the step detect_violation fires sector_excursion; checking
    completion first would label an excursion endpoint a win
    (ROUND3_REVIEW A1 amendment). The remaining episode is provably
    empty (terminal value 0; delivery bonuses already paid as realized
    events): windows reaching the completion collapse to realized
    returns with NO bootstrap (windower terminal collapse, kind
    'completion') and the realized returns-to-go feed the coverage
    trackers exactly like a violation ending.

    ROUND 3 W_rec payment-site scaffold: the terms budget carries a
    'recovery_win' column, included in the reconciliation identity and
    paid 0.0 ALWAYS this round (A2 emits W_rec later, in run_episode
    under this column — never inside priced_env_step, so CF branch
    pricing can never see W_rec; re-review 'W_rec payment site').

    objective_v2 (default ON): no fuel term; the delivery bonus decays
    with transit time via the env-attached DeliveryClock (see the
    OBJECTIVE v2 note). objective_v2=False restores the v1 pricing
    (flat +10, -EPS_FUEL per aircraft-step). The stats dict also carries
    ep_return_disc = sum_t gamma^t r_t, the gamma-discounted return the
    learner actually optimises (B1's gate basis).

    cbp=True replaces the SHAPING TERM ONLY with the counterfactual-
    baselined potential (CBP):
        old: gamma*Phi(s') - Phi(s)               [ambient-dominated]
        new: gamma*(Phi(s_action) - Phi(s_noop))
    where both branches start from the same state s, the action branch
    plays the issued action then all-NOOP, the counterfactual branch (a
    bit-faithful deepcopy) plays all-NOOP throughout, and BOTH are
    measured (1 + cbp_lag) steps after s — see the CBP_LAG note above:
    at lag 0 (the spec-literal one-step construct) the term is provably
    identically zero on this simulator because commands act with one
    sweep of latency. The matched-set convention applies to BOTH branches
    identically (matched over IN_SECTOR at s and the respective measured
    state), so when the issued action IS global NOOP the term is EXACTLY
    0.0 at any lag (asserted by --selftest). ALL OTHER TERMS STAY
    ABSOLUTE (fuel, fees, delivery +10, violation -50) — a full
    difference reward would cancel slow-ripening credit; the ambient
    contamination lives in the potential term specifically. Cost at
    lag 1: two extra deepcopies + three clone steps per training step.

    Budget columns: 'cbp_shaping' is the shaping actually PAID (the CBP
    difference when cbp, the absolute potential difference otherwise), so
    the reconciliation identity
        cbp_shaping + fuel + cmd_cost + delivery_bonus + violation_term
        == ep_return
    holds exactly in both modes (asserted below). The absolute phi_*
    terms are kept as DIAGNOSTICS-ONLY columns (not part of the sum when
    cbp is on).

    step_hook(dict), if given, is called once per step with the step's
    pricing internals (used by --selftest)."""
    obs, info = env.reset(seed=seed)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    # ROUND 3 A1 roster (--completion_terminal): spawn-0 scenarios only —
    # with a nonzero spawn rate delivered == roster can occur with
    # aircraft still airborne, so the flag refuses loudly.
    roster = None
    if completion_terminal:
        sc = env.config.scenario_config
        assert float(sc.get("initial_spawn_rate", 0.0)) == 0.0 and \
            float(sc.get("max_spawn_rate", 0.0)) == 0.0, (
                "--completion_terminal requires a spawn-0 scenario (A1 "
                "is micro-only; A2 owns the full sector) — got spawn "
                f"rates {sc.get('initial_spawn_rate')}/"
                f"{sc.get('max_spawn_rate')}")
        roster = int(sc["num_starter_aircraft"])
        assert roster > 0, f"empty roster ({roster})"
    windower = ControllerWindower(agent.buffer, agent.gamma, nstep) \
        if train else None
    mask_on = getattr(agent, "mask_reissue", False)
    last_issued = {}    # {callsign: instr j}; cleared here at episode reset

    terms = {"cbp_shaping": 0.0,
             "phi_progress": 0.0, "phi_centre": 0.0, "phi_conflict": 0.0,
             "fuel": 0.0, "cmd_cost": 0.0, "delivery_bonus": 0.0,
             "violation_term": 0.0,
             # ROUND 3 W_rec payment-site scaffold: A2's recovery win is
             # paid HERE under its own budget column when it lands;
             # 0.0 always this round. Part of the reconciliation
             # identity below.
             "recovery_win": 0.0}
    ep_states, ep_aidx, ep_strata, ep_rewards = [], [], [], []
    width_sum, width_n = 0.0, 0
    loss_sum, loss_n = 0.0, 0
    n_deliveries = 0
    n_commands = 0
    ep_return = 0.0
    ep_return_disc = 0.0
    violated, violation_kind = False, None
    completed = False   # ROUND 3 A1 completion terminal

    snap = sector_snapshot(env, obs.keys())
    clock = reset_delivery_clock(env) if objective_v2 else None
    step_i = -1
    for step_i in range(maxstep):
        if clock is not None:
            clock.observe(env, snap)
        force_eps = None if train else 0.0
        # stratum BEFORE selection (pure geometry, no RNG): fix 2 needs it
        # at selection time for the count-based bonus; passed only on the
        # training path so scripted/eval shims keep their old signature.
        stratum = snapshot_stratum(snap)
        gen_kw = {"stratum": stratum} if train else {}
        actions, aux = agent.generate_action(env, obs, info, c=c,
                                             force_epsilon=force_eps,
                                             adaptive=adaptive,
                                             last_issued=last_issued,
                                             **gen_kw)
        if aux["callsigns"]:
            width_sum += aux["mean_width"]
            width_n += 1
        issued = aux["cand_idx"] != 0

        # RUN 13 FIX C (--cf_replay, default OFF): generate counter-
        # factual branch targets AT s — BEFORE the live last_issued
        # update below and before the live step, so the CF candidate
        # mask and the branch continuation see EXACTLY the live masked
        # set at s (A13.1). All CF-side randomness lives on agent.cf_rng
        # and the separate agent.cf_counts table (A12.3); the live
        # trajectory is bitwise unaffected (isolation selftest).
        if train and getattr(agent, "cf_replay", False):
            # ROUND 3 B1: the branch step index and the ENV time limit
            # are threaded so --cf_exclude_limit can drop branches whose
            # horizon crosses the limit (inert ints when the flag is off)
            cf_process_state(agent, env, obs, info, snap, stratum,
                             last_issued,
                             (aux["tokens"], aux["callsigns"]),
                             aux["cand_idx"], nstep, cbp, cbp_lag,
                             objective_v2, branch_step=step_i,
                             ep_maxstep=maxstep)

        if mask_on and issued:
            i_ac, j_in = divmod(aux["cand_idx"] - 1, agent.n_instr)
            last_issued[aux["callsigns"][i_ac]] = j_in

        # step + Section-4 pricing (factored for run 13: the CF branch
        # prices through the same function; bitwise-identical op order)
        out = priced_env_step(env, obs, snap, actions, issued,
                              agent.gamma, cbp, cbp_lag, clock,
                              objective_v2)
        next_obs, info = out["next_obs"], out["info"]
        violated, violation_kind = out["violated"], out["violation_kind"]
        deliveries, snap_next = out["deliveries"], out["snap_next"]
        prog, centre, conflict = out["prog"], out["centre"], \
            out["conflict"]
        paid_prog, paid_centre, paid_conf = out["paid_prog"], \
            out["paid_centre"], out["paid_conf"]
        shaping_paid = out["shaping_paid"]
        fuel, cmd, deliv = out["fuel"], out["cmd"], out["deliv"]
        r = out["r"]
        terminal = False
        if violated:
            terms["violation_term"] -= VIOLATION_PENALTY
            terminal = True
        terms["cbp_shaping"] += shaping_paid
        terms["phi_progress"] += prog          # diagnostics-only when cbp
        terms["phi_centre"] += centre
        terms["phi_conflict"] += conflict
        terms["fuel"] += fuel
        terms["cmd_cost"] += cmd
        terms["delivery_bonus"] += deliv
        n_deliveries += deliveries
        n_commands += int(issued)
        # ROUND 3 A1: completion check strictly AFTER the violation
        # branch above — on the OUT_SECTOR corner step (undelivered
        # aircraft leaves obs the same step sector_excursion fires) the
        # violation wins and no completion can be stamped.
        if completion_terminal and not violated and n_deliveries >= roster:
            assert n_deliveries == roster and not violated, (
                f"completion terminal inconsistent: delivered "
                f"{n_deliveries} vs roster {roster}, violated={violated}")
            completed = True
            terminal = True
        ep_return += r
        ep_return_disc += (agent.gamma ** step_i) * r
        if step_hook is not None:
            step_hook({"step": step_i, "issued": issued,
                       "cand_idx": aux["cand_idx"],
                       "shaping_paid": shaping_paid, "r": r,
                       "phi_abs": prog + centre + conflict,
                       # D5 amendment D: per-term components of the
                       # shaping actually PAID (CBP differentials when
                       # cbp, absolute potential differences otherwise);
                       # they sum to shaping_paid exactly.
                       "paid_progress": paid_prog,
                       "paid_centre": paid_centre,
                       "paid_conflict": paid_conf})

        state = (aux["tokens"], aux["callsigns"])
        next_cs = tuple(sorted(next_obs.keys()))
        next_tokens = build_tokens(env, next_obs, info, list(next_cs))
        next_state = (next_tokens, next_cs)
        ep_states.append(state)
        ep_aidx.append(aux["cand_idx"])
        ep_strata.append(stratum)
        ep_rewards.append(r)

        if train:
            next_mask = build_reissue_mask(next_cs, last_issued,
                                           agent.n_instr) if mask_on \
                else np.zeros(1 + agent.n_instr * len(next_cs), dtype=bool)
            windower.add(state, aux["cand_idx"], r, next_state, next_mask,
                         terminal, stratum,
                         terminal_kind=("completion" if completed
                                        else "crash" if violated
                                        else None))
            agent.total_env_steps += 1
            loss = agent.train_step()
            if loss:
                loss_sum += loss
                loss_n += 1

        obs = next_obs
        snap = snap_next
        if violated or completed:
            break

    # reconciliation identity: SUM(budget columns) == ep_return exactly
    # (recovery_win included — ROUND 3 W_rec payment-site scaffold,
    # 0.0 always this round)
    recon = (terms["cbp_shaping"] + terms["fuel"] + terms["cmd_cost"]
             + terms["delivery_bonus"] + terms["violation_term"]
             + terms["recovery_win"])
    assert abs(recon - ep_return) <= 1e-6, (
        f"budget does not reconcile: sum(terms)={recon!r} != "
        f"ep_return={ep_return!r}")

    if train:
        if violated:
            # terminal episode: realized returns-to-go for EVERY step feed
            # the coverage trackers (windower already collapsed at add())
            agent.record_realized_episode(ep_states, ep_aidx, ep_strata,
                                          ep_rewards)
        elif completed:
            # ROUND 3 A1: completion is a TRUE terminal — the remaining
            # episode is provably empty, so every step's return-to-go is
            # fully realized (no bootstrap; windower already collapsed
            # with kind 'completion'). Feed the trackers exactly like a
            # violation ending; count separately.
            agent.record_realized_episode(ep_states, ep_aidx, ep_strata,
                                          ep_rewards)
            agent.completed_episodes += 1
        else:
            # time-limit censoring: bootstrap the tail windows; returns
            # unknown -> counted but NOT fed to the trackers
            windower.flush_censored()
            agent.censored_episodes += 1

    if train and getattr(agent, "cf_replay", False):
        # cf pool aging (A1 Branch A staleness): advance the generation
        # stamp and evict items past the A_max age cap. Recompute-on-
        # expiry via retained clones is NOT implemented (flagged
        # deviation — CF_REPLAY_IMPLEMENTATION.md): expired items DROP.
        agent.cf_gen_ep += 1
        agent.buffer.evict_cf_older_than(agent.cf_gen_ep
                                         - agent.cf_age_cap)

    if train and hasattr(agent.buffer, "gen_episode"):
        # ROUND 3 B2: episode stamp for the crash-row age cap (the
        # cf_gen_ep precedent — pinned to ENV episodes, advanced once
        # per episode). Pure bookkeeping when --class_replay is off;
        # eviction runs only under the flag.
        agent.buffer.gen_episode += 1
        if getattr(agent.buffer, "class_replay", False):
            agent.buffer.evict_crash_older_than(
                agent.buffer.gen_episode - agent.buffer.crash_age_cap)

    steps_done = step_i + 1
    time_to_violation = (steps_done * SEC_PER_STEP if violated
                         else maxstep * SEC_PER_STEP)
    stats_cf = (agent.cf_episode_stats()
                if train and getattr(agent, "cf_replay", False) else None)
    return {
        "steps": steps_done,
        "sim_seconds": steps_done * SEC_PER_STEP,
        "violated": violated,
        "violation_kind": violation_kind,
        "completed": completed,       # ROUND 3 A1 (always False flag-off)
        "time_to_violation": time_to_violation,
        "deliveries": n_deliveries,
        "commands": n_commands,
        "instruction_rate": n_commands / max(1, steps_done),
        "ep_return": round(ep_return, 4),
        "ep_return_disc": round(ep_return_disc, 4),
        "objective_v2": objective_v2,
        **{k: round(v, 4) for k, v in terms.items()},
        "mean_width": width_sum / max(1, width_n),
        "mean_loss": loss_sum / max(1, loss_n),
        **({"cf": stats_cf} if stats_cf is not None else {}),
        # ROUND 3 C2: per-episode tolerance-invoked selection counts
        # (key present only when --noop_tolerance is set)
        **({"noop_tolerance": agent.pop_tolerance_counts()}
           if getattr(agent, "noop_tolerance", None) is not None else {}),
        # ROUND 3 B2: per-class batch-composition telemetry (key present
        # only under --class_replay on the training path)
        **({"batch_kinds": agent.pop_kind_counts()}
           if train and getattr(agent, "class_replay", False) else {}),
    }


# ===========================================================================
# RUN 13 FIX C — COUNTERFACTUAL REPLAY (--cf_replay, default OFF)
# ===========================================================================

class CFBudget:
    """Banked step-equivalent budget for CF-replay (A2.2/A2.3): accrues
    per LIVE training step, spends on branch env steps (1.0 each) and
    deepcopies (CF_CLONE_COST = 1.5 each), and BANKS the balance across
    episodes — a per-episode cap makes CF inert at micro scale and the
    M1 gate unreachable (A2.3). can_afford() gates a whole state's
    estimated cost so a state's K rollouts are never half-funded."""

    def __init__(self):
        self.bank = 0.0
        self.spent_steps = 0
        self.spent_clones = 0

    def accrue(self, rate):
        self.bank += rate

    def charge_steps(self, n):
        self.bank -= n
        self.spent_steps += n

    def charge_clones(self, n):
        self.bank -= n * CF_CLONE_COST
        self.spent_clones += n

    def can_afford(self, est):
        return self.bank >= est


def cf_step_cost(cbp, cbp_lag):
    """(env_steps, deepcopies) one priced branch step consumes through
    priced_env_step: the branch env step itself plus the CBP branches'
    clone steps; deepcopies are charged separately at CF_CLONE_COST.
    Mirrors the live per-step CBP cost exactly (A2-corrected model)."""
    steps = 1
    clones = 0
    if cbp:
        steps += (1 + cbp_lag) + (cbp_lag if cbp_lag > 0 else 0)
        clones = 1 + (1 if cbp_lag > 0 else 0)
    return steps, clones


def cf_select_state(agent, stratum):
    """A5-CORRECTED state selection (stratum semantics NOT inverted:
    snapshot_stratum assigns 0 = <10 nm CONFLICT): stratum 0 always;
    stratum 1 (10-30 nm) at CF_STATE_P1; state-level random floor
    CF_STATE_FLOOR across ALL strata. Strata 1's band probability and
    the floor are folded into one combined draw from the DEDICATED CF
    RNG (A12.3) — stratum 0 consumes no draw."""
    if stratum == 0:
        return True
    p = CF_STATE_FLOOR if stratum == 2 else \
        1.0 - (1.0 - CF_STATE_P1) * (1.0 - CF_STATE_FLOOR)
    return agent.cf_rng.random() < p


def cf_before_entry_mask(env, cs_list, n_instr):
    """Candidate-layout bool mask (True = BEFORE_ENTRY target): a
    clearance to a BEFORE_ENTRY aircraft is a physical no-op (env
    ignores it, fee still paid) — excluded from the CF rollout pool
    (A13.2) and grounded analytically off the shared NOOP probe."""
    m = np.zeros(1 + n_instr * len(cs_list), dtype=bool)
    for i, cs in enumerate(cs_list):
        try:
            td = env.get_tracked_aircraft_data(cs)
        except Exception:
            continue
        if td is None or td.pos_status is None:
            continue
        if getattr(td.pos_status, "name",
                   str(td.pos_status)) == "BEFORE_ENTRY":
            m[1 + n_instr * i: 1 + n_instr * (i + 1)] = True
    return m


def cf_choose_candidates(agent, cl, cu, rmask, be_mask, stratum,
                         taken_idx, k):
    """Section-1 candidate rule 2 at a selected state: mandatory NOOP
    probe first (A1 — the rule never picks NOOP on its own), then
    (a) interval-overlap-with-argmax ambiguity, (b) lowest CF-side count
    cells (the SEPARATE cf_counts table, A12.3), (c) a candidate-level
    random floor CF_CAND_FLOOR. Excluded (A13.2): re-issue-masked and
    BEFORE_ENTRY-target candidates, plus the LIVE taken action (it
    receives a live windower target). Deterministic given cf_counts and
    the CF RNG. Returns (chosen incl. NOOP at [0], argmax idx,
    n_overlap)."""
    n_cand = len(cl)
    scores = cl + agent.c_train * (cu - cl)
    sc = np.where(rmask, -np.inf, scores)
    a_star = 0 if sc[0] == sc.max() else int(sc.argmax())
    l_s, u_s = cl[a_star], cu[a_star]
    eligible = [i for i in range(1, n_cand)
                if not rmask[i] and not be_mask[i] and i != taken_idx]
    overlap = {i for i in eligible
               if i != a_star and not (cu[i] < l_s or cl[i] > u_s)}

    def cell(i):
        return 0 if i == 0 else 1 + (i - 1) % agent.n_instr

    ranked = sorted(
        (i for i in eligible if i != a_star),
        key=lambda i: (0 if i in overlap else 1,
                       int(agent.cf_counts[stratum, cell(i)]),
                       -scores[i], i))
    chosen = [0] + ranked[:max(0, k - 1)]
    rest = ranked[max(0, k - 1):]
    if rest and agent.cf_rng.random() < CF_CAND_FLOOR:
        pick = agent.cf_rng.choice(rest)
        if len(chosen) < k:
            chosen.append(pick)
        else:
            chosen[-1] = pick          # floor replaces the weakest slot
    return chosen, a_star, len(overlap)


def cf_branch_rollout(env, obs, snap, last_issued, agent, first_idx, H,
                      cbp, cbp_lag, objective_v2, budget=None,
                      script=None):
    """One CF branch: deepcopy env AT s, apply the candidate, then run
    the FROZEN CURRENT POLICY greedy (epsilon 0, dedicated CF RNG — the
    net as of branch time; the rollout completes before any further
    update) for the remaining steps, pricing every step through
    priced_env_step — the SAME CBP code path live episodes use (A3).
    H = nstep (A4 matched depth). last_issued is deep-copied at branch
    time into the continuation policy's re-issue mask (A13.1). The
    branch DeliveryClock rides the env deepcopy — it was already
    observed AT s by the live loop, so branch step 0 must NOT observe
    again; every later branch step observes exactly once BEFORE
    stepping (M0's clock regression guard).

    script (M0 teacher-forcing ONLY): a list of (actions_dict,
    cand_idx) replacing both the candidate and the continuation policy
    — M0 replays a RECORDED live action sequence, never re-derives it
    (A4). The synthesized bootstrap fields (s_H tokens, ns_mask,
    a_cont) still come from the production code below, so M0's bitwise
    asserts exercise exactly the machinery training uses.

    Returns dict: g_cf, rewards, disc, ns, ns_mask, next_a, violated,
    steps, deliveries, event, clock_steps, final_clock, li0."""
    env_b = copy.deepcopy(env)
    if budget is not None:
        budget.charge_clones(1)
    li_b = copy.deepcopy(last_issued) if last_issued is not None else {}
    li0 = dict(li_b)
    mask_on = getattr(agent, "mask_reissue", False)
    clock = delivery_clock(env_b) if objective_v2 else None
    step_cost, step_clones = cf_step_cost(cbp, cbp_lag)
    obs_b, snap_b, info_b = obs, snap, None
    rewards = []
    clock_steps = []
    violated = False
    n_deliv = 0
    for kk in range(H):
        if kk > 0 and clock is not None:
            clock.observe(env_b, snap_b)
        if clock is not None:
            clock_steps.append(clock.step)
        if script is not None:
            # COPY the scripted dict: env.step mutates the action dict
            # it is given (the formatter injects entries for aircraft
            # that spawn during the step), and M0's script entries are
            # shared across overlapping windows — replaying a mutated
            # dict at an earlier state KeyErrors on the injected
            # callsign (found the hard way, 19 July 2026)
            actions, a_idx = dict(script[kk][0]), script[kk][1]
        elif kk == 0:
            a_idx = int(first_idx)
            cs_list = sorted(obs_b.keys())
            actions = {cs: 0 for cs in cs_list}
            if a_idx > 0:
                i_ac, j_in = divmod(a_idx - 1, agent.n_instr)
                actions[cs_list[i_ac]] = j_in + 1
        else:
            actions, aux_b = agent.generate_action(
                env_b, obs_b, info_b, force_epsilon=0.0,
                last_issued=li_b, rng=agent.cf_rng)
            a_idx = aux_b["cand_idx"]
        issued = a_idx != 0
        if mask_on and issued:
            cs_list = sorted(obs_b.keys())
            i_ac, j_in = divmod(a_idx - 1, agent.n_instr)
            li_b[cs_list[i_ac]] = j_in
        out = priced_env_step(env_b, obs_b, snap_b, actions, issued,
                              agent.gamma, cbp, cbp_lag, clock,
                              objective_v2)
        if budget is not None:
            budget.charge_steps(step_cost)
            if step_clones:
                budget.charge_clones(step_clones)
        rewards.append(out["r"])
        n_deliv += out["deliveries"]
        obs_b, snap_b, info_b = out["next_obs"], out["snap_next"], \
            out["info"]
        if out["violated"]:
            violated = True
            break
    # windower-identical accumulation (bitwise: same expression form as
    # ControllerWindower._window_return)
    g_cf = sum(agent.gamma ** k2 * r2 for k2, r2 in enumerate(rewards))
    if violated:
        # terminal collapse: same semantics as the live windower —
        # empty next state, NOOP-only mask, disc 0 (bootstrap VANISHES)
        ns = empty_state(agent.token_dim)
        ns_mask = np.zeros(1, dtype=bool)
        disc, next_a = 0.0, 0
    else:
        ncs = tuple(sorted(obs_b.keys()))
        ntoks = build_tokens(env_b, obs_b, info_b, list(ncs))
        ns = (ntoks, ncs)
        ns_mask = build_reissue_mask(ncs, li_b, agent.n_instr) \
            if mask_on else np.zeros(1 + agent.n_instr * len(ncs),
                                     dtype=bool)
        disc = agent.gamma ** len(rewards)
        # a_cont(s_H): the continuation policy's ACTUAL action at s_H
        # (A1 Branch A bootstrap — no NOOP-degenerate tail). One more
        # net forward; no env step.
        if ncs:
            _acts, aux_h = agent.generate_action(
                env_b, obs_b, info_b, force_epsilon=0.0,
                last_issued=li_b, rng=agent.cf_rng)
            next_a = aux_h["cand_idx"]
        else:
            next_a = 0
    return {"g_cf": g_cf, "rewards": rewards, "disc": disc, "ns": ns,
            "ns_mask": ns_mask, "next_a": next_a, "violated": violated,
            "steps": len(rewards), "deliveries": n_deliv,
            "event": bool(violated or n_deliv > 0),
            "clock_steps": clock_steps,
            "final_clock": (clock.step if clock is not None else None),
            "li0": li0}


def cf_process_state(agent, env, obs, info, snap, stratum, last_issued,
                     live_state, taken_idx, H, cbp, cbp_lag,
                     objective_v2, branch_step=None, ep_maxstep=None):
    """CF-replay state hook, called by run_episode AT s BEFORE the live
    last_issued update and the live step. Applies the A5-corrected
    selection rule, the Section-1 candidate rule with A13.2 exclusions,
    runs every chosen candidate through cf_branch_rollout, and banks
    the A13.2 analytic BEFORE_ENTRY targets off the shared NOOP probe.
    Returns a debug record (selftests read it) or None when the state
    is skipped. NEVER touches live RNG, the live count table or the
    live env (A12.3; asserted by the isolation selftest).

    branch_step/ep_maxstep (ROUND 3 B1, threaded by run_episode): the
    live step index the branch would start at and the episode's ENV
    time limit. Under --cf_exclude_limit, states whose H-step branch
    horizon crosses the limit (branch_step + H > ep_maxstep) are
    EXCLUDED entirely — not capped: both cap semantics are biased for a
    time-blind net (terminal-cap teaches near-limit geometry is free;
    censored-cap re-imports the doom tail) per ROUND3_REVIEW B1. The
    check sits BEFORE cf_select_state so excluded states consume no CF
    RNG draw; excluded late states fall back on censored live windows
    (mechanism i) until A4 — the pre-registered coverage cost is H/T of
    the episode's states (~36% at nstep=36 on the run-13 micro T)."""
    bud = agent.cf_budget
    bud.accrue(agent.cf_budget_rate)
    if not obs:
        return None
    if getattr(agent, "cf_exclude_limit", False) \
            and branch_step is not None and ep_maxstep is not None \
            and branch_step + H > ep_maxstep:
        agent.cf_excluded_limit += 1
        return None
    if not cf_select_state(agent, stratum):
        return None
    tokens, cs_tuple = live_state
    cs_list = list(cs_tuple)
    n_instr = agent.n_instr
    n_cand = 1 + n_instr * len(cs_list)
    k = max(CF_K_FLOOR, agent.cf_k)
    step_cost, step_clones = cf_step_cost(cbp, cbp_lag)
    per_step = step_cost + step_clones * CF_CLONE_COST
    est = k * (H * per_step + CF_CLONE_COST)
    if not bud.can_afford(est):
        agent.cf_skipped_budget += 1
        return None
    # the LIVE masked set at s (A13.1) — exactly what generate_action
    # saw for this decision (empty last_issued => nothing masked)
    if agent.mask_reissue and last_issued:
        rmask = build_reissue_mask(cs_list, last_issued, n_instr)
    else:
        rmask = np.zeros(n_cand, dtype=bool)
    be_mask = cf_before_entry_mask(env, cs_list, n_instr)
    cl, cu = agent.candidate_q(tokens)
    chosen, a_star, n_overlap = cf_choose_candidates(
        agent, cl, cu, rmask, be_mask, stratum, taken_idx, k)
    agent.cf_states_selected += 1
    gen_ep = agent.cf_gen_ep
    g_list = []
    recs = {}
    for idx in chosen:
        rec = cf_branch_rollout(env, obs, snap, last_issued, agent, idx,
                                H, cbp, cbp_lag, objective_v2,
                                budget=bud)
        agent.buffer.push_cf(live_state, idx, rec["g_cf"], rec["ns"],
                             rec["ns_mask"], rec["disc"], stratum,
                             rec["next_a"], gen_ep,
                             terminal_kind=("crash" if rec["violated"]
                                            else "live"))
        jt = 0 if idx == 0 else 1 + (idx - 1) % n_instr
        agent.cf_counts[stratum, jt] += 1
        agent.cf_rollouts += 1
        agent.cf_items += 1
        agent.cf_event_flags.append(1.0 if rec["event"] else 0.0)
        g_list.append(rec["g_cf"])
        recs[idx] = rec
    # A3 instrumentation: per-state spread of G_cf across the K
    # candidates, bootstrap excluded — direct measure of ordering
    # content in the returns
    agent.cf_gcf_spreads.append(float(max(g_list) - min(g_list)))
    # A13.2: BEFORE_ENTRY candidates are physical no-ops — store the
    # analytically exact target G_cf(NOOP) - CMD_COST from the shared
    # NOOP probe (env ignores the clearance, only the step-0 fee
    # differs; env deterministic). Zero sim cost.
    noop_rec = recs[0]
    be_idxs = [int(i) for i in np.nonzero(be_mask & ~rmask)[0]]
    for bi in be_idxs:
        agent.buffer.push_cf(live_state, bi,
                             noop_rec["g_cf"] - CMD_COST,
                             noop_rec["ns"], noop_rec["ns_mask"],
                             noop_rec["disc"], stratum,
                             noop_rec["next_a"], gen_ep,
                             terminal_kind=("crash"
                                            if noop_rec["violated"]
                                            else "live"))
        agent.cf_items += 1
        agent.cf_analytic_items += 1
    return {"chosen": chosen, "a_star": a_star, "n_overlap": n_overlap,
            "step0_mask": rmask, "be_idxs": be_idxs, "recs": recs,
            "g_spread": float(max(g_list) - min(g_list))}


# ===========================================================================
# CHECKPOINTS
# ===========================================================================

def save_checkpoint(agent, path, episode, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "controller_frame": True,
        "q_net": agent.q_net.state_dict(),
        "token_dim": agent.token_dim,
        "n_instr": agent.n_instr,
        "c_train": agent.c_train,
        "gamma": agent.gamma,
        "episode": episode,
        "strat_t_values": [tr.t for tr in agent.trackers],
        "strat_coverages": [tr.coverage for tr in agent.trackers],
        "coverage": agent.coverage,
        "mask_reissue": agent.mask_reissue,
        "width_scalars": getattr(agent, "width_scalars", False),
        "e1_width": getattr(agent, "e1_width", False),
        "e1_lambda": getattr(agent, "e1_lambda", 0.5),
        "e1_floor": getattr(agent, "e1_floor", 30.0),
        "bootstrap_support": getattr(agent, "bootstrap_support", "full"),
        "explore_bonus": getattr(agent, "explore_bonus", "off"),
        "explore_bonus_scale": getattr(agent, "explore_bonus_scale", 0.5),
        "explore_bonus_halflife": getattr(agent, "explore_bonus_halflife",
                                          2000.0),
        "explore_counts": getattr(
            agent, "explore_counts",
            np.zeros((N_STRATA, 1 + agent.n_instr), np.int64)).tolist(),
        "tracker_tripwire": getattr(agent, "tracker_tripwire", True),
        # RUN 13 fix C provenance (informational; the cf pool itself is
        # NOT persisted — envs are unpicklable and cf targets are cheap
        # to regenerate relative to their staleness cap)
        "cf_replay": getattr(agent, "cf_replay", False),
        "cf_gen_ep": getattr(agent, "cf_gen_ep", 0),
        # ROUND 3 provenance (all default-OFF flags)
        "cf_exclude_limit": getattr(agent, "cf_exclude_limit", False),
        "class_replay": getattr(agent, "class_replay", False),
        "crash_age_cap": getattr(agent.buffer, "crash_age_cap",
                                 CRASH_AGE_CAP),
        "crash_floor": getattr(agent.buffer, "crash_floor",
                               CRASH_BATCH_FLOOR),
        "noop_tolerance": getattr(agent, "noop_tolerance", None),
        "cf_counts": getattr(
            agent, "cf_counts",
            np.zeros((N_STRATA, 1 + agent.n_instr), np.int64)).tolist(),
        "adaptive_w_mid": getattr(agent, "adaptive_w_mid", None),
        # D5: conflict-pricing mode persisted (amendment A). Replay
        # transitions priced under one mode must not train under the
        # other: fresh runs only; load_agent warns loudly on mismatch.
        "vertical_ramp": VERTICAL_RAMP,
        "delta_conflict": DELTA_CONFLICT,
        "reward_weights": {"alpha": ALPHA_PROGRESS, "beta": BETA_CENTRE,
                           "delta": DELTA_CONFLICT, "eps_fuel": EPS_FUEL,
                           "cmd_cost": CMD_COST,
                           "delivery_bonus": DELIVERY_BONUS,
                           "violation_penalty": VIOLATION_PENALTY},
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_agent(path, device="cpu", **kwargs):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    agent = ControllerAgent(token_dim=ckpt["token_dim"],
                            n_instr=ckpt.get("n_instr", N_INSTR),
                            c_train=ckpt.get("c_train", 0.5),
                            gamma=ckpt.get("gamma", GAMMA),
                            mask_reissue=ckpt.get("mask_reissue", True),
                            width_scalars=ckpt.get("width_scalars", False),
                            e1_width=ckpt.get("e1_width", False),
                            e1_lambda=ckpt.get("e1_lambda", 0.5),
                            e1_floor=ckpt.get("e1_floor", 30.0),
                            bootstrap_support=ckpt.get("bootstrap_support",
                                                       "full"),
                            explore_bonus=ckpt.get("explore_bonus", "off"),
                            explore_bonus_scale=ckpt.get(
                                "explore_bonus_scale", 0.5),
                            explore_bonus_halflife=ckpt.get(
                                "explore_bonus_halflife", 2000.0),
                            tracker_tripwire=ckpt.get("tracker_tripwire",
                                                      True),
                            cf_exclude_limit=ckpt.get("cf_exclude_limit",
                                                      False),
                            class_replay=ckpt.get("class_replay", False),
                            crash_age_cap=ckpt.get("crash_age_cap",
                                                   CRASH_AGE_CAP),
                            crash_floor=ckpt.get("crash_floor",
                                                 CRASH_BATCH_FLOOR),
                            noop_tolerance=ckpt.get("noop_tolerance",
                                                    None),
                            device=device, **kwargs)
    agent.q_net.load_state_dict(ckpt["q_net"])
    agent.target_net.load_state_dict(ckpt["q_net"])
    # D5 (amendment A): pricing mode rides the checkpoint. Legacy
    # checkpoints default to gate/0.2 (the pricing they were trained
    # under). A mismatch with the CURRENT module pricing is legal for
    # probes (they deliberately re-price old nets) but must never pass
    # silently — warn loudly; training resume across modes is refused by
    # policy (fresh runs only; no resume path exists in run_training).
    ckpt.setdefault("vertical_ramp", "gate")
    ckpt.setdefault("delta_conflict", 0.2)
    if (ckpt["vertical_ramp"] != VERTICAL_RAMP
            or ckpt["delta_conflict"] != DELTA_CONFLICT):
        print(f"  [load_agent] NOTE: checkpoint pricing "
              f"({ckpt['vertical_ramp']}, delta={ckpt['delta_conflict']}) "
              f"!= current module {conflict_pricing_str()} — evaluating "
              f"an old net under re-priced objective (intentional for "
              f"probes; do NOT resume training across modes)")
    if ckpt.get("explore_counts") is not None:
        ec = np.asarray(ckpt["explore_counts"], dtype=np.int64)
        assert ec.shape == agent.explore_counts.shape, (
            f"explore_counts shape {ec.shape} != "
            f"{agent.explore_counts.shape}")
        agent.explore_counts = ec
    if ckpt.get("cf_counts") is not None:
        cc = np.asarray(ckpt["cf_counts"], dtype=np.int64)
        if cc.shape == agent.cf_counts.shape:
            agent.cf_counts = cc
        agent.cf_gen_ep = int(ckpt.get("cf_gen_ep", 0))
    for tr, tv in zip(agent.trackers, ckpt.get("strat_t_values", [])):
        tr.t = tv
    if ckpt.get("adaptive_w_mid") is not None:
        agent.adaptive_w_mid = ckpt["adaptive_w_mid"]
    agent.total_env_steps = 10 ** 9   # past warmup: no epsilon at eval
    agent.q_net.eval()
    return agent, ckpt


# ===========================================================================
# TRAINING / EVAL HARNESS
# ===========================================================================

def run_training(args, smoke=False):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 100)
    tag = "SMOKE TEST" if smoke else "TRAINING"
    print(f"BLUEBIRD CONTROLLER INTERVAL DQN — {tag} (seed={args.seed}, "
          f"c_train={args.c_train}, duration={args.duration}s, "
          f"episodes={args.episodes}, gamma={args.gamma}, "
          f"nstep={args.nstep}, cbp={'ON' if args.cbp else 'OFF'} "
          f"(lag={args.cbp_lag}), "
          f"mask_reissue={'ON' if args.mask_reissue else 'OFF'}, "
          f"objective={'v2' if args.objective_v2 else 'v1'}, "
          f"e1_width={'ON' if args.e1_width else 'OFF'}"
          + (f" (lambda={args.e1_lambda}, floor={args.e1_floor})"
             if args.e1_width else "")
          + f", vertical_ramp={VERTICAL_RAMP}"
          + f", delta_conflict={DELTA_CONFLICT}"
          + f", bootstrap_support={args.bootstrap_support}"
          + f", explore_bonus={args.explore_bonus}"
          + (f" (scale={args.explore_bonus_scale}, "
             f"halflife={args.explore_bonus_halflife})"
             if args.explore_bonus != "off" else "")
          + f", tripwire={'ON' if args.tracker_tripwire else 'OFF'}"
          + f", cf_replay={'ON' if args.cf_replay else 'OFF'}"
          + (f" (k={args.cf_k}, rate={args.cf_budget_rate}, "
             f"boost={args.cf_boost}, max_frac={args.cf_max_frac}, "
             f"age_cap={args.cf_age_cap}, H_cf=nstep={args.nstep})"
             if args.cf_replay else "")
          + f", cf_exclude_limit={'ON' if args.cf_exclude_limit else 'OFF'}"
          + f", completion_terminal="
            f"{'ON' if args.completion_terminal else 'OFF'}"
          + f", class_replay={'ON' if args.class_replay else 'OFF'}"
          + (f" (crash_age_cap={args.crash_age_cap}, "
             f"crash_floor={args.crash_floor})"
             if args.class_replay else "")
          + f", noop_tolerance={args.noop_tolerance}" + ")")
    print("=" * 100)

    print("Creating environment...")
    env = make_controller_env(scenario_duration=args.duration,
                              k_nearest=args.k)
    obs, info = env.reset(seed=args.seed)
    obs_dim = int(next(iter(obs.values())).shape[0])
    token_dim = obs_dim + KIN_FEATS
    # the agent's n_instr is DERIVED from the env's action parser (one
    # NOOP + n_instr per-aircraft instructions), so the network and the
    # candidate encoding always match the env; the assert pins the
    # module-level layout contract (checkpoints persist their own
    # n_instr for backward compatibility).
    n_env_actions = int(env.get_action_parser().get_total_num_actions())
    n_instr = n_env_actions - 1
    assert n_env_actions == 1 + N_INSTR, (
        f"expected 1 NOOP + {N_INSTR} instructions, env has {n_env_actions}")
    print(f"obs dim: {obs_dim}, token dim: {token_dim}, "
          f"env actions: {env.get_action_parser().action_formatter_map}")

    agent = ControllerAgent(
        token_dim=token_dim, n_instr=n_instr, lr=args.lr, gamma=args.gamma,
        c_train=args.c_train, target_coverage=args.target_coverage,
        width_reg=args.width_reg, warmup_steps=args.warmup_steps,
        warmup_epsilon=args.warmup_epsilon, buffer_size=args.buffer,
        batch_size=args.batch, device=args.device,
        t_cap_risky=args.t_cap_risky, terminal_boost=args.terminal_boost,
        mask_reissue=args.mask_reissue, width_scalars=args.width_scalars,
        e1_width=args.e1_width, e1_lambda=args.e1_lambda,
        e1_floor=args.e1_floor, bootstrap_support=args.bootstrap_support,
        explore_bonus=args.explore_bonus,
        explore_bonus_scale=args.explore_bonus_scale,
        explore_bonus_halflife=args.explore_bonus_halflife,
        tracker_tripwire=args.tracker_tripwire,
        cf_replay=args.cf_replay, cf_k=args.cf_k,
        cf_boost=args.cf_boost, cf_max_frac=args.cf_max_frac,
        cf_age_cap=args.cf_age_cap,
        cf_budget_rate=args.cf_budget_rate, cf_seed=args.seed,
        cf_exclude_limit=args.cf_exclude_limit,
        class_replay=args.class_replay,
        crash_age_cap=args.crash_age_cap, crash_floor=args.crash_floor,
        noop_tolerance=args.noop_tolerance)
    print(f"network parameters: {agent.param_count()} (target < 100k)")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    prefix = "smoke" if smoke else "train"
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(CHECKPOINT_DIR,
                            f"{prefix}_seed{args.seed}_{run_tag}.jsonl")
    log_file = open(log_path, "w")
    print(f"Logging to {log_path}")
    print("-" * 100)

    total_steps = 0
    t_start = time.time()
    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        t0 = time.time()
        stats = run_episode(env, agent, seed=ep_seed, train=True,
                            nstep=args.nstep, cbp=args.cbp,
                            cbp_lag=args.cbp_lag,
                            objective_v2=args.objective_v2,
                            completion_terminal=args.completion_terminal)
        wall = time.time() - t0
        total_steps += stats["steps"]
        eps = agent._epsilon()

        record = {
            "episode": ep + 1,
            "seed": ep_seed,
            **stats,
            "buffer_size": len(agent.buffer),
            "coverage": round(agent.coverage, 4),
            "bootstrap_coverage": round(agent.bootstrap_coverage, 4),
            "realized_episodes": agent.realized_episodes,
            "censored_episodes": agent.censored_episodes,
            **({"completed_episodes": agent.completed_episodes}
               if args.completion_terminal else {}),
            "strat_coverage": [round(tr.coverage, 4)
                               for tr in agent.trackers],
            "strat_t": [round(tr.t, 4) for tr in agent.trackers],
            "strat_hits": [len(tr.hits) for tr in agent.trackers],
            "t": round(agent.t, 4),
            "epsilon": round(eps, 4),
            "c_train": agent.c_train,
            "steps_per_sec": round(stats["steps"] / max(1e-9, wall), 2),
            "wall_time_s": round(wall, 2),
            # fix 2 logging: per-instruction-type cumulative counts
            # (NOOP + INSTR_NAMES order), summed over strata
            "explore_counts_by_type":
                agent.explore_counts.sum(axis=0).tolist(),
        }
        if agent.bootstrap_support == "taken_noop":
            # fix 1 logging: restricted-argmax winner share + fallbacks
            record["winner_share_taken"] = round(
                agent.winner_share_taken, 4)
            record["noop_fallback_share"] = round(
                agent.noop_fallback_share, 4)
        if agent.tracker_tripwire:
            # fix 3: per-episode tripwire JSONL (pure instrumentation).
            # PRE-REGISTERED TRIGGER: if accuracy_proxy improves >= 30%
            # from its ep500 level while t stays pinned at cap for 1000+
            # episodes, the tracker-feed change (12D doc fix E+F) gets
            # revived.
            record["tripwire"] = {
                "strat_t": [round(tr.t, 4) for tr in agent.trackers],
                "accuracy_proxy": agent.pop_tripwire_proxy(),
                "strat_realized_cov": [round(tr.coverage, 4)
                                       for tr in agent.trackers],
            }
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

        vio = (f"VIOLATION[{stats['violation_kind']}]"
               f"@{stats['time_to_violation']}s"
               if stats["violated"]
               else f"clean({stats['time_to_violation']}s)")
        print(f"  Ep {ep+1:4d} | {vio:>38} | G:{stats['ep_return']:8.2f} "
              f"| del:{stats['deliveries']} cmd:{stats['commands']:3d} "
              f"({stats['instruction_rate']:.2f}/st) "
              f"| W:{stats['mean_width']:.3f} rCvg:{agent.coverage:.2f} "
              f"t:{agent.t:.3f} eps:{eps:.3f} | buf:{len(agent.buffer):6d} "
              f"loss:{stats['mean_loss']:.4f} "
              f"| {record['steps_per_sec']:.1f} st/s")

        if (ep + 1) % args.ckpt_every == 0:
            path = os.path.join(
                CHECKPOINT_DIR,
                f"{prefix}_seed{args.seed}_{run_tag}_ep{ep+1}.pt")
            save_checkpoint(agent, path, ep + 1, extra=_ckpt_extra(args))
            print(f"    Saved checkpoint: {path}")

    final_path = os.path.join(CHECKPOINT_DIR,
                              f"{prefix}_seed{args.seed}_{run_tag}_final.pt")
    save_checkpoint(agent, final_path, args.episodes,
                    extra=_ckpt_extra(args))
    elapsed = time.time() - t_start
    print("-" * 100)
    print(f"Training done: {args.episodes} episodes, {total_steps} env "
          f"steps, {elapsed:.0f}s ({total_steps / max(1e-9, elapsed):.1f} "
          f"steps/s incl. training). Final checkpoint: {final_path}")

    if args.final_eval_episodes > 0:
        print()
        print("=" * 100)
        print(f"FINAL EVALUATION ({args.final_eval_episodes} episodes per "
              f"c, duration {args.duration}s)")
        print("=" * 100)
        for c_val in args.c_eval:
            evaluate(env, agent, c=c_val,
                     n_episodes=args.final_eval_episodes,
                     base_seed=args.seed + 10000,
                     objective_v2=args.objective_v2)
    env.close()
    log_file.close()
    return agent


def _ckpt_extra(args):
    return {"k": args.k, "nstep": args.nstep,
            "duration": args.duration, "seed": args.seed,
            "cbp": args.cbp, "cbp_lag": args.cbp_lag,
            "mask_reissue": args.mask_reissue,
            "objective_v2": args.objective_v2,
            "cf_replay": args.cf_replay, "cf_k": args.cf_k,
            "cf_budget_rate": args.cf_budget_rate,
            "cf_boost": args.cf_boost, "cf_max_frac": args.cf_max_frac,
            "cf_age_cap": args.cf_age_cap,
            "cf_exclude_limit": args.cf_exclude_limit,
            "completion_terminal": args.completion_terminal,
            "class_replay": args.class_replay,
            "crash_age_cap": args.crash_age_cap,
            "crash_floor": args.crash_floor,
            "noop_tolerance": args.noop_tolerance}


def evaluate(env, agent, c, n_episodes=5, base_seed=10042, verbose=True,
             adaptive=False, objective_v2=True):
    """Headline metric: mean time-to-first-violation; plus deliveries and
    instruction economy."""
    if adaptive and not hasattr(agent, "adaptive_w_mid"):
        probe_tokens = []
        p_obs, p_info = env.reset(seed=base_seed - 1)
        for _ in range(int(getattr(env, "maxstep", 100))):
            if not p_obs:
                break
            a, aux = agent.generate_action(env, p_obs, p_info, c=0.0,
                                           force_epsilon=0.0)
            probe_tokens.append(aux["tokens"])
            p_obs, _, _, _, p_info = env.step(a)
        w_mid = agent.calibrate_w_mid(probe_tokens)
        if verbose:
            print(f"  adaptive c: w_mid calibrated to {w_mid:.3f} "
                  f"({len(probe_tokens)} probe steps)")
    ttvs, deliveries, rates, widths = [], [], [], []
    n_violated, kinds = 0, {}
    for i in range(n_episodes):
        stats = run_episode(env, agent, seed=base_seed + i, train=False,
                            c=c, adaptive=adaptive,
                            objective_v2=objective_v2)
        ttvs.append(stats["time_to_violation"])
        deliveries.append(stats["deliveries"])
        rates.append(stats["instruction_rate"])
        widths.append(stats["mean_width"])
        if stats["violated"]:
            n_violated += 1
            kinds[stats["violation_kind"]] = (
                kinds.get(stats["violation_kind"], 0) + 1)
    result = {
        "c": ("adaptive" if adaptive else c),
        "mean_time_to_violation": float(np.mean(ttvs)),
        "std_time_to_violation": float(np.std(ttvs)),
        "violation_rate": n_violated / n_episodes,
        "violation_kinds": kinds,
        "mean_deliveries": float(np.mean(deliveries)),
        "mean_instruction_rate": float(np.mean(rates)),
        "mean_width": float(np.mean(widths)),
        "n_episodes": n_episodes,
    }
    if verbose:
        c_label = "adap" if adaptive else c
        print(f"  c={c_label:<4} | time-to-violation: "
              f"{result['mean_time_to_violation']:7.1f}s "
              f"± {result['std_time_to_violation']:6.1f} | "
              f"violated: {n_violated}/{n_episodes} "
              f"{kinds if kinds else ''} | "
              f"deliveries:{result['mean_deliveries']:.1f} | "
              f"instr rate:{result['mean_instruction_rate']:.3f}/step | "
              f"W:{result['mean_width']:.3f}")
    return result


def run_eval_only(args):
    print(f"Loading checkpoint: {args.ckpt}")
    agent, ckpt = load_agent(args.ckpt, device=args.device)
    print(f"Loaded: token dim {ckpt['token_dim']}, trained "
          f"{ckpt.get('episode', '?')} episodes, "
          f"strat t={[round(t, 3) for t in ckpt.get('strat_t_values', [])]}")
    env = make_controller_env(scenario_duration=args.duration,
                              k_nearest=ckpt.get("k", 3))
    eval_seed = args.seed + 10000
    print("=" * 100)
    print(f"EVALUATION (c={'adaptive' if args.adaptive else args.c}, "
          f"{args.eval_episodes} episodes, base seed {eval_seed})")
    print("=" * 100)
    result = evaluate(env, agent, c=args.c, n_episodes=args.eval_episodes,
                      base_seed=eval_seed, adaptive=args.adaptive,
                      objective_v2=args.objective_v2)
    env.close()
    return result


# ===========================================================================
# SELF-TESTS (mandatory: permutation invariance, padding, telescoping)
# ===========================================================================

def _selftest_permutation(seed=1234):
    """Shuffling aircraft order must change nothing: per-aircraft interval
    outputs permute with the tokens (<= 1e-6), the NOOP interval is
    unchanged, and the chosen clearance (callsign, instruction) is
    identical."""
    print("[selftest] permutation invariance")
    torch.manual_seed(seed)
    np.random.seed(seed)
    D, N = 60, 7
    net = ControllerQNet(D)
    net.eval()
    tokens = torch.randn(1, N, D)
    perm = torch.randperm(N)
    with torch.no_grad():
        l1, u1, nl1, nu1 = net(tokens)
        l2, u2, nl2, nu2 = net(tokens[:, perm])
    assert torch.allclose(l1[:, perm], l2, atol=1e-6), \
        f"FAIL: per-aircraft lower bounds not equivariant " \
        f"(max err {(l1[:, perm] - l2).abs().max():.2e})"
    assert torch.allclose(u1[:, perm], u2, atol=1e-6), \
        "FAIL: per-aircraft upper bounds not equivariant"
    assert torch.allclose(nl1, nl2, atol=1e-6) and \
        torch.allclose(nu1, nu2, atol=1e-6), \
        "FAIL: NOOP interval changed under aircraft permutation"

    # full selection level: same chosen (callsign, instruction) and the
    # same Q interval per (callsign, instruction) after shuffling
    agent = ControllerAgent(token_dim=D, device="cpu")
    agent.q_net.eval()
    cs = [f"AIR-{i:02d}" for i in range(N)]
    toks_np = tokens[0].numpy().astype(np.float32)
    cl1, cu1 = agent.candidate_q(toks_np)
    p = perm.numpy()
    cl2, cu2 = agent.candidate_q(toks_np[p])

    def chosen(cl, cu, cs_order, c=0.5):
        scores = cl + c * (cu - cl)
        best = scores.max()
        idx = 0 if scores[0] == best else int(scores.argmax())
        if idx == 0:
            return ("NOOP", None)
        i, j = divmod(idx - 1, N_INSTR)
        return (cs_order[i], j)

    cs_perm = [cs[i] for i in p]
    assert chosen(cl1, cu1, cs) == chosen(cl2, cu2, cs_perm), \
        "FAIL: chosen clearance changed under aircraft permutation"
    for new_i, old_i in enumerate(p):
        for j in range(N_INSTR):
            a = cl1[1 + N_INSTR * old_i + j]
            b = cl2[1 + N_INSTR * new_i + j]
            assert abs(a - b) <= 1e-6, \
                f"FAIL: Q lower for {cs[old_i]} instr {j} moved " \
                f"({a} vs {b})"
    print(f"  OK  permutation invariance (N={N}, max head err "
          f"{(l1[:, perm] - l2).abs().max():.2e})")


def _selftest_padding(seed=4321):
    """Batching episodes with different N via padding+mask must give
    IDENTICAL per-aircraft outputs to unbatched forwards — including when
    the pad slots contain garbage instead of zeros."""
    print("[selftest] padding equivalence")
    torch.manual_seed(seed)
    D = 60
    net = ControllerQNet(D)
    net.eval()
    t_small = torch.randn(3, D)
    t_big = torch.randn(8, D)
    t_empty = torch.zeros(0, D)   # terminal next-state row

    Nmax = 8
    tok = torch.full((3, Nmax, D), 99.0)   # garbage padding on purpose
    mask = torch.zeros(3, Nmax, dtype=torch.bool)
    tok[0, :3] = t_small
    mask[0, :3] = True
    tok[1, :8] = t_big
    mask[1, :8] = True
    # row 2 stays all-pad (empty state)

    with torch.no_grad():
        bl, bu, bnl, bnu = net(tok, mask)
        sl, su, snl, snu = net(t_small[None])
        gl, gu, gnl, gnu = net(t_big[None])
        el, eu, enl, enu = net(t_empty[None])

    for name, batched, single in [
            ("small lower", bl[0, :3], sl[0]),
            ("small upper", bu[0, :3], su[0]),
            ("big lower", bl[1, :8], gl[0]),
            ("big upper", bu[1, :8], gu[0])]:
        err = (batched - single).abs().max().item()
        assert err <= 1e-6, \
            f"FAIL: padded-batch {name} differs from unbatched by {err:.2e}"
    assert abs(bnl[0] - snl[0]) <= 1e-6 and abs(bnu[0] - snu[0]) <= 1e-6, \
        "FAIL: NOOP head (N=3 row) differs under padding"
    assert abs(bnl[1] - gnl[0]) <= 1e-6 and abs(bnu[1] - gnu[0]) <= 1e-6, \
        "FAIL: NOOP head (N=8 row) differs under padding"
    assert abs(bnl[2] - enl[0]) <= 1e-6 and abs(bnu[2] - enu[0]) <= 1e-6, \
        "FAIL: NOOP head (empty row) differs from empty-state forward"
    assert torch.isfinite(bl).all() and torch.isfinite(bnl).all(), \
        "FAIL: non-finite outputs in padded batch"
    print("  OK  padding equivalence (N=3/8/0 batched vs unbatched, "
          "garbage padding, <=1e-6)")


def _selftest_telescoping(seed=99):
    """Exact gamma-weighted telescoping of the potential terms.

    For any aircraft (or pair) contributing over matched steps a..b, the
    discounted sum of its shaping payments collapses exactly:
      sum_{t=a}^{b-1} gamma^t (gamma*phi_{t+1} - phi_t)
        = gamma^b phi_b - gamma^a phi_a.
    We run a synthetic 9-step sector with churn (entries, exits, a pair
    converging then diverging) through the REAL shaping_terms() and assert
    each term's discounted episode sum equals its analytic endpoint
    expression to <= 1e-9."""
    print("[selftest] gamma-weighted telescoping of Phi terms")
    rng = np.random.default_rng(seed)
    gamma = GAMMA
    T = 9

    def mk(cs, lat, lon, fl, d_exit, centre):
        return AcSnap(cs, lat, lon, fl, d_exit, centre)

    # synthetic trajectories: A present 0..8, B enters at 3, C exits at 5
    snaps = []
    for t in range(T + 1):
        snap = {}
        snap["A"] = mk("A", 50.5 + 0.02 * t, -3.5 + 0.05 * t, 250.0,
                       80.0 - 6.0 * t + float(rng.normal(0, 0.5)),
                       abs(float(rng.normal(2.0, 1.0))))
        if t >= 3:
            snap["B"] = mk("B", 50.5 + 0.02 * t, -3.55 + 0.048 * t, 255.0,
                           70.0 - 5.0 * (t - 3),
                           abs(float(rng.normal(1.0, 0.5))))
        if t <= 5:
            snap["C"] = mk("C", 51.2 - 0.03 * t, -2.8, 280.0,
                           60.0 - 7.0 * t, 0.5 * t)
        snaps.append(snap)

    S_prog = S_centre = S_conf = 0.0
    for t in range(T):
        prog, centre, conf = shaping_terms(snaps[t], snaps[t + 1], gamma)
        S_prog += gamma ** t * prog
        S_centre += gamma ** t * centre
        S_conf += gamma ** t * conf

    def presence(cs):
        """Maximal runs of consecutive matched steps for cs."""
        runs, start = [], None
        for t in range(T):
            here = cs in snaps[t] and cs in snaps[t + 1]
            if here and start is None:
                start = t
            if not here and start is not None:
                runs.append((start, t))
                start = None
        if start is not None:
            runs.append((start, T))
        return runs

    def endpoint_sum(phi_of_snap):
        total = 0.0
        for cs in {"A", "B", "C"}:
            for a, b in presence(cs):
                total += (gamma ** b) * phi_of_snap(snaps[b], cs) \
                         - (gamma ** a) * phi_of_snap(snaps[a], cs)
        return total

    E_prog = endpoint_sum(
        lambda s, cs: -ALPHA_PROGRESS * s[cs].dist_exit)
    E_centre = endpoint_sum(
        lambda s, cs: -BETA_CENTRE * s[cs].centre_off)

    # pair term endpoints: pairs of aircraft present together
    E_conf = 0.0
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    for x, y in pairs:
        runs, start = [], None
        for t in range(T):
            here = all(cs in snaps[t] and cs in snaps[t + 1]
                       for cs in (x, y))
            if here and start is None:
                start = t
            if not here and start is not None:
                runs.append((start, t))
                start = None
        if start is not None:
            runs.append((start, T))
        for a, b in runs:
            E_conf += (gamma ** b) * (
                -DELTA_CONFLICT * pair_conflict_f(snaps[b][x], snaps[b][y])) \
                - (gamma ** a) * (
                -DELTA_CONFLICT * pair_conflict_f(snaps[a][x], snaps[a][y]))

    for name, got, want in [("phi_progress", S_prog, E_prog),
                            ("phi_centre", S_centre, E_centre),
                            ("phi_conflict", S_conf, E_conf)]:
        assert abs(got - want) <= 1e-9, \
            f"FAIL: {name} telescoping broken: discounted sum {got!r} " \
            f"!= endpoint expression {want!r}"
        print(f"  OK  {name}: discounted sum {got:+.6f} == "
              f"gamma-weighted endpoints (err {abs(got - want):.1e})")


def _selftest_smooth_vertical(seed=1717):
    """D5 smooth vertical conflict potential — synthetic selftests
    (review amendments F1, F2 and the synthetic endangerment check F5).

      (F1) telescoping in SMOOTH mode with a CLIMBING aircraft crossing
           another's 20-FL boundary (ramp interior + old cliff region),
           sel != fl throughout the climb, and a MID-RUN selected-FL
           change (commanded factor jumps — a state change like any
           other; identity must hold to <= 1e-9), plus churn.
      (F2) round-trip pump bound: scripted climb-then-descend cycle;
           the two issuance-time CBP conflict payments are computed via
           pair_conflict_f AND via the closed form; assert they match to
           1e-9 and that |net| stays at the reviewer-computed bound
           ~0.010*f_lat (at DELTA=0.2) << 2*CMD_COST. NO anti-symmetry
           is claimed (the payments are NOT equal-and-opposite: g_cur
           geometry differs between the two issuance times).
      (F5) endangerment pays NEGATIVE at issuance:
           (a) descend at exactly-20-FL separation — priced negative via
               the CURRENT factor (the old gate also prices this sliver);
           (b) descend onto a CLIMBING intruder from > 20 FL current
               separation — priced negative purely via the COMMANDED
               factor while GATE mode pays exactly 0.0 (the blend's
               decisive justification).
      Also: gate-vs-smooth pointwise dominance on the old support
      (f_smooth <= f_gate when |dFL| < 20, sel == fl) and zero-set
      consistency spot checks.
    """
    print("[selftest] D5 smooth vertical potential (synthetic)")
    old_mode, old_delta = VERTICAL_RAMP, DELTA_CONFLICT
    try:
        set_conflict_pricing("smooth", 0.2)
        gamma = GAMMA
        rng = np.random.default_rng(seed)

        def mk(cs, lat, lon, fl, sel=None, d_exit=50.0, centre=1.0):
            return AcSnap(cs, lat, lon, fl, d_exit, centre, sel)

        # ---- (F1) telescoping with climb + sel_fl churn ---------------
        T = 12
        snaps = []
        for t in range(T + 1):
            snap = {}
            # A: level FL 250, sel == fl (default None -> falls back)
            snap["A"] = mk("A", 50.6 + 0.01 * t, -3.5 + 0.02 * t, 250.0)
            # B: climbing 2.5 FL/step from FL 226 across A's 20-FL
            # boundary (crosses |dFL|=20 at t~=1.6); sel jumps mid-run
            sel_b = 256.0 if t < 6 else 266.0   # mid-run selected-FL change
            snap["B"] = mk("B", 50.62 + 0.01 * t, -3.55 + 0.021 * t,
                           226.0 + 2.5 * t, sel_b)
            # C: churn — enters at t=4, level with explicit sel != fl
            if t >= 4:
                snap["C"] = mk("C", 50.58 + 0.012 * t, -3.42 + 0.019 * t,
                               244.0, 254.0)
            snaps.append(snap)
        S_conf = 0.0
        f_vals = []
        for t in range(T):
            _p, _c, conf = shaping_terms(snaps[t], snaps[t + 1], gamma)
            S_conf += gamma ** t * conf
            f_vals.append(pair_conflict_f(snaps[t]["A"], snaps[t]["B"]))
        E_conf = 0.0
        for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
            runs, start = [], None
            for t in range(T):
                here = all(cs in snaps[t] and cs in snaps[t + 1]
                           for cs in (x, y))
                if here and start is None:
                    start = t
                if not here and start is not None:
                    runs.append((start, t))
                    start = None
            if start is not None:
                runs.append((start, T))
            for a, b in runs:
                E_conf += (gamma ** b) * (-DELTA_CONFLICT
                                          * pair_conflict_f(snaps[b][x],
                                                            snaps[b][y])) \
                    - (gamma ** a) * (-DELTA_CONFLICT
                                      * pair_conflict_f(snaps[a][x],
                                                        snaps[a][y]))
        assert abs(S_conf - E_conf) <= 1e-9, \
            f"FAIL: smooth-mode telescoping broken ({S_conf!r} vs {E_conf!r})"
        assert len(set(round(v, 9) for v in f_vals)) > 3, \
            "FAIL: smooth f did not vary along the climb (vacuous test)"
        print(f"  OK  smooth telescoping with climb + mid-run sel change "
              f"(err {abs(S_conf - E_conf):.1e}; f varied over "
              f"[{min(f_vals):.4f}, {max(f_vals):.4f}])")

        # ---- (F2) round-trip pump bound -------------------------------
        # co-level pair ~6 nm apart; B climbs +10 (sel jumps at issuance,
        # fl flies 2.75/step), later descends back once level at 260.
        lat_a, lon_a = 50.60, -3.50
        lat_b, lon_b = 50.60, -3.345   # ~6 nm east at this latitude
        A = mk("A", lat_a, lon_a, 250.0, 250.0)
        d_nm = haversine_nm(lat_a, lon_a, lat_b, lon_b)
        f_lat = max(0.0, (CONFLICT_RANGE_NM - d_nm) / CONFLICT_RANGE_NM) ** 2
        ROCD = 2.75    # planning number, FL/step (percentile-sampled live)

        def g_vert(dcur, dcmd):
            gc = max(0.0, (CONFLICT_FL - dcur) / CONFLICT_FL) ** 2
            gm = max(0.0, (CONFLICT_FL_CMD - dcmd) / CONFLICT_FL_CMD) ** 2
            return (1 - VERT_CMD_WEIGHT) * gc + VERT_CMD_WEIGHT * gm

        def cbp_conf(b_noop, b_act):
            """One-time CBP conflict payment for pair (A, B):
            gamma*DELTA*(f_noop - f_act), via pair_conflict_f."""
            return GAMMA * DELTA_CONFLICT * (pair_conflict_f(A, b_noop)
                                             - pair_conflict_f(A, b_act))

        # climb issuance: measured state 1+lag steps on; noop branch
        # stays (250, 250), action branch is (252.75, 260)
        p_climb = cbp_conf(mk("B", lat_b, lon_b, 250.0, 250.0),
                           mk("B", lat_b, lon_b, 250.0 + ROCD, 260.0))
        p_climb_closed = GAMMA * DELTA_CONFLICT * f_lat * (
            g_vert(0.0, 0.0) - g_vert(ROCD, 10.0))
        assert abs(p_climb - p_climb_closed) <= 1e-9, \
            f"FAIL: climb payment {p_climb!r} != closed form " \
            f"{p_climb_closed!r}"
        assert p_climb > 0.0, "FAIL: co-level climb payment not positive"
        # descend-back issuance: B level at 260/sel 260; noop stays,
        # action branch is (257.25, 250)
        p_desc = cbp_conf(mk("B", lat_b, lon_b, 260.0, 260.0),
                          mk("B", lat_b, lon_b, 260.0 - ROCD, 250.0))
        p_desc_closed = GAMMA * DELTA_CONFLICT * f_lat * (
            g_vert(10.0, 10.0) - g_vert(10.0 - ROCD, 0.0))
        assert abs(p_desc - p_desc_closed) <= 1e-9, \
            f"FAIL: descend payment {p_desc!r} != closed form"
        assert p_desc < 0.0, "FAIL: descend-back payment not negative"
        net = p_climb + p_desc
        # reviewer-computed bound: ~0.010 * f_lat at DELTA=0.2; assert
        # |net| within 1.1x of it and << 2*CMD_COST (two fees paid).
        bound = 1.1 * 0.010 * f_lat * (DELTA_CONFLICT / 0.2)
        assert abs(net) <= bound, \
            f"FAIL: round-trip net {net!r} exceeds pump bound {bound!r}"
        assert abs(net) < 0.1 * (2 * CMD_COST), \
            f"FAIL: round-trip net {net!r} not << 2*CMD_COST"
        print(f"  OK  round-trip pump: climb {p_climb:+.5f}, descend "
              f"{p_desc:+.5f}, |net| {abs(net):.5f} <= bound {bound:.5f} "
              f"<< 2*CMD_COST {2 * CMD_COST} (no anti-symmetry claimed); "
              f"climb payment == closed-form p1 to 1e-9")

        # ---- (F5) endangerment pays negative at issuance --------------
        # (a) exactly-20-FL separation, both sel == fl; A descends
        B20 = mk("B", lat_b, lon_b, 250.0, 250.0)
        pay_a = GAMMA * DELTA_CONFLICT * (
            pair_conflict_f(mk("A", lat_a, lon_a, 270.0, 270.0), B20)
            - pair_conflict_f(mk("A", lat_a, lon_a, 270.0 - ROCD, 260.0),
                              B20))
        assert pay_a < 0.0, \
            f"FAIL: descend at 20-FL separation paid {pay_a!r}, not < 0"
        # (b) decisive: intruder B climbing (fl 250, sel 270) toward A
        # (fl 280); current separation stays > 20 in BOTH measured
        # branches, so only the commanded factor can price the descend.
        b_meas = mk("B", lat_b, lon_b, 250.0 + ROCD, 270.0)
        a_noop = mk("A", lat_a, lon_a, 280.0, 280.0)
        a_desc = mk("A", lat_a, lon_a, 280.0 - ROCD, 270.0)
        pay_b = GAMMA * DELTA_CONFLICT * (pair_conflict_f(a_noop, b_meas)
                                          - pair_conflict_f(a_desc, b_meas))
        assert pay_b < 0.0, \
            f"FAIL: descend onto climbing intruder paid {pay_b!r}, not < 0"
        # commanded factor is the only live channel in (b):
        assert abs(a_desc.fl - b_meas.fl) > CONFLICT_FL and \
            abs(a_noop.fl - b_meas.fl) > CONFLICT_FL, "test geometry broken"
        # ...and GATE mode prices (b) at exactly 0.0
        set_conflict_pricing("gate")
        gate_pay_b = GAMMA * DELTA_CONFLICT * (
            pair_conflict_f(a_noop, b_meas) - pair_conflict_f(a_desc, b_meas))
        assert gate_pay_b == 0.0, \
            f"FAIL: gate mode priced case (b) at {gate_pay_b!r}, not 0.0"
        set_conflict_pricing("smooth")
        print(f"  OK  endangerment: descend@20FL pays {pay_a:+.5f} < 0 "
              f"(current factor; old gate prices this sliver too); "
              f"descend onto climbing intruder pays {pay_b:+.5f} < 0 "
              f"purely via the COMMANDED factor while gate mode pays "
              f"exactly 0.0 — the blend's decisive justification")

        # ---- pointwise dominance + zero-set spot checks ----------------
        for _ in range(200):
            fl_a = 250.0 + float(rng.uniform(-25, 25))
            fl_b = 250.0 + float(rng.uniform(-25, 25))
            lon = -3.50 + float(rng.uniform(0.0, 0.45))
            x = mk("A", 50.6, -3.50, fl_a)          # sel == fl (None)
            y = mk("B", 50.6, lon, fl_b)
            set_conflict_pricing("smooth")
            fs = pair_conflict_f(x, y)
            set_conflict_pricing("gate")
            fg = pair_conflict_f(x, y)
            if abs(fl_a - fl_b) < CONFLICT_FL:
                assert fs <= fg + 1e-12, \
                    f"FAIL: smooth f {fs} > gate f {fg} on the old support"
            assert fs >= 0.0 and fg >= 0.0
        set_conflict_pricing("smooth")
        # zero-set: commanded-apart co-level pair keeps >= 0.5*g_cur
        z1 = pair_conflict_f(mk("A", 50.6, -3.50, 250.0, 250.0),
                             mk("B", 50.6, -3.42, 250.0, 260.0))
        z1_floor = 0.5 * max(0.0, (CONFLICT_FL - 0.0) / CONFLICT_FL) ** 2
        d_z = haversine_nm(50.6, -3.50, 50.6, -3.42)
        fl_z = max(0.0, (CONFLICT_RANGE_NM - d_z) / CONFLICT_RANGE_NM) ** 2
        assert abs(z1 - fl_z * z1_floor) <= 1e-12, \
            "FAIL: commanded-apart co-level pair not held at 0.5*g_cur"
        # fully separated-and-commanded-apart pair is exactly 0
        z0 = pair_conflict_f(mk("A", 50.6, -3.50, 250.0, 250.0),
                             mk("B", 50.6, -3.42, 275.0, 275.0))
        assert z0 == 0.0, f"FAIL: safe pair priced {z0!r}"
        print("  OK  pointwise f_smooth <= f_gate on the old support "
              "(200 draws); zero-set spot checks (honesty anchor + "
              "safe-pair zero)")
    finally:
        set_conflict_pricing(old_mode, old_delta)


def _selftest_windower():
    """Controller windower semantics (sanity port of the pilot tests):
    terminal collapse to realized tails, censored gamma^m bootstrap."""
    print("[selftest] controller n-step windows")

    class _Buf:
        def __init__(self):
            self.items = []

        def push(self, *a):
            self.items.append(a)

    gamma = 0.5
    D = 4
    sts = [(np.full((2, D), float(i), dtype=np.float32), (f"X{i}", f"Y{i}"))
           for i in range(6)]
    msk = [np.array([False] + [bool((i + k) % 3 == 0) for k in range(6)])
           for i in range(6)]   # distinct per-state candidate masks
    buf = _Buf()
    w = ControllerWindower(buf, gamma, 2)
    w.add(sts[0], 0, 1.0, sts[1], msk[1], False, 2)
    assert len(buf.items) == 0, "FAIL: pushed before any window completed"
    w.add(sts[1], 3, 2.0, sts[2], msk[2], False, 1)
    # window 0 completes here but is HELD for its next_a (fix D threading)
    assert len(buf.items) == 0 and w.held is not None, \
        "FAIL: completed window must be held one add() for next_a"
    w.add(sts[2], 1, 3.0, sts[3], msk[3], True, 0)   # violation
    # windows: (0: 1+0.5*2 bootstrap g^2, next_a = action taken at s2 = 1),
    # then terminal collapse of 1, 2 (next_a unused -> 0). ROUND 3 B2:
    # every row now carries terminal_kind (9th field) from the collapse
    # site — 'live' for the stamped bootstrap window, 'crash' for the
    # default (violation) terminal collapse.
    exp = [(sts[0], 0, 2.0, sts[2], msk[2], 0.25, 2, 1, "live"),
           (sts[1], 3, 2.0 + 0.5 * 3.0, None, None, 0.0, 1, 0, "crash"),
           (sts[2], 1, 3.0, None, None, 0.0, 0, 0, "crash")]
    assert len(buf.items) == 3, f"FAIL: {len(buf.items)} windows != 3"
    for i, ((s, a, r, ns, nsm, disc, st, nxa, knd),
            (es, ea, er, ens, enm, edisc, est, enxa, eknd)) \
            in enumerate(zip(buf.items, exp)):
        assert s is es and a == ea and st == est, f"FAIL: window {i} ids"
        assert abs(r - er) < 1e-12, f"FAIL: window {i} R {r} != {er}"
        assert abs(disc - edisc) < 1e-12, f"FAIL: window {i} disc"
        assert nxa == enxa, f"FAIL: window {i} next_a {nxa} != {enxa}"
        assert knd == eknd, f"FAIL: window {i} kind {knd} != {eknd}"
        if edisc == 0.0:
            assert ns[0].shape == (0, D) and ns[1] == (), \
                f"FAIL: window {i} terminal next-state not empty"
            assert nsm.shape == (1,) and not nsm.any(), \
                f"FAIL: window {i} terminal ns_mask not NOOP-only/unmasked"
        else:
            assert ns is ens, f"FAIL: window {i} next-state"
            assert nsm is enm, \
                f"FAIL: window {i} ns_mask does not ride with ns"
    buf2 = _Buf()
    w2 = ControllerWindower(buf2, gamma, 2)
    w2.add(sts[0], 0, 1.0, sts[1], msk[1], False, 2)
    w2.add(sts[1], 0, 2.0, sts[2], msk[2], False, 2)
    w2.flush_censored()
    assert len(buf2.items) == 2, \
        "FAIL: flush_censored must push the HELD window too (fix D " \
        "amendment), not silently drop it"
    assert abs(buf2.items[0][2] - 2.0) < 1e-12 \
        and buf2.items[0][5] == 0.25 and buf2.items[0][3] is sts[2] \
        and buf2.items[0][4] is msk[2]
    assert abs(buf2.items[1][2] - 2.0) < 1e-12 \
        and buf2.items[1][5] == 0.5 and buf2.items[1][3] is sts[2] \
        and buf2.items[1][4] is msk[2], \
        "FAIL: censored window must bootstrap at last ns (+its mask) " \
        "with disc gamma^1"
    assert buf2.items[0][7] == 0 and buf2.items[1][7] == 0 \
        and w2.noop_fallbacks == 2, \
        "FAIL: censored windows must carry the NOOP fallback next_a=0 " \
        "and be counted in noop_fallbacks"
    assert all(it[8] == "censored" for it in buf2.items), \
        "FAIL: flush_censored rows (incl. the held window) must carry " \
        "kind 'censored' (ROUND 3 B2 — B3's identifier)"
    print("  OK  terminal collapse + censored gamma^m bootstrap "
          "(+ ns_mask, next_a and terminal_kind threading, "
          "held-window flush)")

    # ---- n=12 synthetic (the 2026-07-09 default) --------------------
    # Brute-force cross-check of EVERY emitted window on a 17-step
    # trajectory ending in a violation, plus a shorter-than-n censored
    # trajectory, plus the terminal-boost pool interaction: one
    # violation must contribute exactly n=12 items to the boosted
    # terminal buffer.
    gamma12, n12, T12 = 0.9, NSTEP, 17
    assert n12 == 12, f"selftest assumes NSTEP == 12, got {n12}"
    sts12 = [(np.full((1, D), float(i), dtype=np.float32), (f"Z{i}",))
             for i in range(T12 + 1)]
    rs12 = [((-1.0) ** i) * (0.1 * i + 0.5) for i in range(T12)]
    buf3 = _Buf()
    w3 = ControllerWindower(buf3, gamma12, n12)
    for i in range(T12):
        w3.add(sts12[i], i % 4, rs12[i], sts12[i + 1], msk[0],
               i == T12 - 1, i % 3)
    assert len(buf3.items) == T12, \
        f"FAIL: n=12 emitted {len(buf3.items)} windows != {T12}"
    for i, (s, a, R, ns, nsm, disc, st, nxa, knd) in enumerate(buf3.items):
        assert s is sts12[i] and a == i % 4 and st == i % 3, \
            f"FAIL: n=12 window {i} ids"
        end = min(i + n12, T12)
        want = sum(gamma12 ** (k - i) * rs12[k] for k in range(i, end))
        assert abs(R - want) < 1e-9, \
            f"FAIL: n=12 window {i} R {R} != brute-force {want}"
        if i < T12 - n12:
            assert disc == gamma12 ** n12 and ns is sts12[i + n12], \
                f"FAIL: n=12 window {i} bootstrap (disc/ns)"
            assert nxa == (i + n12) % 4, \
                f"FAIL: n=12 window {i} next_a {nxa} != action taken at " \
                f"its bootstrap state ({(i + n12) % 4})"
            assert knd == "live", f"FAIL: n=12 window {i} kind {knd}"
        else:
            assert disc == 0.0 and ns[0].shape == (0, D), \
                f"FAIL: n=12 window {i} terminal collapse"
            assert nxa == 0, f"FAIL: n=12 terminal window {i} next_a != 0"
            assert knd == "crash", \
                f"FAIL: n=12 terminal window {i} kind {knd} != 'crash'"
    tiny = ControllerReplayBuffer(capacity=100, terminal_boost=3.0)
    assert tiny.reg_buf.maxlen > 0 and tiny.term_buf.maxlen > 0, \
        "FAIL: tiny-capacity split-pool buffer has an empty pool"
    rb = ControllerReplayBuffer(capacity=10000, terminal_boost=3.0)
    for it in buf3.items:
        rb.push(*it)
    assert len(rb.term_buf) == n12 and len(rb.reg_buf) == T12 - n12, \
        (f"FAIL: terminal-boost pools got {len(rb.term_buf)} terminal / "
         f"{len(rb.reg_buf)} regular items, expected {n12}/{T12 - n12} — "
         f"one violation must feed exactly n=12 boosted items")
    buf4 = _Buf()
    w4 = ControllerWindower(buf4, gamma12, n12)
    T_c = 5   # censored trajectory SHORTER than n
    for i in range(T_c):
        w4.add(sts12[i], 0, rs12[i], sts12[i + 1], msk[0], False, 1)
    w4.flush_censored()
    assert len(buf4.items) == T_c
    for i, (s, a, R, ns, nsm, disc, st, nxa, knd) in enumerate(buf4.items):
        want = sum(gamma12 ** (k - i) * rs12[k] for k in range(i, T_c))
        assert abs(R - want) < 1e-9 and ns is sts12[T_c] \
            and abs(disc - gamma12 ** (T_c - i)) < 1e-15 and nxa == 0 \
            and knd == "censored", \
            f"FAIL: n=12 censored window {i} (R/ns/disc/next_a/kind)"
    print(f"  OK  n={n12} brute-force: {T12}-step violation trajectory "
          f"(every window return, gamma^12 bootstraps, 12-item terminal "
          f"collapse -> boosted pool) + short censored gamma^m tail")


class _FakeSim:
    aircraft = {}


class _FakeEnv:
    """Env stand-in for build_tokens in synthetic selftests: sector bounds
    pre-cached, no aircraft kinematics (zeros appended)."""
    _ctrl_sector_bounds = (50.245, 51.542, -4.650, -2.230)

    def get_simulator_env(self):
        return _FakeSim()


def _selftest_mask(seed=777, n_instr=N_INSTR):
    """Re-issue masking semantics end-to-end (parametrized over n_instr;
    run at both the current N_INSTR=5 and the legacy 3):
      (1) mask bits: after issuing (i, L10), candidate (i, L10) is masked
          next step while every other (i, j) is not; after issuing
          (i, R10), (i, L10) unmasks; NOOP never masked;
      (2) selection: the argmax NEVER lands on a masked candidate, and
          equals the best UNMASKED candidate (checked by masking the
          current argmax); epsilon-sampling never draws a masked one;
      (3) target side, hand-built batch: the stored ns_mask bits divert
          the Double-DQN argmax away from the masked online-argmax
          candidate (padding validity handled independently)."""
    print(f"[selftest] re-issue masking (selection, epsilon, target side) "
          f"@ n_instr={n_instr}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    D, N = 60, 5
    agent = ControllerAgent(token_dim=D, n_instr=n_instr, device="cpu",
                            mask_reissue=True)
    agent.q_net.eval()
    cs_list = [f"AIR-{i:02d}" for i in range(N)]
    toks = np.random.randn(N, D).astype(np.float32)

    # (1) mask-bit semantics
    i_tgt = 2
    cs = cs_list[i_tgt]
    m = build_reissue_mask(cs_list, {cs: 0}, n_instr)   # issued (i, L10)
    assert not m[0], "FAIL: NOOP masked"
    assert m[1 + n_instr * i_tgt + 0], "FAIL: (i, L10) not masked after L10"
    assert not any(m[1 + n_instr * i_tgt + j] for j in range(1, n_instr)), \
        "FAIL: other (i, j) candidates wrongly masked"
    assert m.sum() == 1, "FAIL: unrelated candidates masked"
    m2 = build_reissue_mask(cs_list, {cs: 1}, n_instr)  # then (i, R10)
    assert not m2[1 + n_instr * i_tgt + 0], \
        "FAIL: (i, L10) still masked after intervening R10"
    assert m2[1 + n_instr * i_tgt + 1], "FAIL: (i, R10) not masked after R10"

    # (2) selection excludes masked argmax; picks best unmasked
    idx0, _, _ = agent.select_candidate(toks, c=0.5)
    cl, cu = agent.candidate_q(toks)
    scores = cl + 0.5 * (cu - cl)
    mask_arg = np.zeros(1 + n_instr * N, dtype=bool)
    if idx0 != 0:
        mask_arg[idx0] = True
    idx1, _, _ = agent.select_candidate(toks, c=0.5, reissue_mask=mask_arg)
    assert not mask_arg[idx1], "FAIL: selection returned a masked candidate"
    if idx0 != 0:
        s2 = np.where(mask_arg, -np.inf, scores)
        want = 0 if s2[0] == s2.max() else int(s2.argmax())
        assert idx1 == want, \
            f"FAIL: masked selection {idx1} != best unmasked {want}"
    # epsilon-sampling never draws a masked candidate (fake env: obs are
    # (D - KIN_FEATS)-dim; build_tokens appends the 6 kinematics zeros)
    fake_env = _FakeEnv()
    obs_dict = {c2: toks[k, :D - KIN_FEATS] for k, c2 in enumerate(cs_list)}
    li = {cs: 0}
    for _ in range(300):
        _a, aux = agent.generate_action(fake_env, obs_dict, None,
                                        force_epsilon=1.0, last_issued=li)
        assert aux["cand_idx"] != 1 + n_instr * i_tgt + 0, \
            "FAIL: epsilon-sampling drew a masked re-issue candidate"

    # (3) target side with a hand-built batch
    B = 3
    ntoks = [np.random.randn(n, D).astype(np.float32) for n in (5, 3, 5)]
    next_states = [(t, tuple(cs_list[:t.shape[0]])) for t in ntoks]
    ntok, nmask = pad_state_batch(next_states, agent.device)
    with torch.no_grad():
        o_l, o_u, o_nl, o_nu = agent.q_net(ntok, nmask)
        ocl, ocu, nvalid = candidate_intervals(o_l, o_u, o_nl, o_nu, nmask)
    na0 = agent.double_dqn_argmax(ocl, ocu, nvalid, None, agent.c_train)
    ns_masks = []
    for b in range(B):
        mm = np.zeros(1 + n_instr * ntoks[b].shape[0], dtype=bool)
        if int(na0[b]) != 0:
            mm[int(na0[b])] = True     # mask exactly the unmasked argmax
        ns_masks.append(mm)
    rmask = agent._pad_reissue_masks(ns_masks, ocl.shape[1])
    na1 = agent.double_dqn_argmax(ocl, ocu, nvalid, rmask, agent.c_train)
    nscore = (ocl + agent.c_train * (ocu - ocl)) \
        .masked_fill(~nvalid, -1e9).masked_fill(rmask, -1e9)
    for b in range(B):
        assert not ns_masks[b][int(na1[b])], \
            f"FAIL: target argmax row {b} landed on a masked candidate"
        assert bool(nvalid[b, int(na1[b])]), \
            f"FAIL: target argmax row {b} landed on a padded candidate"
        if int(na0[b]) != 0:
            assert int(na1[b]) != int(na0[b]), \
                f"FAIL: target argmax row {b} ignored the ns_mask"
        assert float(nscore[b, int(na1[b])]) == float(nscore[b].max()), \
            f"FAIL: target argmax row {b} not the best legal candidate"
    print(f"  OK  mask bits, unmask-on-new-instruction, masked "
          f"selection/epsilon (300 draws), target-side Double-DQN argmax "
          f"diverted on all {B} hand-built rows (n_instr={n_instr})")


def _selftest_delivery_bonus():
    """OBJECTIVE v2 delivery-bonus math on a synthetic clock:
      (1) nominal_T capture at entry: dist/speed -> steps (10 nm at
          600 kt = 1 nm/step -> nominal 10);
      (2) on-time delivery -> factor 1.0; late -> floored at 0.3;
          faster-than-nominal -> factor > 1 (uncapped, documented);
      (3) delayed capture: dist unavailable at entry is picked up later
          with elapsed steps added; speed never available -> flat 1.0;
      (4) deepcopy carries the clock (probe rollouts / CBP branches)."""
    print("[selftest] objective v2 delivery bonus (DeliveryClock)")

    class _Ac:
        def __init__(self, tas):
            self.speed_tas = tas

    class _Sim:
        def __init__(self, aircraft):
            self.aircraft = aircraft

    class _Env:
        def __init__(self, aircraft):
            self._sim = _Sim(aircraft)

        def get_simulator_env(self):
            return self._sim

    env = _Env({"A": _Ac(600.0), "B": _Ac(None), "C": _Ac(600.0)})

    def snap_of(**kw):
        # kw: cs -> dist_exit (None allowed)
        return {cs: AcSnap(cs, 51.0, -3.5, 250.0, d, 0.0)
                for cs, d in kw.items()}

    ck = reset_delivery_clock(env)
    assert delivery_clock(env) is ck, "FAIL: clock not env-attached"
    # step 0: A enters with 10 nm to exit at 600 kt (1 nm/step);
    # B enters with distance but NO speed; C enters with NO distance
    ck.observe(env, snap_of(A=10.0, B=10.0, C=None))
    assert ck.entry == {"A": 0, "B": 0, "C": 0}, f"FAIL: {ck.entry}"
    assert abs(ck.nominal["A"] - 10.0) < 1e-12, \
        f"FAIL: nominal_T {ck.nominal.get('A')} != 10.0 " \
        f"(10 nm / (600 kt * 6 s / 3600))"
    assert "B" not in ck.nominal and "C" not in ck.nominal
    # steps 1-2: C's route distance appears at step 3 -> nominal = 3
    # elapsed + 7 nm / (1 nm/step) = 10
    for _ in range(2):
        ck.observe(env, snap_of(A=9.0, B=9.0, C=None))
    ck.observe(env, snap_of(A=7.0, B=7.0, C=7.0))
    assert abs(ck.nominal["C"] - 10.0) < 1e-12, \
        f"FAIL: delayed-capture nominal {ck.nominal.get('C')} != 10.0"
    # advance to 10 observed steps total; deliveries observed after the
    # step that follows the 10th observe -> actual_T = 10 for entry at 0
    for _ in range(6):
        ck.observe(env, snap_of(A=1.0, B=1.0, C=1.0))
    assert ck.step == 10
    f_A = ck.bonus_factor("A")
    assert abs(f_A - 1.0) < 1e-12, f"FAIL: on-time factor {f_A} != 1.0"
    assert ck.bonus_factor("B") == 1.0, \
        "FAIL: no-speed aircraft must fall back to flat 1.0"
    # deepcopy (rollout/CBP branch) carries the history
    env2 = copy.deepcopy(env)
    assert delivery_clock(env2).entry == ck.entry \
        and delivery_clock(env2).nominal.keys() == ck.nominal.keys(), \
        "FAIL: deepcopy did not carry the delivery clock"
    # late delivery: floor at 0.3 (nominal 10, actual 40 -> 0.25 -> 0.3)
    for _ in range(30):
        ck.observe(env, snap_of(A=1.0))
    f_late = ck.bonus_factor("A")
    assert abs(f_late - DELIVERY_FLOOR) < 1e-12, \
        f"FAIL: late factor {f_late} != floor {DELIVERY_FLOOR}"
    # faster than nominal (tailwind): factor > 1, uncapped
    ck2 = DeliveryClock()
    ck2.observe(env, snap_of(A=10.0))
    for _ in range(7):
        ck2.observe(env, snap_of(A=1.0))
    f_fast = ck2.bonus_factor("A")
    assert abs(f_fast - 10.0 / 8.0) < 1e-12, \
        f"FAIL: fast factor {f_fast} != 1.25"
    # never-seen aircraft (paranoia): flat fallback
    assert ck2.bonus_factor("GHOST") == 1.0
    print(f"  OK  nominal capture (entry + delayed), on-time 1.0, late "
          f"floored {DELIVERY_FLOOR}, fast {f_fast:.2f} uncapped, "
          f"no-speed flat 1.0, clock rides deepcopies")


def _e1_synthetic_agent(seed, token_dim=12, **kw):
    """Deterministic small agent + replay for the E1 selftests: same seed
    => identical weights; buffer items are built by the caller."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return ControllerAgent(token_dim=token_dim, batch_size=16,
                           buffer_size=1000, device="cpu", **kw)


def _e1_synthetic_items(seed, n_items=40, token_dim=12):
    """Deterministic replay items (state, a_idx, R, next_state, ns_mask,
    disc, stratum, next_a_idx) with variable aircraft counts."""
    rng = np.random.RandomState(seed)
    items = []
    for _ in range(n_items):
        n = int(rng.randint(1, 6))
        nn_ = int(rng.randint(1, 6))
        s = (rng.randn(n, token_dim).astype(np.float32),
             tuple(f"A{k}" for k in range(n)))
        ns = (rng.randn(nn_, token_dim).astype(np.float32),
              tuple(f"B{k}" for k in range(nn_)))
        a_idx = int(rng.randint(0, 1 + N_INSTR * n))
        next_a = int(rng.randint(0, 1 + N_INSTR * nn_))
        items.append((s, a_idx, float(rng.randn()), ns, None,
                      GAMMA ** NSTEP, int(rng.randint(0, N_STRATA)),
                      next_a))
    return items


def _selftest_e1_default_off(seed=2026):
    """With e1_width False the train_step code path is EXACTLY the
    pre-E1 one: no extra RNG draws, no extra loss term. Two agents with
    identical weights/buffers — one default, one with junk e1_lambda/
    e1_floor but the flag off — must produce bitwise-identical losses,
    post-step weights and RNG states."""
    print("[selftest] E1 default OFF: train_step bitwise unchanged")
    a1 = _e1_synthetic_agent(seed)
    a2 = _e1_synthetic_agent(seed, e1_width=False, e1_lambda=123.0,
                             e1_floor=1.0)
    a2.q_net.load_state_dict(a1.q_net.state_dict())
    a2.target_net.load_state_dict(a1.target_net.state_dict())
    for it in _e1_synthetic_items(seed + 1):
        a1.buffer.push(*it)
        a2.buffer.push(*it)
    losses, rng_states = [], []
    for ag in (a1, a2):
        torch.manual_seed(seed + 2)
        np.random.seed(seed + 2)
        random.seed(seed + 2)
        losses.append(ag.train_step())
        rng_states.append(torch.random.get_rng_state())
    assert losses[0] == losses[1], \
        f"FAIL: flag-off loss differs bitwise ({losses[0]!r} vs {losses[1]!r})"
    assert torch.equal(rng_states[0], rng_states[1]), \
        "FAIL: flag-off train_step consumed extra torch RNG"
    for (k1, p1), (k2, p2) in zip(a1.q_net.state_dict().items(),
                                  a2.q_net.state_dict().items()):
        assert k1 == k2 and torch.equal(p1, p2), \
            f"FAIL: post-step weight {k1} differs bitwise with flag off"
    print(f"  OK  e1_width=False is inert (loss {losses[0]:.6f} bitwise "
          f"equal, all post-step weights bitwise equal)")


def _selftest_e1_negatives(seed=515):
    """build_e1_negatives: aircraft-swap rows are real rows from >=2
    distinct source states (never padding), counts come from the batch's
    real counts; field-shuffle preserves per-feature marginals exactly."""
    print("[selftest] E1 negative builder (aircraft-swap + field-shuffle)")
    torch.manual_seed(seed)
    B, Nmax, D = 8, 6, 5
    counts = [2, 3, 1, 4, 2, 3, 5, 2]
    tok = torch.full((B, Nmax, D), -777.0)   # padding sentinel
    mask = torch.zeros(B, Nmax, dtype=torch.bool)
    for s, n in enumerate(counts):
        for i in range(n):
            for d in range(D):
                tok[s, i, d] = 100 * s + 10 * i + d   # provenance encoding
        mask[s, :n] = True

    seen_swap_counts = set()
    for trial in range(50):
        neg_tok, neg_mask, info = build_e1_negatives(tok, mask)
        n_swap = info["n_swap"]
        assert n_swap == B // 2, f"FAIL: 50/50 split broken ({n_swap})"
        assert (-777.0 != neg_tok[neg_mask]).all(), \
            "FAIL: padding sentinel leaked into a real negative row"
        # swap half: every real row is an exact copy of a real source row
        for j in range(n_swap):
            rows = neg_tok[j][neg_mask[j]]
            n_j = rows.shape[0]
            srcs = set()
            for r in rows:
                v = int(r[0].item())
                s, i = divmod(v, 100)
                i //= 10
                assert bool(mask[s, i]) and torch.equal(r, tok[s, i]), \
                    f"FAIL: swap row is not a verbatim real source row"
                srcs.add(s)
            assert n_j in set(counts), \
                f"FAIL: swap count {n_j} not a batch real count"
            seen_swap_counts.add(n_j)
            assert srcs == set(info["swap_src"][j]), \
                "FAIL: swap_src provenance mismatch"
            if n_j >= 2:
                assert len(srcs) >= 2, \
                    f"FAIL: swap negative {j} drawn from ONE state"
        # field-shuffle half: same counts, exact per-feature marginals
        shuf = neg_tok[n_swap:]
        shuf_m = neg_mask[n_swap:]
        assert [int(m.sum()) for m in shuf_m] == counts[B // 2:], \
            "FAIL: field-shuffle counts changed"
        pool2 = torch.cat([tok[b][mask[b]] for b in range(B // 2, B)], 0)
        got = shuf[shuf_m]
        for d in range(D):
            assert torch.equal(got[:, d].sort().values,
                               pool2[:, d].sort().values), \
                f"FAIL: field-shuffle feature {d} marginal not preserved"
    assert len(seen_swap_counts) >= 3, \
        f"FAIL: swap count distribution degenerate ({seen_swap_counts})"
    print(f"  OK  negatives verbatim-real, swap always multi-source, "
          f"swap counts {sorted(seen_swap_counts)} from batch counts, "
          f"shuffle marginals exact (50 trials)")


def _selftest_e1_hinge(seed=808):
    """Bounded hinge: tiny widths => 0 < term <= 1 (so the loss term is
    capped at e1_lambda); widths >= floor => term EXACTLY 0 with an
    exactly-zero gradient (runaway impossible by construction)."""
    print("[selftest] E1 hinge boundedness + dead gradient at the floor")
    agent = _e1_synthetic_agent(seed, token_dim=6, e1_width=True,
                                e1_lambda=0.5, e1_floor=30.0)
    net = agent.q_net

    def force_raw(bias_val):
        with torch.no_grad():
            lin_i = net.instr_head[-1]
            lin_i.weight.zero_()
            lin_i.bias[0::2] = 0.0          # lower channels
            lin_i.bias[1::2] = bias_val     # delta_raw channels
            lin_n = net.noop_head[-1]
            lin_n.weight.zero_()
            lin_n.bias[0] = 0.0
            lin_n.bias[1] = bias_val

    tok = torch.randn(4, 5, 6)
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[0, 3:] = False

    force_raw(-30.0)                        # widths ~ 1e-6 << floor
    term = agent.e1_hinge(tok, mask)
    assert 0.0 < term.item() <= 1.0, \
        f"FAIL: tiny-width hinge term {term.item()} outside (0, 1]"
    assert agent.e1_lambda * term.item() <= agent.e1_lambda, \
        "FAIL: E1 loss term exceeds lambda"

    force_raw(200.0)                        # widths = 200 >= floor 30
    term = agent.e1_hinge(tok, mask)
    assert term.item() == 0.0, \
        f"FAIL: term {term.item()!r} not EXACTLY 0 with widths >= floor"
    net.zero_grad()
    (agent.e1_lambda * term).backward()
    for name, p in net.named_parameters():
        assert p.grad is None or float(p.grad.abs().max()) == 0.0, \
            f"FAIL: nonzero gradient through a satisfied hinge ({name})"
    print("  OK  hinge in (0, 1] under tiny widths, exactly 0 with zero "
          "gradient at/above the floor")


def _selftest_e1_no_contamination(seed=3033):
    """An E1-on train_step must differ from E1-off ONLY in the loss/
    gradient: coverage trackers (t values, hit histories), bootstrap-hit
    log and replay buffer are untouched by the negatives pass."""
    print("[selftest] E1 no-contamination (trackers, buffer, bootstrap log)")
    a_off = _e1_synthetic_agent(seed)
    a_on = _e1_synthetic_agent(seed, e1_width=True, e1_lambda=0.5,
                               e1_floor=30.0)
    a_on.q_net.load_state_dict(a_off.q_net.state_dict())
    a_on.target_net.load_state_dict(a_off.target_net.state_dict())
    for it in _e1_synthetic_items(seed + 1):
        a_off.buffer.push(*it)
        a_on.buffer.push(*it)
    n_before = len(a_off.buffer)
    for ag in (a_off, a_on):
        torch.manual_seed(seed + 2)
        np.random.seed(seed + 2)
        random.seed(seed + 2)
        loss = ag.train_step()
        assert np.isfinite(loss), "FAIL: non-finite train_step loss"
    assert [tr.t for tr in a_off.trackers] == \
           [tr.t for tr in a_on.trackers], \
        "FAIL: E1 changed a coverage tracker's t"
    assert [list(tr.hits) for tr in a_off.trackers] == \
           [list(tr.hits) for tr in a_on.trackers], \
        "FAIL: E1 fed hits into a coverage tracker"
    assert list(a_off.bootstrap_hits) == list(a_on.bootstrap_hits), \
        "FAIL: E1 negatives leaked into the bootstrap-coverage log"
    assert len(a_on.buffer) == len(a_off.buffer) == n_before, \
        "FAIL: train_step changed the replay buffer length"
    print("  OK  identical tracker t/hits, bootstrap log and buffer "
          "length with E1 on vs off")


def _selftest_e1_params():
    """E1 adds NO parameters: 59276 base at n_instr=5 (59016 at the
    legacy 3 — the vertical actions add 64*4+4=260 instr-head weights),
    +18 with width_scalars (+12 at n_instr=3), invariant to e1 flags."""
    print("[selftest] E1 parameter neutrality")
    torch.manual_seed(0)
    base = sum(p.numel() for p in ControllerQNet(60).parameters())
    ws = sum(p.numel() for p in
             ControllerQNet(60, width_scalars=True).parameters())
    assert base == 59276, f"FAIL: base param count {base} != 59276"
    assert ws == 59276 + 18, f"FAIL: width_scalars count {ws} != 59294"
    base3 = sum(p.numel() for p in
                ControllerQNet(60, n_instr=3).parameters())
    ws3 = sum(p.numel() for p in
              ControllerQNet(60, n_instr=3, width_scalars=True).parameters())
    assert base3 == 59016 and ws3 == 59016 + 12, \
        f"FAIL: legacy n_instr=3 counts moved ({base3}, {ws3})"
    a_on = _e1_synthetic_agent(1, token_dim=60, e1_width=True)
    a_off = _e1_synthetic_agent(1, token_dim=60)
    assert a_on.param_count() == a_off.param_count() == base, \
        "FAIL: e1 flags changed the parameter count"
    print(f"  OK  {base} params @5 / {base3} @3 (+18/+12 width_scalars), "
          f"unchanged by E1")


def _selftest_candidate_roundtrip(seed=1313):
    """Candidate encode/decode round-trip at BOTH n_instr=3 (legacy
    checkpoints) and n_instr=5 (vertical actions):
      (1) for every candidate idx, decode (i, j) = divmod(idx-1, n) and
          re-encode 1 + n*i + j == idx; the env action is j+1 for the
          chosen aircraft and 0 for all others;
      (2) generate_action produces exactly that action dict for a forced
          candidate (checked by masking every other candidate);
      (3) candidate_intervals column 1 + n*i + j carries aircraft i's
          instruction-j head output (flattening order contract)."""
    print("[selftest] candidate encode/decode round-trip @ n_instr=3 and 5")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    D, N = 60, 4
    fake_env = _FakeEnv()
    for n in (3, 5):
        agent = ControllerAgent(token_dim=D, n_instr=n, device="cpu")
        agent.q_net.eval()
        cs_list = [f"AIR-{i:02d}" for i in range(N)]
        toks = np.random.randn(N, D).astype(np.float32)
        n_cand = 1 + n * N
        # (1) pure index arithmetic
        for idx in range(1, n_cand):
            i, j = divmod(idx - 1, n)
            assert 0 <= i < N and 0 <= j < n and 1 + n * i + j == idx, \
                f"FAIL: round-trip broke at idx {idx} (n_instr={n})"
        # (2) generate_action realizes the decode: 300 epsilon=1.0 draws
        # must produce action dicts that match divmod-decoding of the
        # reported cand_idx (chosen aircraft gets j+1, everyone else 0),
        # and every candidate index must be legal
        obs_dict = {c2: toks[k, :D - KIN_FEATS]
                    for k, c2 in enumerate(cs_list)}
        seen = set()
        for _ in range(300):
            actions, aux = agent.generate_action(fake_env, obs_dict, None,
                                                 force_epsilon=1.0)
            idx = aux["cand_idx"]
            seen.add(idx)
            assert 0 <= idx < n_cand, f"FAIL: cand_idx {idx} out of range"
            if idx == 0:
                assert all(v == 0 for v in actions.values()), \
                    "FAIL: NOOP candidate issued a clearance"
            else:
                i, j = divmod(idx - 1, n)
                assert actions[aux["callsigns"][i]] == j + 1 and \
                    sum(v != 0 for v in actions.values()) == 1, \
                    f"FAIL: action dict wrong for candidate {idx} " \
                    f"(n_instr={n})"
        assert len(seen) == n_cand, \
            f"FAIL: epsilon draws covered {len(seen)}/{n_cand} candidates"
        # (3) flattening order: candidate column == per-aircraft head cell
        with torch.no_grad():
            tok_t = torch.from_numpy(toks[None])
            mask_t = torch.ones(1, N, dtype=torch.bool)
            ac_l, ac_u, nl, nu = agent.q_net(tok_t, mask_t)
            cl, cu, _ = candidate_intervals(ac_l, ac_u, nl, nu, mask_t)
        for i in range(N):
            for j in range(n):
                assert float(cl[0, 1 + n * i + j]) == float(ac_l[0, i, j]) \
                    and float(cu[0, 1 + n * i + j]) == float(ac_u[0, i, j]), \
                    f"FAIL: flatten order broke at (i={i}, j={j}, n={n})"
        assert float(cl[0, 0]) == float(nl[0]), "FAIL: NOOP not column 0"
        print(f"  OK  n_instr={n}: {n_cand} candidates round-trip "
              f"(indices, epsilon-drawn action dicts, head-flattening "
              f"order)")


def _selftest_old_ckpt_compat(tmp_name="_selftest_ckpt_compat.pt"):
    """Backward compatibility with 3-instruction checkpoints:
      (1) synthetic: save a n_instr=3 agent through save_checkpoint,
          load through load_agent — n_instr must come back 3, the net
          must produce 1 + 3*N candidates and select_candidate must work;
      (2) real: if a pre-vertical train_*.pt exists in CHECKPOINT_DIR,
          load it and run a forward pass at its own token_dim (evaluates
          the actual 12b net under the new module constants)."""
    print("[selftest] old 3-instruction checkpoint compatibility")
    torch.manual_seed(4242)
    np.random.seed(4242)
    D, N = 60, 6
    agent3 = ControllerAgent(token_dim=D, n_instr=3, device="cpu")
    path = os.path.join(CHECKPOINT_DIR, tmp_name)
    save_checkpoint(agent3, path, episode=1)
    try:
        loaded, ckpt = load_agent(path, device="cpu")
        assert ckpt["n_instr"] == 3 and loaded.n_instr == 3, \
            f"FAIL: persisted n_instr {ckpt.get('n_instr')} != 3"
        assert loaded.q_net.n_instr == 3, "FAIL: net head width != 3"
        toks = np.random.randn(N, D).astype(np.float32)
        cl, cu = loaded.candidate_q(toks)
        assert cl.shape == (1 + 3 * N,), \
            f"FAIL: candidate vector {cl.shape} != (1 + 3*{N},)"
        assert np.all(cu >= cl), "FAIL: interval inversion in loaded net"
        idx, _, _ = loaded.select_candidate(toks, c=0.5)
        assert 0 <= idx < 1 + 3 * N, "FAIL: selection out of range"
        # weights round-trip bitwise
        for (k1, p1), (k2, p2) in zip(agent3.q_net.state_dict().items(),
                                      loaded.q_net.state_dict().items()):
            assert k1 == k2 and torch.equal(p1, p2), \
                f"FAIL: weight {k1} did not round-trip"
        print(f"  OK  synthetic n_instr=3 checkpoint: save/load round-trip, "
              f"{1 + 3 * N}-candidate forward + selection")
    finally:
        if os.path.exists(path):
            os.remove(path)
    import glob as _glob
    real = sorted(_glob.glob(os.path.join(CHECKPOINT_DIR, "train_*.pt")),
                  key=os.path.getmtime)
    if real:
        rp = real[-1]
        ragent, rckpt = load_agent(rp, device="cpu")
        rtoks = np.random.randn(5, rckpt["token_dim"]).astype(np.float32)
        rcl, rcu = ragent.candidate_q(rtoks)
        want = 1 + ragent.n_instr * 5
        assert rcl.shape == (want,) and np.all(rcu >= rcl), \
            f"FAIL: real ckpt forward broke ({rcl.shape} vs ({want},))"
        idx, _, _ = ragent.select_candidate(rtoks, c=0.0)
        assert 0 <= idx < want
        print(f"  OK  real checkpoint {os.path.basename(rp)} "
              f"(n_instr={ragent.n_instr}, episode "
              f"{rckpt.get('episode', '?')}) loads and evaluates")
    else:
        print("  --  no real train_*.pt found; synthetic check only")


def _selftest_cbp(seed=10043, n_steps=50):
    """CBP invariants on the REAL env and the REAL run_episode path:
      (1) clone side-effect-freeness: the live env fingerprint is
          unchanged by cbp_noop_snapshot's counterfactual step;
      (2) NOOP-zero invariant: over a 50-step all-NOOP episode run
          through run_episode(cbp=True), the CBP shaping term is EXACTLY
          0.0 (bitwise) at every step — both branches are the same
          determinized computation;
      (3) budget reconciliation (asserted inside run_episode) holds."""
    print(f"[selftest] CBP: clone side-effect-freeness + NOOP-zero "
          f"invariant over {n_steps} steps (builds a real env; ~2 min)")
    env = make_controller_env(scenario_duration=n_steps * SEC_PER_STEP)

    # (1) side-effect-freeness of the counterfactual step
    obs, info = env.reset(seed=seed)
    fp0 = env_fingerprint(env)
    _snap_cf = cbp_noop_snapshot(env, obs)
    fp1 = env_fingerprint(env)
    assert fp0 == fp1, ("FAIL: counterfactual NOOP step mutated the live "
                        "env (fingerprint changed)")
    print("  OK  live env fingerprint unchanged by counterfactual step")

    # (2) all-NOOP episode through run_episode itself, cbp=True
    class _NoopAgent:
        gamma = GAMMA
        n_instr = N_INSTR
        mask_reissue = False
        buffer = None

        def generate_action(self, env, obs_dict, info_dict, c=None,
                            force_epsilon=None, adaptive=False,
                            last_issued=None):
            aux = {"cand_idx": 0,
                   "tokens": np.zeros((0, 1), dtype=np.float32),
                   "callsigns": (), "mean_width": 0.0, "c_used": 0.0}
            return {cs: 0 for cs in obs_dict}, aux

    seen = []
    stats = run_episode(env, _NoopAgent(), seed=seed, train=False,
                        cbp=True, step_hook=seen.append)
    assert len(seen) >= min(n_steps, 40), \
        f"FAIL: only {len(seen)} steps observed"
    for h in seen:
        assert not h["issued"], "FAIL: NOOP shim issued a clearance"
        assert h["shaping_paid"] == 0.0, (
            f"FAIL: CBP shaping term at NOOP step {h['step']} is "
            f"{h['shaping_paid']!r}, not exactly 0.0 — clones are not "
            f"bit-faithful")
    assert stats["cbp_shaping"] == 0.0, \
        f"FAIL: episode CBP total {stats['cbp_shaping']!r} != 0.0"
    assert stats["objective_v2"] and stats["fuel"] == 0.0, (
        f"FAIL: objective v2 fuel term {stats['fuel']!r} != 0.0 — the "
        f"global fuel term must be REMOVED under v2")
    n_nonzero_phi = sum(1 for h in seen if h["phi_abs"] != 0.0)
    assert n_nonzero_phi > 0, ("FAIL: absolute phi is zero all episode — "
                               "the invariant test is vacuous")
    print(f"  OK  CBP shaping EXACTLY 0.0 at all {len(seen)} NOOP steps "
          f"(absolute phi nonzero at {n_nonzero_phi} of them); budget "
          f"reconciliation asserted inside run_episode")

    # (4) NON-DEGENERACY at the default lag: an actually-issued clearance
    # must produce a NONZERO CBP term at some issuing step (this is the
    # check the lag-0 construct fails: command latency makes it
    # identically zero; see CBP_LAG note).
    class _IssueAgent(_NoopAgent):
        def generate_action(self, env, obs_dict, info_dict, c=None,
                            force_epsilon=None, adaptive=False,
                            last_issued=None):
            actions = {cs: 0 for cs in obs_dict}
            tgt = None
            for cs in sorted(obs_dict):
                try:
                    td = env.get_tracked_aircraft_data(cs)
                except Exception:
                    continue
                if td is not None and td.pos_status is not None and \
                        getattr(td.pos_status, "name", "") == "IN_SECTOR":
                    tgt = cs
                    break
            aux = {"cand_idx": 0,
                   "tokens": np.zeros((0, 1), dtype=np.float32),
                   "callsigns": (), "mean_width": 0.0, "c_used": 0.0}
            if tgt is not None:
                actions[tgt] = 1                    # L10 every sweep
                aux = dict(aux, cand_idx=1)
            return actions, aux

    seen2 = []
    run_episode(env, _IssueAgent(), seed=seed, train=False,
                cbp=True, step_hook=seen2.append)
    issued_terms = [h["shaping_paid"] for h in seen2 if h["issued"]]
    noop_terms = [h["shaping_paid"] for h in seen2 if not h["issued"]]
    assert issued_terms, "FAIL: issuing shim never issued"
    assert all(v == 0.0 for v in noop_terms), \
        "FAIL: CBP term nonzero at a global-NOOP step of the issuing run"
    mx = max(abs(v) for v in issued_terms)
    assert mx > 0.0, (
        f"FAIL: CBP term is zero at ALL {len(issued_terms)} issuing steps "
        f"at lag {CBP_LAG} — the counterfactual is degenerate (this is "
        f"exactly the lag-0 pathology)")
    env.close()
    print(f"  OK  CBP non-degeneracy at lag {CBP_LAG}: issued clearances "
          f"produce nonzero terms (max |term| {mx:.2e} over "
          f"{len(issued_terms)} issuing steps; still exactly 0.0 at all "
          f"{len(noop_terms)} NOOP steps)")


def _selftest_d5_cbp_env(seed=10043, n_steps=78):
    """D5 on the REAL env + REAL run_episode CBP path (review amendments
    F3/F4):
      (F3) GATE-mode invariant, corrected per review AND then corrected
           once more by measurement (2026-07-18): the review predicted
           the conflict-term CBP component would be 0.0 EXACTLY at a
           climb decision step (vertical gate common-mode, "lateral
           geometry is common-mode"). MEASURED: 7.2e-6, NOT exactly
           zero — the climb's CAS->TAS speed-schedule change moves the
           climbed aircraft's LATERAL along-track position, and f_lat
           picks that drift up; the same mechanism the review licensed
           for the TOTAL ("small-but-nonzero") also leaks into the
           conflict term. What IS exactly invariant: the vertical gate
           indicator 1[|dFL|<20] is identical across both measured
           branches for every matched pair (asserted), so the entire
           gate-mode conflict component is lateral-drift leakage —
           asserted equal to the gate-formula recomputation to 1e-6
           relative and < 5% of the smooth-mode p1. total==0 is
           deliberately NOT asserted (per review).
      (F4) SMOOTH-mode: the same climb pays a POSITIVE conflict-term CBP
           at issuance, equal to the analytic p1 recomputed from the
           measured branch states with an INDEPENDENT inline formula
           (not pair_conflict_f) to 1e-6 relative.
    """
    print(f"[selftest] D5 gate/smooth CBP on the real env (seed {seed}; "
          f"builds a real env; ~2 min)")
    old_mode, old_delta = VERTICAL_RAMP, DELTA_CONFLICT
    env = make_controller_env(scenario_duration=n_steps * SEC_PER_STEP)
    try:
        set_conflict_pricing("gate", 0.2)
        # ---- scan: all-NOOP trace; pick the climb step/target ----------
        # earliest step with a vertically-proximate (<10 FL) pair inside
        # 13 nm; target = the HIGHER member (a +10 climb opens its gap
        # monotonically). On seed 10043 this is the violating co-level
        # pair AIR-00/AIR-02 entering ~12 nm at step ~71 (NOOP LoS at
        # step 76), so the climb is priced at genuinely threatening
        # geometry with a few steps of margin.
        obs, _info = env.reset(seed=seed)
        cand = None   # (d, step, target_cs)
        for step in range(n_steps - CBP_LAG - 4):
            snap = sector_snapshot(env, obs.keys())
            css = sorted(snap)
            for i in range(len(css)):
                for j in range(i + 1, len(css)):
                    a, b = snap[css[i]], snap[css[j]]
                    if abs(a.fl - b.fl) >= 10.0:
                        continue
                    d = haversine_nm(a.lat, a.lon, b.lat, b.lon)
                    if d < 13.0 and step >= 2 and \
                            (cand is None or d < cand[0]):
                        hi = a if a.fl >= b.fl else b
                        cand = (d, step, hi.cs)
            if cand is not None:
                break
            obs = env.step({cs: 0 for cs in obs})[0]
        assert cand is not None, \
            "FAIL: no vertically-proximate pair (<10 FL, <13 nm) found " \
            "in the scan window — pick another seed"
        d0, t_star, target = cand
        print(f"  climb probe: step {t_star}, target {target} "
              f"(pair sep {d0:.2f} nm, |dFL| < 10)")

        class _ClimbAgent:
            gamma = GAMMA
            n_instr = N_INSTR
            mask_reissue = False
            buffer = None

            def __init__(self):
                self.t = -1

            def generate_action(self, env_, obs_dict, info_dict, c=None,
                                force_epsilon=None, adaptive=False,
                                last_issued=None):
                self.t += 1
                actions = {cs: 0 for cs in obs_dict}
                aux = {"cand_idx": 0,
                       "tokens": np.zeros((0, 1), dtype=np.float32),
                       "callsigns": (), "mean_width": 0.0, "c_used": 0.0}
                if self.t == t_star and target in obs_dict:
                    actions[target] = 4          # simple_fl_climb +10
                    aux = dict(aux, cand_idx=1)
                return actions, aux

        def run_mode(mode):
            set_conflict_pricing(mode)
            hooks = []
            run_episode(env, _ClimbAgent(), seed=seed, train=False,
                        cbp=True, step_hook=hooks.append)
            h = next(h for h in hooks if h["step"] == t_star)
            assert h["issued"], "FAIL: climb was not issued at t_star"
            for o in hooks:
                if not o["issued"]:
                    assert o["shaping_paid"] == 0.0, \
                        f"FAIL: NOOP step {o['step']} paid " \
                        f"{o['shaping_paid']!r} ({mode})"
                assert abs((o["paid_progress"] + o["paid_centre"]
                            + o["paid_conflict"]) - o["shaping_paid"]) \
                    <= 1e-12, "FAIL: per-term components do not sum to " \
                              "shaping_paid"
            return h

        # ---- both modes through the REAL run_episode CBP path ----------
        h_gate = run_mode("gate")
        h_smooth = run_mode("smooth")
        assert h_smooth["paid_conflict"] > 0.0, (
            f"FAIL: smooth-mode conflict-term CBP at the climb step is "
            f"{h_smooth['paid_conflict']!r}, not > 0")
        # independent mirror: rebuild s, the action branch and the NOOP
        # branch by hand, price the conflict term with an inline formula
        obs_m, _ = env.reset(seed=seed)
        for _ in range(t_star):
            obs_m = env.step({cs: 0 for cs in obs_m})[0]
        snap_s = sector_snapshot(env, obs_m.keys())
        env_a = copy.deepcopy(env)
        acts = {cs: 0 for cs in obs_m}
        acts[target] = 4
        o_a = env_a.step(acts)[0]
        for _ in range(CBP_LAG):
            o_a = env_a.step({cs: 0 for cs in o_a})[0]
        snap_act = sector_snapshot(env_a, o_a.keys())
        env_n = copy.deepcopy(env)
        o_n = env_n.step({cs: 0 for cs in obs_m})[0]
        for _ in range(CBP_LAG):
            o_n = env_n.step({cs: 0 for cs in o_n})[0]
        snap_cf = sector_snapshot(env_n, o_n.keys())

        def f_inline(x, y, mode):
            dd = haversine_nm(x.lat, x.lon, y.lat, y.lon)
            m = max(0.0, (15.0 - dd) / 15.0)
            if mode == "gate":
                return (m * m) if abs(x.fl - y.fl) < 20.0 else 0.0
            gc = max(0.0, (20.0 - abs(x.fl - y.fl)) / 20.0) ** 2
            sx = x.sel_fl if x.sel_fl is not None else x.fl
            sy = y.sel_fl if y.sel_fl is not None else y.fl
            gm = max(0.0, (10.0 - abs(sx - sy)) / 10.0) ** 2
            return m * m * (0.5 * gc + 0.5 * gm)

        def conf_term(snap_t, snap_t1, mode):
            matched = [cs for cs in snap_t if cs in snap_t1]
            tot = 0.0
            for i in range(len(matched)):
                for j in range(i + 1, len(matched)):
                    x, y = matched[i], matched[j]
                    tot += GAMMA * (-0.2 * f_inline(snap_t1[x],
                                                    snap_t1[y], mode)) \
                        - (-0.2 * f_inline(snap_t[x], snap_t[y], mode))
            return tot

        # (F3, measurement-corrected) gate mode: the vertical gate is
        # EXACTLY common-mode across the branches...
        matched_a = [cs for cs in snap_s if cs in snap_act]
        matched_n = [cs for cs in snap_s if cs in snap_cf]
        assert set(matched_a) == set(matched_n), \
            "FAIL: branch matched sets differ — pick another step"
        for i in range(len(matched_a)):
            for j in range(i + 1, len(matched_a)):
                x, y = matched_a[i], matched_a[j]
                ga = abs(snap_act[x].fl - snap_act[y].fl) < 20.0
                gn = abs(snap_cf[x].fl - snap_cf[y].fl) < 20.0
                assert ga == gn, (
                    f"FAIL: pair ({x},{y}) straddles the 20-FL gate "
                    f"between branches — the gate-invariance premise "
                    f"does not hold at this step")
        # ...so the whole gate-mode conflict component is lateral CAS
        # drift: equal to the gate-formula recomputation, and tiny
        # against the smooth p1.
        p_gate_formula = conf_term(snap_s, snap_act, "gate") \
            - conf_term(snap_s, snap_cf, "gate")
        rel_g = abs(h_gate["paid_conflict"] - p_gate_formula) \
            / max(abs(p_gate_formula), 1e-12)
        assert rel_g <= 1e-6, (
            f"FAIL: gate conflict-term CBP {h_gate['paid_conflict']!r} != "
            f"gate-formula recomputation {p_gate_formula!r} "
            f"(rel {rel_g:.2e})")
        assert abs(h_gate["paid_conflict"]) <= 1e-4 and \
            abs(h_gate["paid_conflict"]) < 0.05 * h_smooth["paid_conflict"], (
            f"FAIL: gate-mode conflict-term CBP "
            f"{h_gate['paid_conflict']!r} is not tiny vs the smooth p1 "
            f"{h_smooth['paid_conflict']!r} — vertical blindness "
            f"regression proof broke")
        print(f"  OK  gate mode: vertical gate exactly common-mode across "
              f"branches (all pairs); conflict-term CBP "
              f"{h_gate['paid_conflict']:+.2e} is pure lateral CAS-drift "
              f"leakage (== gate formula, rel {rel_g:.1e}; "
              f"{abs(h_gate['paid_conflict']) / h_smooth['paid_conflict'] * 100:.2f}% "
              f"of smooth p1) — MEASUREMENT CORRECTION to the review's "
              f"'exactly 0.0' claim; total shaping "
              f"{h_gate['shaping_paid']:+.2e} not asserted zero")

        # (F4) smooth mode == independent formula recomputation
        p1_formula = conf_term(snap_s, snap_act, "smooth") \
            - conf_term(snap_s, snap_cf, "smooth")
        rel = abs(h_smooth["paid_conflict"] - p1_formula) \
            / max(abs(p1_formula), 1e-12)
        assert rel <= 1e-6, (
            f"FAIL: smooth conflict-term CBP {h_smooth['paid_conflict']!r} "
            f"!= independent formula {p1_formula!r} (rel {rel:.2e})")
        # report the flown-FL planning number for this scenario
        tgt_dfl = abs(snap_act[target].fl - snap_s[target].fl) \
            if target in snap_act and target in snap_s else float("nan")
        print(f"  OK  smooth mode: conflict-term CBP "
              f"{h_smooth['paid_conflict']:+.5f} > 0 at the climb step, "
              f"== independent inline formula (rel err {rel:.1e}); "
              f"measured flown dFL at the measured state {tgt_dfl:.2f} "
              f"(planning number 2.75)")
    finally:
        set_conflict_pricing(old_mode, old_delta)
        env.close()


def _selftest_bootstrap_support(seed=6006):
    """Fix D (--bootstrap_support taken_noop):
      (1) restricted argmax on a hand-built batch picks ONLY from
          {NOOP, next_a}, equals the better of the two, resolves exact
          ties to NOOP, and still respects padding + re-issue masks;
      (2) full-mode default: taken_noop machinery is inert — train_step
          under 'full' is bitwise identical whether next_a fields carry
          real indices or zeros (targets never read them);
      (3) a taken_noop train_step populates the winner-share/fallback
          logs and never targets a candidate outside the allowed set;
      (4) legacy checkpoints (no new fields) load with full/off/tripwire
          defaults and zero counts."""
    print("[selftest] trained-support bootstrap (--bootstrap_support)")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    D, N = 60, 5
    agent = ControllerAgent(token_dim=D, device="cpu",
                            bootstrap_support="taken_noop")
    agent.q_net.eval()

    # (1) hand-built batch through restricted_support_argmax
    ntoks = [np.random.randn(n, D).astype(np.float32) for n in (5, 3, 4)]
    next_states = [(t, tuple(f"AIR-{i}" for i in range(t.shape[0])))
                   for t in ntoks]
    ntok, nmask = pad_state_batch(next_states, agent.device)
    with torch.no_grad():
        o_l, o_u, o_nl, o_nu = agent.q_net(ntok, nmask)
        ocl, ocu, nvalid = candidate_intervals(o_l, o_u, o_nl, o_nu, nmask)
    n_cands = [1 + N_INSTR * t.shape[0] for t in ntoks]
    for trial in range(50):
        next_as = torch.LongTensor(
            [random.randrange(nc) for nc in n_cands])
        na = agent.restricted_support_argmax(ocl, ocu, nvalid, None,
                                             agent.c_train, next_as)
        nscore = ocl + agent.c_train * (ocu - ocl)
        for b in range(len(ntoks)):
            got, nxa = int(na[b]), int(next_as[b])
            assert got in (0, nxa), \
                f"FAIL: restricted argmax {got} outside {{0, {nxa}}}"
            s0, sa = float(nscore[b, 0]), float(nscore[b, nxa])
            want = nxa if sa > s0 else 0     # ties -> NOOP (lower index)
            assert got == want, \
                f"FAIL: restricted argmax {got} != better-of-two {want}"
    # exact tie -> NOOP: duplicate NOOP score into the taken column
    tie_cl, tie_cu = ocl.clone(), ocu.clone()
    tie_cl[:, 3] = ocl[:, 0]
    tie_cu[:, 3] = ocu[:, 0]
    na_tie = agent.restricted_support_argmax(
        tie_cl, tie_cu, nvalid, None, agent.c_train,
        torch.LongTensor([3, 3, 3]))
    assert (na_tie == 0).all(), "FAIL: exact tie did not resolve to NOOP"
    # re-issue mask on the taken action forces NOOP
    rmask = torch.zeros_like(nvalid)
    rmask[:, 3] = True
    na_m = agent.restricted_support_argmax(
        ocl, ocu, nvalid, rmask, agent.c_train,
        torch.LongTensor([3, 3, 3]))
    assert (na_m == 0).all(), \
        "FAIL: re-issue-masked taken action must fall back to NOOP"

    # (2) 'full' ignores next_a bitwise
    items = _e1_synthetic_items(seed + 1)
    items_zeroed = [it[:7] + (0,) for it in items]
    losses, weights = [], []
    for variant in (items, items_zeroed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        ag = ControllerAgent(token_dim=12, batch_size=16,
                             buffer_size=1000, device="cpu")
        assert ag.bootstrap_support == "full", "FAIL: default not 'full'"
        for it in variant:
            ag.buffer.push(*it)
        torch.manual_seed(seed + 2)
        np.random.seed(seed + 2)
        random.seed(seed + 2)
        losses.append(ag.train_step())
        weights.append(ag.q_net.state_dict())
    assert losses[0] == losses[1], \
        "FAIL: 'full' train_step read the next_a field"
    for (k1, p1), (k2, p2) in zip(weights[0].items(), weights[1].items()):
        assert k1 == k2 and torch.equal(p1, p2), \
            f"FAIL: 'full' weights differ ({k1}) when next_a is zeroed"
    assert not ControllerAgent(token_dim=12, device="cpu"
                               ).bsupport_taken_wins, \
        "FAIL: winner-share log pre-populated"

    # (3) taken_noop train_step populates the winner-share logs
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    ag_tn = ControllerAgent(token_dim=12, batch_size=16, buffer_size=1000,
                            device="cpu", bootstrap_support="taken_noop")
    for it in items:
        ag_tn.buffer.push(*it)
    loss_tn = ag_tn.train_step()
    assert np.isfinite(loss_tn), "FAIL: taken_noop loss non-finite"
    assert len(ag_tn.bsupport_fallbacks) > 0, \
        "FAIL: fallback log empty after a taken_noop train_step"
    assert 0.0 <= ag_tn.winner_share_taken <= 1.0 or \
        len(ag_tn.bsupport_taken_wins) == 0, "FAIL: winner share invalid"

    # (4) legacy checkpoint (new fields stripped) loads with defaults
    path = os.path.join(CHECKPOINT_DIR, "_selftest_bsupport_legacy.pt")
    save_checkpoint(agent, path, episode=1)
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        for k in ("bootstrap_support", "explore_bonus",
                  "explore_bonus_scale", "explore_bonus_halflife",
                  "explore_counts", "tracker_tripwire"):
            ck.pop(k, None)
        torch.save(ck, path)
        legacy, ck2 = load_agent(path, device="cpu")
        assert legacy.bootstrap_support == "full" \
            and legacy.explore_bonus == "off" \
            and legacy.tracker_tripwire is True \
            and int(legacy.explore_counts.sum()) == 0, \
            "FAIL: legacy checkpoint did not default to " \
            "full/off/tripwire-on/zero-counts"
        # and a NEW-format checkpoint round-trips the mode
        save_checkpoint(agent, path, episode=1)
        again, _ = load_agent(path, device="cpu")
        assert again.bootstrap_support == "taken_noop", \
            "FAIL: bootstrap_support did not round-trip"
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("  OK  restricted argmax (subset/better-of-two/tie->NOOP/mask "
          "fallback, 50 trials), 'full' ignores next_a bitwise, "
          "winner-share logs live, legacy + new checkpoints load")


def _selftest_explore_bonus(seed=7007):
    """Fix 2 (--explore_bonus count):
      (1) bonus decays with counts: scale at 0, exactly scale/2 at
          halflife, monotone decreasing;
      (2) selection prefers an undersampled instruction type under a
          large scale, and a masked candidate is NEVER resurrected by
          the bonus; counts update on taken actions (incl. NOOP);
      (3) E1-pattern: same seed/weights/buffer, flag on vs off ->
          train_step losses, post-step weights, RNG states, trackers and
          buffer contents bitwise identical (bonus is SELECTION ONLY);
      (4) counts persist through save/load."""
    print("[selftest] count-based exploration bonus (--explore_bonus)")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    D, N = 60, 4
    agent = ControllerAgent(token_dim=D, device="cpu",
                            explore_bonus="count",
                            explore_bonus_scale=0.5,
                            explore_bonus_halflife=2000.0)
    agent.q_net.eval()

    # (1) decay shape
    b0 = float(agent.explore_bonus_value(0))
    bh = float(agent.explore_bonus_value(2000))
    assert abs(b0 - 0.5) < 1e-12, f"FAIL: bonus at count 0 is {b0} != scale"
    assert abs(bh - 0.25) < 1e-12, \
        f"FAIL: bonus at halflife is {bh} != scale/2"
    bs = agent.explore_bonus_value(np.arange(0, 10001, 100))
    assert np.all(np.diff(bs) < 0), "FAIL: bonus not strictly decreasing"

    # (2) selection steering + mask supremacy
    toks = np.random.randn(N, D).astype(np.float32)
    stratum = 1
    big = ControllerAgent(token_dim=D, device="cpu",
                          explore_bonus="count",
                          explore_bonus_scale=1e4,
                          explore_bonus_halflife=2000.0)
    big.q_net.eval()
    big.explore_counts[stratum, :] = 10 ** 9     # everything exhausted...
    big.explore_counts[stratum, 1 + 2] = 0       # ...except type j=2
    idx, _, _ = big.select_candidate(toks, c=0.5, bonus_stratum=stratum)
    assert idx != 0 and (idx - 1) % big.n_instr == 2, \
        f"FAIL: huge-bonus selection {idx} not of the undersampled type"
    # masked candidates stay dead no matter the bonus
    rmask = np.zeros(1 + big.n_instr * N, dtype=bool)
    for i in range(N):
        rmask[1 + big.n_instr * i + 2] = True    # mask ALL of type j=2
    idx_m, _, _ = big.select_candidate(toks, c=0.5, reissue_mask=rmask,
                                       bonus_stratum=stratum)
    assert not rmask[idx_m], \
        "FAIL: bonus resurrected a re-issue-masked candidate"
    # counts update on the TAKEN action via generate_action(stratum=...)
    fake_env = _FakeEnv()
    obs_dict = {f"AIR-{k:02d}": toks[k, :D - KIN_FEATS] for k in range(N)}
    before = agent.explore_counts.copy()
    _a, aux = agent.generate_action(fake_env, obs_dict, None,
                                    force_epsilon=0.0, stratum=2)
    jt = 0 if aux["cand_idx"] == 0 else 1 + (aux["cand_idx"] - 1) \
        % agent.n_instr
    diff = agent.explore_counts - before
    assert diff.sum() == 1 and diff[2, jt] == 1, \
        f"FAIL: taken-action count not incremented at (2, {jt})"
    _a, _aux = agent.generate_action(fake_env, obs_dict, None,
                                     force_epsilon=0.0)   # no stratum
    assert (agent.explore_counts - before).sum() == 1, \
        "FAIL: counts moved without a stratum (eval path)"

    # (3) E1-pattern bitwise: selection-only, targets/replay untouched
    a_off = _e1_synthetic_agent(seed)
    a_on = _e1_synthetic_agent(seed, explore_bonus="count",
                               explore_bonus_scale=123.0,
                               explore_bonus_halflife=7.0)
    a_on.q_net.load_state_dict(a_off.q_net.state_dict())
    a_on.target_net.load_state_dict(a_off.target_net.state_dict())
    for it in _e1_synthetic_items(seed + 1):
        a_off.buffer.push(*it)
        a_on.buffer.push(*it)
    a_on.explore_counts[:, :] = 5   # nonzero table; must still be inert
    losses, rng_states = [], []
    for ag in (a_off, a_on):
        torch.manual_seed(seed + 2)
        np.random.seed(seed + 2)
        random.seed(seed + 2)
        losses.append(ag.train_step())
        rng_states.append(torch.random.get_rng_state())
    assert losses[0] == losses[1], \
        f"FAIL: explore_bonus changed train_step loss " \
        f"({losses[0]!r} vs {losses[1]!r})"
    assert torch.equal(rng_states[0], rng_states[1]), \
        "FAIL: explore_bonus consumed extra torch RNG in train_step"
    for (k1, p1), (k2, p2) in zip(a_off.q_net.state_dict().items(),
                                  a_on.q_net.state_dict().items()):
        assert k1 == k2 and torch.equal(p1, p2), \
            f"FAIL: post-step weight {k1} differs with bonus on"
    assert [tr.t for tr in a_off.trackers] == \
           [tr.t for tr in a_on.trackers] and \
           [list(tr.hits) for tr in a_off.trackers] == \
           [list(tr.hits) for tr in a_on.trackers], \
        "FAIL: explore_bonus touched a coverage tracker"
    assert len(a_off.buffer) == len(a_on.buffer), \
        "FAIL: explore_bonus changed the replay buffer"

    # (4) checkpoint round-trip of the counts table + params
    path = os.path.join(CHECKPOINT_DIR, "_selftest_explore_bonus.pt")
    agent.explore_counts[0, 0] = 17
    save_checkpoint(agent, path, episode=1)
    try:
        loaded, ck = load_agent(path, device="cpu")
        assert loaded.explore_bonus == "count" \
            and loaded.explore_bonus_scale == 0.5 \
            and loaded.explore_bonus_halflife == 2000.0 \
            and np.array_equal(loaded.explore_counts,
                               agent.explore_counts), \
            "FAIL: explore_bonus fields/counts did not round-trip"
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("  OK  decay (scale @0, scale/2 @halflife, monotone), "
          "selection steering + mask supremacy, taken-action counting, "
          "bitwise-inert train_step, checkpoint round-trip")


def _selftest_tripwire(seed=8008):
    """Fix 3 (--tracker_tripwire): pure instrumentation.
      (1) ON vs OFF train_step: losses, post-step weights, torch RNG,
          trackers and buffer bitwise identical (no behavior change);
      (2) ON accumulates one finite nonneg proxy per batch; pop returns
          the mean and clears; OFF accumulates nothing."""
    print("[selftest] tracker tripwire (pure instrumentation)")
    a_on = _e1_synthetic_agent(seed)                     # default ON
    a_off = _e1_synthetic_agent(seed, tracker_tripwire=False)
    assert a_on.tracker_tripwire and not a_off.tracker_tripwire
    a_off.q_net.load_state_dict(a_on.q_net.state_dict())
    a_off.target_net.load_state_dict(a_on.target_net.state_dict())
    for it in _e1_synthetic_items(seed + 1):
        a_on.buffer.push(*it)
        a_off.buffer.push(*it)
    losses, rng_states = [], []
    for ag in (a_on, a_off):
        torch.manual_seed(seed + 2)
        np.random.seed(seed + 2)
        random.seed(seed + 2)
        for _ in range(3):
            losses.append(ag.train_step())
        rng_states.append(torch.random.get_rng_state())
    assert losses[:3] == losses[3:], \
        f"FAIL: tripwire changed train_step losses ({losses})"
    assert torch.equal(rng_states[0], rng_states[1]), \
        "FAIL: tripwire consumed torch RNG"
    for (k1, p1), (k2, p2) in zip(a_on.q_net.state_dict().items(),
                                  a_off.q_net.state_dict().items()):
        assert k1 == k2 and torch.equal(p1, p2), \
            f"FAIL: post-step weight {k1} differs with tripwire on"
    assert [tr.t for tr in a_on.trackers] == \
           [tr.t for tr in a_off.trackers], \
        "FAIL: tripwire moved a tracker t"
    assert len(a_on._tripwire_devs) == 3, \
        f"FAIL: expected 3 accumulated proxies, got " \
        f"{len(a_on._tripwire_devs)}"
    assert all(np.isfinite(v) and v >= 0.0 for v in a_on._tripwire_devs), \
        "FAIL: non-finite/negative accuracy proxy"
    v = a_on.pop_tripwire_proxy()
    assert v is not None and np.isfinite(v) and not a_on._tripwire_devs, \
        "FAIL: pop_tripwire_proxy did not return-and-clear"
    assert a_on.pop_tripwire_proxy() is None, \
        "FAIL: empty accumulator must pop None"
    assert a_off._tripwire_devs == [], \
        "FAIL: tripwire OFF still accumulated proxies"
    # flag round-trips through checkpoints
    path = os.path.join(CHECKPOINT_DIR, "_selftest_tripwire.pt")
    save_checkpoint(a_off, path, episode=1)
    try:
        loaded, _ck = load_agent(path, device="cpu")
        assert loaded.tracker_tripwire is False, \
            "FAIL: tracker_tripwire=False did not round-trip"
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("  OK  bitwise-identical training on/off (3 steps), per-batch "
          "proxy accumulation + pop semantics, flag round-trip")


def _selftest_cf_flag_off(seed=9101):
    """RUN 13 fix C, no-contamination gate (deliverable 4a): with the cf
    pool empty — flag OFF, or flag ON before any cf item exists — the
    training path is BITWISE the pre-cf code:
      (1) train_step: losses, post-step weights, torch RNG and the live
          python-random stream identical between a default agent and a
          cf-configured agent (flag on, junk cf params, empty pool);
      (2) sampler: ControllerReplayBuffer.sample() with an empty cf
          pool draws the SAME items with the SAME RNG consumption as a
          literal reimplementation of the legacy two-pool sampler."""
    print("[selftest] cf-replay flag OFF: bitwise no-contamination")
    a1 = _e1_synthetic_agent(seed)
    a2 = _e1_synthetic_agent(seed, cf_replay=True, cf_k=9,
                             cf_budget_rate=123.0, cf_age_cap=1)
    a2.q_net.load_state_dict(a1.q_net.state_dict())
    a2.target_net.load_state_dict(a1.target_net.state_dict())
    for it in _e1_synthetic_items(seed + 1):
        a1.buffer.push(*it)
        a2.buffer.push(*it)
    losses, rng_t, rng_py = [], [], []
    for ag in (a1, a2):
        torch.manual_seed(seed + 2)
        np.random.seed(seed + 2)
        random.seed(seed + 2)
        losses.append(ag.train_step())
        rng_t.append(torch.random.get_rng_state())
        rng_py.append(random.getstate())
    assert losses[0] == losses[1], \
        f"FAIL: cf-off loss differs bitwise ({losses[0]!r} vs {losses[1]!r})"
    assert torch.equal(rng_t[0], rng_t[1]) and rng_py[0] == rng_py[1], \
        "FAIL: cf machinery consumed live RNG with an empty cf pool"
    for (k1, p1), (k2, p2) in zip(a1.q_net.state_dict().items(),
                                  a2.q_net.state_dict().items()):
        assert k1 == k2 and torch.equal(p1, p2), \
            f"FAIL: post-step weight {k1} differs with cf configured"
    assert list(a1.bootstrap_hits) == list(a2.bootstrap_hits) and \
        a1._tripwire_devs == a2._tripwire_devs, \
        "FAIL: cf machinery touched the live logs with an empty pool"
    assert not a2.cf_bootstrap_hits and not a2._cf_tripwire_devs, \
        "FAIL: cf_* series populated with no cf rows"

    # (2) sampler RNG parity against a legacy reimplementation
    rb = ControllerReplayBuffer(capacity=10000, terminal_boost=3.0)
    items = _e1_synthetic_items(seed + 3, n_items=60)
    for j, it in enumerate(items):
        rb.push(*(it[:5] + (0.0 if j % 5 == 0 else it[5],) + it[6:]))
    random.seed(seed + 4)
    b_new = rb.sample(24)
    s_new = random.getstate()
    assert rb.last_cf_mask is None, \
        "FAIL: empty-cf sample() set a cf mask"
    random.seed(seed + 4)
    n_term, n_reg = len(rb.term_buf), len(rb.reg_buf)
    share = n_term / max(1, n_term + n_reg)
    target = min(0.5, rb.terminal_boost * share)
    k_t = min(n_term, int(round(24 * target)))
    k_r = 24 - k_t
    if k_r > n_reg:
        k_r = n_reg
        k_t = min(n_term, 24 - k_r)
    b_leg = (random.sample(rb.term_buf, k_t)
             + random.sample(rb.reg_buf, k_r))
    assert [id(x) for x in b_new] == [id(x) for x in b_leg], \
        "FAIL: empty-cf sample() draws differ from the legacy sampler"
    assert random.getstate() == s_new, \
        "FAIL: empty-cf sample() consumed extra RNG vs legacy"
    print("  OK  train_step bitwise identical (loss/weights/RNG/logs), "
          "sampler draw-for-draw legacy with empty cf pool")


def _selftest_cf_budget_selection(seed=9202):
    """RUN 13 fix C budget + selection rules (synthetic):
      (1) CFBudget: banked accrual across 'episodes', deepcopies charged
          at CF_CLONE_COST = 1.5 step-equivalents (A2.2/A2.3);
      (2) cf_step_cost mirrors the live CBP per-step cost;
      (3) cf_select_state: stratum 0 (<10 nm CONFLICT — A5 semantics
          NOT inverted) always selected with NO RNG draw; stratum 1 at
          the combined band+floor rate; stratum 2 at the floor;
      (4) cf_choose_candidates: NOOP mandatory first; re-issue-masked,
          BEFORE_ENTRY and the live taken action never chosen;
          interval-overlap ambiguity ranks first, then lowest CF-side
          count cells; the candidate floor draws from the leftovers."""
    print("[selftest] cf-replay budget + state/candidate selection")
    b = CFBudget()
    b.accrue(10.0)
    b.accrue(10.0)                       # banked, never reset
    assert b.bank == 20.0
    b.charge_steps(4)
    b.charge_clones(2)                   # 2 * 1.5 = 3.0
    assert b.bank == 13.0 and b.spent_steps == 4 and b.spent_clones == 2
    assert b.can_afford(13.0) and not b.can_afford(13.1)
    assert cf_step_cost(True, 1) == (4, 2), "FAIL: cbp lag-1 step cost"
    assert cf_step_cost(True, 0) == (2, 1), "FAIL: cbp lag-0 step cost"
    assert cf_step_cost(False, 1) == (1, 0), "FAIL: no-cbp step cost"

    agent = _e1_synthetic_agent(seed)
    st0 = agent.cf_rng.getstate()
    assert cf_select_state(agent, 0) is True and \
        agent.cf_rng.getstate() == st0, \
        "FAIL: stratum 0 not always-selected draw-free"
    n = 20000
    f1 = sum(cf_select_state(agent, 1) for _ in range(n)) / n
    f2 = sum(cf_select_state(agent, 2) for _ in range(n)) / n
    p1 = 1.0 - (1.0 - CF_STATE_P1) * (1.0 - CF_STATE_FLOOR)
    assert abs(f1 - p1) < 0.02, f"FAIL: stratum-1 rate {f1} != ~{p1}"
    assert abs(f2 - CF_STATE_FLOOR) < 0.01, \
        f"FAIL: stratum-2 floor rate {f2} != ~{CF_STATE_FLOOR}"

    # (4) hand-built candidate geometry: N=4 aircraft, 21 candidates
    N = 4
    n_cand = 1 + N_INSTR * N
    cl = np.zeros(n_cand, dtype=np.float32)
    cu = np.ones(n_cand, dtype=np.float32)          # far below argmax
    cl[5], cu[5] = 10.0, 12.0                       # argmax (score 11.0)
    cl[3], cu[3] = 9.0, 11.0                        # overlaps (score 10.0)
    cl[7], cu[7] = 9.5, 11.5                        # overlaps (score 10.5)
    cl[9], cu[9] = 5.0, 6.0                         # high but no overlap
    rmask = np.zeros(n_cand, dtype=bool)
    rmask[4] = True
    be_mask = np.zeros(n_cand, dtype=bool)
    be_mask[11:16] = True                           # aircraft 2 BEFORE_ENTRY
    taken = 20
    agent.c_train = 0.5
    agent.cf_counts[:] = 0
    # deterministic ranking checks: pin the CF RNG to a seed whose
    # first draw does NOT fire the candidate floor
    s_nofire = next(s for s in range(1000)
                    if random.Random(s).random() >= CF_CAND_FLOOR)
    agent.cf_rng = random.Random(s_nofire)
    chosen, a_star, n_ov = cf_choose_candidates(
        agent, cl, cu, rmask, be_mask, 0, taken, 3)
    assert a_star == 5, f"FAIL: argmax {a_star} != 5"
    assert chosen[0] == 0, "FAIL: mandatory NOOP probe not first"
    assert set(chosen[1:]) == {3, 7}, \
        f"FAIL: overlap-ambiguous candidates not chosen ({chosen})"
    for bad in [4, 11, 12, 13, 14, 15, taken, 5]:
        assert bad not in chosen, \
            f"FAIL: excluded candidate {bad} was chosen"
    # count steering: k=4 -> the 3rd probe comes from the lowest
    # CF-side count cell among the non-overlap leftovers
    agent.cf_counts[0, :] = 100
    agent.cf_counts[0, 1 + (9 - 1) % N_INSTR] = 0   # cell of idx 9
    agent.cf_rng = random.Random(s_nofire)          # floor stays quiet
    chosen4, _, _ = cf_choose_candidates(
        agent, cl, cu, rmask, be_mask, 0, taken, 4)
    assert chosen4[:3] == [0, 3, 7] or set(chosen4[:3]) == {0, 3, 7}, \
        f"FAIL: overlap candidates lost priority ({chosen4})"
    assert (chosen4[3] - 1) % N_INSTR == (9 - 1) % N_INSTR, \
        f"FAIL: lowest-count cell not preferred ({chosen4})"
    # candidate floor: a seed whose first draw fires the floor
    s_fire = next(s for s in range(1000)
                  if random.Random(s).random() < CF_CAND_FLOOR)
    agent.cf_rng = random.Random(s_fire)
    chosen_f, _, _ = cf_choose_candidates(
        agent, cl, cu, rmask, be_mask, 0, taken, 3)
    assert chosen_f[0] == 0 and len(chosen_f) == 3, \
        f"FAIL: floor broke the NOOP slot or K ({chosen_f})"
    assert chosen_f[-1] not in (3, 7), \
        f"FAIL: floor did not draw from the leftovers ({chosen_f})"
    assert not rmask[chosen_f[-1]] and not be_mask[chosen_f[-1]] and \
        chosen_f[-1] != taken and chosen_f[-1] != 5, \
        f"FAIL: floor drew an excluded candidate ({chosen_f})"
    print(f"  OK  budget banking + 1.5x clone pricing, stratum rates "
          f"(s1 {f1:.3f}~{p1:.3f}, s2 {f2:.3f}~{CF_STATE_FLOOR}), "
          f"NOOP-first, exclusions, overlap/count ranking, floor")


def _selftest_cf_pool(seed=9303):
    """RUN 13 fix C cf pool + A14 sampler (synthetic):
      (1) push_cf stamps generation episodes; sample() draws cf rows at
          min(cf_max_frac, cf_boost * share), flags them via
          last_cf_mask (aligned), records sampled ages and increments
          per-item replay counts;
      (2) age eviction drops exactly the items older than the cap;
      (3) SEVERED pool (cf_sample_enabled False): draws and RNG
          consumption identical to a cf-free buffer (A12.3)."""
    print("[selftest] cf pool: boosted sampler, ages, eviction, sever")
    items = _e1_synthetic_items(seed, n_items=36)
    rb = ControllerReplayBuffer(capacity=5000, terminal_boost=3.0)
    for j, it in enumerate(items):
        rb.push(*(it[:5] + (0.0 if j % 6 == 0 else it[5],) + it[6:]))
    cf_items = _e1_synthetic_items(seed + 1, n_items=12)
    for g, it in enumerate(cf_items):
        # sentinel stratum 2 is irrelevant here; identity via object id
        rb.push_cf(*it[:8], gen_episode=g)
    n_live = len(rb.term_buf) + len(rb.reg_buf)
    share_cf = 12 / (12 + n_live)
    k_c_exp = int(round(24 * min(rb.cf_max_frac,
                                 rb.cf_boost * share_cf)))
    random.seed(seed + 2)
    batch = rb.sample(24)
    assert rb.last_cf_mask is not None and len(rb.last_cf_mask) == 24
    assert sum(rb.last_cf_mask) == k_c_exp, \
        f"FAIL: cf rows {sum(rb.last_cf_mask)} != expected {k_c_exp}"
    cf_ids = {id(e[0]) for e in rb.cf_buf}
    for it, is_cf in zip(batch, rb.last_cf_mask):
        assert (id(it) in cf_ids) == is_cf, \
            "FAIL: last_cf_mask misaligned with the drawn rows"
    assert len(rb.last_cf_ages) == k_c_exp and \
        all(0 <= a < 12 for a in rb.last_cf_ages), \
        "FAIL: sampled cf ages not recorded"
    assert sum(rb.cf_replay_counts()) == k_c_exp, \
        "FAIL: per-item replay counts not incremented"
    rb.sample(24)
    assert sum(rb.cf_replay_counts()) == 2 * k_c_exp, \
        "FAIL: replay counts must accumulate across samples"
    n_ev = rb.evict_cf_older_than(6)
    assert n_ev == 6 and len(rb.cf_buf) == 6 and \
        all(e[1] >= 6 for e in rb.cf_buf), \
        f"FAIL: age eviction wrong ({n_ev} evicted)"
    # (3) severed == cf-free, draw for draw
    rb2 = ControllerReplayBuffer(capacity=5000, terminal_boost=3.0)
    for j, it in enumerate(items):
        rb2.push(*(it[:5] + (0.0 if j % 6 == 0 else it[5],) + it[6:]))
    rb.cf_sample_enabled = False
    random.seed(seed + 3)
    b_sev = rb.sample(24)
    s_sev = random.getstate()
    random.seed(seed + 3)
    b_ref = rb2.sample(24)
    assert rb.last_cf_mask is None, "FAIL: severed sample flagged cf rows"
    assert all(x is y or x == y for x, y in zip(b_sev, b_ref)) and \
        random.getstate() == s_sev, \
        "FAIL: severed sampler differs from a cf-free buffer"
    print(f"  OK  boosted share draw ({k_c_exp}/24 rows), aligned "
          f"flags, ages, replay counts, eviction (6), severed parity")


def _selftest_cf_train_quarantine(seed=9404):
    """RUN 13 fix C train-side gates (deliverables 2 + 3):
      (1) BOOTSTRAP: a cf row's target is r + disc * Q_target(s_H,
          a_cont) with a_cont = the STORED continuation action — the
          returned loss equals a manual recomputation with pre-step
          nets, and DIFFERS from the argmax-bootstrap alternative;
      (2) QUARANTINE (A12.1): cf rows skip bootstrap_hits and
          _tripwire_devs (parallel cf_* series fed instead) and, under
          taken_noop, skip the bsupport_* logs."""
    print("[selftest] cf train_step: forced a_cont bootstrap + "
          "tracker quarantine")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    D = 12
    agent = ControllerAgent(token_dim=D, batch_size=16, buffer_size=1000,
                            device="cpu", cf_replay=True)
    rng = np.random.RandomState(seed + 1)
    s_live = (rng.randn(2, D).astype(np.float32), ("A0", "A1"))
    ns_live = (rng.randn(3, D).astype(np.float32), ("B0", "B1", "B2"))
    live_item = (s_live, 3, 0.7, ns_live, None, GAMMA ** NSTEP, 1, 2)
    s_cf = (rng.randn(2, D).astype(np.float32), ("C0", "C1"))
    ns_cf = (rng.randn(2, D).astype(np.float32), ("D0", "D1"))
    # pick a_cont != the Double-DQN argmax at ns_cf so the forced
    # bootstrap is observably different
    ntok, nmask = pad_state_batch([ns_cf], agent.device)
    with torch.no_grad():
        o_l, o_u, o_nl, o_nu = agent.q_net(ntok, nmask)
        ocl0, ocu0, nval0 = candidate_intervals(o_l, o_u, o_nl, o_nu,
                                                nmask)
    na_full = int(agent.double_dqn_argmax(ocl0, ocu0, nval0, None,
                                          agent.c_train)[0])
    a_cont = (na_full + 3) % (1 + N_INSTR * 2)
    if a_cont == na_full:
        a_cont = (na_full + 1) % (1 + N_INSTR * 2)
    cf_item = (s_cf, 6, -1.3, ns_cf, None, GAMMA ** NSTEP, 0, a_cont)
    for _ in range(16):
        agent.buffer.push(*live_item)
    for _ in range(32):
        agent.buffer.push_cf(*cf_item, gen_episode=0)
    # expected composition: share_cf = 32/48 -> capped 0.25 -> 4 cf rows
    q_pre = copy.deepcopy(agent.q_net)
    t_pre = copy.deepcopy(agent.target_net)
    t_vals = [tr.t for tr in agent.trackers]
    cov_gt = [1.0 if tr.coverage > tr.target else 0.0
              for tr in agent.trackers]
    random.seed(seed + 2)
    torch.manual_seed(seed + 2)
    loss = agent.train_step()
    n_cf_rows = int(round(np.sum(agent.buffer.last_cf_mask))) \
        if agent.buffer.last_cf_mask else 4
    assert agent.buffer.last_cf_mask is not None and n_cf_rows == 4, \
        f"FAIL: expected 4 cf rows, mask {agent.buffer.last_cf_mask}"
    # (2) quarantine
    assert len(agent.bootstrap_hits) == 12 and \
        len(agent.cf_bootstrap_hits) == 4, \
        (f"FAIL: bootstrap-hit quarantine broken "
         f"({len(agent.bootstrap_hits)} live / "
         f"{len(agent.cf_bootstrap_hits)} cf)")
    assert len(agent._tripwire_devs) == 1 and \
        len(agent._cf_tripwire_devs) == 1, \
        "FAIL: tripwire quarantine broken"
    assert len(agent.cf_batch_frac) == 1 and \
        abs(agent.cf_batch_frac[0] - 4 / 16) < 1e-12, \
        "FAIL: realized cf batch fraction not logged"
    # (1) manual recomputation with pre-step nets
    batch = [live_item] * 12 + [cf_item] * 4
    cf_mask = torch.tensor([False] * 12 + [True] * 4)
    with torch.no_grad():
        tok, msk = pad_state_batch([b[0] for b in batch], agent.device)
        ac_l, ac_u, nl, nu = q_pre(tok, msk)
        cl, cu, _ = candidate_intervals(ac_l, ac_u, nl, nu, msk)
        a_idxs = torch.LongTensor([b[1] for b in batch])
        low = cl.gather(1, a_idxs.unsqueeze(1)).squeeze(1)
        upp = cu.gather(1, a_idxs.unsqueeze(1)).squeeze(1)
        ntok2, nmask2 = pad_state_batch([b[3] for b in batch],
                                        agent.device)
        o_l, o_u, o_nl, o_nu = q_pre(ntok2, nmask2)
        ocl, ocu, nvalid = candidate_intervals(o_l, o_u, o_nl, o_nu,
                                               nmask2)
        rmask_t = agent._pad_reissue_masks([b[4] for b in batch],
                                           ocl.shape[1])
        na = agent.double_dqn_argmax(ocl, ocu, nvalid, rmask_t,
                                     agent.c_train)
        next_as = torch.LongTensor([b[7] for b in batch])
        na_forced = torch.where(cf_mask, next_as, na)
        tl_, tu_, tnl, tnu = t_pre(ntok2, nmask2)
        tcl, tcu, _ = candidate_intervals(tl_, tu_, tnl, tnu, nmask2)
        discs = torch.FloatTensor([b[5] for b in batch])
        rews = torch.FloatTensor([b[2] for b in batch])

        def manual_loss(na_sel):
            tgt_l = rews + discs * tcl.gather(
                1, na_sel.unsqueeze(1)).squeeze(1)
            tgt_u = rews + discs * tcu.gather(
                1, na_sel.unsqueeze(1)).squeeze(1)
            Nn = agent.N_TARGET_SAMPLES
            alphas = torch.linspace(0, 1, Nn)
            tv = torch.FloatTensor([t_vals[b[6]] for b in batch])
            wm = torch.FloatTensor([cov_gt[b[6]] for b in batch])
            tot = torch.zeros(len(batch))
            for i in range(Nn):
                t_i = tgt_l + alphas[i] * (tgt_u - tgt_l)
                inside = (t_i >= low) & (t_i <= upp)
                dl = (t_i - low) ** 2
                du = (t_i - upp) ** 2
                tot += tv * torch.where(inside, torch.zeros_like(dl),
                                        torch.min(dl, du)) \
                    + (1 - tv) * torch.max(dl, du)
            tot /= Nn
            tot = tot + agent.width_reg * (upp - low) ** 2 * wm
            return float(tot.mean())

        loss_forced = manual_loss(na_forced)
        loss_argmax = manual_loss(na)
    assert abs(loss - loss_forced) < 1e-5, \
        (f"FAIL: cf bootstrap != G_cf + disc * Q_target(s_H, a_cont) "
         f"(loss {loss!r} vs manual {loss_forced!r})")
    assert abs(loss - loss_argmax) > 1e-7, \
        "FAIL: forcing a_cont made no difference — the override is dead"
    # taken_noop variant: bsupport logs must count LIVE rows only
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    ag2 = ControllerAgent(token_dim=D, batch_size=16, buffer_size=1000,
                          device="cpu", cf_replay=True,
                          bootstrap_support="taken_noop")
    for _ in range(16):
        ag2.buffer.push(*live_item)
    for _ in range(32):
        ag2.buffer.push_cf(*cf_item, gen_episode=0)
    random.seed(seed + 2)
    torch.manual_seed(seed + 2)
    ag2.train_step()
    assert len(ag2.bsupport_fallbacks) == 12, \
        (f"FAIL: bsupport_fallbacks counted "
         f"{len(ag2.bsupport_fallbacks)} rows, want 12 live-only")
    assert len(ag2.bsupport_taken_wins) == 12, \
        "FAIL: bsupport_taken_wins not live-only"
    print(f"  OK  forced-a_cont target verified against pre-step nets "
          f"(|d|={abs(loss - loss_forced):.1e}; argmax alternative "
          f"differs by {abs(loss - loss_argmax):.1e}), quarantine holds "
          f"(12 live / 4 cf rows, bsupport live-only)")


# ===========================================================================
# ROUND 3 SELFTESTS (build steps 2-4 + C2 scaffold, 2026-07-20)
# ===========================================================================

def _selftest_terminal_kind_schema(seed=11001):
    """ROUND 3 B2 schema (re-review 'windower/buffer contract'):
      (1) the WINDOWER COLLAPSE SITE stamps the tag: completion collapse
          pushes kind 'completion'; the recovery enum values are accepted
          end-to-end by the machinery (A2 emits them later; nothing
          emits them this round); invalid kinds fail loudly;
      (2) routing: legacy (class_replay off) routes disc==0.0 rows to
          term_buf REGARDLESS of kind (bitwise legacy); --class_replay
          routes ONLY crash rows to the boosted pool — completion and
          recovery rows go to the regular pool (wins must never capture
          the 3x boost built for sparse -50 anchors);
      (3) gen_episode stamps ride term_buf appends in lockstep and the
          crash age cap evicts exactly the stale rows (cf mirror)."""
    print("[selftest] ROUND 3 terminal_kind schema (windower tag origin, "
          "class routing, crash age cap)")

    class _Buf:
        def __init__(self):
            self.items = []

        def push(self, *a):
            self.items.append(a)

    D = 4
    sts = [(np.full((1, D), float(i), dtype=np.float32), (f"K{i}",))
           for i in range(5)]
    msk = np.zeros(1 + N_INSTR, dtype=bool)
    # (1) completion collapse stamps 'completion'; earlier stamped
    # window stays 'live'
    buf = _Buf()
    w = ControllerWindower(buf, 0.9, 2)
    w.add(sts[0], 0, 1.0, sts[1], msk, False, 2)
    w.add(sts[1], 2, 1.0, sts[2], msk, False, 2)
    w.add(sts[2], 0, 1.0, sts[3], msk, True, 1,
          terminal_kind="completion")
    kinds = [it[8] for it in buf.items]
    assert kinds == ["live", "completion", "completion"], \
        f"FAIL: completion collapse kinds {kinds}"
    assert all(it[5] == 0.0 for it in buf.items[1:]), \
        "FAIL: completion rows must carry disc 0 (no bootstrap)"
    # recovery enum values pass the collapse site (nothing emits them
    # this round — machinery-only)
    for rk in ("armed_recovery", "unarmed_recovery"):
        bufr = _Buf()
        wr = ControllerWindower(bufr, 0.9, 2)
        wr.add(sts[0], 0, 1.0, sts[1], msk, True, 0, terminal_kind=rk)
        assert [it[8] for it in bufr.items] == [rk], \
            f"FAIL: recovery kind {rk} did not thread"
    # invalid kind fails loudly at both sites
    for bad_call in (
            lambda: ControllerWindower(_Buf(), 0.9, 2).add(
                sts[0], 0, 1.0, sts[1], msk, True, 0,
                terminal_kind="junk"),
            lambda: ControllerReplayBuffer(1000, 3.0).push(
                sts[0], 0, 1.0, sts[1], msk, 0.0, 0, 0, "junk")):
        try:
            bad_call()
            raise RuntimeError("no assert")
        except AssertionError:
            pass

    # (2) routing: legacy vs --class_replay
    rows = [  # (disc, kind, boosted_under_legacy, boosted_under_class)
        (0.0, "crash", True, True),
        (0.0, "completion", True, False),
        (GAMMA ** 3, "live", False, False),
        (GAMMA ** 2, "censored", False, False),
        (0.0, "armed_recovery", True, False),
        (GAMMA ** 3, "unarmed_recovery", False, False),
    ]
    for class_on in (False, True):
        rb = ControllerReplayBuffer(capacity=1000, terminal_boost=3.0,
                                    class_replay=class_on)
        for disc, kind, leg_b, cls_b in rows:
            rb.push(sts[0], 0, 0.5, sts[1], msk, disc, 0, 0, kind)
        want_term = sum((cls_b if class_on else leg_b)
                        for _d, _k, leg_b, cls_b in rows)
        assert len(rb.term_buf) == want_term and \
            len(rb.reg_buf) == len(rows) - want_term, \
            (f"FAIL: routing (class_replay={class_on}) "
             f"{len(rb.term_buf)}/{len(rb.reg_buf)}")
        term_kinds = {it[8] for it in rb.term_buf}
        if class_on:
            assert term_kinds == {"crash"}, \
                f"FAIL: non-crash rows in the boosted pool ({term_kinds})"

    # (3) gen stamps + crash age cap (cf mirror)
    rb = ControllerReplayBuffer(capacity=1000, terminal_boost=3.0,
                                class_replay=True, crash_age_cap=400)
    for g in range(10):
        rb.gen_episode = g
        rb.push(sts[0], 0, -50.0, sts[1], msk, 0.0, 0, 0, "crash")
    assert list(rb.term_gen) == list(range(10)), \
        "FAIL: gen_episode stamps did not ride term_buf appends"
    n_ev = rb.evict_crash_older_than(6)
    assert n_ev == 6 and len(rb.term_buf) == 4 and \
        all(g >= 6 for g in rb.term_gen), \
        f"FAIL: crash age cap evicted {n_ev} (want 6)"
    print("  OK  completion/recovery kinds threaded from the collapse "
          "site, invalid kinds rejected, class-aware routing (crash-only "
          "boost), gen stamps + age-cap eviction")


def _selftest_class_replay(seed=11002):
    """ROUND 3 B2 (--class_replay), sampling side:
      (1) FLAG OFF is bitwise legacy even where the floor WOULD bind:
          sample() draws and RNG consumption identical to a literal
          legacy-sampler reimplementation on a crash-starved pool;
          train_step with junk crash params (flag off) bitwise matches a
          default agent (losses, weights, RNG, trackers);
      (2) FLAG ON: the violation-row batch floor lifts the crash draw to
          round(rest * crash_floor) when the boost target sits below it;
          last_kind_counts tallies the batch per class;
      (3) agent-side telemetry: train_step accumulates the sampler's
          per-class counts; pop_kind_counts returns-and-clears."""
    print("[selftest] ROUND 3 class replay: flag-off bitwise, batch "
          "floor, composition telemetry")
    rng = np.random.RandomState(seed)
    D = 8

    def mk_rows(n, disc, kind):
        out = []
        for _ in range(n):
            s = (rng.randn(1, D).astype(np.float32), ("A0",))
            ns = (rng.randn(1, D).astype(np.float32), ("B0",))
            out.append((s, 0, float(rng.randn()), ns, None, disc, 0, 0,
                        kind))
        return out

    crash = mk_rows(5, 0.0, "crash")
    live = mk_rows(500, GAMMA ** NSTEP, "live")

    # (1) flag-off sampler parity on a crash-starved pool (boost target
    # k_t = 1 < floor 2 — the floor branch must NOT fire)
    rb = ControllerReplayBuffer(capacity=10000, terminal_boost=3.0)
    for it in crash + live:
        rb.push(*it)
    random.seed(seed + 1)
    b_new = rb.sample(24)
    s_new = random.getstate()
    assert rb.last_kind_counts is None, \
        "FAIL: flag-off sample() tallied kind counts"
    random.seed(seed + 1)
    n_term, n_reg = len(rb.term_buf), len(rb.reg_buf)
    share = n_term / max(1, n_term + n_reg)
    target = min(0.5, rb.terminal_boost * share)
    k_t = min(n_term, int(round(24 * target)))
    k_r = 24 - k_t
    if k_r > n_reg:
        k_r = n_reg
        k_t = min(n_term, 24 - k_r)
    b_leg = (random.sample(rb.term_buf, k_t)
             + random.sample(rb.reg_buf, k_r))
    assert [id(x) for x in b_new] == [id(x) for x in b_leg] and \
        random.getstate() == s_new, \
        "FAIL: flag-off sampler differs from legacy (floor contaminated)"
    n_crash_off = sum(1 for it in b_new if it[8] == "crash")
    # train_step parity: default agent vs junk-crash-params flag-off
    a1 = _e1_synthetic_agent(seed)
    a2 = _e1_synthetic_agent(seed, class_replay=False, crash_floor=0.99,
                             crash_age_cap=1)
    a2.q_net.load_state_dict(a1.q_net.state_dict())
    a2.target_net.load_state_dict(a1.target_net.state_dict())
    for it in _e1_synthetic_items(seed + 2):
        a1.buffer.push(*it)
        a2.buffer.push(*it)
    losses, rng_states = [], []
    for ag in (a1, a2):
        torch.manual_seed(seed + 3)
        np.random.seed(seed + 3)
        random.seed(seed + 3)
        losses.append(ag.train_step())
        rng_states.append(torch.random.get_rng_state())
    assert losses[0] == losses[1] and \
        torch.equal(rng_states[0], rng_states[1]), \
        "FAIL: flag-off class-replay params contaminated train_step"
    for (k1, p1), (k2, p2) in zip(a1.q_net.state_dict().items(),
                                  a2.q_net.state_dict().items()):
        assert k1 == k2 and torch.equal(p1, p2), \
            f"FAIL: post-step weight {k1} differs (class flag off)"

    # (2) flag ON: floor lifts k_t from the boost target (1) to
    # round(24 * 0.10) = 2; composition telemetry tallies the batch
    rb_on = ControllerReplayBuffer(capacity=10000, terminal_boost=3.0,
                                   class_replay=True, crash_floor=0.10)
    for it in crash + live:
        rb_on.push(*it)
    random.seed(seed + 1)
    b_on = rb_on.sample(24)
    n_crash_on = sum(1 for it in b_on if it[8] == "crash")
    assert n_crash_off == 1 and n_crash_on == 2, \
        (f"FAIL: violation-row floor (crash rows off/on = "
         f"{n_crash_off}/{n_crash_on}, want 1/2)")
    kc = rb_on.last_kind_counts
    assert kc is not None and sum(kc.values()) == 24 and \
        kc.get("crash", 0) == 2 and kc.get("live", 0) == 22, \
        f"FAIL: batch composition telemetry {kc}"

    # (3) agent-side accumulation + pop
    torch.manual_seed(seed + 4)
    np.random.seed(seed + 4)
    random.seed(seed + 4)
    ag_on = ControllerAgent(token_dim=D, batch_size=16, buffer_size=2000,
                            device="cpu", class_replay=True)
    for it in mk_rows(4, 0.0, "crash") + mk_rows(60, GAMMA ** NSTEP,
                                                 "live"):
        ag_on.buffer.push(*it)
    ag_on.train_step()
    counts = ag_on.pop_kind_counts()
    assert sum(counts.values()) == 16 and counts.get("crash", 0) >= \
        int(round(16 * CRASH_BATCH_FLOOR)), \
        f"FAIL: agent kind-count accumulation {counts}"
    assert ag_on.pop_kind_counts() == {}, \
        "FAIL: pop_kind_counts did not clear"
    print(f"  OK  flag-off draw-for-draw legacy + bitwise train_step; "
          f"flag-on floor 1->2 crash rows/24, composition {kc}, "
          f"agent accumulation + pop")


def _selftest_cf_exclude_limit(seed=11003):
    """ROUND 3 B1 (--cf_exclude_limit) gate semantics, synthetic (no env
    is ever touched on the excluded path — env=None proves it):
      (1) flag ON: branch_step + H > ep_maxstep is EXCLUDED before any
          CF RNG draw or env access; the counter increments;
      (2) boundary: branch_step + H == ep_maxstep passes the gate (it
          matches live-window feasibility — the last full live window
          bootstraps at s_maxstep);
      (3) flag OFF (default): the same late state passes the gate
          (legacy behavior), with no extra RNG consumption either way."""
    print("[selftest] ROUND 3 cf exclude-limit gate (synthetic)")
    D = 8
    tokens = np.zeros((1, D), dtype=np.float32)
    live_state = (tokens, ("A0",))

    def probe(flag, b_step, mstep, stratum=1):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        ag = ControllerAgent(token_dim=D, device="cpu", cf_replay=True,
                             cf_seed=seed, cf_budget_rate=0.0,
                             cf_exclude_limit=flag)
        st0 = ag.cf_rng.getstate()
        rec = cf_process_state(ag, None, {"A0": 1}, None, None, stratum,
                               {}, live_state, 0, 36, False, 1, False,
                               branch_step=b_step, ep_maxstep=mstep)
        return ag, rec, st0

    # (1) excluded: stratum 1 would normally draw the selection RNG —
    # the gate sits BEFORE cf_select_state, so no draw happens
    ag, rec, st0 = probe(True, 70, 100, stratum=1)
    assert rec is None and ag.cf_excluded_limit == 1 and \
        ag.cf_states_selected == 0 and len(ag.buffer.cf_buf) == 0, \
        "FAIL: crossing branch not excluded"
    assert ag.cf_rng.getstate() == st0, \
        "FAIL: excluded state consumed a CF RNG draw"
    assert ag.cf_episode_stats()["excluded_limit"] == 1, \
        "FAIL: excluded_limit telemetry missing"
    # (2) boundary passes the gate (stratum 0: draw-free selection,
    # zero budget -> stops at the budget gate, env untouched)
    ag, rec, st0 = probe(True, 64, 100, stratum=0)
    assert rec is None and ag.cf_excluded_limit == 0 and \
        ag.cf_skipped_budget == 1, \
        "FAIL: boundary branch (step + H == maxstep) wrongly excluded"
    # (3) flag off: late state passes through (legacy)
    ag, rec, st0 = probe(False, 70, 100, stratum=0)
    assert rec is None and ag.cf_excluded_limit == 0 and \
        ag.cf_skipped_budget == 1 and \
        ag.cf_rng.getstate() == st0, \
        "FAIL: flag-off gate not inert"
    print("  OK  crossing branches excluded pre-RNG (counter + "
          "telemetry), boundary passes, flag-off inert")


def _selftest_noop_tolerance(seed=11004):
    """ROUND 3 C2 scaffold (--noop_tolerance):
      (1) default None is OFF: selection identical to a base agent, no
          counters move;
      (2) tolerance semantics on a stubbed candidate head: raw margin
          <= tol flips the pick to NOOP (tol_invoked_ep); margin > tol
          leaves the argmax; the re-issue mask applies to the RAW copy
          (masked best candidate cannot hold the margin open);
      (3) STRATUM GUARD: never fires on stratum 0 (in-band) nor when no
          stratum is supplied (probe/eval callers);
      (4) bonus interplay (raw PRE-BONUS copy): a bonus-driven argmax
          change stands as logged exploration (tol_bonus_override_ep),
          so the count bonus can never be silently vetoed nor override
          the tolerance unlogged;
      (5) pop_tolerance_counts returns-and-clears."""
    print("[selftest] ROUND 3 NOOP tolerance (raw pre-bonus copy, "
          "stratum guard, bonus override logging)")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    D, N = 12, 2
    n_cand = 1 + N_INSTR * N
    toks = np.zeros((N, D), dtype=np.float32)

    def mk_agent(**kw):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return ControllerAgent(token_dim=D, device="cpu", **kw)

    base = mk_agent()
    assert base.noop_tolerance is None, "FAIL: default not OFF"
    tol = mk_agent(noop_tolerance=0.5)
    tol.q_net.load_state_dict(base.q_net.state_dict())
    # (1) default OFF: same pick as base on real heads, counters silent
    i_b, _, _ = base.select_candidate(toks, c=0.5, bonus_stratum=1)
    i_t, _, _ = tol.select_candidate(toks, c=0.5, bonus_stratum=None)
    assert i_b == i_t and base.tol_invoked_ep == 0 and \
        tol.tol_invoked_ep == 0, \
        "FAIL: no-stratum/off selection diverged or counted"
    # stub the candidate head for controlled margins
    cl = np.zeros(n_cand, dtype=np.float32)
    cl[0], cl[3], cl[5] = 1.0, 1.3, 1.2      # margin 0.3 (idx 3)
    cu = cl.copy()
    for ag in (tol,):
        ag.candidate_q = lambda t: (cl.copy(), cu.copy())
    # (2) margin 0.3 <= tol 0.5 -> NOOP, counted
    idx, _, _ = tol.select_candidate(toks, c=0.0, bonus_stratum=1)
    assert idx == 0 and tol.tol_invoked_ep == 1, \
        f"FAIL: tolerance did not flip to NOOP (idx {idx})"
    # margin > tol: pick stands
    tol.noop_tolerance = 0.2
    idx, _, _ = tol.select_candidate(toks, c=0.0, bonus_stratum=1)
    assert idx == 3 and tol.tol_invoked_ep == 1, \
        f"FAIL: tolerance fired above threshold (idx {idx})"
    # re-issue mask applies to the RAW copy: masking idx 3 leaves best
    # raw margin ~0.2 (idx 5) <= tol -> flips (tol 0.25 clears the
    # float32 representation of the 0.2 margin)
    tol.noop_tolerance = 0.25
    rmask = np.zeros(n_cand, dtype=bool)
    rmask[3] = True
    idx, _, _ = tol.select_candidate(toks, c=0.0, reissue_mask=rmask,
                                     bonus_stratum=1)
    assert idx == 0 and tol.tol_invoked_ep == 2, \
        f"FAIL: raw copy ignored the re-issue mask (idx {idx})"
    # (3) stratum guard: stratum 0 and stratum None never fire
    tol.noop_tolerance = 0.5
    idx0, _, _ = tol.select_candidate(toks, c=0.0, bonus_stratum=0)
    idxn, _, _ = tol.select_candidate(toks, c=0.0)
    assert idx0 == 3 and idxn == 3 and tol.tol_invoked_ep == 2, \
        f"FAIL: stratum guard breached (idx0 {idx0}, idxn {idxn})"
    # (4) bonus interplay: bonus moves the argmax to an undersampled
    # type -> exploration stands, logged as bonus_override
    tolb = mk_agent(noop_tolerance=0.5, explore_bonus="count",
                    explore_bonus_scale=10.0)
    tolb.candidate_q = lambda t: (cl.copy(), cu.copy())
    tolb.explore_counts[1, :] = 10 ** 9
    tolb.explore_counts[1, 1 + 4] = 0        # type j=4 -> cols 5 and 10
    idx, _, _ = tolb.select_candidate(toks, c=0.0, bonus_stratum=1)
    assert idx == 5 and tolb.tol_bonus_override_ep == 1 and \
        tolb.tol_invoked_ep == 0, \
        (f"FAIL: bonus-driven pick not preserved/logged "
         f"(idx {idx}, ov {tolb.tol_bonus_override_ep})")
    # same state with the bonus decayed to nothing -> tolerance governs
    tolb.explore_counts[1, :] = 10 ** 9
    idx, _, _ = tolb.select_candidate(toks, c=0.0, bonus_stratum=1)
    assert idx == 0 and tolb.tol_invoked_ep == 1, \
        f"FAIL: tolerance did not govern once the bonus decayed ({idx})"
    # (5) pop semantics
    got = tol.pop_tolerance_counts()
    assert got == {"invoked": 2, "bonus_override": 0} and \
        tol.pop_tolerance_counts() == {"invoked": 0,
                                       "bonus_override": 0}, \
        f"FAIL: pop_tolerance_counts ({got})"
    print("  OK  default OFF, raw-margin flip + threshold + masked-raw "
          "semantics, stratum-0/None guard, bonus override logged, "
          "pop clears")


class _ScriptTD:
    """Tracked-aircraft-data stub for the scripted A1/B1 env."""

    class _Pos:
        def __init__(self, lat, lon):
            self.lat, self.lon = lat, lon

    def __init__(self, status, lat, lon, fl):
        self.pos_status = status            # plain string (str() path)
        self.position = self._Pos(lat, lon)
        self.flight_level = fl
        self.track_dist_to_exit_cr = 30.0
        self.centreline_info_cr = (0.5,)
        self.centreline_info_fr = None


class _ScriptAc:
    """Simulator-aircraft stub (kinematics for tokens/snapshots)."""

    def __init__(self, lat, lon, fl):
        self.lat, self.lon, self.fl = lat, lon, fl
        self.heading = 90.0
        self.speed_tas = 450.0
        self.selected_fl = fl


class _ScriptSimEnv:
    def __init__(self, aircraft):
        self.aircraft = aircraft


class _ScriptCfg:
    def __init__(self, spawn_rate, roster, maxstep):
        self.scenario_config = {"initial_spawn_rate": spawn_rate,
                                "max_spawn_rate": spawn_rate,
                                "num_starter_aircraft": roster}
        # run_episode's getattr fallback evaluates eagerly even though
        # _ScriptEnv.maxstep wins — the attribute must exist
        self.scenario_duration = maxstep * SEC_PER_STEP


class _ScriptEnv:
    """Deterministic spawn-0 micro env for the ROUND 3 A1/B1 selftests.
    frames[t] = {callsign: (status, lat, lon, fl)}; an aircraft is in
    obs at t iff its status there is IN_SECTOR; EXIT_REACHED/OUT_SECTOR
    appear in the step info exactly once (the transition step), which is
    what priced_env_step's delivery scan and detect_violation read.
    env.step ignores actions (positions are scripted); deepcopy-able so
    CF branch rollouts can step a clone past the live frame (the final
    frame is held)."""
    _ctrl_sector_bounds = (50.245, 51.542, -4.650, -2.230)

    def __init__(self, frames, roster, maxstep, obs_dim=4,
                 spawn_rate=0.0):
        self.frames = frames
        self.maxstep = maxstep
        self.obs_dim = obs_dim
        self.t = 0
        self.config = _ScriptCfg(spawn_rate, roster, maxstep)

    def _frame(self, t):
        return self.frames[min(t, len(self.frames) - 1)]

    def _obs(self, t):
        return {cs: np.full(self.obs_dim, (ord(cs[-1]) % 7) / 10.0,
                            dtype=np.float32)
                for cs, (st, *_rest) in self._frame(t).items()
                if st == "IN_SECTOR"}

    def _info(self, t):
        fr = self._frame(t)
        info = {cs: {"pos_status": st}
                for cs, (st, *_rest) in fr.items()}
        info["simulator_environment"] = self.get_simulator_env()
        return info

    def get_simulator_env(self):
        fr = self._frame(self.t)
        return _ScriptSimEnv({cs: _ScriptAc(lat, lon, fl)
                              for cs, (st, lat, lon, fl) in fr.items()
                              if st == "IN_SECTOR"})

    def get_tracked_aircraft_data(self, cs):
        e = self._frame(self.t).get(cs)
        if e is None:
            return None
        st, lat, lon, fl = e
        return _ScriptTD(st, lat, lon, fl)

    def reset(self, seed=None):
        self.t = 0
        return self._obs(0), self._info(0)

    def step(self, actions):
        self.t += 1
        return self._obs(self.t), 0.0, False, False, self._info(self.t)


def _selftest_round3_env(seed=11005):
    """ROUND 3 A1 + W_rec scaffold + B1 integration on the scripted
    spawn-0 env (run_episode end-to-end, no real simulator):
      (A1) roster completion: episode ENDS as a completion when
          delivered == roster; windows collapse to realized returns
          with kind 'completion' and no bootstrap; the coverage
          trackers are fed (realized episode) and censored_episodes
          does NOT move; --class_replay routes the completion rows to
          the regular pool while legacy routing boosts them
          (documented dilution, B2's motivation);
      (A1 corner) an undelivered aircraft going OUT_SECTOR ends the
          episode as a VIOLATION (crash rows) — the completion check
          runs strictly AFTER the violation branch;
      (A1 guard) a nonzero spawn rate refuses --completion_terminal;
      (W_rec) the stats carry recovery_win == 0.0 and the
          reconciliation identity holds with the column present;
      (B1) run_episode threads the branch step index: with
          --cf_exclude_limit the states whose nstep-horizon crosses
          maxstep are excluded (counter == 2 here) and the cf pool
          shrinks by exactly those states' branches vs the flag-off
          twin."""
    print("[selftest] ROUND 3 scripted-env suite (A1 completion, "
          "OUT_SECTOR corner, recovery_win column, B1 threading)")
    D_OBS, NSTEP3 = 4, 3
    A = ("IN_SECTOR", 50.50, -4.00, 200.0)
    B = ("IN_SECTOR", 50.6333, -4.00, 215.0)   # ~8 nm, dFL 15: stratum 0,
    #                                            no LoS (needs <5nm & <10FL)

    def frames_completion():
        return [
            {"AAA": A, "BBB": B},                                   # 0
            {"AAA": A, "BBB": B},                                   # 1
            {"AAA": A, "BBB": B},                                   # 2
            {"AAA": ("EXIT_REACHED",) + A[1:], "BBB": B},           # 3
            {"BBB": B},                                             # 4
            {"BBB": B},                                             # 5
            {"BBB": ("EXIT_REACHED",) + B[1:]},                     # 6
            {},                                                     # 7
        ]

    def mk_agent(**kw):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return ControllerAgent(token_dim=D_OBS + KIN_FEATS, device="cpu",
                               warmup_steps=0, **kw)

    def run(env, agent, **kw):
        torch.manual_seed(seed + 1)
        np.random.seed(seed + 1)
        random.seed(seed + 1)
        return run_episode(env, agent, seed=0, train=True, nstep=NSTEP3,
                           cbp=False, objective_v2=False, **kw)

    # ---- (A1) completion ------------------------------------------------
    for class_on in (False, True):
        env = _ScriptEnv(frames_completion(), roster=2, maxstep=10)
        ag = mk_agent(class_replay=class_on)
        stats = run(env, ag, completion_terminal=True)
        assert stats["completed"] and not stats["violated"] and \
            stats["steps"] == 6 and stats["deliveries"] == 2, \
            f"FAIL: completion episode {stats}"
        assert ag.completed_episodes == 1 and ag.censored_episodes == 0 \
            and ag.realized_episodes == 1, \
            "FAIL: completion episode not counted as realized"
        rows = list(ag.buffer.term_buf) + list(ag.buffer.reg_buf)
        comp = [it for it in rows if it[8] == "completion"]
        assert len(rows) == 6 and len(comp) == 3 and \
            all(it[5] == 0.0 for it in comp), \
            "FAIL: completion collapse rows (want 3, disc 0)"
        assert sum(1 for it in rows if it[8] == "live") == 3, \
            "FAIL: pre-collapse windows must stay kind 'live'"
        if class_on:
            assert len(ag.buffer.term_buf) == 0, \
                "FAIL: completion rows captured the boost under " \
                "--class_replay"
        else:
            assert len(ag.buffer.term_buf) == 3, \
                "FAIL: legacy routing changed (disc==0 -> term_buf)"
        # W_rec payment-site scaffold: column present, 0.0, identity held
        # (run_episode asserts the identity internally; recompute here)
        assert stats["recovery_win"] == 0.0, "FAIL: recovery_win != 0"
        recon = (stats["cbp_shaping"] + stats["fuel"] + stats["cmd_cost"]
                 + stats["delivery_bonus"] + stats["violation_term"]
                 + stats["recovery_win"])
        assert abs(recon - stats["ep_return"]) < 1e-3, \
            "FAIL: reconciliation with recovery_win column"

    # ---- (A1 corner) OUT_SECTOR on the last undelivered aircraft ------
    fr = frames_completion()
    fr[6] = {"BBB": ("OUT_SECTOR",) + B[1:]}
    env = _ScriptEnv(fr, roster=2, maxstep=10)
    ag = mk_agent()
    stats = run(env, ag, completion_terminal=True)
    assert stats["violated"] and \
        stats["violation_kind"] == "sector_excursion" and \
        not stats["completed"] and stats["deliveries"] == 1, \
        f"FAIL: OUT_SECTOR corner not a violation {stats}"
    rows = list(ag.buffer.term_buf) + list(ag.buffer.reg_buf)
    assert sum(1 for it in rows if it[8] == "crash") == 3 and \
        not any(it[8] == "completion" for it in rows), \
        "FAIL: corner rows must collapse as 'crash', never 'completion'"

    # ---- (A1 guard) nonzero spawn rate refuses the flag ----------------
    env = _ScriptEnv(frames_completion(), roster=2, maxstep=10,
                     spawn_rate=0.01)
    try:
        run(env, mk_agent(), completion_terminal=True)
        raise RuntimeError("no assert")
    except AssertionError as e:
        assert "spawn-0" in str(e), f"FAIL: wrong guard message ({e})"

    # ---- (B1) branch-step threading through run_episode ----------------
    def frames_censored():
        return [{"AAA": A, "BBB": B}] * 7
    pools, excluded = {}, {}
    for flag in (False, True):
        env = _ScriptEnv(frames_censored(), roster=2, maxstep=6)
        ag = mk_agent(cf_replay=True, cf_seed=seed, cf_budget_rate=50.0,
                      cf_exclude_limit=flag)
        stats = run(env, ag)
        assert not stats["violated"] and stats["steps"] == 6, \
            f"FAIL: censored cf episode {stats}"
        pools[flag] = len(ag.buffer.cf_buf)
        excluded[flag] = ag.cf_excluded_limit
        assert stats["cf"]["excluded_limit"] == ag.cf_excluded_limit, \
            "FAIL: excluded_limit missing from the cf episode stats"
    # stratum-0 states are always selected: 6 states flag-off, 4 with
    # the two crossing states (step 4, 5: step + 3 > 6) excluded
    assert excluded[False] == 0 and excluded[True] == 2, \
        f"FAIL: exclusion counts {excluded}"
    assert pools[False] == 6 * 3 and pools[True] == 4 * 3, \
        (f"FAIL: cf pools {pools} — want 18 flag-off (horizon-crossing "
         f"branches included: the B1 bug) vs 12 excluded")
    print("  OK  completion terminal (realized, kind-tagged, both "
          "routings), OUT_SECTOR corner -> violation, spawn guard, "
          "recovery_win column reconciles, B1 exclusion 2 states / "
          "18->12 cf items")


def _np_state_equal(s0, s1):
    return (s0[0] == s1[0] and np.array_equal(s0[1], s1[1])
            and s0[2:] == s1[2:])


class _M0Recorder(ControllerAgent):
    """M0 harness (A4): a normal ControllerAgent that snapshots, at
    every decision, the pre-step env deepcopy, the last_issued dict, the
    delivery-clock counter (POST-observe — run_episode observes before
    it selects) and the action it returned. The recorded ACTION SEQUENCE
    is what M0 teacher-forces through the CF pricing path — never
    re-derived from the live policy. force_noop makes every decision a
    global NOOP (recorded consistently) — the fallback that guarantees a
    violation for the collapsed-semantics assert on seeds where the
    random-init greedy net flies clean."""

    def __init__(self, *a, **kw):
        self.force_noop = kw.pop("force_noop", False)
        super().__init__(*a, **kw)
        self.rec = []

    def generate_action(self, env, obs_dict, info_dict, **kw):
        li = kw.get("last_issued") or {}
        pre = {"env": copy.deepcopy(env), "li": dict(li),
               "clock": delivery_clock(env).step, "obs": obs_dict}
        actions, aux = super().generate_action(env, obs_dict, info_dict,
                                               **kw)
        if self.force_noop:
            actions = {cs: 0 for cs in obs_dict}
            aux = dict(aux, cand_idx=0)
        pre["actions"] = dict(actions)
        pre["cand_idx"] = aux["cand_idx"]
        self.rec.append(pre)
        return actions, aux


def _selftest_cf_env_suite(seed=10043, n_steps=90, n_m0=6):
    """RUN 13 fix C on the REAL env (one env build shared by three
    gates; ~3-6 min):
      (A13.1) mask threading: the CF machinery's candidate mask ==
          the live masked set at s, li deep-copied at branch time,
          BEFORE_ENTRY candidates excluded with analytic targets, and
          cf generation side-effect-free on the live env AND on every
          live RNG stream / count table (A12.3);
      (A12.3) flag-ON isolation: identically-seeded training episodes,
          cf ON with the optimizer channel SEVERED vs cf OFF — live
          trajectory, weights and counters bitwise identical while cf
          generation demonstrably ran;
      (A4/M0) CF-identity, gating: teacher-forced replay of the
          fixed-seed episode's RECORDED action sequence through the CF
          pricing path at every taken (s, a) with a bootstrap window:
          bitwise G_cf == the windower's n-step return, synthesized
          ns tokens/callsigns/ns_mask/next_a == the stored live 8-tuple
          fields, branch delivery clock advanced exactly once per step
          and matching the live counter at s_{i+n}; terminal windows
          get the separate collapsed-semantics assert (disc 0, empty
          next state, NOOP-only mask)."""
    print(f"[selftest] cf-replay env suite (seed {seed}; builds a real "
          f"env; ~3-6 min)")
    env = make_controller_env(scenario_duration=n_steps * SEC_PER_STEP)
    try:
        obs, info = env.reset(seed=seed)
        obs_dim = int(next(iter(obs.values())).shape[0])
        token_dim = obs_dim + KIN_FEATS

        # ---- (A13.1) mask threading + BE analytic targets ------------
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        agent = ControllerAgent(token_dim=token_dim, device="cpu",
                                cf_replay=True, cf_seed=seed)
        agent.q_net.eval()
        clock = reset_delivery_clock(env)
        snap = sector_snapshot(env, obs.keys())
        li, tgt = {}, None
        steps_after_issue = 0
        for step in range(40):
            clock.observe(env, snap)
            actions = {cs: 0 for cs in obs}
            if tgt is None:
                # issue L10 to the FIRST aircraft to enter the sector
                for cs in sorted(obs):
                    try:
                        td = env.get_tracked_aircraft_data(cs)
                    except Exception:
                        continue
                    if td is not None and td.pos_status is not None and \
                            getattr(td.pos_status, "name",
                                    "") == "IN_SECTOR":
                        tgt = cs
                        actions[tgt] = 1            # L10
                        li[tgt] = 0
                        break
            else:
                steps_after_issue += 1
            obs = env.step(actions)[0]
            snap = sector_snapshot(env, obs.keys())
            if steps_after_issue >= 2:
                break
        assert tgt is not None and tgt in obs, \
            "FAIL: no IN_SECTOR aircraft appeared in the prefix window"
        clock.observe(env, snap)     # live loop observes AT s pre-branch
        cs_list = sorted(obs)
        tokens = build_tokens(env, obs, None, cs_list)
        fp0 = env_fingerprint(env)
        st_py = random.getstate()
        st_np = np.random.get_state()
        st_torch = torch.random.get_rng_state()
        ec0 = agent.explore_counts.copy()
        li_snap = dict(li)
        agent.cf_budget.bank = 1e9
        rec = cf_process_state(agent, env, obs, None, snap, 0, li,
                               (tokens, tuple(cs_list)), 0, 4, True,
                               CBP_LAG, True)
        assert rec is not None, "FAIL: stratum-0 state not selected"
        live_mask = build_reissue_mask(cs_list, li, agent.n_instr)
        assert np.array_equal(rec["step0_mask"], live_mask), \
            "FAIL (A13.1): CF step-0 masked set != live masked set at s"
        masked_idx = np.nonzero(live_mask)[0]
        assert masked_idx.size >= 1, "FAIL: test mask vacuous"
        assert all(mi not in rec["chosen"] for mi in masked_idx), \
            "FAIL: a re-issue-masked candidate was CF-probed"
        assert rec["chosen"][0] == 0, "FAIL: NOOP probe missing"
        for r_ in rec["recs"].values():
            assert r_["li0"] == li_snap, \
                "FAIL: branch last_issued not the live dict at branch time"
        be = cf_before_entry_mask(env, cs_list, agent.n_instr)
        n_expected = len(rec["chosen"]) + len(rec["be_idxs"])
        if bool(be.any()):
            assert all(int(i) not in rec["chosen"]
                       for i in np.nonzero(be)[0]), \
                "FAIL: BEFORE_ENTRY candidate was rolled out"
            g_noop = rec["recs"][0]["g_cf"]
            be_items = [e for e in agent.buffer.cf_buf
                        if e[0][1] in rec["be_idxs"]]
            assert len(be_items) == len(rec["be_idxs"]) and all(
                e[0][2] == g_noop - CMD_COST for e in be_items), \
                "FAIL: analytic BE target != G_cf(NOOP) - CMD_COST"
        assert len(agent.buffer.cf_buf) == n_expected, \
            "FAIL: cf pool size != chosen + analytic items"
        assert int(agent.cf_counts.sum()) == len(rec["chosen"]), \
            "FAIL: CF-side count table not fed per probed candidate"
        # side-effect freedom on everything live (A12.3)
        assert env_fingerprint(env) == fp0, \
            "FAIL: cf generation mutated the live env"
        assert random.getstate() == st_py and \
            _np_state_equal(st_np, np.random.get_state()) and \
            torch.equal(st_torch, torch.random.get_rng_state()), \
            "FAIL: cf generation consumed a LIVE RNG stream"
        assert np.array_equal(agent.explore_counts, ec0), \
            "FAIL: cf generation wrote the live count table"
        assert li == li_snap, "FAIL: cf generation mutated last_issued"
        print(f"  OK  A13.1 mask threading ({masked_idx.size} masked, "
              f"{len(rec['chosen'])} probes incl. NOOP, "
              f"{len(rec['be_idxs'])} analytic BE targets), live "
              f"env/RNG/tables untouched")

        # ---- (A12.3) flag-ON isolation, optimizer channel severed ----
        pair = []
        for cf_on in (False, True):
            torch.manual_seed(seed + 1)
            np.random.seed(seed + 1)
            random.seed(seed + 1)
            ag = ControllerAgent(token_dim=token_dim, device="cpu",
                                 batch_size=8, buffer_size=4000,
                                 warmup_steps=10, cf_replay=cf_on,
                                 cf_seed=seed, cf_budget_rate=50.0)
            if cf_on:
                ag.buffer.cf_sample_enabled = False    # SEVERED
            pair.append(ag)
        a_off, a_on = pair
        a_on.q_net.load_state_dict(a_off.q_net.state_dict())
        a_on.target_net.load_state_dict(a_off.target_net.state_dict())
        hooks, stats_pair = {}, {}
        for name, ag in (("off", a_off), ("on", a_on)):
            torch.manual_seed(seed + 2)
            np.random.seed(seed + 2)
            random.seed(seed + 2)
            hs = []
            stats_pair[name] = run_episode(env, ag, seed=seed,
                                           train=True, nstep=3,
                                           cbp=True,
                                           step_hook=hs.append)
            hooks[name] = hs
        assert len(hooks["off"]) == len(hooks["on"]), \
            "FAIL: episode lengths differ with cf ON (severed)"
        for h0, h1 in zip(hooks["off"], hooks["on"]):
            assert h0["cand_idx"] == h1["cand_idx"] and \
                h0["r"] == h1["r"] and \
                h0["shaping_paid"] == h1["shaping_paid"], \
                (f"FAIL: live trajectory diverged at step {h0['step']} "
                 f"with cf ON (severed)")
        assert stats_pair["off"]["ep_return"] == \
            stats_pair["on"]["ep_return"], "FAIL: ep_return diverged"
        for (k1, p1), (k2, p2) in zip(a_off.q_net.state_dict().items(),
                                      a_on.q_net.state_dict().items()):
            assert k1 == k2 and torch.equal(p1, p2), \
                f"FAIL: post-episode weight {k1} diverged (severed cf)"
        assert np.array_equal(a_off.explore_counts, a_on.explore_counts)
        assert [tr.t for tr in a_off.trackers] == \
               [tr.t for tr in a_on.trackers] and \
            list(a_off.bootstrap_hits) == list(a_on.bootstrap_hits), \
            "FAIL: trackers diverged (severed cf)"
        assert len(a_off.buffer) == len(a_on.buffer), \
            "FAIL: live buffer sizes diverged"
        assert a_on.cf_states_selected > 0 and \
            len(a_on.buffer.cf_buf) > 0, \
            "FAIL: isolation test vacuous — cf generated nothing"
        print(f"  OK  A12.3 isolation: {len(hooks['on'])} live steps "
              f"bitwise identical while cf generated "
              f"{len(a_on.buffer.cf_buf)} items from "
              f"{a_on.cf_states_selected} states (severed)")

        # ---- (A4/M0) teacher-forced CF-identity ----------------------
        torch.manual_seed(seed + 3)
        np.random.seed(seed + 3)
        random.seed(seed + 3)
        m0 = _M0Recorder(token_dim=token_dim, device="cpu",
                         warmup_steps=0, batch_size=10 ** 6)
        st = run_episode(env, m0, seed=seed, train=True, nstep=n_m0,
                         cbp=True)
        if not st["violated"]:
            # random-init greedy flew clean on this seed: fall back to
            # the recorded all-NOOP sequence (violates ~step 76 on seed
            # 10043) so the terminal asserts stay exercised
            print("  --  greedy episode censored; falling back to "
                  "forced-NOOP recording for M0")
            torch.manual_seed(seed + 3)
            np.random.seed(seed + 3)
            random.seed(seed + 3)
            m0 = _M0Recorder(token_dim=token_dim, device="cpu",
                             warmup_steps=0, batch_size=10 ** 6,
                             force_noop=True)
            st = run_episode(env, m0, seed=seed, train=True, nstep=n_m0,
                             cbp=True)
        assert st["violated"], \
            "FAIL: no violating M0 episode — pick another seed/duration"
        T = st["steps"]
        rec_l = m0.rec
        assert len(rec_l) == T
        reg_items = list(m0.buffer.reg_buf)
        term_items = list(m0.buffer.term_buf)
        assert len(reg_items) == max(0, T - n_m0) and \
            len(term_items) == min(T, n_m0), \
            f"FAIL: window pools {len(reg_items)}/{len(term_items)}"
        n_boot = 0
        for i in range(T - n_m0):
            item = reg_items[i]
            script = [(rec_l[j]["actions"], rec_l[j]["cand_idx"])
                      for j in range(i, i + n_m0)]
            env_i = rec_l[i]["env"]
            snap_i = sector_snapshot(env_i, rec_l[i]["obs"].keys())
            br = cf_branch_rollout(env_i, rec_l[i]["obs"], snap_i,
                                   rec_l[i]["li"], m0, None, n_m0,
                                   True, CBP_LAG, True, script=script)
            assert not br["violated"], \
                f"FAIL: M0 branch {i} violated but live did not"
            assert br["g_cf"] == item[2], \
                (f"FAIL: M0 window {i}: G_cf {br['g_cf']!r} != windower "
                 f"n-step return {item[2]!r} (bitwise)")
            assert br["disc"] == item[5], f"FAIL: M0 window {i} disc"
            assert br["ns"][1] == item[3][1] and \
                np.array_equal(br["ns"][0], item[3][0]), \
                f"FAIL: M0 window {i} synthesized s_H != stored ns"
            assert np.array_equal(br["ns_mask"], item[4]), \
                f"FAIL: M0 window {i} synthesized ns_mask != stored"
            assert br["next_a"] == item[7], \
                (f"FAIL: M0 window {i} synthesized next_a "
                 f"{br['next_a']} != stored {item[7]}")
            assert br["clock_steps"] == [rec_l[i + k]["clock"]
                                         for k in range(n_m0)], \
                f"FAIL: M0 window {i} branch clock did not advance " \
                f"once per step"
            # recorder counters are POST-observe; the branch stops
            # before observing at s_{i+n}, so its final counter is the
            # live PRE-observe counter there
            assert br["final_clock"] == rec_l[i + n_m0]["clock"] - 1, \
                f"FAIL: M0 window {i} final branch clock != live " \
                f"counter at s_(i+n)"
            n_boot += 1
        n_term = 0
        for w, i in enumerate(range(max(0, T - n_m0), T)):
            item = term_items[w]
            L = T - i
            script = [(rec_l[j]["actions"], rec_l[j]["cand_idx"])
                      for j in range(i, T)]
            env_i = rec_l[i]["env"]
            snap_i = sector_snapshot(env_i, rec_l[i]["obs"].keys())
            br = cf_branch_rollout(env_i, rec_l[i]["obs"], snap_i,
                                   rec_l[i]["li"], m0, None, L,
                                   True, CBP_LAG, True, script=script)
            assert br["violated"] and br["steps"] == L, \
                f"FAIL: M0 terminal window {i} did not violate at +{L}"
            assert br["g_cf"] == item[2], \
                (f"FAIL: M0 terminal window {i}: collapsed G_cf "
                 f"{br['g_cf']!r} != stored {item[2]!r}")
            assert item[5] == 0.0 and br["disc"] == 0.0, \
                f"FAIL: M0 terminal window {i} disc"
            assert br["ns"][0].shape == (0, token_dim) and \
                br["ns"][1] == () and item[3][0].shape[0] == 0, \
                f"FAIL: M0 terminal window {i} next state not empty"
            assert br["ns_mask"].shape == (1,) and \
                not br["ns_mask"].any() and \
                item[4].shape == (1,) and not item[4].any(), \
                f"FAIL: M0 terminal window {i} mask not NOOP-only"
            assert br["next_a"] == 0 == item[7], \
                f"FAIL: M0 terminal window {i} next_a"
            n_term += 1
        print(f"  OK  M0 CF-identity: {n_boot} bootstrap windows "
              f"bitwise (G_cf, disc, s_H tokens/callsigns, ns_mask, "
              f"next_a, clock) + {n_term} terminal windows collapsed "
              f"(T={T}, n={n_m0}, violation at "
              f"{st['time_to_violation']}s)")
    finally:
        env.close()


def run_selftest():
    print("=" * 100)
    print("CONTROLLER-FRAME SELF-TESTS")
    print("=" * 100)
    _selftest_permutation()
    _selftest_padding()
    _selftest_telescoping()
    _selftest_smooth_vertical()
    _selftest_windower()
    _selftest_mask()                 # n_instr = N_INSTR (5)
    _selftest_mask(n_instr=3)        # legacy layout
    _selftest_candidate_roundtrip()
    _selftest_old_ckpt_compat()
    _selftest_delivery_bonus()
    _selftest_e1_default_off()
    _selftest_e1_negatives()
    _selftest_e1_hinge()
    _selftest_e1_no_contamination()
    _selftest_e1_params()
    _selftest_bootstrap_support()
    _selftest_explore_bonus()
    _selftest_tripwire()
    _selftest_cf_flag_off()
    _selftest_cf_budget_selection()
    _selftest_cf_pool()
    _selftest_cf_train_quarantine()
    # ROUND 3 (build steps 2-4 + C2 scaffold, 2026-07-20)
    _selftest_terminal_kind_schema()
    _selftest_class_replay()
    _selftest_cf_exclude_limit()
    _selftest_noop_tolerance()
    _selftest_round3_env()
    net = ControllerQNet(60)
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params < 100000, f"FAIL: {n_params} params >= 100k budget"
    print(f"[selftest] parameter budget: {n_params} < 100000  OK")
    _selftest_cbp()
    _selftest_d5_cbp_env()
    _selftest_cf_env_suite()
    print("SELFTEST PASSED")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Controller-frame Interval DQN on BluebirdATC "
                    "(one clearance per sweep; CONTROLLER_ARCHITECTURE.md)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny end-to-end run (120 s, few episodes)")
    parser.add_argument("--train", action="store_true",
                        help="full training (defaults: run-12a pilot, "
                             "1500 eps @ 1200 s)")
    parser.add_argument("--eval", action="store_true",
                        help="eval-only from a checkpoint")
    parser.add_argument("--selftest", action="store_true",
                        help="mandatory unit tests: permutation invariance, "
                             "padding equivalence, gamma-weighted "
                             "telescoping identity")
    parser.add_argument("--cbp", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="counterfactual-baselined potential: training "
                             "shaping term = gamma*(Phi(s'_action) - "
                             "Phi(s'_noop)) via a deepcopy NOOP branch; all "
                             "other reward terms stay absolute. Default ON "
                             "for training (--no-cbp to disable). ~2-4x "
                             "env cost per training step (see --cbp_lag).")
    parser.add_argument("--cbp_lag", type=int, default=CBP_LAG,
                        help="extra all-NOOP steps both CBP branches take "
                             "before Phi is measured (default 1: minimal "
                             "latency-matched counterfactual — commands "
                             "act with one sweep of latency, so the "
                             "spec-literal lag 0 term is provably "
                             "identically zero; see CBP_LAG note).")
    parser.add_argument("--objective_v2",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="OBJECTIVE v2 (default ON): no global fuel "
                             "term; delivery bonus decays with transit "
                             "time, 10 * max(0.3, nominal_T/actual_T). "
                             "--no-objective_v2 restores the v1 pricing "
                             "(flat +10, -0.005/aircraft-step fuel).")
    parser.add_argument("--width_scalars",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="unnormalized count/density scalars into the "
                             "width heads (zero-init; JK directive 8 July)")
    parser.add_argument("--e1_width",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="E1-ATC contrastive off-manifold width term "
                             "(default OFF): bounded hinge "
                             "lambda*relu(1 - w/floor)^2 pushing candidate "
                             "widths on aircraft-swap + field-shuffle "
                             "batch negatives up to an ABSOLUTE floor "
                             "(WIDTH_MECHANISM_PROBES.md, ATC transfer "
                             "spec; 12b decision). Adds no parameters.")
    parser.add_argument("--e1_lambda", type=float, default=0.5,
                        help="E1 hinge weight (term is capped at "
                             "e1_lambda by construction)")
    parser.add_argument("--e1_floor", type=float, default=30.0,
                        help="E1 absolute width floor; default 30.0 = 3x "
                             "the measured median on-dist candidate width "
                             "(10.1) of the 12b ep1500 net on the a4v2 "
                             "frozen probe states. Never keyed to live "
                             "batch widths (LL attempt-1 runaway lesson).")
    parser.add_argument("--bootstrap_support", type=str, default="full",
                        choices=["full", "taken_noop"],
                        help="Double-DQN target argmax support at s' "
                             "(default full = legacy). 'taken_noop' (fix D, "
                             "12D_CREDIT_DESIGN.md 2.4) restricts it to "
                             "{NOOP, the action actually taken at s'} — "
                             "n-step SARSA with a NOOP floor; removes "
                             "max-over-untrained-candidates hallucination "
                             "noise from every backup hop.")
    parser.add_argument("--explore_bonus", type=str, default="off",
                        choices=["off", "count"],
                        help="count-based exploration bonus at ACTION "
                             "SELECTION ONLY (never targets, never stored "
                             "rewards): score += scale * sqrt(1/(1 + "
                             "3*count/halflife)) per (risk-stratum, "
                             "instruction-type incl. NOOP) count. Counts "
                             "persist in checkpoints; re-issue masking is "
                             "respected. Default off.")
    parser.add_argument("--explore_bonus_scale", type=float, default=0.5,
                        help="bonus at count 0 (default 0.5)")
    parser.add_argument("--explore_bonus_halflife", type=float,
                        default=2000.0,
                        help="counts at which the bonus halves "
                             "(default 2000)")
    parser.add_argument("--tracker_tripwire",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="per-episode tripwire JSONL block (default ON; "
                             "pure instrumentation, no behavior change): "
                             "per-stratum t, prediction-accuracy proxy "
                             "mean |bootstrap-target mid - predicted mid| "
                             "over the episode's batches, per-stratum "
                             "realized coverage. Pre-registered E+F "
                             "revival trigger lives in train_step.")
    parser.add_argument("--vertical_ramp", type=str, default="gate",
                        choices=["gate", "smooth"],
                        help="vertical response of the pair-conflict "
                             "potential (D5). 'gate' (default) = binary "
                             "|dFL|<20 gate, bit-identical 12c pricing; "
                             "'smooth' = product of quadratic ramps over "
                             "current (V=20) and commanded/selected "
                             "(V=10) FL separation, w=0.5 blend "
                             "(D5_VERTICAL_POTENTIAL_DESIGN.md).")
    parser.add_argument("--delta_conflict", type=float,
                        default=DELTA_CONFLICT,
                        help="conflict-potential weight DELTA_CONFLICT "
                             "(default 0.2; JK approved 0.5 for the D5 "
                             "smooth-ramp composite battery).")
    parser.add_argument("--cf_replay",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="RUN 13 fix C counterfactual replay "
                             "(default OFF): at selected live states, "
                             "roll out K untaken candidates (mandatory "
                             "NOOP probe) on env deepcopies under the "
                             "frozen current policy for H_cf = nstep "
                             "steps, priced by the live CBP layer; "
                             "targets y = G_cf + gamma^H * "
                             "Q_target(s_H, a_cont). Trackers are "
                             "quarantined from cf rows "
                             "(RUN13_FIX_C_DESIGN.md Section 1).")
    parser.add_argument("--cf_k", type=int, default=3,
                        help="CF candidates per selected state incl. "
                             "the mandatory NOOP probe (floor 3)")
    parser.add_argument("--cf_budget_rate", type=float,
                        default=CF_BUDGET_RATE,
                        help="step-equivalents banked per live training "
                             "step for CF rollouts (banked ACROSS "
                             "episodes, A2.3; deepcopies charged at "
                             f"{CF_CLONE_COST} step-equivalents, A2.2)")
    parser.add_argument("--cf_boost", type=float, default=CF_BOOST,
                        help="share-proportional cf-pool sampling boost "
                             "(terminal-boost pattern, A14)")
    parser.add_argument("--cf_max_frac", type=float, default=CF_MAX_FRAC,
                        help="cf batch-fraction cap (1:3 cf:real — a "
                             "never-expected-to-bind upper bound, A14)")
    parser.add_argument("--cf_age_cap", type=int, default=CF_AGE_CAP,
                        help="evict cf items older than this many "
                             "episodes (A1 Branch A staleness)")
    parser.add_argument("--cf_exclude_limit",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="ROUND 3 B1 (default OFF): EXCLUDE (never "
                             "cap) CF branch windows whose H-step "
                             "horizon crosses the ENV time limit "
                             "(branch_step + nstep > maxstep). Both cap "
                             "semantics are biased for a time-blind net; "
                             "pre-registered coverage cost is H/T of the "
                             "episode's states (~36% at nstep=36 on the "
                             "run-13 micro T). Excluded-state count "
                             "logged as cf.excluded_limit.")
    parser.add_argument("--completion_terminal",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="ROUND 3 A1 (default OFF; spawn-0 micros "
                             "only): end the episode as a TRUE COMPLETION "
                             "when delivered_count == num_starter_aircraft "
                             "(roster), checked strictly AFTER the "
                             "violation branch (OUT_SECTOR same-step "
                             "corner). Windows reaching it collapse to "
                             "realized returns, no bootstrap (kind "
                             "'completion'); refuses non-spawn-0 configs.")
    parser.add_argument("--class_replay",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="ROUND 3 B2 (default OFF): per-terminal_kind "
                             "buffer policy — only crash rows keep the "
                             "terminal boost and are age-capped "
                             "(--crash_age_cap); completion/win rows go "
                             "to the regular pool; violation-row batch "
                             "floor (--crash_floor); per-class batch "
                             "composition logged per episode "
                             "(batch_kinds).")
    parser.add_argument("--crash_age_cap", type=int,
                        default=CRASH_AGE_CAP,
                        help="evict crash rows older than this many "
                             "episodes under --class_replay (default "
                             f"{CRASH_AGE_CAP}, mirroring the cf age cap)")
    parser.add_argument("--crash_floor", type=float,
                        default=CRASH_BATCH_FLOOR,
                        help="violation-row batch floor under "
                             f"--class_replay (default {CRASH_BATCH_FLOOR}"
                             ", pre-registered draft)")
    parser.add_argument("--noop_tolerance", type=float, default=None,
                        help="ROUND 3 C2 scaffold (default None=OFF): "
                             "NOOP-preference tolerance in "
                             "select_candidate, evaluated on a RAW "
                             "pre-bonus copy of the Hurwicz scores; "
                             "never applies on stratum-0 (in-band) "
                             "states; tolerance-invoked counts logged "
                             "per episode. The THRESHOLD is a measured "
                             "quantity (ROUND3_REREVIEW C2 derivation) "
                             "— leave unset until measured.")
    parser.add_argument("--mask_reissue",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="mask candidates that re-issue an aircraft's "
                             "still-active instruction (selection AND "
                             "Double-DQN target argmax). Default ON "
                             "(--no-mask_reissue to disable).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None,
                        help="training episodes (default 1500; smoke 3)")
    parser.add_argument("--duration", type=int, default=None,
                        help="scenario duration in sim seconds "
                             "(default 1200; smoke 120)")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--c_train", type=float, default=0.5)
    parser.add_argument("--c_eval", type=float, nargs="+",
                        default=[0.0, 0.5])
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer", type=int, default=BUFFER_SIZE)
    parser.add_argument("--target_coverage", type=float, default=0.85)
    parser.add_argument("--width_reg", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="epsilon warmup env steps (default 1000; "
                             "smoke 20)")
    parser.add_argument("--warmup_epsilon", type=float, default=0.5)
    parser.add_argument("--nstep", type=int, default=NSTEP)
    parser.add_argument("--terminal_boost", type=float,
                        default=TERMINAL_BOOST)
    parser.add_argument("--t_cap_risky", type=float, default=0.99)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--c", type=float, default=0.0,
                        help="Hurwicz c for --eval")
    parser.add_argument("--adaptive", action="store_true",
                        help="eval with width-conditioned c")
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--final_eval_episodes", type=int, default=None)
    parser.add_argument("--ckpt_every", type=int, default=25)
    args = parser.parse_args()

    # D5: apply the conflict-pricing flags before ANY pricing happens
    # (training, smoke, eval; selftests manage modes internally but
    # honour the flag as the starting mode).
    set_conflict_pricing(args.vertical_ramp, args.delta_conflict)

    if args.selftest:
        run_selftest()
    elif args.smoke:
        args.episodes = args.episodes if args.episodes is not None else 3
        args.duration = args.duration if args.duration is not None else 120
        args.warmup_steps = (args.warmup_steps
                             if args.warmup_steps is not None else 20)
        args.final_eval_episodes = (args.final_eval_episodes
                                    if args.final_eval_episodes is not None
                                    else 1)
        args.buffer = min(args.buffer, 5000)
        args.batch = min(args.batch, 32)
        run_training(args, smoke=True)
        print("\nSMOKE TEST PASSED")
    elif args.eval:
        if args.ckpt is None:
            parser.error("--eval requires --ckpt PATH")
        args.duration = args.duration if args.duration is not None else 1200
        run_eval_only(args)
    else:
        args.episodes = args.episodes if args.episodes is not None else 1500
        args.duration = args.duration if args.duration is not None else 1200
        args.warmup_steps = (args.warmup_steps
                             if args.warmup_steps is not None else 1000)
        args.final_eval_episodes = (args.final_eval_episodes
                                    if args.final_eval_episodes is not None
                                    else 3)
        run_training(args, smoke=False)
