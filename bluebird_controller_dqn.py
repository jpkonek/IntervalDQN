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

    idx 0            : global NOOP  (env action 0 for every aircraft)
    idx 1 + 3*i + j  : aircraft CS[i], instruction j, where
                       j=0 -> env action 1 (simple_heading_left  10 = L10)
                       j=1 -> env action 2 (simple_heading_right 10 = R10)
                       j=2 -> env action 3 (simple_heading_route_parallel)

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

# Sector-objective weights (spec Section 4, JK-approved)
ALPHA_PROGRESS = 0.05        # per nm along-track distance to exit
BETA_CENTRE = 0.02           # per nm centreline offset
DELTA_CONFLICT = 0.2         # pair-margin potential weight (see docstring)
CONFLICT_RANGE_NM = 15.0     # f ramps up inside this lateral range
CONFLICT_FL = 20.0           # pairs closer than this vertically count
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

N_INSTR = 3    # L10, R10, route_parallel  (env actions 1, 2, 3)

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
    cfg.action_config = {"simple_heading_left": [10],
                         "simple_heading_right": [10],
                         "simple_heading_route_parallel": True}
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
    """Per-aircraft slice of simulator state used by the reward layer."""
    __slots__ = ("cs", "lat", "lon", "fl", "dist_exit", "centre_off")

    def __init__(self, cs, lat, lon, fl, dist_exit, centre_off):
        self.cs = cs
        self.lat = lat
        self.lon = lon
        self.fl = fl
        self.dist_exit = dist_exit      # along-track nm to exit (>= 0)
        self.centre_off = centre_off    # nm off the current-route centreline


def sector_snapshot(env, callsigns):
    """{callsign: AcSnap} for the aircraft among `callsigns` that are
    IN_SECTOR right now, read from the env's tracked-aircraft data (same
    source detect_violation uses via info; tracked data works at reset
    too, before any step info exists)."""
    snap = {}
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
        snap[cs] = AcSnap(cs, float(pos.lat), float(pos.lon),
                          float(td.flight_level),
                          float(d_exit) if d_exit is not None else None,
                          centre)
    return snap


def pair_conflict_f(a, b):
    """f = max(0, (15 - d_nm)/15)^2 for vertically-proximate pairs
    (|delta FL| < 20), else 0."""
    if abs(a.fl - b.fl) >= CONFLICT_FL:
        return 0.0
    d = haversine_nm(a.lat, a.lon, b.lat, b.lon)
    m = max(0.0, (CONFLICT_RANGE_NM - d) / CONFLICT_RANGE_NM)
    return m * m


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


# CBP LATENCY FINDING (2026-07-08, measured): BlueBird clearances act with
# EXACTLY one sweep of latency — issuing L10/R10/route_parallel leaves the
# simulator state BIT-IDENTICAL to all-NOOP after one step and divergent
# only from the second step (verified directly with env_fingerprint on
# seed 10043; consistent with the old battery's B3, where every 1-step
# candidate delta equalled the -0.1 fee exactly, and with a 40-episode
# tiny train whose per-episode CBP term was 0.0 bitwise). Because global
# NOOP does NOT cancel an aircraft's active command, the two branches of
# the lag-0 counterfactual share all previously-issued commands and are
# therefore identical FOREVER: the spec-literal CBP term
# gamma*(Phi(s'_action) - Phi(s'_noop)) is identically zero — it deletes
# the potential term from training instead of de-ambienting it.
# CBP_LAG extends BOTH branches by `lag` extra all-NOOP steps so the
# potential difference is measured at the first state the new command can
# have acted on (lag 1 = minimal latency-matched counterfactual;
# lag 0 = the spec-literal, provably-zero construct). The NOOP-zero
# invariant is unaffected: for a global-NOOP action both branches are the
# same determinized computation at ANY lag (asserted by --selftest).
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
    kinematics), used by --selftest to prove the CBP counterfactual step
    is side-effect-free on the live env."""
    sim = env.get_simulator_env()
    fp = []
    for cs in sorted(sim.aircraft.keys()):
        ac = sim.aircraft[cs]
        fp.append((cs, ac.lat, ac.lon, ac.heading, ac.fl, ac.speed_tas))
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
    opaque controller states. Items:
    (state, a_idx, R_n, next_state, ns_mask, disc, stratum)
    where ns_mask is the re-issue candidate mask AT next_state (same
    layout as the s' candidate matrix, True = masked), applied to the
    Double-DQN argmax on the target side."""

    def __init__(self, capacity=BUFFER_SIZE, terminal_boost=TERMINAL_BOOST):
        self.terminal_boost = float(terminal_boost)
        if self.terminal_boost > 1.0:
            # cap the floor at capacity//2 so tiny buffers keep a regular pool
            term_cap = max(min(1000, capacity // 2), capacity // 5)
            self.term_buf = deque(maxlen=term_cap)
            self.reg_buf = deque(maxlen=capacity - term_cap)
            self.buffer = None
        else:
            self.buffer = deque(maxlen=capacity)

    def push(self, state, a_idx, reward, next_state, ns_mask, disc, stratum):
        item = (state, a_idx, reward, next_state, ns_mask, disc, stratum)
        if self.buffer is not None:
            self.buffer.append(item)
        elif disc == 0.0:
            self.term_buf.append(item)
        else:
            self.reg_buf.append(item)

    def sample(self, batch_size):
        if self.buffer is not None:
            return random.sample(self.buffer, batch_size)
        n_term, n_reg = len(self.term_buf), len(self.reg_buf)
        share = n_term / max(1, n_term + n_reg)
        target = min(0.5, self.terminal_boost * share)
        k_t = min(n_term, int(round(batch_size * target)))
        k_r = batch_size - k_t
        if k_r > n_reg:
            k_r = n_reg
            k_t = min(n_term, batch_size - k_r)
        return (random.sample(self.term_buf, k_t)
                + random.sample(self.reg_buf, k_r))

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
    """

    def __init__(self, buffer, gamma, n):
        self.buffer = buffer
        self.gamma = gamma
        self.n = n
        self.pending = []   # [(state, a_idx, r, stratum)]
        self.last_ns = None
        self.last_ns_mask = None

    def _window_return(self, i):
        return sum(self.gamma ** k * r
                   for k, (_s, _a, r, _st) in enumerate(self.pending[i:]))

    def add(self, s, a_idx, r, ns, ns_mask, terminal, stratum):
        self.pending.append((s, a_idx, r, stratum))
        self.last_ns = ns
        self.last_ns_mask = ns_mask
        if terminal:
            dim = s[0].shape[1]
            zeros = empty_state(dim)
            zeros_mask = np.zeros(1, dtype=bool)   # NOOP-only candidate set
            for i in range(len(self.pending)):
                s_i, a_i, _, st_i = self.pending[i]
                self.buffer.push(s_i, a_i, self._window_return(i),
                                 zeros, zeros_mask, 0.0, st_i)
            self.pending.clear()
        elif len(self.pending) == self.n:
            s_0, a_0, _, st_0 = self.pending[0]
            self.buffer.push(s_0, a_0, self._window_return(0), ns, ns_mask,
                             self.gamma ** self.n, st_0)
            self.pending.pop(0)

    def flush_censored(self):
        for i in range(len(self.pending)):
            s_i, a_i, _, st_i = self.pending[i]
            m = len(self.pending) - i
            self.buffer.push(s_i, a_i, self._window_return(i),
                             self.last_ns, self.last_ns_mask,
                             self.gamma ** m, st_i)
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
                 e1_floor=30.0):
        self.token_dim = token_dim
        self.n_instr = n_instr
        self.gamma = gamma
        self.c_train = c_train
        self.width_reg = width_reg
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.mask_reissue = mask_reissue
        self.width_scalars = width_scalars
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
        self.buffer = ControllerReplayBuffer(buffer_size,
                                             terminal_boost=terminal_boost)

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
                         reissue_mask=None):
        """Hurwicz l + c*(u - l) over ALL candidates; adaptive c is a
        SINGLE width-conditioned c from the mean candidate width (pilot
        sigmoid rule); exact ties involving NOOP resolve to NOOP.
        reissue_mask (candidate layout, True = masked) excludes re-issue
        candidates from the argmax (score forced to -inf); NOOP is never
        masked so the argmax always has a candidate.
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
        if reissue_mask is not None:
            scores = np.where(reissue_mask, -np.inf, scores)
        best = scores.max()
        idx = 0 if scores[0] == best else int(scores.argmax())
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
                        last_issued=None):
        """Compute ONE clearance (aircraft, instruction) or global NOOP.

        Returns (action_dict {callsign: int}, aux) where aux carries the
        token array, sorted callsign tuple, chosen candidate index, mean
        candidate width and the c used — everything replay needs, so the
        buffer never touches the env. Epsilon warmup: uniform over the
        candidate set (NOOP + 3 per aircraft), minus masked re-issues.
        last_issued ({callsign: instr j}) enables re-issue masking when
        self.mask_reissue is on: candidates repeating an aircraft's
        still-active instruction are excluded from BOTH the argmax and
        epsilon-sampling."""
        cs_list = sorted(obs_dict.keys())
        tokens = build_tokens(env, obs_dict, info_dict, cs_list)
        rmask = None
        if self.mask_reissue and last_issued:
            rmask = build_reissue_mask(cs_list, last_issued, self.n_instr)
        idx, mean_width, c_used = self.select_candidate(
            tokens, c=c, adaptive=adaptive, reissue_mask=rmask)
        eps = self._epsilon(force_epsilon)
        if random.random() < eps:
            n_cand = 1 + self.n_instr * len(cs_list)
            if rmask is not None:
                idx = random.choice(
                    [k for k in range(n_cand) if not rmask[k]])
            else:
                idx = random.randrange(n_cand)
        actions = {cs: 0 for cs in cs_list}
        if idx > 0:
            i, j = divmod(idx - 1, self.n_instr)
            actions[cs_list[i]] = j + 1     # env actions 1..3
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
                              t_vec, width_mask):
        N = self.N_TARGET_SAMPLES
        batch_size = lower.shape[0]
        alphas = torch.linspace(0, 1, N, device=lower.device)
        targets = target_lower.unsqueeze(0) + alphas.unsqueeze(1) * (
            target_upper - target_lower).unsqueeze(0)

        # bootstrap-target coverage: LOGGING ONLY (t runs on realized)
        target_mid = (target_lower + target_upper) / 2
        inside_mid = (target_mid >= lower) & (target_mid <= upper)
        self.bootstrap_hits.extend(
            inside_mid.detach().cpu().numpy().astype(np.float32).tolist())

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
        states = [b[0] for b in batch]
        a_idxs = torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_states = [b[3] for b in batch]
        ns_masks = [b[4] for b in batch]
        discs = torch.FloatTensor([b[5] for b in batch]).to(self.device)
        strata = [b[6] for b in batch]

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
            na = self.double_dqn_argmax(ocl, ocu, nvalid, rmask,
                                        self.c_train)
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

        t_vec = torch.FloatTensor(
            [self.trackers[si].t for si in strata]).to(self.device)
        width_mask = torch.FloatTensor(
            [1.0 if (self.trackers[si].coverage > self.trackers[si].target)
             else 0.0 for si in strata]).to(self.device)
        loss = self.interval_loss_sampled(lower_a, upper_a,
                                          target_l, target_u,
                                          t_vec, width_mask)
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

def run_episode(env, agent, seed, train=True, c=None, adaptive=False,
                nstep=NSTEP, cbp=False, cbp_lag=CBP_LAG, step_hook=None,
                objective_v2=True):
    """One controller episode; ends at the first violation (terminal, -50)
    or the time limit (censored). Returns a stats dict with the per-term
    additive reward budget (credit tracing per spec Section 4).

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
    windower = ControllerWindower(agent.buffer, agent.gamma, nstep) \
        if train else None
    mask_on = getattr(agent, "mask_reissue", False)
    last_issued = {}    # {callsign: instr j}; cleared here at episode reset

    terms = {"cbp_shaping": 0.0,
             "phi_progress": 0.0, "phi_centre": 0.0, "phi_conflict": 0.0,
             "fuel": 0.0, "cmd_cost": 0.0, "delivery_bonus": 0.0,
             "violation_term": 0.0}
    ep_states, ep_aidx, ep_strata, ep_rewards = [], [], [], []
    width_sum, width_n = 0.0, 0
    loss_sum, loss_n = 0.0, 0
    n_deliveries = 0
    n_commands = 0
    ep_return = 0.0
    ep_return_disc = 0.0
    violated, violation_kind = False, None

    snap = sector_snapshot(env, obs.keys())
    clock = reset_delivery_clock(env) if objective_v2 else None
    step_i = -1
    for step_i in range(maxstep):
        if clock is not None:
            clock.observe(env, snap)
        force_eps = None if train else 0.0
        actions, aux = agent.generate_action(env, obs, info, c=c,
                                             force_epsilon=force_eps,
                                             adaptive=adaptive,
                                             last_issued=last_issued)
        stratum = snapshot_stratum(snap)
        if aux["callsigns"]:
            width_sum += aux["mean_width"]
            width_n += 1
        issued = aux["cand_idx"] != 0
        if mask_on and issued:
            i_ac, j_in = divmod(aux["cand_idx"] - 1, agent.n_instr)
            last_issued[aux["callsigns"][i_ac]] = j_in

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
        prog, centre, conflict = shaping_terms(snap, snap_next, agent.gamma)
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
                                                         agent.gamma)
            else:
                prog_a, centre_a, conf_a = prog, centre, conflict
            prog_n, centre_n, conf_n = shaping_terms(snap, snap_cf,
                                                     agent.gamma)
            shaping_paid = ((prog_a - prog_n) + (centre_a - centre_n)
                            + (conf_a - conf_n))
        else:
            shaping_paid = prog + centre + conflict
        fuel = 0.0 if objective_v2 else -EPS_FUEL * len(snap)
        cmd = -CMD_COST if issued else 0.0
        if objective_v2:
            deliv = sum(DELIVERY_BONUS * clock.bonus_factor(cs)
                        for cs in delivered_cs)
        else:
            deliv = DELIVERY_BONUS * deliveries
        r = shaping_paid + fuel + cmd + deliv
        terminal = False
        if violated:
            r -= VIOLATION_PENALTY
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
        ep_return += r
        ep_return_disc += (agent.gamma ** step_i) * r
        if step_hook is not None:
            step_hook({"step": step_i, "issued": issued,
                       "cand_idx": aux["cand_idx"],
                       "shaping_paid": shaping_paid, "r": r,
                       "phi_abs": prog + centre + conflict})

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
                         terminal, stratum)
            agent.total_env_steps += 1
            loss = agent.train_step()
            if loss:
                loss_sum += loss
                loss_n += 1

        obs = next_obs
        snap = snap_next
        if violated:
            break

    # reconciliation identity: SUM(budget columns) == ep_return exactly
    recon = (terms["cbp_shaping"] + terms["fuel"] + terms["cmd_cost"]
             + terms["delivery_bonus"] + terms["violation_term"])
    assert abs(recon - ep_return) <= 1e-6, (
        f"budget does not reconcile: sum(terms)={recon!r} != "
        f"ep_return={ep_return!r}")

    if train:
        if violated:
            # terminal episode: realized returns-to-go for EVERY step feed
            # the coverage trackers (windower already collapsed at add())
            agent.record_realized_episode(ep_states, ep_aidx, ep_strata,
                                          ep_rewards)
        else:
            # time-limit censoring: bootstrap the tail windows; returns
            # unknown -> counted but NOT fed to the trackers
            windower.flush_censored()
            agent.censored_episodes += 1

    steps_done = step_i + 1
    time_to_violation = (steps_done * SEC_PER_STEP if violated
                         else maxstep * SEC_PER_STEP)
    return {
        "steps": steps_done,
        "sim_seconds": steps_done * SEC_PER_STEP,
        "violated": violated,
        "violation_kind": violation_kind,
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
    }


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
        "adaptive_w_mid": getattr(agent, "adaptive_w_mid", None),
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
                            device=device, **kwargs)
    agent.q_net.load_state_dict(ckpt["q_net"])
    agent.target_net.load_state_dict(ckpt["q_net"])
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
             if args.e1_width else "") + ")")
    print("=" * 100)

    print("Creating environment...")
    env = make_controller_env(scenario_duration=args.duration,
                              k_nearest=args.k)
    obs, info = env.reset(seed=args.seed)
    obs_dim = int(next(iter(obs.values())).shape[0])
    token_dim = obs_dim + KIN_FEATS
    n_env_actions = int(env.get_action_parser().get_total_num_actions())
    assert n_env_actions == 1 + N_INSTR, (
        f"expected 1 NOOP + {N_INSTR} instructions, env has {n_env_actions}")
    print(f"obs dim: {obs_dim}, token dim: {token_dim}, "
          f"env actions: {env.get_action_parser().action_formatter_map}")

    agent = ControllerAgent(
        token_dim=token_dim, lr=args.lr, gamma=args.gamma,
        c_train=args.c_train, target_coverage=args.target_coverage,
        width_reg=args.width_reg, warmup_steps=args.warmup_steps,
        warmup_epsilon=args.warmup_epsilon, buffer_size=args.buffer,
        batch_size=args.batch, device=args.device,
        t_cap_risky=args.t_cap_risky, terminal_boost=args.terminal_boost,
        mask_reissue=args.mask_reissue, width_scalars=args.width_scalars,
        e1_width=args.e1_width, e1_lambda=args.e1_lambda,
        e1_floor=args.e1_floor)
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
                            objective_v2=args.objective_v2)
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
            "strat_coverage": [round(tr.coverage, 4)
                               for tr in agent.trackers],
            "strat_t": [round(tr.t, 4) for tr in agent.trackers],
            "strat_hits": [len(tr.hits) for tr in agent.trackers],
            "t": round(agent.t, 4),
            "epsilon": round(eps, 4),
            "c_train": agent.c_train,
            "steps_per_sec": round(stats["steps"] / max(1e-9, wall), 2),
            "wall_time_s": round(wall, 2),
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
            "objective_v2": args.objective_v2}


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
    w.add(sts[1], 3, 2.0, sts[2], msk[2], False, 1)
    w.add(sts[2], 1, 3.0, sts[3], msk[3], True, 0)   # violation
    # windows: (0: 1+0.5*2 bootstrap g^2), then terminal collapse of 1, 2
    exp = [(sts[0], 0, 2.0, sts[2], msk[2], 0.25, 2),
           (sts[1], 3, 2.0 + 0.5 * 3.0, None, None, 0.0, 1),
           (sts[2], 1, 3.0, None, None, 0.0, 0)]
    assert len(buf.items) == 3, f"FAIL: {len(buf.items)} windows != 3"
    for i, ((s, a, r, ns, nsm, disc, st),
            (es, ea, er, ens, enm, edisc, est)) \
            in enumerate(zip(buf.items, exp)):
        assert s is es and a == ea and st == est, f"FAIL: window {i} ids"
        assert abs(r - er) < 1e-12, f"FAIL: window {i} R {r} != {er}"
        assert abs(disc - edisc) < 1e-12, f"FAIL: window {i} disc"
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
    assert len(buf2.items) == 2
    assert abs(buf2.items[0][2] - 2.0) < 1e-12 \
        and buf2.items[0][5] == 0.25 and buf2.items[0][3] is sts[2] \
        and buf2.items[0][4] is msk[2]
    assert abs(buf2.items[1][2] - 2.0) < 1e-12 \
        and buf2.items[1][5] == 0.5 and buf2.items[1][3] is sts[2] \
        and buf2.items[1][4] is msk[2], \
        "FAIL: censored window must bootstrap at last ns (+its mask) " \
        "with disc gamma^1"
    print("  OK  terminal collapse + censored gamma^m bootstrap "
          "(+ ns_mask threading)")

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
    for i, (s, a, R, ns, nsm, disc, st) in enumerate(buf3.items):
        assert s is sts12[i] and a == i % 4 and st == i % 3, \
            f"FAIL: n=12 window {i} ids"
        end = min(i + n12, T12)
        want = sum(gamma12 ** (k - i) * rs12[k] for k in range(i, end))
        assert abs(R - want) < 1e-9, \
            f"FAIL: n=12 window {i} R {R} != brute-force {want}"
        if i < T12 - n12:
            assert disc == gamma12 ** n12 and ns is sts12[i + n12], \
                f"FAIL: n=12 window {i} bootstrap (disc/ns)"
        else:
            assert disc == 0.0 and ns[0].shape == (0, D), \
                f"FAIL: n=12 window {i} terminal collapse"
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
    for i, (s, a, R, ns, nsm, disc, st) in enumerate(buf4.items):
        want = sum(gamma12 ** (k - i) * rs12[k] for k in range(i, T_c))
        assert abs(R - want) < 1e-9 and ns is sts12[T_c] \
            and abs(disc - gamma12 ** (T_c - i)) < 1e-15, \
            f"FAIL: n=12 censored window {i} (R/ns/disc)"
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


def _selftest_mask(seed=777):
    """Re-issue masking semantics end-to-end:
      (1) mask bits: after issuing (i, L10), candidate (i, L10) is masked
          next step while (i, R10) / (i, route_parallel) are not; after
          issuing (i, R10), (i, L10) unmasks; NOOP never masked;
      (2) selection: the argmax NEVER lands on a masked candidate, and
          equals the best UNMASKED candidate (checked by masking the
          current argmax); epsilon-sampling never draws a masked one;
      (3) target side, hand-built batch: the stored ns_mask bits divert
          the Double-DQN argmax away from the masked online-argmax
          candidate (padding validity handled independently)."""
    print("[selftest] re-issue masking (selection, epsilon, target side)")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    D, N = 60, 5
    agent = ControllerAgent(token_dim=D, device="cpu", mask_reissue=True)
    agent.q_net.eval()
    cs_list = [f"AIR-{i:02d}" for i in range(N)]
    toks = np.random.randn(N, D).astype(np.float32)

    # (1) mask-bit semantics
    i_tgt = 2
    cs = cs_list[i_tgt]
    m = build_reissue_mask(cs_list, {cs: 0})            # issued (i, L10)
    assert not m[0], "FAIL: NOOP masked"
    assert m[1 + 3 * i_tgt + 0], "FAIL: (i, L10) not masked after L10"
    assert not m[1 + 3 * i_tgt + 1] and not m[1 + 3 * i_tgt + 2], \
        "FAIL: (i, R10)/(i, route_parallel) wrongly masked"
    assert m.sum() == 1, "FAIL: unrelated candidates masked"
    m2 = build_reissue_mask(cs_list, {cs: 1})           # then (i, R10)
    assert not m2[1 + 3 * i_tgt + 0], \
        "FAIL: (i, L10) still masked after intervening R10"
    assert m2[1 + 3 * i_tgt + 1], "FAIL: (i, R10) not masked after R10"

    # (2) selection excludes masked argmax; picks best unmasked
    idx0, _, _ = agent.select_candidate(toks, c=0.5)
    cl, cu = agent.candidate_q(toks)
    scores = cl + 0.5 * (cu - cl)
    mask_arg = np.zeros(1 + 3 * N, dtype=bool)
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
        assert aux["cand_idx"] != 1 + 3 * i_tgt + 0, \
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
        mm = np.zeros(1 + 3 * ntoks[b].shape[0], dtype=bool)
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
          f"diverted on all {B} hand-built rows")


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
    disc, stratum) with variable aircraft counts."""
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
        items.append((s, a_idx, float(rng.randn()), ns, None,
                      GAMMA ** NSTEP, int(rng.randint(0, N_STRATA))))
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
    """E1 adds NO parameters: 59016 base, +12 with width_scalars,
    invariant to the e1 flags."""
    print("[selftest] E1 parameter neutrality")
    torch.manual_seed(0)
    base = sum(p.numel() for p in ControllerQNet(60).parameters())
    ws = sum(p.numel() for p in
             ControllerQNet(60, width_scalars=True).parameters())
    assert base == 59016, f"FAIL: base param count {base} != 59016"
    assert ws == 59016 + 12, f"FAIL: width_scalars count {ws} != 59028"
    a_on = _e1_synthetic_agent(1, token_dim=60, e1_width=True)
    a_off = _e1_synthetic_agent(1, token_dim=60)
    assert a_on.param_count() == a_off.param_count() == base, \
        "FAIL: e1 flags changed the parameter count"
    print(f"  OK  {base} params (+12 width_scalars), unchanged by E1")


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


def run_selftest():
    print("=" * 100)
    print("CONTROLLER-FRAME SELF-TESTS")
    print("=" * 100)
    _selftest_permutation()
    _selftest_padding()
    _selftest_telescoping()
    _selftest_windower()
    _selftest_mask()
    _selftest_delivery_bonus()
    _selftest_e1_default_off()
    _selftest_e1_negatives()
    _selftest_e1_hinge()
    _selftest_e1_no_contamination()
    _selftest_e1_params()
    net = ControllerQNet(60)
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params < 100000, f"FAIL: {n_params} params >= 100k budget"
    print(f"[selftest] parameter budget: {n_params} < 100000  OK")
    _selftest_cbp()
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
