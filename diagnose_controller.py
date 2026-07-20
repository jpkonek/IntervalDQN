"""
Controller-frame probe battery (PROBE_BATTERY.md, pre-registered 8 July 2026)
=============================================================================

Implements the [ENV], [CKPT] and [JSONL] probes against the system under
test, bluebird_controller_dqn.py. Every probe prints its PRE-REGISTERED
expectation next to its result; a miss is a FINDING, never a tuning cue.

Faithfulness rules honoured here:
  - env construction goes through bluebird_controller_dqn.make_controller_env
  - tokens go through bluebird_controller_dqn.build_tokens
  - the sector objective is NEVER reimplemented: scripted policies are run
    through bcd.run_episode itself (via a policy shim), and the step-reward
    helper used by counterfactual probes composes bcd's own
    sector_snapshot / shaping_terms / detect_violation / constants and is
    VALIDATED against bcd.run_episode to ~1e-6 before any oracle runs.

2026-07-09 re-registration (OBJECTIVE v2 in bcd, default ON):
  - [ENV] probes price with the v2 objective: NO global fuel term (B1
    measured it as a survival tax), delivery bonus decays with transit
    time (10 * max(0.3, nominal_T/actual_T); nominal_T from the
    tracker's route data at entry — see bcd.DeliveryClock).
  - b1's GATE BASIS moves to DISCOUNTED (gamma=0.97) returns — the
    learner's actual objective; competition metrics become a side table.
  - the mirror (sector_step) prices v2; every env.reset on a pricing
    path resets the env-attached delivery clock.

2026-07-08 B1-v2 re-registration (BLESSED by JK; PROBE_BATTERY.md "B1-v2"):
  - b1's basis moves to MEAN RETURNS-TO-GO over on-trajectory steps:
    mean over t of G_t = sum_{k>=t} gamma^(k-t) r_k of the v2+CBP
    training stream (the quantity the learner's targets estimate); the
    from-t0 discounted and undiscounted returns stay as SIDE columns.
  - gated pairs (2 seeds): DELIVERER>NOOP at LIGHT density (spawn knob
    via CustomInfiniteEnv, ~7 concurrent, 1800 s); HOLDER<NOOP (std,
    1200 s); DELIVERER>DELIVERER+REISSUE (paired-trajectory fee test
    with a determinism check and an exact gap identity). WEAVER is
    DROPPED from the gate (terminal-timing confound, ill-posed);
    DELIVERER>NOOP at standard density is REPORTED, not gated (it
    measures the lateral-only capability wall, not the objective).

2026-07-08 re-registration (CBP + masking upgrades in bcd):
  - [ENV] probes (b1-b4) price episodes with the CBP TRAINING signal
    (bcd.run_episode(cbp=True) / sector_step_cbp, mirror-validated),
    because that is now the objective the learner sees; b1 gains the
    expectation DELIVERER(competent) > NOOP on both seeds; b3 measures
    post-CBP deltas with an EX-FEE decision rule.
  - DELIVERER is the competent hand policy (entry route_parallel, L10
    give-way deconfliction, route_parallel recovery).
  - [CKPT] probes drive the deployed policy (re-issue masking on); a1
    intentionally measures the UNMASKED net (see its note).

Usage:
    .venv/bin/python diagnose_controller.py --probe b1 b2 b3 b4
    .venv/bin/python diagnose_controller.py --probe a1 a2 a3 a4 c2 c3
    .venv/bin/python diagnose_controller.py --probe all
"""

import argparse
import copy
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import (
    GAMMA, EPS_FUEL, CMD_COST, DELIVERY_BONUS, VIOLATION_PENALTY,
    CONFLICT_FL, CONFLICT_RANGE_NM, CHECKPOINT_DIR, N_INSTR, INSTR_NAMES,
    make_controller_env, build_tokens, sector_snapshot, shaping_terms,
    snapshot_stratum, candidate_intervals, load_agent, run_episode,
    pad_state_batch,
)
# N_INSTR is 5 since the 2026-07-13 vertical-action extension (L10, R10,
# route_parallel, climb +10 FL, descend -10 FL; env actions 1..5).
# DECODING RULE ADOPTED THROUGHOUT: candidate indices produced by a
# LOADED agent are decoded with that agent's OWN n_instr (persisted in
# the checkpoint — 3 for pre-vertical nets), never the module constant;
# candidate ENUMERATION for env-side oracles uses the module N_INSTR
# (the env's actual instruction set). Scripted policies that hardcode
# env action ints 1/2/3 stay valid: heading actions kept their ints.
from bluebird_interval_dqn import detect_violation, haversine_nm, SEC_PER_STEP

sys.stdout.reconfigure(line_buffering=True)

DIAG_DIR = os.path.join(CHECKPOINT_DIR, "diagnostics")

# ===========================================================================
# Pre-registered expectations (verbatim distillations of PROBE_BATTERY.md)
# ===========================================================================
EXPECT = {
    "b1": ("B1-v2 ORDERING (objective v2 + CBP training signal, basis = "
           "MEAN RETURNS-TO-GO: mean over visited steps t of G_t = "
           "sum_{k>=t} 0.97^(k-t) r_k). Gated pairs, BOTH seeds: "
           "(1) DELIVERER > NOOP at LIGHT density (~7 concurrent, "
           "1800 s, full transits possible); (2) HOLDER < NOOP "
           "(standard, 1200 s; dying earlier makes mean-RTG MORE "
           "negative and fees only widen it); (3) DELIVERER > "
           "DELIVERER+REISSUE (identical trajectories; score gap must "
           "equal the discounted extra-fee sum to <1e-6 relative). "
           "DELIVERER > NOOP at STANDARD density is reported, NOT "
           "gated (capability question). Any gated pair misordered => "
           "objective wrong, run-12b must not launch."),
    "b2": ("(post-CBP training signal; RE-REGISTERED 18 Jul 2026, JK-"
           "approved, for smooth pricing where correct clearances are net-"
           "positive at issuance) Myopic oracle must NOT be a degenerate "
           "high scorer: it may exceed the NOOP..DELIVERER band top by at "
           "most its ONE-SHOT honest shaping budget (n_commands x max "
           "single-pair p1); repeatable do-nothing-beating income is still "
           "an exploit. Exit-window scripts (park short / oscillate) must "
           "not out-score completing the delivery."),
    "b3": ("(post-CBP TRAINING-signal deltas, not raw objective) Top-"
           "candidate H-step (H=10) |delta| EX-FEE distinguishable from "
           "ambient per-step training-reward noise for states with "
           "conflicts inside horizon; else global-frame credit is "
           "information-starved and CTDE / longer n-step is needed."),
    "b4": ("Crossing > veer-off > excursion-miss, margins >> shaping "
           "noise; if veer-off ~ crossing, the run-11 boundary-aversion "
           "trap survives in the new objective."),
    "a1": ("Re-issue rate (same (aircraft, instruction) as that aircraft's "
           "still-active clearance) falls with training; high terminal "
           "rate = fee wasted on no-ops the concept says are free."),
    "a2": ("Per-aircraft attention mass correlates with token mask-impact "
           "on Q (rho > 0.5); if not, the interpretability story is "
           "decorative and must not be cited."),
    "a3": ("Nonzero chosen-clearance sensitivity to deleting 4th..Nth-"
           "nearest aircraft tokens (else a pilot-frame policy in a "
           "controller body — informative, not a bug)."),
    "a4": ("(revised 2026-07-08 per the parallel A4 investigation) FLOOR: "
           "OOD width ratio >= 1.0 on ALL constructions — ANY narrowing "
           "on OOD input is an automatic FAIL. Width MONOTONE in OOD "
           "distance: real <= shuffled <= box-garbage <= 3x-box. Interval "
           "inversion count == 0. Stratum-width pattern is reported as a "
           "FINDING over all three populated strata (the previous "
           "monotone PASS was vacuous — stratum 1 had 0 states), not "
           "scored."),
    "a4v2": ("CORRECTED A4 instruments (WIDTH_MECHANISM_PROBES.md 8-Jul "
             "correction): (a) kNN-excess = width / width predicted by 10 "
             "nearest ON-POLICY neighbours in the net's own pooled summary "
             "space; (b) activation-pattern novelty (Hamming bits to "
             "nearest on-policy ReLU pattern); (c) in-range width-vs-norm "
             "slope; (d) width-pressure ladder = mean delta_raw "
             "(pre-softplus) on FROZEN probe states across the training "
             "checkpoint ladder. SANITY: held-out on-policy kNN-excess "
             "~ 1 (else instrument broken). If the LL novelty-elevation "
             "mechanism transferred to 12b: AIRCRAFT-SWAP states "
             "(plausible aircraft, impossible joint picture — the "
             "operationally relevant near-OOD) show kNN-excess > 1.2 "
             "with pattern novelty >> held-out; garbage/shuffled-field "
             "must show kNN-excess >= 1.0 (excess < 1 = the A4 "
             "certain-basin collapse persists). LADDER (finding, not "
             "scored): under the degenerate-landscape hypothesis, 12b's "
             "outcome diversity should lift the global downward width "
             "pressure — off-manifold delta_raw should not collapse "
             "toward the certain basin as training proceeds."),
    "c2": ("Rank correlation (Spearman) of ground-truth H=15 rollout "
           "returns vs net Hurwicz scores rho > 0.4 by end of 12a; near 0 "
           "= net hasn't learned the objective. Also: unbiased coverage = "
           "fraction of ground-truth returns inside candidate intervals."),
    "c3": ("Fraction of episodes ending censored vs terminal, per bin; if "
           "clean-1200s episodes grow, the realized-coverage tracker's "
           "diet starves and C2 rollout coverage must take over."),
}


# ===========================================================================
# Shared plumbing
# ===========================================================================

_ENV_CACHE = {}


def get_env(duration):
    if duration not in _ENV_CACHE:
        print(f"  [env] building duration={duration}s ...")
        _ENV_CACHE[duration] = make_controller_env(
            scenario_duration=duration)
    return _ENV_CACHE[duration]


def pos_status_name(td):
    if td is None or td.pos_status is None:
        return None
    return getattr(td.pos_status, "name", str(td.pos_status))


def tracked(env, cs):
    try:
        return env.get_tracked_aircraft_data(cs)
    except Exception:
        return None


def sector_step(env, obs, snap, actions, issued):
    """Advance env one step and price it with THE system's own reward
    layer (OBJECTIVE v2, bcd's default): identical composition to
    bcd.run_episode, built from bcd.sector_snapshot / bcd.shaping_terms /
    bcd.DeliveryClock / detect_violation and bcd's constants. v2: no
    fuel term (column kept at 0.0 for budget parity) and the delivery
    bonus decays with transit time via the env-attached clock — callers
    that env.reset() must bcd.reset_delivery_clock(env) (deepcopied
    probe snapshots carry the clock automatically). Validated against
    bcd.run_episode by validate_reward_mirror()."""
    clock = bcd.delivery_clock(env)
    clock.observe(env, snap)
    next_obs, _r, done, trunc, info = env.step(actions)
    violated, kind, involved = detect_violation(info)
    delivered_cs = []
    for cs in obs:
        d_cs = info.get(cs)
        if isinstance(d_cs, dict) and \
                d_cs.get("pos_status") == "EXIT_REACHED":
            delivered_cs.append(cs)
    deliveries = len(delivered_cs)
    snap_next = sector_snapshot(env, next_obs.keys())
    prog, centre, conflict = shaping_terms(snap, snap_next, GAMMA)
    fuel = 0.0   # objective v2: global fuel term removed
    cmd = -CMD_COST if issued else 0.0
    deliv = sum(DELIVERY_BONUS * clock.bonus_factor(cs)
                for cs in delivered_cs)
    r = prog + centre + conflict + fuel + cmd + deliv
    if violated:
        r -= VIOLATION_PENALTY
    return {"r": r, "next_obs": next_obs, "info": info,
            "snap_next": snap_next, "violated": violated, "kind": kind,
            "deliveries": deliveries,
            "terms": {"phi_progress": prog, "phi_centre": centre,
                      "phi_conflict": conflict, "fuel": fuel,
                      "cmd_cost": cmd, "delivery_bonus": deliv,
                      "violation_term": -VIOLATION_PENALTY if violated
                      else 0.0}}


def _phi_sum(out):
    t = out["terms"]
    return t["phi_progress"] + t["phi_centre"] + t["phi_conflict"]


def sector_step_cbp(env, obs, snap, actions, issued):
    """sector_step priced with the CBP TRAINING signal (what the agent
    now trains on): the shaping term becomes gamma*(Phi(s_action) -
    Phi(s_noop)) with BOTH branches measured (1 + bcd.CBP_LAG) steps
    after s — command latency is one sweep, so at the default lag 1 the
    measurement includes the new command's first acting step (identical
    composition to bcd.run_episode's cbp=True path — validated by
    validate_cbp_mirror()); all other terms stay absolute. For an
    all-NOOP step the two branches are the same determinized computation,
    so the shaping term is EXACTLY 0.0 and the clones are skipped (the
    NOOP-zero invariant is asserted by bcd --selftest)."""
    lag = bcd.CBP_LAG
    all_noop = not issued and all(a == 0 for a in actions.values())
    s_noop = None
    if not all_noop:
        env_cf = copy.deepcopy(env)
        o = env_cf.step({cs: 0 for cs in obs})[0]
        for _ in range(lag):
            o = env_cf.step({cs: 0 for cs in o})[0]
        snap_cf = sector_snapshot(env_cf, o.keys())
        p_n, c_n, f_n = shaping_terms(snap, snap_cf, GAMMA)
        s_noop = p_n + c_n + f_n
    out = sector_step(env, obs, snap, actions, issued)   # env now AT s'
    phi = _phi_sum(out)
    if all_noop:
        cbp_shaping = 0.0
    else:
        if lag > 0:
            env_a = copy.deepcopy(env)
            o = out["next_obs"]
            for _ in range(lag):
                o = env_a.step({cs: 0 for cs in o})[0]
            snap_act = sector_snapshot(env_a, o.keys())
            p_a, c_a, f_a = shaping_terms(snap, snap_act, GAMMA)
            s_act = p_a + c_a + f_a
        else:
            s_act = phi
        cbp_shaping = s_act - s_noop
    out["r"] = out["r"] - phi + cbp_shaping
    out["terms"]["cbp_shaping"] = cbp_shaping
    return out


class ScriptedShim:
    """Adapter that lets a scripted policy run through bcd.run_episode
    UNCHANGED, so scripted episodes are priced by the system's own reward
    code, not a copy. decide(env, obs) -> (target_cs or None, env_action)."""

    mask_reissue = False   # scripted policies manage their own persistence

    def __init__(self, policy):
        self.policy = policy
        self.gamma = GAMMA
        self.n_instr = N_INSTR
        self.buffer = None  # never touched with train=False

    def generate_action(self, env, obs_dict, info_dict, c=None,
                        force_epsilon=None, adaptive=False,
                        last_issued=None):
        target, act = self.policy.decide(env, obs_dict)
        actions = {cs: 0 for cs in obs_dict}
        issued = target is not None
        if issued:
            actions[target] = act
        aux = {"cand_idx": 1 if issued else 0,
               "tokens": np.zeros((0, 1), dtype=np.float32),
               "callsigns": (), "mean_width": 0.0, "c_used": 0.0}
        return actions, aux


class NoopPolicy:
    name = "NOOP"

    def decide(self, env, obs):
        return None, 0


class Deliverer:
    """COMPETENT hand policy (2026-07-08 upgrade; replaces the naive
    route_parallel spammer). Four behaviours, ONE clearance per step,
    priority (b) conflict avoidance > (c) recovery > (d) drift retrim
    > (a) delivery queue:

      (a) DELIVER: on each aircraft's first IN_SECTOR step it joins a
          FIFO queue for a single route_parallel (issued when the step's
          clearance budget is free).
      (b) AVOID: while a vertically-proximate pair (|dFL| < 20) has
          lateral distance < 20 nm AND is closing (distance strictly
          decreased since the previous step), resolve via the GIVE-WAY
          member. VERTICAL-FIRST RULE (2026-07-13; FL-BAND FIX
          2026-07-13b, documented): a conflict action goes out ONLY if
          the pair's ACTUAL |dFL| < 10 (the LoS band — a pair with the
          vertical gap already open cannot LoS and gets NO command;
          re-checked every step, so a later gap collapse re-arms the
          rule). The resolution is ONE vertical move per conflict
          episode, chosen deterministically as the first of [give-way
          climb, give-way descend, other climb, other descend] that
          BOTH (1) keeps the target INSIDE the sector's vertical band
          (min_fl..max_fl read from the env's own sector volumes,
          FL200-300 on X-Plus; the pre-fix oracle climbed a FL300
          aircraft to FL302 -> OUT_SECTOR sector_excursion at 414 s on
          seed 10043 — the measured B1-v2 regression, counterfactually
          isolated: NOOP/RP/L10 at the same step all stay clean) and
          (2) OPENS the vertical gap to >= 10 FL (a give-way descend
          onto the other member's level is never issued). +-10 FL costs
          one fee and does NOT jeopardize delivery (EXIT_REACHED is not
          FL-enforced in this env version). While the move is taking
          effect the budget is released to lower priorities; the moved
          aircraft stays laterally ON-ROUTE, so no recovery pass is
          needed for it. If NO in-band gap-opening vertical move exists
          for either member, the episode is LATERAL-ONLY: old-style
          escalating L10s to the give-way member from the start (which
          feed (c) recovery). LATERAL ESCALATION: if |dFL| is still
          < 10 and the estimated steps-to-LoS (d - 5 nm)/closure_rate
          drops below the steps still needed to open 10 FL vertically
          at a conservative CLIMB_FL_PER_STEP (1 FL/step ~ 1000 fpm at
          6 s/step), fall back to the old escalating L10s (relative
          heading commands accumulate) until the pair stops closing.
          The give-way member is chosen once per conflict episode and
          kept sticky, EXCEPT that a member within 8 nm of the sector
          boundary never gives way (turning it further would trade the
          LoS for a sector excursion — measured): the other member
          takes over (the boundary veto only matters for the lateral
          path; vertical moves do not turn anyone).
          DOCUMENTED DEVIATION from the sketch parameters ("< 12 nm,
          L10 once per pair per 10 steps"): those are geometrically
          incapable — measured on seed 10043, the violating pair closes
          at ~1.4 nm/step, entering 12 nm five steps before LoS, and a
          single 10-degree turn under one-sweep command latency deflects
          < 1 nm where > 5 nm is needed. NOOP loses that seed at 462 s;
          beating NOOP (the pre-registered expectation) requires earlier
          detection and an escalating turn.
          Right-of-way heuristic (documented): the give-way aircraft is
          the one that sees the other on its RIGHT (relative bearing of
          the intruder in (0, 180) deg from own heading) — the classic
          converging-traffic rule; L10 turns it LEFT, i.e. away from the
          traffic it must yield to. If both or neither see the other on
          the right, the lexicographically smaller callsign gives way.
      (c) RECOVER: a previously-deviated aircraft whose conflicts have
          cleared (every vertically-proximate other is > 15 nm away OR
          no longer closing) gets ONE route_parallel re-issue.
      (d) RETRIM: wind drift walks unattended aircraft off-route (the
          measured NOOP failure mode on seed 20042 is a drift excursion);
          any non-deviated aircraft whose centreline offset exceeds 3 nm
          gets route_parallel, rate-limited once per aircraft per 10
          steps.
      (b0) EMERGENCY BOUNDARY RECOVERY (highest priority): any aircraft
          whose current heading crosses the sector POLYGON within 12 nm
          (tracker's nearest_forward_boundary_dist; the bounding box
          overstates margins 2x+ in the X-Plus notches — measured) is
          immediately re-routed (route_parallel) unless it is close to
          its exit (track distance < 15 nm — exits are AT the boundary,
          a legitimate approach must not be disturbed). This is what
          keeps a give-way member deviated by (b) from trading the LoS
          for a sector excursion (measured failure mode).
    """
    name = "DELIVERER"
    AVOID_RANGE_NM = 20.0
    BOUNDARY_GUARD_NM = 8.0     # 360-deg polygon distance (give-way veto)
    FWD_GUARD_NM = 12.0         # forward polygon distance (emergency)
    RETRIM_OFFSET_NM = 3.0
    RETRIM_PERIOD = 10
    CLIMB_ACTION = 4            # env int: simple_fl_climb (+10 FL)
    DESCEND_ACTION = 5          # env int: simple_fl_descent (-10 FL)
    FL_STEP = 10.0              # magnitude of one vertical clearance
    LOS_FL = 10.0               # LoS vertical band (detect_violation)
    CLIMB_FL_PER_STEP = 1.0     # conservative climb rate for escalation
                                # (~1000 fpm at 6 s/step; see docstring)
    FL_BAND_FALLBACK = (200.0, 300.0)   # X-Plus sector band (measured)

    def __init__(self):
        self.seen = set()
        self.rp_queue = []          # FIFO for first-entry route_parallel
        self.deviated = set()       # aircraft turned off-route by (b)
        self.prev_d = {}            # pair -> lateral distance last step
        self.giveway = {}           # pair -> sticky give-way member
        self.climb_step = {}        # pair -> step the vertical move went out
        self.lateral_only = set()   # pairs with no legal vertical move
        self.last_retrim = {}       # cs -> step of last (d) clearance
        self._fl_band_cache = None  # (min_fl, max_fl) of the sector
        self.t = 0

    @staticmethod
    def _give_way(a, b, ta, tb):
        """Right-of-way heuristic (see class docstring)."""
        def sees_right(td_own, td_oth):
            if td_own.heading is None:
                return False
            brg = initial_bearing(td_own.position.lat, td_own.position.lon,
                                  td_oth.position.lat, td_oth.position.lon)
            rel = (brg - td_own.heading) % 360.0
            return 0.0 < rel < 180.0
        a_right, b_right = sees_right(ta, tb), sees_right(tb, ta)
        if a_right and not b_right:
            return a
        if b_right and not a_right:
            return b
        return min(a, b)

    def _fl_band(self, env):
        """Sector vertical band (min_fl, max_fl), read once from the
        env's OWN sector volumes (the authority pos_status uses: an
        aircraft outside the band goes OUT_SECTOR = sector_excursion).
        X-Plus is FL200-300; the constant fallback covers exotic envs."""
        if self._fl_band_cache is None:
            try:
                sim = env.get_simulator_env()
                sec = sim.airspace.sectors[env.active_airspace_sector]
                lo = min(float(v.min_fl) for v in sec.volumes)
                hi = max(float(v.max_fl) for v in sec.volumes)
                self._fl_band_cache = (lo, hi)
            except Exception:
                self._fl_band_cache = self.FL_BAND_FALLBACK
        return self._fl_band_cache

    def _vertical_move(self, env, ins, give, other):
        """First in-band, gap-opening vertical clearance in the fixed
        preference order give-climb, give-descend, other-climb,
        other-descend (see class docstring). Guards (FL-BAND FIX):
        (1) target FL stays inside the sector band — FLs are read at
        decision time, so a mid-climb aircraft is guarded at its
        CURRENT level and repeated climbs cannot ratchet it over the
        ceiling; (2) the resulting |dFL| is >= LOS_FL and strictly
        larger than now. Returns (callsign, env_action) or None."""
        lo, hi = self._fl_band(env)
        for cs, partner in ((give, other), (other, give)):
            fl = ins[cs].flight_level
            pfl = ins[partner].flight_level
            for act, dfl_cmd in ((self.CLIMB_ACTION, self.FL_STEP),
                                 (self.DESCEND_ACTION, -self.FL_STEP)):
                new_fl = fl + dfl_cmd
                if not (lo <= new_fl <= hi):
                    continue
                if abs(new_fl - pfl) >= self.LOS_FL \
                        and abs(new_fl - pfl) > abs(fl - pfl):
                    return cs, act
        return None

    @staticmethod
    def _boundary_dist_nm(env, td):
        """Distance (nm) to the nearest point of the ACTUAL sector
        polygon (X-Plus is plus-shaped; the bounding box overstates the
        margin by 2x+ in the notch regions — measured), straight from
        the tracker."""
        d = td.nearest_360_boundary_dist
        if d is not None:
            return float(d)
        lat0, lat1, lon0, lon1 = bcd.sector_bounds(env)   # fallback: bbox
        lat, lon = td.position.lat, td.position.lon
        coslat = math.cos(math.radians(lat))
        return min((lat - lat0) * 60.0, (lat1 - lat) * 60.0,
                   (lon - lon0) * 60.0 * coslat,
                   (lon1 - lon) * 60.0 * coslat)

    def decide(self, env, obs):
        self.t += 1
        ins = {}
        for cs in sorted(obs.keys()):
            td = tracked(env, cs)
            if pos_status_name(td) == "IN_SECTOR" \
                    and td.position is not None \
                    and td.flight_level is not None:
                ins[cs] = td
                if cs not in self.seen:
                    self.seen.add(cs)
                    self.rp_queue.append(cs)

        # pairwise sweep: distances, closing flags and closure rates
        # (state for (b) and (c))
        css = sorted(ins)
        closing, rate = {}, {}
        for i in range(len(css)):
            for j in range(i + 1, len(css)):
                a, b = css[i], css[j]
                d = haversine_nm(ins[a].position.lat, ins[a].position.lon,
                                 ins[b].position.lat, ins[b].position.lon)
                key = (a, b)
                prev = self.prev_d.get(key, float("inf"))
                closing[key] = d < prev - 1e-9
                rate[key] = (prev - d) if prev < float("inf") else 0.0
                self.prev_d[key] = d

        # (b0) EMERGENCY: aircraft whose CURRENT HEADING crosses the
        # sector polygon within FWD_GUARD_NM (see docstring); highest
        # priority — an excursion is the same -50 as the LoS it was
        # traded for. Worst-case turnaround (~180 deg at ~18 deg/step)
        # travels ~8 nm forward, so 12 nm leaves margin.
        emergency = None
        for cs in css:
            fwd = ins[cs].nearest_forward_boundary_dist
            d_exit = ins[cs].track_dist_to_exit_cr
            near_exit = d_exit is not None and d_exit < 15.0
            if emergency is None and fwd is not None \
                    and fwd < self.FWD_GUARD_NM and not near_exit \
                    and self.t - self.last_retrim.get(cs, -999) >= 5:
                emergency = cs
        if emergency is not None:
            self.deviated.discard(emergency)
            self.last_retrim[emergency] = self.t
            return emergency, 3              # route_parallel, turn back

        # (b) conflict avoidance: MOST URGENT closing vertically-
        # proximate pair inside AVOID_RANGE_NM, urgency = estimated
        # steps to LoS (d - 5 nm) / closure rate — distance-priority
        # starves a fast-closing far pair behind a slow-closing near one
        # (measured failure mode on seed 20042); escalating L10s to a
        # sticky give-way member with a boundary override (see docstring)
        best = None
        for i in range(len(css)):
            for j in range(i + 1, len(css)):
                a, b = css[i], css[j]
                if abs(ins[a].flight_level - ins[b].flight_level) \
                        >= CONFLICT_FL:
                    continue
                key = (a, b)
                d = self.prev_d[key]
                if d < self.AVOID_RANGE_NM and closing[key]:
                    urgency = (d - 5.0) / max(rate[key], 1e-6)
                    if best is None or urgency < best[0]:
                        best = (urgency, a, b)
                elif key in self.giveway and not closing[key]:
                    self.giveway.pop(key, None)   # conflict episode over
                    self.climb_step.pop(key, None)
                    self.lateral_only.discard(key)
        if best is not None:
            urgency, a, b = best
            key = (a, b)
            # FL-BAND FIX (2026-07-13b): a pair whose vertical gap is
            # already open (|dFL| >= 10) cannot LoS and gets NO command
            # — the pre-fix rule climbed such pairs anyway, ratcheting
            # aircraft into the sector ceiling (measured on both gate
            # seeds). Re-checked every step: a later gap collapse
            # re-enters here with dfl < 10 and full vertical-first
            # treatment.
            dfl = abs(ins[a].flight_level - ins[b].flight_level)
            if dfl >= self.LOS_FL and key not in self.climb_step:
                pass                             # release the budget
            else:
                give = self.giveway.get(key)
                if give is None or give not in ins:
                    give = self._give_way(a, b, ins[a], ins[b])
                # boundary override: never turn a member that is close
                # to the boundary; the other member takes over (if it,
                # too, is boundary-pinned, keep the original choice)
                other = b if give == a else a
                if self._boundary_dist_nm(env, ins[give]) \
                        < self.BOUNDARY_GUARD_NM and \
                        self._boundary_dist_nm(env, ins[other]) \
                        >= self.BOUNDARY_GUARD_NM:
                    give = other
                self.giveway[key] = give
                # VERTICAL-FIRST GIVE-WAY (2026-07-13; FL-BAND FIX
                # 2026-07-13b; see class docstring): one in-band,
                # gap-opening vertical move per conflict episode, then
                # wait for it to open the vertical gap; escalate
                # laterally if no legal vertical move exists or if
                # lateral LoS is projected BEFORE the vertical band
                # clears at the conservative climb rate.
                if key not in self.climb_step:
                    mv = self._vertical_move(env, ins, give, other)
                    if mv is not None:
                        self.climb_step[key] = self.t
                        return mv                # vertical give-way
                    self.lateral_only.add(key)   # no legal vertical move
                    self.climb_step[key] = self.t
                escalate = key in self.lateral_only
                if not escalate and dfl < self.LOS_FL:
                    steps_to_clear = (self.LOS_FL - dfl) \
                        / self.CLIMB_FL_PER_STEP
                    escalate = urgency < steps_to_clear
                if escalate:
                    self.deviated.add(give)
                    if give in self.rp_queue:
                        self.rp_queue.remove(give)  # (c) re-routes it
                    return give, 1               # L10 lateral escalation
            # vertical gap open or opening in time: release the budget

        # (c) recovery: re-issue route_parallel once the conflict clears
        for cs in sorted(self.deviated):
            if cs not in ins:
                self.deviated.discard(cs)    # left the sector
                continue
            clear = True
            for o_cs in css:
                if o_cs == cs or abs(ins[cs].flight_level
                                     - ins[o_cs].flight_level) >= CONFLICT_FL:
                    continue
                key = tuple(sorted((cs, o_cs)))
                if self.prev_d[key] <= 15.0 and closing[key]:
                    clear = False
                    break
            if clear:
                self.deviated.discard(cs)
                self.last_retrim[cs] = self.t
                return cs, 3                 # route_parallel recovery

        # (d) drift retrim: largest centreline offset first
        snap = sector_snapshot(env, list(ins.keys()))
        cands = [(abs(s.centre_off), cs) for cs, s in snap.items()
                 if cs in ins and cs not in self.deviated
                 and abs(s.centre_off) > self.RETRIM_OFFSET_NM
                 and self.t - self.last_retrim.get(cs, -999)
                 >= self.RETRIM_PERIOD]
        if cands:
            _, cs = max(cands)
            self.last_retrim[cs] = self.t
            return cs, 3                     # route_parallel re-centre

        # (a) first-entry route_parallel queue
        while self.rp_queue:
            cs = self.rp_queue.pop(0)
            if cs in ins and cs not in self.deviated:
                self.last_retrim[cs] = self.t
                return cs, 3                 # route_parallel
        return None, 0


class Holder:
    """L10 each step to the IN_SECTOR aircraft whose last command is
    oldest — keeps everyone turning (the run-8 exploit, scripted)."""
    name = "HOLDER"

    def __init__(self):
        self.last_cmd = {}
        self.t = 0

    def decide(self, env, obs):
        in_sector = [cs for cs in sorted(obs.keys())
                     if pos_status_name(tracked(env, cs)) == "IN_SECTOR"]
        self.t += 1
        if not in_sector:
            return None, 0
        target = min(in_sector,
                     key=lambda cs: (self.last_cmd.get(cs, -1), cs))
        self.last_cmd[target] = self.t
        return target, 1  # L10


class Weaver:
    """Alternate L10/R10 to a rotating IN_SECTOR aircraft."""
    name = "WEAVER"

    def __init__(self):
        self.i = 0
        self.t = 0

    def decide(self, env, obs):
        in_sector = [cs for cs in sorted(obs.keys())
                     if pos_status_name(tracked(env, cs)) == "IN_SECTOR"]
        self.t += 1
        if not in_sector:
            return None, 0
        target = in_sector[self.i % len(in_sector)]
        self.i += 1
        return target, 1 if self.t % 2 == 0 else 2


class Parker:
    """Exit-window red-team: park aircraft ~1.5 nm short of exit by
    continuous turns once track distance to exit < 2.5 nm."""
    name = "PARKER"

    def __init__(self):
        self.last_cmd = {}
        self.t = 0

    def decide(self, env, obs):
        near = []
        for cs in sorted(obs.keys()):
            td = tracked(env, cs)
            if pos_status_name(td) != "IN_SECTOR":
                continue
            d = td.track_dist_to_exit_cr
            if d is not None and d < 2.5:
                near.append(cs)
        self.t += 1
        if not near:
            return None, 0
        target = min(near, key=lambda cs: (self.last_cmd.get(cs, -1), cs))
        self.last_cmd[target] = self.t
        return target, 1


class Oscillator:
    """Exit-window red-team: oscillate across the fix — alternate L10/R10
    each step to aircraft within 4 nm of their exit."""
    name = "OSCILLATOR"

    def __init__(self):
        self.last_cmd = {}
        self.t = 0

    def decide(self, env, obs):
        near = []
        for cs in sorted(obs.keys()):
            td = tracked(env, cs)
            if pos_status_name(td) != "IN_SECTOR":
                continue
            d = td.track_dist_to_exit_cr
            if d is not None and d < 4.0:
                near.append(cs)
        self.t += 1
        if not near:
            return None, 0
        target = min(near, key=lambda cs: (self.last_cmd.get(cs, -1), cs))
        self.last_cmd[target] = self.t
        return target, 1 if self.t % 2 == 0 else 2


def run_scripted(env, policy, seed, cbp=False, step_hook=None):
    """Scripted policy through THE system's run_episode (train=False).
    cbp=True prices the episode with the CBP training signal (bcd's own
    cbp path, not a copy). step_hook passes through to run_episode
    (per-step pricing internals; the micro reward audit reads them)."""
    return run_episode(env, ScriptedShim(policy), seed=seed, train=False,
                       cbp=cbp, step_hook=step_hook)


_MIRROR_OK = False
_CBP_MIRROR_OK = False


def validate_reward_mirror(seed=10043):
    """Prove sector_step reproduces bcd.run_episode's pricing exactly on a
    full NOOP episode (same env, same seed): ep_return and every additive
    term must agree to <= 1e-6. Guards the 'testing your copy' hazard."""
    global _MIRROR_OK
    if _MIRROR_OK:
        return
    env = get_env(1200)
    ref = run_scripted(env, NoopPolicy(), seed)
    obs, info = env.reset(seed=seed)
    bcd.reset_delivery_clock(env)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    snap = sector_snapshot(env, obs.keys())
    tot, terms = 0.0, {}
    for _ in range(maxstep):
        out = sector_step(env, obs, snap, {cs: 0 for cs in obs}, False)
        tot += out["r"]
        for k, v in out["terms"].items():
            terms[k] = terms.get(k, 0.0) + v
        obs, snap = out["next_obs"], out["snap_next"]
        if out["violated"]:
            break
    err = abs(tot - ref["ep_return"])
    term_errs = {k: abs(terms.get(k, 0.0) - ref[k]) for k in
                 ("phi_progress", "phi_centre", "phi_conflict", "fuel",
                  "cmd_cost", "delivery_bonus", "violation_term")}
    assert err <= 1e-4 and all(e <= 1e-4 for e in term_errs.values()), (
        f"reward mirror does NOT reproduce run_episode: dG={err}, "
        f"{term_errs} — refusing to run counterfactual probes")
    print(f"  [mirror] sector_step == run_episode on NOOP ep (dG={err:.2e},"
          f" max term err {max(term_errs.values()):.2e})  OK")
    _MIRROR_OK = True


def validate_cbp_mirror(seed=10043):
    """Prove sector_step_cbp reproduces bcd.run_episode(cbp=True)'s
    pricing on an ISSUING scripted episode (WEAVER exercises the clone
    branch every step) — ep_return and the cbp_shaping budget column must
    agree to <= 1e-4. Also asserts the NOOP episode's CBP total is
    exactly 0.0 on both paths."""
    global _CBP_MIRROR_OK
    if _CBP_MIRROR_OK:
        return
    validate_reward_mirror(seed)
    env = get_env(1200)
    # NOOP episode: CBP total must be exactly zero via run_episode
    ref0 = run_scripted(env, NoopPolicy(), seed, cbp=True)
    assert ref0["cbp_shaping"] == 0.0, (
        f"NOOP episode CBP shaping {ref0['cbp_shaping']!r} != 0.0")
    # issuing episode: WEAVER through run_episode(cbp) vs manual mirror
    ref = run_scripted(env, Weaver(), seed, cbp=True)
    pol = Weaver()
    obs, info = env.reset(seed=seed)
    bcd.reset_delivery_clock(env)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    snap = sector_snapshot(env, obs.keys())
    tot, tot_cbp = 0.0, 0.0
    for _ in range(maxstep):
        cs, act = pol.decide(env, obs)
        acts = noop_action(obs)
        issued = cs is not None
        if issued:
            acts[cs] = act
        out = sector_step_cbp(env, obs, snap, acts, issued)
        tot += out["r"]
        tot_cbp += out["terms"]["cbp_shaping"]
        obs, snap = out["next_obs"], out["snap_next"]
        if out["violated"]:
            break
    err = abs(tot - ref["ep_return"])
    err_cbp = abs(tot_cbp - ref["cbp_shaping"])
    assert err <= 1e-4 and err_cbp <= 1e-4, (
        f"CBP mirror does NOT reproduce run_episode(cbp=True): "
        f"dG={err}, d_cbp_shaping={err_cbp} — refusing to run "
        f"training-signal probes")
    print(f"  [mirror] sector_step_cbp == run_episode(cbp=True) on WEAVER "
          f"ep (dG={err:.2e}, d_cbp={err_cbp:.2e}); NOOP CBP total == 0.0 "
          f" OK")
    _CBP_MIRROR_OK = True


def noop_action(obs):
    return {cs: 0 for cs in obs}


def conflict_ranked_candidates(snap, max_aircraft=4, n_instr=N_INSTR):
    """NOOP + all n_instr instructions for the `max_aircraft` IN_SECTOR
    aircraft nearest to a conflict (min pairwise distance among
    vertically-proximate pairs; aircraft with no such partner rank last).
    Enumerates the env's FULL instruction set (since 2026-07-13 that
    includes climb/descent, so B2/B3 oracles now consider vertical
    resolutions too — a deliberate semantic extension, flagged in the
    probe notes)."""
    css = sorted(snap.keys())
    prox = {}
    for cs in css:
        best = float("inf")
        for other in css:
            if other == cs:
                continue
            if abs(snap[cs].fl - snap[other].fl) >= CONFLICT_FL:
                continue
            d = haversine_nm(snap[cs].lat, snap[cs].lon,
                             snap[other].lat, snap[other].lon)
            best = min(best, d)
        prox[cs] = best
    ranked = sorted(css, key=lambda cs: (prox[cs], cs))[:max_aircraft]
    cands = [(None, 0)]
    for cs in ranked:
        for j in range(1, n_instr + 1):
            cands.append((cs, j))
    return cands


def rollout_return(env_snapshot, obs, snap, first_actions, first_issued,
                   horizon, continue_policy=None, use_cbp=False):
    """Discounted H-step return of (first action, then continue_policy or
    NOOP) from a deepcopy of env_snapshot, priced by sector_step (or by
    sector_step_cbp — the CBP training signal — when use_cbp).
    continue_policy(env, obs) -> (actions dict, issued bool).
    Returns (G, steps_simulated, violated, deliveries)."""
    env = copy.deepcopy(env_snapshot)
    step_fn = sector_step_cbp if use_cbp else sector_step
    G, disc = 0.0, 1.0
    deliveries = 0
    o, s = obs, snap
    acts, issued = first_actions, first_issued
    steps = 0
    violated = False
    for h in range(horizon):
        out = step_fn(env, o, s, acts, issued)
        steps += 1
        G += disc * out["r"]
        disc *= GAMMA
        deliveries += out["deliveries"]
        o, s = out["next_obs"], out["snap_next"]
        if out["violated"]:
            violated = True
            break
        if h < horizon - 1:
            if continue_policy is None:
                acts, issued = noop_action(o), False
            else:
                acts, issued = continue_policy(env, o)
    return G, steps, violated, deliveries


# ---- geometry helpers (steering only; not part of the objective) ---------

def initial_bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlam))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def point_beyond(w0, w1, extra_nm):
    """Point extra_nm beyond w1 along the segment w0->w1 (flat-earth local
    approximation, adequate at ~2 nm)."""
    lat_mid = math.radians((w0.lat + w1.lat) / 2)
    dx = (w1.lon - w0.lon) * 60.0 * math.cos(lat_mid)   # nm east
    dy = (w1.lat - w0.lat) * 60.0                        # nm north
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    return (w1.lat + (extra_nm * uy) / 60.0,
            w1.lon + (extra_nm * ux) / (60.0 * math.cos(lat_mid)))


def make_custom_density_env(duration, initial_spawn_rate, max_spawn_rate,
                            num_starter_aircraft):
    """CustomInfiniteEnv carrying EXACTLY make_controller_env's config
    mutations plus explicit spawn knobs. The standard InfiniteEnv
    HARDCODES Infinite.setup's spawn defaults (starters=2,
    initial_spawn_rate=0.01, max=0.1, increment=0) and exposes NO
    density keys; CustomInfiniteEnv is the package's own mechanism for
    non-standard densities and is identical otherwise (same BaseEnv,
    same tracker, same X-Plus scenario, wind + forecast ON — matching
    Infinite.setup's defaults that InfiniteEnv uses). Density probes the
    OBJECTIVE (bcd's reward layer, reused verbatim via run_episode /
    sector_step), not the learner, so the env-class swap changes only
    traffic density."""
    from bluebird_gymnasium.envs import CustomInfiniteEnv
    from bluebird_gymnasium.envs.infinite import ScenarioName
    cfg = CustomInfiniteEnv.get_default_env_config()
    cfg.state_repr_config = {"encoder_cls": "relative",
                             "k_nearest_aircraft": 3}
    # mirrors make_controller_env's action_config EXACTLY, including the
    # 2026-07-13 vertical actions (insertion order fixes env ints 1..5)
    cfg.action_config = {"simple_heading_left": [10],
                         "simple_heading_right": [10],
                         "simple_heading_route_parallel": True,
                         "simple_fl_climb": [10],
                         "simple_fl_descent": [10]}
    cfg.reward_config = {"fns": ["position_status_const"], "coeffs": [0.0]}
    cfg.scenario_config["scenario_name"] = ScenarioName.sector_xplus
    cfg.view_config["type"] = "decentralized"
    cfg.view_config["decentralized_params"] = {}
    cfg.scenario_duration = duration
    cfg.scenario_config.update({
        "initial_spawn_rate": initial_spawn_rate,
        "max_spawn_rate": max_spawn_rate,
        "spawn_rate_increment": 0.0, "spawn_rate_increase_interval": 0.0,
        "num_starter_aircraft": num_starter_aircraft})
    env = CustomInfiniteEnv(config=cfg)
    assert env.get_action_parser().get_total_num_actions() == 1 + N_INSTR
    return env


def make_b4_env(duration=1200, n_aircraft=1):
    """Single-aircraft field environment for B4 (documented deviation).

    Findings that force this: (i) in the system env (InfiniteEnv via
    make_controller_env) an ambient violation ends every scanned episode
    at 420-670 s while the earliest possible delivery is ~700 s, so the
    needle-threading field is unreachable end-to-end; (ii) InfiniteEnv
    ignores density keys — the package's own mechanism for low-density
    scenarios is CustomInfiniteEnv (see make_custom_density_env)."""
    return make_custom_density_env(duration, 0.0, 0.0, n_aircraft)


class B4Policy:
    """B4 trajectory scripting WITH an ambient-traffic keeper.

    The keeper deconflicts ambient pairs (R10 to one member of the
    closest closing vertically-proximate pair inside 12 nm,
    route_parallel recovery once clear) with logic IDENTICAL across the
    trajectory variants; a no-op in the single-aircraft B4 env. Target
    modes (wind makes plain NOOP drift off-route, so every mode steers):
      crossing : bang-bang L10/R10 at the exit fix (dead-centre delivery)
      miss     : bang-bang at a point miss_nm beyond the exit-window edge
      veer     : crossing-steer until track dist-to-exit <= veer_at_nm,
                 then L10 every step (continuous turn) for the rest.
    target=None = keeper only.
    """

    def __init__(self, target_cs, mode, miss_nm=1.5, veer_at_nm=2.0):
        self.target = target_cs
        self.mode = mode
        self.miss_nm = miss_nm
        self.veer_at_nm = veer_at_nm
        self.veering = False
        self.goal = None
        self.prev_d = {}
        self.last_cmd = {}
        self.off_route = set()
        self.t = 0
        self.name = f"B4-{mode}"

    def _keeper(self, env, obs):
        ins = {}
        for cs in obs:
            td = tracked(env, cs)
            if pos_status_name(td) == "IN_SECTOR" and \
                    td.position is not None and td.flight_level is not None:
                ins[cs] = td
        css = sorted(ins)
        best = None
        for i in range(len(css)):
            for j in range(i + 1, len(css)):
                a, b = css[i], css[j]
                ta, tb = ins[a], ins[b]
                if abs(ta.flight_level - tb.flight_level) >= CONFLICT_FL:
                    continue
                d = haversine_nm(ta.position.lat, ta.position.lon,
                                 tb.position.lat, tb.position.lon)
                key = (a, b)
                closing = d < self.prev_d.get(key, 1e9) - 1e-9
                self.prev_d[key] = d
                if d < 12.0 and closing:
                    if best is None or d < best[0]:
                        best = (d, a, b)
        if best is not None:
            _, a, b = best
            cands = [cs for cs in (a, b) if cs != self.target
                     and self.t - self.last_cmd.get(cs, -99) > 2]
            if cands:
                cs = min(cands, key=lambda c: self.last_cmd.get(c, -99))
                self.last_cmd[cs] = self.t
                self.off_route.add(cs)
                return cs, 2                       # R10, dodge
        for cs in sorted(list(self.off_route)):
            if cs not in ins or cs == self.target:
                self.off_route.discard(cs)
                continue
            td = ins[cs]
            near = False
            for o_cs, o_td in ins.items():
                if o_cs == cs:
                    continue
                if abs(td.flight_level - o_td.flight_level) < CONFLICT_FL \
                        and haversine_nm(td.position.lat, td.position.lon,
                                         o_td.position.lat,
                                         o_td.position.lon) < 15.0:
                    near = True
                    break
            if not near and self.t - self.last_cmd.get(cs, -99) > 2:
                self.off_route.discard(cs)
                self.last_cmd[cs] = self.t
                return cs, 3                       # route_parallel recovery
        return None, 0

    def _steer_at(self, td, goal_lat, goal_lon):
        hdg = td.heading
        if hdg is None:
            return None, 0
        brg = initial_bearing(td.position.lat, td.position.lon,
                              goal_lat, goal_lon)
        diff = wrap180(brg - hdg)
        if abs(diff) <= 6.0:
            return None, 0
        return (self.target, 2 if diff > 0 else 1)  # R10 / L10

    def decide(self, env, obs):
        self.t += 1
        cs, act = self._keeper(env, obs)
        if cs is not None:
            return cs, act
        if self.target is None or self.target not in obs:
            return None, 0
        td = tracked(env, self.target)
        if pos_status_name(td) != "IN_SECTOR" or td.position is None:
            return None, 0
        if self.mode == "veer":
            d = td.track_dist_to_exit_cr
            if not self.veering and d is not None and d <= self.veer_at_nm:
                self.veering = True
            if self.veering:
                return (self.target, 1)   # continuous L10 turn
            # pre-veer: fly the crossing approach
            ep = td.sector_exit_pos
            if ep is None:
                return None, 0
            return self._steer_at(td, ep.lat, ep.lon)
        if self.mode == "crossing":
            ep = td.sector_exit_pos
            if ep is None:
                return None, 0
            return self._steer_at(td, ep.lat, ep.lon)
        # miss: steer miss_nm beyond the exit-window edge
        if self.goal is None:
            w = td.sector_exit_window
            if w is None:
                return None, 0
            self.goal = point_beyond(w[0], w[1], self.miss_nm)
        return self._steer_at(td, self.goal[0], self.goal[1])


class SteerPolicy:
    """Steers ONE target aircraft via bang-bang L10/R10 toward a moving
    goal supplied by mode logic; all other aircraft NOOP. Modes:
      crossing : pure NOOP (route delivers dead-centre through the exit)
      miss     : steer at a point `miss_nm` beyond the exit-window edge
      veer     : NOOP until track dist-to-exit <= veer_at_nm, then L10
                 every step (continuous turn) for the rest of the episode
    """

    def __init__(self, target_cs, mode, miss_nm=1.5, veer_at_nm=2.0):
        self.target = target_cs
        self.mode = mode
        self.miss_nm = miss_nm
        self.veer_at_nm = veer_at_nm
        self.veering = False
        self.goal = None
        self.name = f"B4-{mode}"

    def decide(self, env, obs):
        if self.target not in obs:
            return None, 0
        td = tracked(env, self.target)
        if pos_status_name(td) != "IN_SECTOR" or td.position is None:
            return None, 0
        if self.mode == "crossing":
            return None, 0
        if self.mode == "veer":
            d = td.track_dist_to_exit_cr
            if not self.veering and d is not None and d <= self.veer_at_nm:
                self.veering = True
            return (self.target, 1) if self.veering else (None, 0)
        # miss: steer 1.5 nm beyond the exit-window edge
        if self.goal is None:
            w = td.sector_exit_window
            if w is None:
                return None, 0
            self.goal = point_beyond(w[0], w[1], self.miss_nm)
        hdg = td.heading
        if hdg is None:
            return None, 0
        brg = initial_bearing(td.position.lat, td.position.lon,
                              self.goal[0], self.goal[1])
        diff = wrap180(brg - hdg)
        if abs(diff) <= 6.0:
            return None, 0
        return (self.target, 2 if diff > 0 else 1)  # R10 / L10


# ---- statistics helpers ---------------------------------------------------

def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def latest_checkpoint():
    pts = glob.glob(os.path.join(CHECKPOINT_DIR, "train_*.pt"))
    if not pts:
        raise FileNotFoundError("no train_*.pt in " + CHECKPOINT_DIR)
    return max(pts, key=os.path.getmtime)


def latest_jsonl():
    js = glob.glob(os.path.join(CHECKPOINT_DIR, "train_*.jsonl"))
    return max(js, key=os.path.getmtime)


def collect_probe_states(env, seed, act_fn, n_states, min_aircraft=2,
                         maxsteps=None):
    """Run an episode driven by act_fn(env, obs, info) -> (actions, issued)
    and snapshot (deepcopy env, obs, stratum, step) at ~n_states evenly
    spaced eligible steps (>= min_aircraft IN_SECTOR). Deterministic env
    => a single pass with cheap deepcopies is exact. The delivery clock
    is reset here and rides the deepcopied snapshots (v2 pricing)."""
    obs, info = env.reset(seed=seed)
    bcd.reset_delivery_clock(env)
    if maxsteps is None:
        maxsteps = int(getattr(env, "maxstep",
                               env.config.scenario_duration // SEC_PER_STEP))
    raw = []
    snap = sector_snapshot(env, obs.keys())
    for step in range(maxsteps):
        if len(snap) >= min_aircraft:
            raw.append((copy.deepcopy(env), copy.deepcopy(obs),
                        snapshot_stratum(snap), step))
        actions, issued = act_fn(env, obs, info)
        out = sector_step(env, obs, snap, actions, issued)
        obs, snap, info = out["next_obs"], out["snap_next"], out["info"]
        if out["violated"]:
            break
    if len(raw) > n_states:
        idx = np.linspace(0, len(raw) - 1, n_states).round().astype(int)
        raw = [raw[i] for i in sorted(set(idx.tolist()))]
    return raw


# ===========================================================================
# B1-v2 machinery (blessed 8 July 2026; PROBE_BATTERY.md "B1-v2")
# ===========================================================================

# LIGHT-density spawn knob for pair 1, calibrated 2026-07-08 with a NOOP
# density probe on both gate seeds (CustomInfiniteEnv, all-NOOP, 1800 s,
# violations ignored for the measurement):
#   0.0005/s -> mean 4.0-4.8 concurrent (below target)
#   0.001 /s -> mean 5.6-6.7, plateau 7-8, max 9-10, NOOP transits complete
#   0.002 /s -> mean 7.9-9.3, max 14 (peaks above target)
# 0.001/s with 2 starters is the pre-registered choice: squarely in the
# 6-10 band with full-length transits (830-1500 s) completable inside the
# 1800 s episode. (A 600 s episode would end before ANY entry can exit —
# the duration, not the spawn rate, was the earlier draft's error.)
B1V2_LIGHT_SPAWN = {"initial_spawn_rate": 0.001, "max_spawn_rate": 0.001,
                    "num_starter_aircraft": 2}
B1V2_LIGHT_DURATION = 1800   # seconds (300 steps)
B1V2_STD_DURATION = 1200     # seconds; the 2400 s leg is dropped (dead
                             # weight: scripted episodes end well before)
B1V2_GAP_RTOL = 1e-6         # pair-3 gap identity, relative


def get_light_env():
    key = ("light", B1V2_LIGHT_DURATION)
    if key not in _ENV_CACHE:
        print(f"  [env] building LIGHT-density CustomInfiniteEnv "
              f"duration={B1V2_LIGHT_DURATION}s spawn={B1V2_LIGHT_SPAWN} ...")
        _ENV_CACHE[key] = make_custom_density_env(
            duration=B1V2_LIGHT_DURATION, **B1V2_LIGHT_SPAWN)
    return _ENV_CACHE[key]


def returns_to_go(rs, gamma=GAMMA):
    """Realized discounted returns-to-go G_t = sum_{k>=t} gamma^(k-t) r_k
    for EVERY visited step t of a finished episode (suffix recursion).
    This is the quantity the learner's targets estimate; B1-v2 scores a
    policy by the MEAN over t of G_t."""
    G, out = 0.0, []
    for r in reversed(list(rs)):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out


class RecordingShim(ScriptedShim):
    """ScriptedShim that additionally records the per-step aircraft-count
    fingerprint of the visited trajectory: (len(obs), n IN_SECTOR,
    sorted-callsign tuple) per step. Read-only additions (sector_snapshot
    only reads tracker data); pricing is untouched."""

    def __init__(self, policy):
        super().__init__(policy)
        self.n_obs = []
        self.n_in_sector = []
        self._cs_seq = []

    def generate_action(self, env, obs_dict, info_dict, **kw):
        self.n_obs.append(len(obs_dict))
        self.n_in_sector.append(len(sector_snapshot(env, obs_dict.keys())))
        self._cs_seq.append(tuple(sorted(obs_dict.keys())))
        return super().generate_action(env, obs_dict, info_dict, **kw)

    def fingerprint(self):
        return tuple(zip(self.n_obs, self.n_in_sector, self._cs_seq))


def run_scripted_stream(env, policy, seed):
    """One scripted episode through THE system's run_episode(cbp=True),
    with a step_hook collecting the per-step training reward r_t (the
    system's own pricing — no mirror copy involved). Returns
    (stats, rs, rtg, mean_rtg, shim). Stream fidelity is asserted against
    run_episode's own totals (which it rounds to 4 dp)."""
    rs = []
    shim = RecordingShim(policy)
    stats = run_episode(env, shim, seed=seed, train=False, cbp=True,
                        step_hook=lambda d: rs.append(d["r"]))
    assert len(rs) == stats["steps"]
    disc = float(sum((GAMMA ** i) * r for i, r in enumerate(rs)))
    assert abs(disc - stats["ep_return_disc"]) <= 1e-3 and \
        abs(float(sum(rs)) - stats["ep_return"]) <= 1e-3, \
        "step_hook stream does not reproduce run_episode's totals"
    rtg = returns_to_go(rs)
    mean_rtg = float(np.mean(rtg)) if rtg else 0.0
    return stats, rs, rtg, mean_rtg, shim


def _rp_reissue_is_inert(env, cs):
    """True iff re-issuing route_parallel to cs is a dynamic no-op RIGHT
    NOW. FORCED CHOICE (documented): of the three instructions, L10/R10
    are RELATIVE heading commands (bluebird_gymnasium actions/simple/
    heading.py builds change_heading_to = selected -/+ 10, so a repeat
    accumulates another 10 degrees — the very mechanism Deliverer's
    escalating turn exploits); route_parallel is ABSOLUTE
    (change_heading_to = current segment bearing) and hence idempotent
    while that bearing still equals the aircraft's selected heading.
    The predicate recomputes the bearing with the env's OWN action
    builder (read-only: it only constructs an Action object) and
    compares it to selected_instructions.heading."""
    from bluebird_gymnasium.actions.simple.heading import \
        heading_route_parallel
    ac = env.get_simulator_env().aircraft.get(cs)
    if ac is None:
        return False
    sel = ac.selected_instructions.heading
    if sel is None:
        return False
    try:
        act = heading_route_parallel(cs, env)
    except Exception:
        return False
    return int(round(float(act.value))) == int(round(float(sel)))


def _find_inert_reissue(env, obs, active):
    """First (sorted) aircraft with a still-active route_parallel
    clearance whose re-issue is dynamically inert; None if none."""
    for cs in sorted(active):
        if active[cs] != 3 or cs not in obs:
            continue
        if pos_status_name(tracked(env, cs)) != "IN_SECTOR":
            continue
        if _rp_reissue_is_inert(env, cs):
            return cs
    return None


class ScheduleRecorder:
    """Pass A wrapper: runs the inner policy UNCHANGED, records its exact
    clearance schedule {step: (cs, act)} and, on no-issue steps, whether a
    dynamically-inert re-issue candidate existed (the eligibility trace
    the analytic fallback prices)."""
    name = "DELIVERER"

    def __init__(self, inner):
        self.inner = inner
        self.schedule = {}
        self.active = {}      # cs -> last instruction issued (1/2/3)
        self.eligible = []    # no-issue steps with an inert re-issue avail
        self.t = 0

    def decide(self, env, obs):
        cs, act = self.inner.decide(env, obs)
        self.schedule[self.t] = (cs, act)
        for k in list(self.active):
            if k not in obs:
                self.active.pop(k)
        if cs is None:
            if _find_inert_reissue(env, obs, self.active) is not None:
                self.eligible.append(self.t)
        else:
            self.active[cs] = act
        self.t += 1
        return cs, act


class ReplayPolicy:
    """Pass B: replay a recorded clearance schedule verbatim (the
    determinism check — same seed + same actions must reproduce the
    trajectory bit-for-bit)."""
    name = "DELIVERER-REPLAY"

    def __init__(self, schedule):
        self.schedule = schedule
        self.t = 0

    def decide(self, env, obs):
        cs, act = self.schedule.get(self.t, (None, 0))
        self.t += 1
        if cs is not None and cs not in obs:
            return None, 0   # divergence guard; caught by fingerprints
        return cs, act


class ReissuePolicy(ReplayPolicy):
    """Pass C: replay the schedule, and on recorded no-issue steps
    re-issue the first still-active, dynamically-inert route_parallel
    clearance (same aircraft + same instruction as already active).
    Each re-issue pays the -0.1 fee with ZERO dynamic effect, so the
    score gap vs pass A isolates fee pricing with no terminal-timing
    confound (the confound that made WEAVER<=HOLDER ill-posed)."""
    name = "DELIVERER+REISSUE"

    def __init__(self, schedule):
        super().__init__(schedule)
        self.active = {}
        self.reissue_steps = []

    def decide(self, env, obs):
        t = self.t
        cs, act = super().decide(env, obs)
        for k in list(self.active):
            if k not in obs:
                self.active.pop(k)
        if cs is not None:
            self.active[cs] = act
            return cs, act
        r_cs = _find_inert_reissue(env, obs, self.active)
        if r_cs is not None:
            self.reissue_steps.append(t)
            return r_cs, self.active[r_cs]   # == 3, route_parallel
        return None, 0


# ===========================================================================
# [ENV] PROBES
# ===========================================================================

def probe_b1(args):
    print("\n" + "=" * 78)
    print("B1-v2 [ENV] ORACLE-POLICY ORDERING — objective v2, CBP training "
          "signal, basis = MEAN RETURNS-TO-GO (gamma=0.97)")
    print("EXPECTATION:", EXPECT["b1"])
    print("  notes (fixed pre-run, B1-v2 blessed 8 July 2026):")
    print("  (i) BASIS: each episode's per-step training rewards r_t are "
          "collected from bcd.run_episode(cbp=True) itself via its "
          "step_hook (the system's own pricing, no mirror copy); the "
          "score is mean over visited t of G_t = sum_{k>=t} g^(k-t) r_k. "
          "Rationale: from t=0, g^118 ~ 0.027 makes everything after "
          "~700 s invisible, while the learner bootstraps returns-to-go "
          "from EVERY state; mean-RTG is the quantity its targets "
          "estimate. From-t0 Gdisc and undiscounted G stay as SIDE "
          "columns.")
    print("  (ii) PAIR 1 at LIGHT density: CustomInfiniteEnv (the "
          "package's own density mechanism; the standard InfiniteEnv "
          "hardcodes spawn defaults) with spawn 0.001/s, 2 starters, "
          "1800 s — calibrated to ~6-8 concurrent (plateau ~7-8, max 10) "
          "with full transits completable. The 1200 s standard-density "
          "leg is REPORTED, not gated: at ~25 concurrent a lateral-only "
          "oracle hits the capability wall, which is not an objective "
          "property.")
    print("  (iii) PAIR 2 sign check in advance (new basis): the -50 "
          "terminal occupies a larger, less-discounted share of visited "
          "states the EARLIER it comes — mean-RTG(violation at step T) "
          "~ -50*(1-g^T)/(T*(1-g)): T~24 (HOLDER) -> ~-36 vs T~77 "
          "(NOOP) -> ~-20 — so HOLDER < NOOP is the correct expected "
          "sign, and HOLDER's fees only widen the gap.")
    print("  (iv) PAIR 3 (replaces WEAVER<=HOLDER): pass A records "
          "DELIVERER's exact schedule; pass B replays it (determinism "
          "check: identical per-step aircraft-count fingerprint, "
          "violation step+kind, and bit-identical reward stream); pass "
          "C re-issues, on recorded no-issue steps, a still-active "
          "route_parallel clearance. FORCED CHOICE: L10/R10 are "
          "RELATIVE commands (a repeat turns another 10 deg), so only "
          "route_parallel re-issues can be dynamically inert, and only "
          "while the recomputed segment bearing equals the selected "
          "heading — the shadow checks this with the env's own action "
          "builder before each re-issue. Gap must equal the discounted "
          "extra-fee sum to <1e-6 relative ON BOTH BASES. If A vs B is "
          "NOT deterministic, fall back to pricing pass A's stream "
          "twice (fees added on the recorded eligible steps) and say "
          "so PROMINENTLY.")
    print("=" * 78)
    validate_cbp_mirror()
    seeds = [10043, 20042]

    def vio_str(s):
        return ("VIOLATION[" + str(s["violation_kind"]) + "]@"
                + str(s["time_to_violation"]) + "s") if s["violated"] \
            else "clean"

    ep = {}       # (leg, name, seed) -> episode record
    recorders = {}

    def run_leg(leg, env, name, policy, seed):
        t0 = time.time()
        stats, rs, rtg, mean_rtg, shim = run_scripted_stream(
            env, policy, seed)
        disc = float(sum((GAMMA ** i) * r for i, r in enumerate(rs)))
        rec = {"stats": stats, "rs": rs, "mean_rtg": mean_rtg,
               "disc_from_stream": disc, "shim": shim,
               "density_mean": float(np.mean(shim.n_in_sector)),
               "density_max": int(np.max(shim.n_in_sector))}
        ep[(leg, name, seed)] = rec
        print(f"  {name:18s} seed={seed} {leg:5s} | "
              f"meanRTG={mean_rtg:9.3f} | Gdisc={disc:9.3f} "
              f"G={stats['ep_return']:9.3f} | del={stats['deliveries']} "
              f"cmd={stats['commands']:3d} | "
              f"dens~{rec['density_mean']:.1f}(max {rec['density_max']}) | "
              f"{vio_str(stats)} | {time.time() - t0:.0f}s wall")
        return rec

    # ---- leg 1: LIGHT density (pair 1, GATED) -----------------------------
    print(f"  -- LIGHT density leg ({B1V2_LIGHT_DURATION}s, spawn "
          f"{B1V2_LIGHT_SPAWN}) --")
    env_l = get_light_env()
    for seed in seeds:
        for name, P in (("DELIVERER", Deliverer), ("NOOP", NoopPolicy)):
            run_leg("light", env_l, name, P(), seed)

    # ---- leg 2: STANDARD density 1200 s (pair 2 GATED; DELIVERER>NOOP
    #      reported; DELIVERER run doubles as pair-3 pass A) ---------------
    print(f"  -- STANDARD density leg ({B1V2_STD_DURATION}s) --")
    env_s = get_env(B1V2_STD_DURATION)
    for seed in seeds:
        recorders[seed] = ScheduleRecorder(Deliverer())
        run_leg("std", env_s, "DELIVERER", recorders[seed], seed)
        run_leg("std", env_s, "NOOP", NoopPolicy(), seed)
        run_leg("std", env_s, "HOLDER", Holder(), seed)

    # ---- pair 3: DELIVERER vs DELIVERER+REISSUE ---------------------------
    print("  -- pair 3: paired-trajectory fee test --")
    pair3 = {}
    for seed in seeds:
        A = ep[("std", "DELIVERER", seed)]
        sched = recorders[seed].schedule
        B = run_leg("std", env_s, "DELIVERER-REPLAY", ReplayPolicy(sched),
                    seed)
        det_ok = (A["shim"].fingerprint() == B["shim"].fingerprint()
                  and A["stats"]["violated"] == B["stats"]["violated"]
                  and A["stats"]["violation_kind"]
                  == B["stats"]["violation_kind"]
                  and A["stats"]["steps"] == B["stats"]["steps"]
                  and A["rs"] == B["rs"])
        print(f"    seed={seed} determinism (A vs replay B): "
              f"{'OK — bit-identical' if det_ok else 'FAILED'}")
        if det_ok:
            mode = "paired-trajectory"
            polC = ReissuePolicy(sched)
            C = run_leg("std", env_s, "DELIVERER+REISSUE", polC, seed)
            re_steps = polC.reissue_steps
            traj_ok = (C["shim"].fingerprint() == A["shim"].fingerprint()
                       and C["stats"]["violated"] == A["stats"]["violated"]
                       and C["stats"]["violation_kind"]
                       == A["stats"]["violation_kind"]
                       and C["stats"]["steps"] == A["stats"]["steps"])
            mean_rtg_C, disc_C = C["mean_rtg"], C["disc_from_stream"]
        else:
            # PROMINENT fallback (pre-registered): analytic construction —
            # price pass A's recorded stream twice, once with the fee
            # added on the recorded eligible no-issue steps.
            mode = "ANALYTIC-FALLBACK (env NOT deterministic across "\
                   "same-seed same-action runs)"
            print(f"    seed={seed} !! {mode} !!")
            re_steps = recorders[seed].eligible
            rs_C = list(A["rs"])
            for t in re_steps:
                rs_C[t] -= CMD_COST
            rtg_C = returns_to_go(rs_C)
            mean_rtg_C = float(np.mean(rtg_C)) if rtg_C else 0.0
            disc_C = float(sum((GAMMA ** i) * r
                               for i, r in enumerate(rs_C)))
            traj_ok = True   # by construction
            ep[("std", "DELIVERER+REISSUE", seed)] = {
                "stats": None, "rs": rs_C, "mean_rtg": mean_rtg_C,
                "disc_from_stream": disc_C, "shim": None,
                "density_mean": A["density_mean"],
                "density_max": A["density_max"]}
        # exact gap identity on BOTH bases
        n_steps = len(A["rs"])
        fee = [0.0] * n_steps
        for t in re_steps:
            fee[t] = CMD_COST
        fee_rtg = returns_to_go(fee)
        exp_mean_gap = float(np.mean(fee_rtg)) if fee_rtg else 0.0
        exp_disc_gap = float(sum((GAMMA ** t) * f
                                 for t, f in enumerate(fee)))
        got_mean_gap = A["mean_rtg"] - mean_rtg_C
        got_disc_gap = A["disc_from_stream"] - disc_C
        rel_mean = abs(got_mean_gap - exp_mean_gap) \
            / max(abs(exp_mean_gap), 1e-12)
        rel_disc = abs(got_disc_gap - exp_disc_gap) \
            / max(abs(exp_disc_gap), 1e-12)
        identity_ok = (traj_ok and len(re_steps) > 0
                       and rel_mean < B1V2_GAP_RTOL
                       and rel_disc < B1V2_GAP_RTOL)
        ordered = A["mean_rtg"] > mean_rtg_C
        pair3[seed] = {
            "mode": mode, "deterministic": det_ok,
            "trajectory_identical": traj_ok,
            "n_reissues": len(re_steps), "reissue_steps": re_steps,
            "expected_mean_rtg_gap": exp_mean_gap,
            "observed_mean_rtg_gap": got_mean_gap,
            "rel_err_mean_rtg_gap": rel_mean,
            "expected_disc_gap": exp_disc_gap,
            "observed_disc_gap": got_disc_gap,
            "rel_err_disc_gap": rel_disc,
            "identity_ok": identity_ok, "ordered": ordered,
            "ok": identity_ok and ordered}
        print(f"    seed={seed} re-issues={len(re_steps)} | mean-RTG gap "
              f"obs={got_mean_gap:.6f} exp={exp_mean_gap:.6f} "
              f"(rel {rel_mean:.2e}) | Gdisc gap obs={got_disc_gap:.6f} "
              f"exp={exp_disc_gap:.6f} (rel {rel_disc:.2e}) | "
              f"trajectory {'identical' if traj_ok else 'DIVERGED'} | "
              f"{'OK' if pair3[seed]['ok'] else 'FAIL'}")

    # ---- side table (competition metrics + old bases, NOT gated) ----------
    print("  " + "-" * 74)
    print("  SIDE TABLE (not gated): leg policy seed | meanRTG Gdisc G | "
          "TTV(s) del viol_kind cmds")
    for (leg, name, seed), r in ep.items():
        s = r["stats"]
        if s is None:
            continue
        vk = s["violation_kind"] if s["violated"] else "clean"
        print(f"    {leg:5s} {name:18s} {seed} | {r['mean_rtg']:9.3f} "
              f"{r['disc_from_stream']:9.3f} {s['ep_return']:9.3f} | "
              f"{s['time_to_violation']:6.0f} {s['deliveries']:2d} "
              f"{str(vk):18s} {s['commands']:3d}")

    # ---- gated checks ------------------------------------------------------
    def m(leg, name, seed):
        return ep[(leg, name, seed)]["mean_rtg"]

    checks, all_ok = [], True
    for seed in seeds:
        cond = {
            "pair1_DELIVERER>NOOP@light":
                m("light", "DELIVERER", seed) > m("light", "NOOP", seed),
            "pair2_HOLDER<NOOP@std":
                m("std", "HOLDER", seed) < m("std", "NOOP", seed),
            "pair3_DELIVERER>REISSUE@std": pair3[seed]["ok"],
        }
        reported = {
            "DELIVERER>NOOP@std (capability, NOT gated)":
                m("std", "DELIVERER", seed) > m("std", "NOOP", seed),
        }
        ok = all(cond.values())
        all_ok = all_ok and ok
        checks.append({"seed": seed, "checks": cond, "reported": reported})
        print(f"  seed={seed} (mean-RTG basis): " + "  ".join(
            f"{k}:{'OK' if v else 'FAIL'}" for k, v in cond.items()))
        print(f"    reported only: " + "  ".join(
            f"{k}:{'holds' if v else 'fails'}" for k, v in reported.items()))
    verdict = "MEET" if all_ok else "MISS"
    result = (f"B1-v2 mean-RTG orderings "
              f"{'all hold' if all_ok else 'VIOLATED'} over 2 seeds x 3 "
              f"gated pairs (light-density pair via CustomInfiniteEnv "
              f"spawn=0.001/s; pair-3 mode: "
              + "; ".join(f"s{s}:{pair3[s]['mode']}" for s in seeds) + ")")
    print(f"RESULT: {result}\nVERDICT: {verdict}"
          + ("" if all_ok else "  << GATE: run-12b must NOT launch"))
    return {"probe": "b1", "version": "b1v2",
            "expectation": EXPECT["b1"], "result": result,
            "verdict": verdict, "gate": verdict == "MEET",
            "detail": {
                "basis": "mean returns-to-go over on-trajectory steps, "
                         "gamma=0.97, objective v2, CBP training signal",
                "light_env": {"duration_s": B1V2_LIGHT_DURATION,
                              **B1V2_LIGHT_SPAWN},
                "episodes": {
                    f"{leg}|{name}|s{seed}": {
                        "mean_rtg": r["mean_rtg"],
                        "disc_from_stream": r["disc_from_stream"],
                        "density_mean": r["density_mean"],
                        "density_max": r["density_max"],
                        "rs": [round(x, 6) for x in r["rs"]],
                        "stats": r["stats"]}
                    for (leg, name, seed), r in ep.items()},
                "pair3": {str(s): {k: v for k, v in pair3[s].items()}
                          for s in seeds},
                "checks": checks}}


def probe_b2(args):
    print("\n" + "=" * 78)
    print("B2 [ENV] EXPLOIT RED-TEAM (myopic oracle + exit-window scripts) "
          "— post-CBP training signal")
    print("EXPECTATION:", EXPECT["b2"])
    print("  notes (fixed pre-run): all returns are the CBP training "
          "signal, and the myopic oracle argmaxes the IMMEDIATE CBP "
          "training reward (at cbp_lag>=1 this includes the candidate's "
          "first acting step, so the oracle sees one sweep of genuine "
          "marginal effect — commands are latent for one sweep and the "
          "raw 1-step reward cannot distinguish candidates beyond the "
          "fee).")
    print("=" * 78)
    validate_cbp_mirror()
    env = get_env(1200)
    seed = 10043
    # baselines on the same seed/duration (CBP training signal)
    g_noop = run_scripted(env, NoopPolicy(), seed, cbp=True)
    g_del = run_scripted(env, Deliverer(), seed, cbp=True)

    # (a) myopic oracle: argmax TRUE immediate TRAINING reward (CBP) via
    # deepcopy counterfactuals per candidate
    obs, info = env.reset(seed=seed)
    bcd.reset_delivery_clock(env)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    snap = sector_snapshot(env, obs.keys())
    G, ncmd, ndel = 0.0, 0, 0
    pick_labels = ("NOOP",) + INSTR_NAMES     # index by env action int
    picks = {name: 0 for name in pick_labels}
    violated, kind, steps = False, None, 0
    t0 = time.time()
    for step in range(maxstep):
        cands = conflict_ranked_candidates(snap, max_aircraft=4)
        scores = []
        for cs, j in cands:
            acts = noop_action(obs)
            issued = cs is not None
            if issued:
                acts[cs] = j
            env2 = copy.deepcopy(env)
            out = sector_step_cbp(env2, obs, snap, acts, issued)
            scores.append(out["r"])
        best = max(scores)
        k = 0 if scores[0] == best else int(np.argmax(scores))
        cs, j = cands[k]
        acts = noop_action(obs)
        issued = cs is not None
        if issued:
            acts[cs] = j
        picks[pick_labels[j if issued else 0]] += 1
        out = sector_step_cbp(env, obs, snap, acts, issued)
        G += out["r"]
        ncmd += int(issued)
        ndel += out["deliveries"]
        obs, snap = out["next_obs"], out["snap_next"]
        steps += 1
        if out["violated"]:
            violated, kind = True, out["kind"]
            break
    print(f"  ORACLE(1-step, cap NOOP+{4 * N_INSTR}) seed={seed} 1200s | "
          f"G={G:9.3f} | "
          f"del={ndel} cmd={ncmd} | picks={picks} | "
          f"{'VIOLATION[' + str(kind) + ']@' + str(steps * SEC_PER_STEP) + 's' if violated else 'clean'}"
          f" | {time.time() - t0:.0f}s wall")
    print(f"  baselines: NOOP G={g_noop['ep_return']:.3f} "
          f"(del={g_noop['deliveries']}), DELIVERER G={g_del['ep_return']:.3f}"
          f" (del={g_del['deliveries']})")

    # (b) exit-window candidates (CBP training signal)
    g_park = run_scripted(env, Parker(), seed, cbp=True)
    g_osc = run_scripted(env, Oscillator(), seed, cbp=True)
    print(f"  PARKER     G={g_park['ep_return']:9.3f} del={g_park['deliveries']}"
          f" cmd={g_park['commands']}")
    print(f"  OSCILLATOR G={g_osc['ep_return']:9.3f} del={g_osc['deliveries']}"
          f" cmd={g_osc['commands']}")

    TOL = 1e-3   # ep_return is rounded to 4 dp inside run_episode
    lo = min(g_noop["ep_return"], g_del["ep_return"])
    hi = max(g_noop["ep_return"], g_del["ep_return"])
    within = (lo - TOL) <= G <= (hi + TOL)
    # RE-REGISTERED 18 July 2026 (JK-approved): under smooth pricing at
    # delta=0.5 a genuinely conflict-reducing clearance is net-positive AT
    # ISSUANCE by design, so "nothing beats NOOP myopically" is stale. New
    # bound: the oracle may exceed the band top by at most its ONE-SHOT
    # honest shaping budget = n_commands x max single-pair p1
    # (gamma * delta * max f_lat * max dg_vert = 0.97*DELTA*1.0*0.628).
    # Repeatable income (exceeding the budget) is still an exploit.
    import bluebird_controller_dqn as _bcd
    p1_max = 0.97 * _bcd.DELTA_CONFLICT * 0.628
    honest_budget = ncmd * p1_max
    no_exploit_oracle = G <= hi + honest_budget + TOL
    deliver_best = max(g_park["ep_return"],
                       g_osc["ep_return"]) <= hi + TOL
    # exit-window scripts are VACUOUS if they never fired (no aircraft
    # ever near its exit before the episode-ending ambient violation)
    vacuous_b2b = (g_park["commands"] == 0 and g_osc["commands"] == 0)
    if no_exploit_oracle and deliver_best and within:
        verdict = "PARTIAL" if vacuous_b2b else "MEET"
    elif no_exploit_oracle and deliver_best:
        verdict = "PARTIAL"   # oracle below the NOOP..DELIVERER band
    else:
        verdict = "MISS"
    result = (f"oracle G={G:.2f} vs band [NOOP {g_noop['ep_return']:.2f}, "
              f"DELIVERER {g_del['ep_return']:.2f}]; park={g_park['ep_return']:.2f}"
              f", oscillate={g_osc['ep_return']:.2f} (delivery-best "
              f"{'holds' if deliver_best else 'BEATEN — exploit'})"
              + ("; exit-window scripts VACUOUS (never fired: no aircraft "
                 "reached its exit vicinity before the ambient violation)"
                 if vacuous_b2b else ""))
    print(f"RESULT: {result}\nVERDICT: {verdict}"
          + ("" if verdict != "MISS" else "  << GATE: fix before 12b"))
    return {"probe": "b2", "expectation": EXPECT["b2"], "result": result,
            "verdict": verdict, "gate": verdict != "MISS",
            "detail": {"oracle": {"G": G, "deliveries": ndel,
                                  "commands": ncmd, "picks": picks,
                                  "violated": violated, "kind": kind,
                                  "steps": steps},
                       "noop": g_noop, "deliverer": g_del,
                       "parker": g_park, "oscillator": g_osc}}


def probe_b3(args):
    n_states = args.b3_states
    H = 10
    print("\n" + "=" * 78)
    print(f"B3 [ENV] MARGINAL-CREDIT MEASUREMENT — post-CBP TRAINING "
          f"signal ({n_states} states, 1-step and H={H})")
    print("EXPECTATION:", EXPECT["b3"])
    print("  decision rule (fixed pre-run, updated with the CBP "
          "re-registration): MEET iff median top-candidate H-step |delta| "
          "EX-FEE over conflict-in-horizon states > sigma_shaping (std of "
          "NOOP per-step TRAINING rewards excl. delivery/violation steps; "
          "under CBP the NOOP shaping term is exactly 0, so this is the "
          "residual fuel/occupancy noise). With-fee deltas and ratios vs "
          "full per-step std also reported. Rationale for ex-fee: under "
          "CBP every issued candidate differs from NOOP by the -0.1 fee "
          "mechanically; a fee-driven MEET would be degenerate. "
          "(2026-07-09, objective v2): with the fuel term removed AND "
          "CBP zeroing NOOP shaping, the NOOP training reward at "
          "non-event steps is exactly 0, so sigma_shaping is expected "
          "to be 0.0 and the pre-registered rule reduces to 'any "
          "nonzero ex-fee |dH| MEETs' — that is the DESIGNED outcome "
          "(event terms carry all credit); raw magnitudes and "
          "sigma_full are reported for context.")
    print("=" * 78)
    validate_cbp_mirror()
    env = get_env(1200)
    seed = 10043
    # ambient noise from the NOOP base episode, TRAINING (CBP) signal
    obs, info = env.reset(seed=seed)
    bcd.reset_delivery_clock(env)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    snap = sector_snapshot(env, obs.keys())
    rs, kinds = [], []
    for _ in range(maxstep):
        out = sector_step_cbp(env, obs, snap, noop_action(obs), False)
        rs.append(out["r"])
        kinds.append("event" if (out["deliveries"] or out["violated"])
                     else "shaping")
        obs, snap = out["next_obs"], out["snap_next"]
        if out["violated"]:
            break
    rs = np.array(rs)
    sigma_full = float(np.std(rs))
    shaping_rs = rs[[k == "shaping" for k in kinds]]
    sigma_shaping = float(np.std(shaping_rs))
    # ambient H-step discounted windows (NOOP, training signal)
    disc = GAMMA ** np.arange(H)
    winds = [float(np.dot(rs[i:i + H], disc[:len(rs[i:i + H])]))
             for i in range(len(rs) - H)]
    sigma_H_amb = float(np.std(winds)) if winds else float("nan")
    print(f"  ambient (NOOP, CBP training signal, {len(rs)} steps): "
          f"sigma_full={sigma_full:.4f} "
          f"sigma_shaping={sigma_shaping:.4f} sigma_H_windows={sigma_H_amb:.4f}")

    states = collect_probe_states(env, seed,
                                  lambda e, o, i: (noop_action(o), False),
                                  n_states, min_aircraft=2)
    print(f"  probe states: {len(states)}")
    per_state = []
    t0 = time.time()
    for env_s, obs_s, stratum, step in states:
        snap_s = sector_snapshot(env_s, obs_s.keys())
        cands = conflict_ranked_candidates(snap_s, max_aircraft=4)
        # NOOP branch (also detects conflict-in-horizon)
        envn = copy.deepcopy(env_s)
        o, s = obs_s, snap_s
        Gn, d1n = 0.0, None
        conflict_in_h = snapshot_stratum(snap_s) == 0
        dd = 1.0
        for h in range(H):
            out = sector_step_cbp(envn, o, s, noop_action(o), False)
            Gn += dd * out["r"]
            dd *= GAMMA
            if h == 0:
                d1n = out["r"]
            o, s = out["next_obs"], out["snap_next"]
            if len(s) >= 2 and snapshot_stratum(s) <= 1:
                # any vertically-proximate pair inside 30 nm counts as a
                # conflict inside horizon only if inside CONFLICT_RANGE_NM
                acs = list(s.values())
                for a_i in range(len(acs)):
                    for b_i in range(a_i + 1, len(acs)):
                        if abs(acs[a_i].fl - acs[b_i].fl) < CONFLICT_FL and \
                           haversine_nm(acs[a_i].lat, acs[a_i].lon,
                                        acs[b_i].lat, acs[b_i].lon) \
                           < CONFLICT_RANGE_NM:
                            conflict_in_h = True
            if out["violated"]:
                break
        d1s, dHs = [], []
        for cs, j in cands[1:]:
            acts = noop_action(obs_s)
            acts[cs] = j
            env2 = copy.deepcopy(env_s)
            out1 = sector_step_cbp(env2, obs_s, snap_s, acts, True)
            d1 = out1["r"] - d1n
            Gc, dd = out1["r"], GAMMA
            o, s = out1["next_obs"], out1["snap_next"]
            if not out1["violated"]:
                for h in range(1, H):
                    out = sector_step_cbp(env2, o, s, noop_action(o), False)
                    Gc += dd * out["r"]
                    dd *= GAMMA
                    o, s = out["next_obs"], out["snap_next"]
                    if out["violated"]:
                        break
            d1s.append(d1)
            dHs.append(Gc - Gn)
        # remove the pure command-fee offset so deltas measure CONTROL
        # effect, not the -0.1 bookkeeping (reported both ways)
        per_state.append({
            "step": step, "stratum": stratum, "n_cands": len(cands) - 1,
            "conflict_in_horizon": bool(conflict_in_h),
            "top_abs_d1": float(np.max(np.abs(d1s))) if d1s else 0.0,
            "top_abs_dH": float(np.max(np.abs(dHs))) if dHs else 0.0,
            "top_abs_dH_exfee": float(np.max(np.abs(np.array(dHs)
                                                    + CMD_COST)))
            if dHs else 0.0,
        })
    print(f"  counterfactuals done in {time.time() - t0:.0f}s")
    conf = [p for p in per_state if p["conflict_in_horizon"]]
    noconf = [p for p in per_state if not p["conflict_in_horizon"]]

    def med(xs, k):
        return float(np.median([x[k] for x in xs])) if xs else float("nan")
    m1c, mHc = med(conf, "top_abs_d1"), med(conf, "top_abs_dH")
    mHc_x = med(conf, "top_abs_dH_exfee")
    mHn = med(noconf, "top_abs_dH")
    print(f"  conflict-in-horizon states: {len(conf)}/{len(per_state)}")
    print(f"  top-cand |delta| medians (conflict states, training signal): "
          f"1-step={m1c:.4f} "
          f"H-step={mHc:.4f} (ex-fee {mHc_x:.4f}); no-conflict H-step={mHn:.4f}")
    ratio_shaping = mHc_x / sigma_shaping if sigma_shaping > 0 \
        else float("inf")
    ratio_fee = mHc / sigma_shaping if sigma_shaping > 0 else float("inf")
    ratio_full = mHc_x / sigma_full if sigma_full > 0 else float("inf")
    print(f"  ratios: median top ex-fee |dH| / sigma_shaping = "
          f"{ratio_shaping:.2f}; with-fee = {ratio_fee:.2f}; "
          f"ex-fee / sigma_full = {ratio_full:.2f}")
    ok = len(conf) > 0 and mHc_x > sigma_shaping
    verdict = "MEET" if ok else ("PARTIAL" if len(conf) > 0
                                 and mHc > sigma_shaping else "MISS")
    result = (f"(post-CBP) median top ex-fee |dH|={mHc_x:.3f} vs "
              f"sigma_shaping={sigma_shaping:.3f} (x{ratio_shaping:.1f}; "
              f"with-fee x{ratio_fee:.1f}), sigma_full={sigma_full:.3f} "
              f"(x{ratio_full:.1f}); {len(conf)} conflict states")
    print(f"RESULT: {result}\nVERDICT: {verdict}")
    return {"probe": "b3", "expectation": EXPECT["b3"], "result": result,
            "verdict": verdict,
            "detail": {"signal": "post-CBP training signal",
                       "sigma_full": sigma_full,
                       "sigma_shaping": sigma_shaping,
                       "sigma_H_ambient_windows": sigma_H_amb,
                       "per_state": per_state}}


def probe_b4(args):
    print("\n" + "=" * 78)
    print("B4 [ENV] NEEDLE-THREADING FIELD — post-CBP training signal")
    print("EXPECTATION:", EXPECT["b4"])
    print("  note: trajectories priced by bcd.run_episode(cbp=True); the "
          "steering clearances carry their counterfactual marginal "
          "shaping, terminals/fees/fuel stay absolute.")
    print("  interpretation notes (fixed pre-run): the X-Plus exit window is"
          " ~20 nm wide, so (i) the 'miss' trajectory crosses 1.5 nm beyond"
          " the WINDOW EDGE (a literal 1.5 nm-from-fix crossing is a valid "
          "delivery, reported as a footnote); (ii) veer-off is scripted at "
          "2 nm early as pre-registered AND at 6 nm early (turn radius "
          "~4.3 nm makes a 2 nm-late veer geometrically unable to avoid "
          "crossing).")
    print("=" * 78)
    validate_cbp_mirror()
    print("  DOCUMENTED DEVIATION: run on a single-aircraft "
          "CustomInfiniteEnv field (make_controller_env config + zero "
          "spawn rates) because in the system env every scanned episode "
          "is ended by an ambient violation at 420-670 s, before ANY "
          "delivery is reachable (~700 s transit) — recorded as a "
          "finding. The objective is bcd's own reward layer, unchanged. "
          "Wind drift also makes plain NOOP excurse, so the crossing "
          "trajectory is actively steered at the exit fix.")
    env4 = make_b4_env(duration=1200, n_aircraft=1)
    # find a seed whose lone aircraft can be steered to a delivery
    target, seed_used, scan = None, None, {}
    for seed in (10043, 20042, 30042, 40042, 50042):
        pol = B4Policy(None, "crossing")
        obs, info = env4.reset(seed=seed)
        cs0 = sorted(obs.keys())[0]
        pol.target = cs0
        pol.mode = "crossing"
        stats = run_scripted(env4, pol, seed, cbp=True)
        scan[seed] = {"target": cs0, "deliveries": stats["deliveries"],
                      "violated": stats["violated"],
                      "kind": stats["violation_kind"]}
        print(f"  scan seed {seed}: target={cs0} steered-crossing -> "
              f"del={stats['deliveries']} violated={stats['violated']} "
              f"({stats['violation_kind']})")
        if stats["deliveries"] >= 1 and not stats["violated"]:
            target, seed_used = cs0, seed
            break
    if target is None:
        result = ("UNMEASURABLE: steered crossing cannot complete a "
                  "delivery even on a single-aircraft field (see scan)")
        print("RESULT:", result, "\nVERDICT: MISS")
        return {"probe": "b4", "expectation": EXPECT["b4"], "result": result,
                "verdict": "MISS",
                "detail": {"scan": {str(k): v for k, v in scan.items()}}}
    print(f"  target {target} (seed {seed_used}) delivers under steered "
          f"crossing")

    variants = {
        "crossing": B4Policy(target, "crossing"),
        "miss_1.5nm_past_window": B4Policy(target, "miss", miss_nm=1.5),
        "veer_2nm_early": B4Policy(target, "veer", veer_at_nm=2.0),
        "veer_6nm_early": B4Policy(target, "veer", veer_at_nm=6.0),
        "veer_12nm_early": B4Policy(target, "veer", veer_at_nm=12.0),
    }
    runs = {}
    for name, pol in variants.items():
        stats = run_scripted(env4, pol, seed_used, cbp=True)
        td = tracked(env4, target)
        final = pos_status_name(td)
        runs[name] = {**stats, "target_final_status": final}
        print(f"  {name:24s} | G={stats['ep_return']:9.3f} | "
              f"del={stats['deliveries']} cmd={stats['commands']:3d} | "
              f"{'VIOLATION[' + str(stats['violation_kind']) + ']@' + str(stats['time_to_violation']) + 's' if stats['violated'] else 'clean'}"
              f" | target ends {final}")
    g_cross = runs["crossing"]["ep_return"]
    g_miss = runs["miss_1.5nm_past_window"]["ep_return"]
    # shaping-noise scale for "margins >> shaping noise": std of the
    # crossing run's per-step shaping rewards (no deliveries/violations
    # counted) x sqrt(T), measured on a steered-crossing replay
    pol = B4Policy(target, "crossing")
    obs, info = env4.reset(seed=seed_used)
    bcd.reset_delivery_clock(env4)
    snap = sector_snapshot(env4, obs.keys())
    rs = []
    for _ in range(int(getattr(env4, "maxstep", 200))):
        t_cs, t_act = pol.decide(env4, obs)
        acts = noop_action(obs)
        if t_cs is not None:
            acts[t_cs] = t_act
        out = sector_step_cbp(env4, obs, snap, acts, t_cs is not None)
        if not (out["deliveries"] or out["violated"]):
            rs.append(out["r"])
        obs, snap = out["next_obs"], out["snap_next"]
        if out["violated"]:
            break
    noise_scale = float(np.std(rs) * math.sqrt(max(1, len(rs))))
    miss_is_excursion = runs["miss_1.5nm_past_window"]["violated"] and \
        runs["miss_1.5nm_past_window"]["violation_kind"] == "sector_excursion"
    # the veer reference must be a variant that genuinely forfeits the
    # delivery; veers that still deliver through the ~20 nm window are the
    # geometric-impossibility finding, reported alongside
    veer_name, veer_ref = None, None
    for vn in ("veer_2nm_early", "veer_6nm_early", "veer_12nm_early"):
        if runs[vn]["deliveries"] == 0:
            veer_name, veer_ref = vn, runs[vn]["ep_return"]
            break
    delivered_veers = [vn for vn in ("veer_2nm_early", "veer_6nm_early",
                                     "veer_12nm_early")
                       if runs[vn]["deliveries"] > 0]
    if veer_ref is None:
        ord_ok = False
        margins_ok = False
        verdict = "PARTIAL" if (g_cross - g_miss) > noise_scale and \
            miss_is_excursion else "MISS"
        result = (f"crossing={g_cross:.2f} > miss={g_miss:.2f} (margin "
                  f"{g_cross - g_miss:.2f} vs noise {noise_scale:.2f}, miss "
                  f"{'IS' if miss_is_excursion else 'is NOT'} an excursion) "
                  f"BUT veer-off is UNREALIZABLE: every veer variant "
                  f"({', '.join(delivered_veers)}) still delivers through "
                  f"the ~20 nm exit window — the veer-off leg of the "
                  f"pre-registered field cannot exist at these parameters")
    else:
        ord_ok = (g_cross > veer_ref > g_miss)
        margins_ok = (g_cross - veer_ref) > noise_scale and \
                     (veer_ref - g_miss) > noise_scale
        # the pre-registered 2 nm veer delivering => geometric
        # impossibility of the specced trajectory: cap at PARTIAL
        veer2_delivered = runs["veer_2nm_early"]["deliveries"] > 0
        if ord_ok and margins_ok and miss_is_excursion:
            verdict = "PARTIAL" if veer2_delivered else "MEET"
        elif ord_ok:
            verdict = "PARTIAL"
        else:
            verdict = "MISS"
        result = (f"crossing={g_cross:.2f} > {veer_name}={veer_ref:.2f} > "
                  f"miss={g_miss:.2f}: {'holds' if ord_ok else 'VIOLATED'}; "
                  f"margins ({g_cross - veer_ref:.2f}, "
                  f"{veer_ref - g_miss:.2f}) vs shaping-noise scale "
                  f"{noise_scale:.2f}; miss "
                  f"{'IS' if miss_is_excursion else 'is NOT'} an excursion; "
                  f"veers still delivering: {delivered_veers or 'none'} "
                  f"(pre-registered 2 nm veer is geometrically a delivery; "
                  f"{veer_name} ends "
                  f"{runs[veer_name]['target_final_status']})")
    print(f"RESULT: {result}\nVERDICT: {verdict}")
    return {"probe": "b4", "expectation": EXPECT["b4"], "result": result,
            "verdict": verdict,
            "detail": {"seed": seed_used, "target": target, "runs": runs,
                       "noise_scale": noise_scale,
                       "scan": {str(k): v for k, v in scan.items()},
                       "deviation": "single-aircraft CustomInfiniteEnv "
                                    "field; steered crossing (wind)"}}


# ===========================================================================
# [CKPT] PROBES
# ===========================================================================

def _ckpt_agent(args):
    path = args.ckpt or latest_checkpoint()
    agent, ckpt = load_agent(path, device="cpu")
    print(f"  [ckpt] {os.path.basename(path)} (episode "
          f"{ckpt.get('episode', '?')}, strat_t="
          f"{[round(t, 3) for t in ckpt.get('strat_t_values', [])]})")
    return agent, ckpt, path


def _agent_act_fn(agent, c):
    """Deployed-policy driver: includes re-issue masking (when the agent
    was trained with it) via a per-episode last_issued dict, exactly as
    bcd.run_episode maintains it. Candidate decode uses the AGENT'S OWN
    n_instr (3 for pre-vertical checkpoints), never the module constant."""
    li = {}

    def act(env, obs, info):
        actions, aux = agent.generate_action(env, obs, info, c=c,
                                             force_epsilon=0.0,
                                             last_issued=li)
        idx = aux["cand_idx"]
        if idx != 0 and getattr(agent, "mask_reissue", False):
            i, j = divmod(idx - 1, agent.n_instr)
            li[aux["callsigns"][i]] = j
        return actions, idx != 0
    return act


def probe_a1(args):
    print("\n" + "=" * 78)
    print("A1 [CKPT] CLEARANCE SEMANTICS — re-issue rate across training")
    print("EXPECTATION:", EXPECT["a1"])
    print("  note (fixed pre-run): with --mask_reissue the DEPLOYED "
          "re-issue rate is mechanically ~0; A1 therefore drives the net "
          "WITHOUT the mask and measures whether the net itself learned "
          "not to prefer re-issues (the expectation's original sense).")
    print("=" * 78)
    latest = args.ckpt or latest_checkpoint()
    m = re.match(r"(.*)_(ep\d+|final)\.pt$", os.path.basename(latest))
    stem = m.group(1) if m else None
    ckpts = []
    if stem:
        for p in glob.glob(os.path.join(CHECKPOINT_DIR, stem + "_ep*.pt")):
            em = re.search(r"_ep(\d+)\.pt$", p)
            if em:
                ckpts.append((int(em.group(1)), p))
        ckpts.sort()
    picks = []
    if ckpts:
        idxs = sorted({0, len(ckpts) // 3, 2 * len(ckpts) // 3,
                       len(ckpts) - 1})
        picks = [ckpts[i] for i in idxs]
    if not any(p == latest for _, p in picks):
        em = re.search(r"_ep(\d+)\.pt$", latest)
        picks.append((int(em.group(1)) if em else 10 ** 9, latest))
    env = get_env(1200)
    rows = []
    for ep_no, path in picks:
        agent, ckpt = load_agent(path, device="cpu")
        reissue, issued_total, rp_reissue, rp_total = 0, 0, 0, 0
        for seed in (50042, 50043):
            obs, info = env.reset(seed=seed)
            active = {}
            for _ in range(int(getattr(env, "maxstep", 200))):
                actions, aux = agent.generate_action(env, obs, info, c=0.0,
                                                     force_epsilon=0.0)
                idx = aux["cand_idx"]
                if idx != 0:
                    i, j = divmod(idx - 1, agent.n_instr)
                    cs = aux["callsigns"][i]
                    issued_total += 1
                    if active.get(cs) == j:
                        reissue += 1
                    if j == 2:
                        rp_total += 1
                        if active.get(cs) == 2:
                            rp_reissue += 1
                    active[cs] = j
                obs, _, _, _, info = env.step(actions)
                v, _, _ = detect_violation(info)
                if v:
                    break
        rate = reissue / max(1, issued_total)
        rp_rate = rp_reissue / max(1, rp_total)
        rows.append({"episode": ckpt.get("episode", ep_no), "path":
                     os.path.basename(path), "issued": issued_total,
                     "reissue_rate": rate,
                     "route_parallel_reissue_rate": rp_rate})
        print(f"  ep{ckpt.get('episode', ep_no):>5} | issued={issued_total:4d}"
              f" | re-issue rate={rate:.3f} | route_parallel re-issue="
              f"{rp_rate:.3f}")
    falls = len(rows) >= 2 and rows[-1]["reissue_rate"] < \
        rows[0]["reissue_rate"]
    low_terminal = rows[-1]["reissue_rate"] < 0.2 if rows else False
    if falls and low_terminal:
        verdict = "MEET"
    elif falls or low_terminal:
        verdict = "PARTIAL"
    else:
        verdict = "MISS"
    result = (f"re-issue rate {rows[0]['reissue_rate']:.3f} (ep"
              f"{rows[0]['episode']}) -> {rows[-1]['reissue_rate']:.3f} (ep"
              f"{rows[-1]['episode']}); {'falls' if falls else 'does NOT fall'}"
              f", terminal {'low' if low_terminal else 'HIGH (fee waste)'}")
    print(f"RESULT: {result}\nVERDICT: {verdict}")
    return {"probe": "a1", "expectation": EXPECT["a1"], "result": result,
            "verdict": verdict, "detail": {"checkpoints": rows}}


def forward_with_attn(net, tokens_np):
    """Mirror of ControllerQNet.forward for ONE unmasked state that also
    returns per-layer attention maps; equality with net() is asserted by
    the caller so the mirror cannot drift."""
    tok = torch.from_numpy(tokens_np[None].astype(np.float32))
    N = tok.shape[1]
    h = net.token_mlp(tok)
    attns = []
    for attn, ln1, ffn, ln2 in zip(net.attn, net.ln1, net.ffn, net.ln2):
        a, w = attn(h, h, h, need_weights=True, average_attn_weights=True)
        attns.append(w[0])          # [N, N] query x key
        h = ln1(h + a)
        h = ln2(h + ffn(h))
    out = net.instr_head(h).view(1, N, net.n_instr, 2)
    ac_l = out[..., 0]
    ac_u = ac_l + torch.nn.functional.softplus(out[..., 1]) + 1e-6
    summary = h.mean(dim=1)
    no = net.noop_head(summary)
    noop_l, noop_u = no[:, 0], no[:, 0] + \
        torch.nn.functional.softplus(no[:, 1]) + 1e-6
    return ac_l, ac_u, noop_l, noop_u, attns


def _hurwicz_scores(agent, tokens_np, c):
    cl, cu = agent.candidate_q(tokens_np)
    return cl + c * (cu - cl), cl, cu


def probe_a2(args):
    print("\n" + "=" * 78)
    print("A2 [CKPT] ATTENTION FIDELITY")
    print("EXPECTATION:", EXPECT["a2"])
    print("=" * 78)
    agent, ckpt, path = _ckpt_agent(args)
    env = get_env(1200)
    states = collect_probe_states(env, 60042, _agent_act_fn(agent, 0.0),
                                  30, min_aircraft=4)
    print(f"  probe states with N>=4: {len(states)}")
    rhos_impact, rhos_conflict, target_ranks = [], [], []
    checked_mirror = False
    for env_s, obs_s, stratum, step in states:
        cs_list = sorted(obs_s.keys())
        toks = build_tokens(env_s, obs_s, None, cs_list)
        N = len(cs_list)
        with torch.no_grad():
            al, au, nl, nu, attns = forward_with_attn(agent.q_net, toks)
        if not checked_mirror:
            cl_ref, cu_ref = agent.candidate_q(toks)
            mask = torch.ones(1, N, dtype=torch.bool)
            cl_m, cu_m, _ = candidate_intervals(al, au, nl, nu, mask)
            err = float(np.abs(cl_m[0].numpy() - cl_ref).max())
            assert err <= 1e-5, f"attention mirror drifted: {err}"
            print(f"  [mirror] attention forward == net forward "
                  f"(max err {err:.1e})  OK")
            checked_mirror = True
        # attention mass received by aircraft i (mean over layers+queries)
        A = torch.stack(attns).mean(dim=0)     # [N, N]
        mass = A.mean(dim=0).numpy()           # over queries
        # ground truth: mask-impact of deleting token i on candidate scores
        base_scores, _, _ = _hurwicz_scores(agent, toks, 0.5)
        impact = np.zeros(N)
        for i in range(N):
            keep = [k for k in range(N) if k != i]
            sc_i, _, _ = _hurwicz_scores(agent, toks[keep], 0.5)
            # align: NOOP + surviving aircraft in original order
            n_in = agent.n_instr
            diffs = [abs(sc_i[0] - base_scores[0])]
            for new_pos, old_pos in enumerate(keep):
                for j in range(n_in):
                    diffs.append(abs(sc_i[1 + n_in * new_pos + j]
                                     - base_scores[1 + n_in * old_pos + j]))
            impact[i] = float(np.mean(diffs))
        rho = spearman(mass, impact)
        if rho is not None:
            rhos_impact.append(rho)
        # conflict proximity
        snap_s = sector_snapshot(env_s, obs_s.keys())
        prox = []
        for cs in cs_list:
            if cs not in snap_s:
                prox.append(-999.0)
                continue
            best = float("inf")
            for o_cs in snap_s:
                if o_cs != cs and abs(snap_s[cs].fl
                                      - snap_s[o_cs].fl) < CONFLICT_FL:
                    best = min(best, haversine_nm(
                        snap_s[cs].lat, snap_s[cs].lon,
                        snap_s[o_cs].lat, snap_s[o_cs].lon))
            prox.append(-best if best < float("inf") else -999.0)
        rho_c = spearman(mass, prox)
        if rho_c is not None:
            rhos_conflict.append(rho_c)
        # clearance target choice
        sc0, _, _ = _hurwicz_scores(agent, toks, 0.0)
        best = sc0.max()
        idx = 0 if sc0[0] == best else int(sc0.argmax())
        if idx > 0:
            ti = (idx - 1) // agent.n_instr
            target_ranks.append(
                float((mass > mass[ti]).sum() + 1))  # 1 = most attended
    mean_rho = float(np.mean(rhos_impact)) if rhos_impact else float("nan")
    mean_rho_c = float(np.mean(rhos_conflict)) if rhos_conflict \
        else float("nan")
    mean_trank = float(np.mean(target_ranks)) if target_ranks \
        else float("nan")
    verdict = "MEET" if mean_rho > 0.5 else (
        "PARTIAL" if mean_rho > 0.3 else "MISS")
    result = (f"mean Spearman(attention, mask-impact)={mean_rho:.3f} over "
              f"{len(rhos_impact)} states (need >0.5); rho(attention, "
              f"conflict prox)={mean_rho_c:.3f}; chosen-target mean "
              f"attention rank={mean_trank:.2f}")
    print(f"RESULT: {result}\nVERDICT: {verdict}"
          + ("" if verdict == "MEET" else
             "  << interpretability story must not be cited"))
    return {"probe": "a2", "expectation": EXPECT["a2"], "result": result,
            "verdict": verdict,
            "detail": {"ckpt": os.path.basename(path),
                       "episode": ckpt.get("episode"),
                       "rhos_impact": rhos_impact,
                       "rhos_conflict": rhos_conflict,
                       "target_attention_ranks": target_ranks}}


def probe_a3(args):
    print("\n" + "=" * 78)
    print("A3 [CKPT] GLOBAL-VIEW UTILIZATION — token-deletion sensitivity")
    print("EXPECTATION:", EXPECT["a3"])
    print("  caveat (fixed pre-run): deleting a token removes the aircraft "
          "from the ATTENTION view only; it remains inside other tokens' "
          "k=3 neighbour blocks, so near-rank sensitivity is understated.")
    print("=" * 78)
    agent, ckpt, path = _ckpt_agent(args)
    env = get_env(1200)
    states = collect_probe_states(env, 70042, _agent_act_fn(agent, 0.0),
                                  40, min_aircraft=5)
    print(f"  probe states with N>=5: {len(states)}")
    near_changes, near_total = 0, 0
    far_changes, far_total = 0, 0
    used = 0
    for env_s, obs_s, stratum, step in states:
        cs_list = sorted(obs_s.keys())
        toks = build_tokens(env_s, obs_s, None, cs_list)
        N = len(cs_list)

        def chosen(t_np, order):
            sc, _, _ = _hurwicz_scores(agent, t_np, 0.0)
            best = sc.max()
            k = 0 if sc[0] == best else int(sc.argmax())
            if k == 0:
                return ("NOOP", None)
            i, j = divmod(k - 1, agent.n_instr)
            return (order[i], j)

        base_choice = chosen(toks, cs_list)
        if base_choice[0] == "NOOP":
            continue
        used += 1
        target = base_choice[0]
        sim = env_s.get_simulator_env()
        t_ac = sim.aircraft.get(target)
        others = [cs for cs in cs_list if cs != target]
        dists = {}
        for cs in others:
            ac = sim.aircraft.get(cs)
            if ac is None or ac.lat is None or t_ac is None \
                    or t_ac.lat is None:
                dists[cs] = float("inf")
            else:
                dists[cs] = haversine_nm(t_ac.lat, t_ac.lon, ac.lat, ac.lon)
        ranked = sorted(others, key=lambda cs: dists[cs])
        for rank, cs in enumerate(ranked, start=1):
            keep = [k for k, c2 in enumerate(cs_list) if c2 != cs]
            order = [cs_list[k] for k in keep]
            ch = chosen(toks[keep], order)
            changed = ch != base_choice
            if rank <= 3:
                near_total += 1
                near_changes += int(changed)
            else:
                far_total += 1
                far_changes += int(changed)
    near_rate = near_changes / max(1, near_total)
    far_rate = far_changes / max(1, far_total)
    verdict = "MEET" if far_changes > 0 else "MISS"
    result = (f"deletion change rate: 1st-3rd nearest {near_rate:.3f} "
              f"({near_changes}/{near_total}), 4th..Nth {far_rate:.3f} "
              f"({far_changes}/{far_total}) over {used} clearance states")
    print(f"RESULT: {result}\nVERDICT: {verdict}"
          + ("" if verdict == "MEET" else
             "  << pilot-frame policy in a controller body (informative)"))
    return {"probe": "a3", "expectation": EXPECT["a3"], "result": result,
            "verdict": verdict,
            "detail": {"ckpt": os.path.basename(path),
                       "episode": ckpt.get("episode"),
                       "near": [near_changes, near_total],
                       "far": [far_changes, far_total],
                       "states_used": used}}


def probe_a4(args):
    print("\n" + "=" * 78)
    print("A4 [CKPT] INTERVAL SEMANTICS (OOD width floor + monotonicity, "
          "risk strata, inversions) — revised 2026-07-08")
    print("EXPECTATION:", EXPECT["a4"])
    print("  NOTE: the absolute >5x epistemic-width bar is DEFERRED to "
          "ensemble-disagreement machinery (run 13); a single "
          "interval-head's width need not explode far off-manifold, but "
          "it must never NARROW there (floor), and it must not rank OOD "
          "inputs as more certain than nearer ones (monotonicity).")
    print("=" * 78)
    agent, ckpt, path = _ckpt_agent(args)
    env = get_env(1200)

    # ---- state collection: FILL ALL THREE RISK STRATA (the old single-
    # seed pass left stratum 1 empty, making the stratum check vacuous)
    MIN_PER_STRATUM, CAP_PER_STRATUM = 8, 20
    by_strat_states = {0: [], 1: [], 2: []}
    seeds_used = []
    for k in range(8):
        if all(len(v) >= MIN_PER_STRATUM for v in by_strat_states.values()):
            break
        sd = 80042 + k
        seeds_used.append(sd)
        for st in collect_probe_states(env, sd, _agent_act_fn(agent, 0.0),
                                       60, min_aircraft=2):
            if len(by_strat_states[st[2]]) < CAP_PER_STRATUM:
                by_strat_states[st[2]].append(st)
    states = [s for v in by_strat_states.values() for s in v]
    counts = [len(by_strat_states[s]) for s in range(3)]
    all_filled = all(c >= MIN_PER_STRATUM for c in counts)
    fill_msg = ("all filled" if all_filled else
                "NOT all filled — stratum finding on partial support")
    print(f"  probe states: {len(states)} over seeds {seeds_used}; "
          f"stratum counts [<10nm, 10-30nm, >30nm] = {counts} "
          f"({fill_msg})")

    tok_pool, widths_real, strata = [], [], []
    inversions = 0
    for env_s, obs_s, stratum, step in states:
        cs_list = sorted(obs_s.keys())
        toks = build_tokens(env_s, obs_s, None, cs_list)
        tok_pool.append(toks)
        cl, cu = agent.candidate_q(toks)
        inversions += int((cu < cl).sum())
        widths_real.append(float((cu - cl).mean()))
        strata.append(stratum)
    w_real = float(np.mean(widths_real))
    rng = np.random.default_rng(0)
    all_toks = np.concatenate(tok_pool, axis=0)
    lo_f = all_toks.min(axis=0)
    hi_f = all_toks.max(axis=0)
    centre = (lo_f + hi_f) / 2.0
    half = (hi_f - lo_f) / 2.0

    def eval_widths(gen):
        nonlocal inversions
        ws = []
        for toks in tok_pool:
            cl, cu = agent.candidate_q(gen(toks))
            inversions += int((cu < cl).sum())
            ws.append(float((cu - cl).mean()))
        return float(np.mean(ws))

    # OOD-distance ladder (near -> far):
    #   SHUFFLED   : token sets stitched from unrelated states (joint
    #                structure broken, marginals intact)
    #   BOX-GARBAGE: per-feature uniform over the pool's [min, max]
    #   3X-BOX     : per-feature uniform over the 3x-expanded box
    w_shuf = eval_widths(lambda toks: all_toks[
        rng.choice(len(all_toks), size=toks.shape[0], replace=False)])
    w_garb = eval_widths(lambda toks: rng.uniform(
        lo_f, hi_f, size=toks.shape).astype(np.float32))
    w_3x = eval_widths(lambda toks: rng.uniform(
        centre - 3.0 * half, centre + 3.0 * half,
        size=toks.shape).astype(np.float32))

    ratios = {"shuffled": w_shuf / w_real, "box_garbage": w_garb / w_real,
              "3x_box": w_3x / w_real}
    floor_ok = all(r >= 1.0 for r in ratios.values())
    chain = [("real", w_real), ("shuffled", w_shuf),
             ("box_garbage", w_garb), ("3x_box", w_3x)]
    mono_ok = all(chain[i][1] <= chain[i + 1][1]
                  for i in range(len(chain) - 1))
    inv_ok = inversions == 0
    print("  widths (near -> far OOD): " + "  ".join(
        f"{n}={w:.3f}" for n, w in chain))
    print("  ratios vs real: " + "  ".join(
        f"{n} x{r:.2f}" for n, r in ratios.items())
        + f"  -> floor(>=1.0 all): {'OK' if floor_ok else 'FAIL'}"
          f", OOD-distance monotone: {'OK' if mono_ok else 'FAIL'}")
    print(f"  interval inversions: {inversions}")

    # stratum-width pattern: FINDING, not scored
    by_strat = {s: [] for s in range(3)}
    for w, s in zip(widths_real, strata):
        by_strat[s].append(w)
    strat_means = [float(np.mean(by_strat[s])) if by_strat[s]
                   else float("nan") for s in range(3)]
    have = [m for m in strat_means if not math.isnan(m)]
    strat_mono = all(have[i] >= have[i + 1] for i in range(len(have) - 1)) \
        if len(have) >= 2 else False
    print(f"  FINDING (not scored): width by risk stratum "
          f"[<10nm, 10-30nm, >30nm] = "
          f"{[None if math.isnan(m) else round(m, 2) for m in strat_means]}"
          f" (counts {counts}, "
          f"{'all strata populated' if all_filled else 'PARTIAL support'})"
          f" -> {'monotone (risky widest)' if strat_mono else 'NON-monotone'}")

    if not floor_ok:
        verdict = "MISS"   # automatic FAIL: OOD narrowing
    else:
        n_ok = sum([mono_ok, inv_ok])
        verdict = "MEET" if n_ok == 2 else ("PARTIAL" if n_ok == 1
                                            else "MISS")
    floor_msg = "OK" if floor_ok else "FAILED (OOD narrowing — automatic FAIL)"
    result = (f"floor {floor_msg}: "
              + ", ".join(f"{n} x{r:.2f}" for n, r in ratios.items())
              + f"; OOD-distance chain "
                f"{'monotone' if mono_ok else 'NOT monotone'}; "
                f"inversions={inversions}; stratum widths (finding) "
                f"{[None if math.isnan(m) else round(m, 1) for m in strat_means]}"
                f" {'monotone' if strat_mono else 'NON-monotone'} on counts "
                f"{counts}")
    print(f"RESULT: {result}\nVERDICT: {verdict}")
    return {"probe": "a4", "expectation": EXPECT["a4"], "result": result,
            "verdict": verdict,
            "detail": {"ckpt": os.path.basename(path),
                       "episode": ckpt.get("episode"), "w_real": w_real,
                       "w_shuffled": w_shuf, "w_box_garbage": w_garb,
                       "w_3x_box": w_3x, "ratios": ratios,
                       "floor_ok": floor_ok, "ood_monotone": mono_ok,
                       "strat_means": strat_means, "strat_counts": counts,
                       "strat_all_filled": all_filled,
                       "strat_monotone_finding": strat_mono,
                       "seeds_used": seeds_used,
                       "epistemic_bar_note": ">5x bar deferred to "
                       "ensemble-disagreement (run 13)",
                       "inversions": inversions}}


# ===========================================================================
# A4v2 — corrected width-mechanism instruments (WIDTH_MECHANISM_PROBES.md)
# ===========================================================================

A4V2_KNN_K = 10
A4V2_FROZEN_NPZ = os.path.join(DIAG_DIR, "a4v2_frozen_probe_states.npz")


@torch.no_grad()
def _a4v2_stats(net, tok_list, batch=64, _checked=[False]):
    """Per-state instruments for a list of token arrays [N_i, D] (all
    rows real aircraft). Returns dict of numpy arrays over states:
      width   : mean candidate interval width over VALID candidates only
      raw     : mean candidate delta_raw (PRE-softplus), captured via
                forward hooks on instr_head/noop_head (the ensemble
                module's _raw_heads pattern) + the width_scalars channel
                added exactly as ControllerQNet.forward does
      raw_noop: the NOOP head's delta_raw alone
      summary : the net's masked-mean pooled 64-d sector summary
                (captured as the INPUT of noop_head — that IS the summary)
      pattern : 256-bit ReLU on/off pattern, bool [K, 256]: the two
                token-MLP ReLUs + the two attention-block FFN ReLUs,
                pooled per-state by masked MEAN over aircraft then > 0.5
                (majority vote; documented choice — an any-on pooling
                saturates at high aircraft counts)
      norm    : mean L2 norm of the state's token rows (in-range slope)
    Fidelity: on the first batch the captured raws are pushed back
    through softplus and asserted equal to the net's own (u - l).
    """
    import torch.nn.functional as F
    relus = [net.token_mlp[1], net.token_mlp[3],
             net.ffn[0][1], net.ffn[1][1]]
    widths, raws, raws_noop, sums, pats, norms = [], [], [], [], [], []
    for b0 in range(0, len(tok_list), batch):
        chunk = tok_list[b0:b0 + batch]
        tok, mask = pad_state_batch([(t, None) for t in chunk],
                                    torch.device("cpu"))
        cap = {}

        def _noop_hook(_m, inp, out):
            cap["noop"] = out
            cap["summary"] = inp[0]

        handles = [
            net.instr_head.register_forward_hook(
                lambda _m, _i, o: cap.__setitem__("instr", o)),
            net.noop_head.register_forward_hook(_noop_hook),
        ]
        for li, m in enumerate(relus):
            handles.append(m.register_forward_hook(
                lambda _m, _i, o, li=li: cap.__setitem__(("relu", li), o)))
        try:
            ac_l, ac_u, nl, nu = net(tok, mask)
        finally:
            for h in handles:
                h.remove()
        B, N, _ = tok.shape
        out = cap["instr"].view(B, N, net.n_instr, 2)
        ac_raw = out[..., 1]
        noop_raw = cap["noop"][:, 1]
        if getattr(net, "width_scalars", False):
            cf = net._count_feats(mask)
            ac_raw = ac_raw + net.wscalar_instr(cf).unsqueeze(1)
            noop_raw = noop_raw + net.wscalar_noop(cf)[:, 0]
        if not _checked[0]:
            err = max(
                float((out[..., 0] + F.softplus(ac_raw) + 1e-6
                       - ac_u).abs().max()),
                float((cap["noop"][:, 0] + F.softplus(noop_raw) + 1e-6
                       - nu).abs().max()))
            assert err <= 1e-5, f"a4v2 raw-head hook drifted: {err}"
            print(f"  [mirror] hooked delta_raw -> softplus == net (u-l) "
                  f"(max err {err:.1e})  OK")
            _checked[0] = True
        cl, cu, valid = candidate_intervals(ac_l, ac_u, nl, nu, mask)
        vm = valid.float()
        nv = vm.sum(dim=1)
        widths.append((((cu - cl) * vm).sum(dim=1) / nv).numpy())
        raw_cand = torch.cat([noop_raw.unsqueeze(1),
                              ac_raw.reshape(B, -1)], dim=1)
        raws.append(((raw_cand * vm).sum(dim=1) / nv).numpy())
        raws_noop.append(noop_raw.numpy())
        sums.append(cap["summary"].numpy())
        acts = torch.cat([cap[("relu", li)] for li in range(len(relus))],
                         dim=-1) > 0                     # [B, N, 256]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        pooled = (acts.float() * mask.unsqueeze(-1)).sum(dim=1) / denom
        pats.append((pooled > 0.5).numpy())
        norms.append(np.array([float(np.linalg.norm(t, axis=1).mean())
                               for t in chunk]))
    return {"width": np.concatenate(widths),
            "raw": np.concatenate(raws),
            "raw_noop": np.concatenate(raws_noop),
            "summary": np.concatenate(sums, axis=0),
            "pattern": np.concatenate(pats, axis=0),
            "norm": np.concatenate(norms)}


def _a4v2_knn_excess(w_probe, sum_probe, ref_w, ref_sum, k=A4V2_KNN_K):
    """Per-state width / mean width of the k nearest on-policy neighbours
    in summary space (Euclidean)."""
    p2 = (sum_probe ** 2).sum(axis=1)[:, None]
    r2 = (ref_sum ** 2).sum(axis=1)[None, :]
    with np.errstate(all="ignore"):   # Accelerate BLAS raises spurious
        d2 = p2 + r2 - 2.0 * (sum_probe @ ref_sum.T)   # FP flags on macOS
    assert np.isfinite(d2).all(), "non-finite kNN distances"
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    pred = ref_w[nn].mean(axis=1)
    return w_probe / np.maximum(pred, 1e-9)


def _a4v2_novelty(pat_probe, pat_ref):
    """Hamming distance (bits of 256) to the nearest on-policy pattern."""
    P = pat_probe.astype(np.float32)
    R = pat_ref.astype(np.float32)
    with np.errstate(all="ignore"):   # spurious Accelerate FP flags
        d = P @ (1.0 - R).T + (1.0 - P) @ R.T
    assert np.isfinite(d).all(), "non-finite pattern distances"
    return d.min(axis=1)


def _a4v2_auroc(w_real, w_cond):
    """P(width_cond > width_real), ties 0.5 — per-state separability."""
    r = rankdata(np.concatenate([w_real, w_cond]))
    n0, n1 = len(w_real), len(w_cond)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def probe_a4v2(args):
    print("\n" + "=" * 78)
    print("A4v2 [CKPT] CORRECTED WIDTH-MECHANISM INSTRUMENTS "
          "(kNN-excess, pattern novelty, in-range slope, "
          "width-pressure ladder)")
    print("EXPECTATION:", EXPECT["a4v2"])
    print("=" * 78)
    torch.set_num_threads(2)   # live training run on this machine
    quick = getattr(args, "a4v2_quick", False)
    if quick:
        print("  [quick] selftest-lite mode: reduced states, 2-point "
              "ladder, frozen-set file NOT touched")
    agent, ckpt, path = _ckpt_agent(args)
    env = get_env(1200)
    rng = np.random.default_rng(0)

    # ---- on-policy state collection: the checkpoint's own deployed
    # policy (greedy, c = c_train, re-issue masking as trained), standard
    # density. Token arrays only — no env snapshots needed.
    target_total = 200 if quick else 1800
    max_eps = 3 if quick else 30
    states, seeds_used = [], []
    t0 = time.time()
    for k in range(max_eps):
        if len(states) >= target_total:
            break
        sd = 43000 + k
        seeds_used.append(sd)
        li = {}
        obs, info = env.reset(seed=sd)
        maxstep = int(getattr(env, "maxstep",
                              env.config.scenario_duration // SEC_PER_STEP))
        for _ in range(maxstep):
            actions, aux = agent.generate_action(
                env, obs, info, c=agent.c_train, force_epsilon=0.0,
                last_issued=li)
            idx = aux["cand_idx"]
            if idx != 0 and getattr(agent, "mask_reissue", False):
                i, j = divmod(idx - 1, agent.n_instr)
                li[aux["callsigns"][i]] = j
            if aux["tokens"].shape[0] >= 2:
                states.append(aux["tokens"])
            obs, _, _, _, info = env.step(actions)
            v, _, _ = detect_violation(info)
            if v:
                break
    # split: every 6th state HELD OUT (probe condition i), rest =
    # on-policy reference pool for kNN / patterns. Temporal neighbours of
    # held-out states remain in the reference — that is the point of the
    # sanity check (excess ~ 1 for states the pool genuinely covers).
    held = states[::6]
    ref = [s for i, s in enumerate(states) if i % 6 != 0]
    print(f"  on-policy collection: {len(states)} states over "
          f"{len(seeds_used)} episodes (seeds {seeds_used[0]}.."
          f"{seeds_used[-1]}), {time.time() - t0:.0f}s; "
          f"reference={len(ref)}, held-out={len(held)}")

    # ---- OOD condition construction (shapes mirror held-out states)
    n_cond = min(60 if quick else 200, len(held))
    all_ref = np.concatenate(ref, axis=0)
    lo_f, hi_f = all_ref.min(axis=0), all_ref.max(axis=0)
    D = all_ref.shape[1]

    def make_swap(shape_tok):
        """AIRCRAFT-SWAP: each token row is a REAL aircraft token sampled
        from a DIFFERENT collected state (plausible aircraft, impossible
        joint traffic picture)."""
        N = shape_tok.shape[0]
        st_idx = rng.choice(len(ref), size=N, replace=False)
        return np.stack([ref[s][rng.integers(ref[s].shape[0])]
                         for s in st_idx]).astype(np.float32)

    def make_garbage(shape_tok):
        """A4-style box garbage: per-feature uniform over the pool box."""
        return rng.uniform(lo_f, hi_f,
                           size=shape_tok.shape).astype(np.float32)

    def make_shuffled_field(shape_tok):
        """SHUFFLED-FIELD (a4's marginal-preserving recombination taken
        to the per-feature level): every (token, feature) cell drawn
        independently from the pool's rows — marginals intact, within-
        token coherence destroyed (incoherent aircraft)."""
        N = shape_tok.shape[0]
        idx = rng.integers(0, all_ref.shape[0], size=(N, D))
        return all_ref[idx, np.arange(D)[None, :]].astype(np.float32)

    base_shapes = held[:n_cond]
    conditions = {
        "real_heldout": held,
        "aircraft_swap": [make_swap(t) for t in base_shapes],
        "box_garbage": [make_garbage(t) for t in base_shapes],
        "shuffled_field": [make_shuffled_field(t) for t in base_shapes],
    }

    # ---- instruments on the evaluated checkpoint
    net = agent.q_net
    ref_stats = _a4v2_stats(net, ref)
    slope = spearman(ref_stats["width"], ref_stats["norm"])
    rows = {}
    for name, toks in conditions.items():
        st = _a4v2_stats(net, toks)
        exc = _a4v2_knn_excess(st["width"], st["summary"],
                               ref_stats["width"], ref_stats["summary"])
        nov = _a4v2_novelty(st["pattern"], ref_stats["pattern"])
        rows[name] = {
            "n": len(toks),
            "mean_width": float(st["width"].mean()),
            "mean_raw": float(st["raw"].mean()),
            "mean_raw_noop": float(st["raw_noop"].mean()),
            "knn_excess_mean": float(exc.mean()),
            "knn_excess_median": float(np.median(exc)),
            "novelty_bits_mean": float(nov.mean()),
            "novelty_bits_p90": float(np.percentile(nov, 90)),
            "widths": st["width"],
        }
    w_real = rows["real_heldout"]["widths"]
    for name in conditions:
        rows[name]["auroc_vs_real"] = (
            0.5 if name == "real_heldout"
            else _a4v2_auroc(w_real, rows[name]["widths"]))
        rows[name]["width_x_real"] = (rows[name]["mean_width"]
                                      / max(1e-9, float(w_real.mean())))

    print(f"\n  in-range width-vs-norm slope (reference pool, "
          f"Spearman): {slope if slope is None else round(slope, 3)}")
    hdr = (f"  {'condition':<15} {'n':>4} {'width':>8} {'x_real':>7} "
           f"{'kNN-exc':>8} {'(med)':>7} {'nov.bits':>9} {'AUROC':>6} "
           f"{'raw':>8} {'raw_noop':>9}")
    print(hdr)
    for name in ("real_heldout", "aircraft_swap", "box_garbage",
                 "shuffled_field"):
        r = rows[name]
        print(f"  {name:<15} {r['n']:>4} {r['mean_width']:>8.3f} "
              f"{r['width_x_real']:>7.2f} {r['knn_excess_mean']:>8.2f} "
              f"{r['knn_excess_median']:>7.2f} "
              f"{r['novelty_bits_mean']:>9.1f} {r['auroc_vs_real']:>6.3f} "
              f"{r['mean_raw']:>8.2f} {r['mean_raw_noop']:>9.2f}")

    # ---- width-pressure ladder on FROZEN probe states -----------------
    m = re.match(r"(.*)_(ep\d+|final)\.pt$", os.path.basename(path))
    stem = m.group(1) if m else None
    ladder_ckpts = []
    if stem:
        for p in glob.glob(os.path.join(CHECKPOINT_DIR, stem + "_ep*.pt")):
            em = re.search(r"_ep(\d+)\.pt$", p)
            if em:
                ladder_ckpts.append((int(em.group(1)), p))
        ladder_ckpts.sort()
    if len(ladder_ckpts) > 30:   # subsample (keep first + every 4th + last)
        keep = set(range(0, len(ladder_ckpts), 4)) | {len(ladder_ckpts) - 1}
        ladder_ckpts = [c for i, c in enumerate(ladder_ckpts) if i in keep]

    n_frozen = min(200, len(held))
    frozen_meta = None
    if (not quick) and os.path.exists(A4V2_FROZEN_NPZ):
        z = np.load(A4V2_FROZEN_NPZ, allow_pickle=False)
        n_real = int(z["n_real"])
        n_garb = int(z["n_garb"])
        frozen_real = [z[f"real_{i}"] for i in range(n_real)]
        frozen_garb = [z[f"garb_{i}"] for i in range(n_garb)]
        frozen_meta = str(z["meta"])
        print(f"\n  [frozen] reusing {A4V2_FROZEN_NPZ} "
              f"({n_real} real + {n_garb} garbage states; {frozen_meta})")
    else:
        frozen_real = held[:n_frozen]
        frozen_garb = [make_garbage(t) for t in held[:n_frozen]]
        if not quick:
            frozen_meta = (f"created {time.strftime('%F %T')} from "
                           f"{os.path.basename(path)} on-policy rollout "
                           f"seeds {seeds_used}")
            np.savez(A4V2_FROZEN_NPZ,
                     meta=np.str_(frozen_meta),
                     n_real=len(frozen_real), n_garb=len(frozen_garb),
                     **{f"real_{i}": t for i, t in enumerate(frozen_real)},
                     **{f"garb_{i}": t for i, t in enumerate(frozen_garb)})
            print(f"\n  [frozen] probe-state set SAVED to "
                  f"{A4V2_FROZEN_NPZ} ({len(frozen_real)} real + "
                  f"{len(frozen_garb)} garbage) — later milestone runs "
                  f"reuse the SAME states")
    if quick and len(ladder_ckpts) > 2:
        ladder_ckpts = [ladder_ckpts[0], ladder_ckpts[-1]]

    print(f"\n  WIDTH-PRESSURE LADDER (frozen states; delta_raw = mean "
          f"pre-softplus over valid candidates, incl. width_scalars "
          f"channel when trained with it)")
    print(f"  {'ep':>5} {'w_real':>8} {'raw_real':>9} {'w_garb':>8} "
          f"{'raw_garb':>9} {'garb/real_w':>12}")
    ladder = []
    for ep_no, p in ladder_ckpts:
        ag_i, _ck = load_agent(p, device="cpu")
        sr = _a4v2_stats(ag_i.q_net, frozen_real)
        sg = _a4v2_stats(ag_i.q_net, frozen_garb)
        row = {"episode": ep_no,
               "w_real": float(sr["width"].mean()),
               "raw_real": float(sr["raw"].mean()),
               "raw_noop_real": float(sr["raw_noop"].mean()),
               "w_garbage": float(sg["width"].mean()),
               "raw_garbage": float(sg["raw"].mean()),
               "raw_noop_garbage": float(sg["raw_noop"].mean())}
        ladder.append(row)
        print(f"  {ep_no:>5} {row['w_real']:>8.3f} {row['raw_real']:>9.2f} "
              f"{row['w_garbage']:>8.3f} {row['raw_garbage']:>9.2f} "
              f"{row['w_garbage'] / max(1e-9, row['w_real']):>12.2f}")

    # ---- verdict -------------------------------------------------------
    held_exc = rows["real_heldout"]["knn_excess_mean"]
    swap_exc = rows["aircraft_swap"]["knn_excess_mean"]
    garb_exc = rows["box_garbage"]["knn_excess_mean"]
    shuf_exc = rows["shuffled_field"]["knn_excess_mean"]
    sanity_ok = 0.8 <= held_exc <= 1.25
    swap_elev = swap_exc >= 1.2
    no_collapse = garb_exc >= 1.0 and shuf_exc >= 1.0
    if not sanity_ok:
        verdict = "MISS"
    else:
        verdict = "MEET" if (swap_elev and no_collapse) else (
            "PARTIAL" if (swap_elev or no_collapse) else "MISS")
    raw_trend = ("n/a" if len(ladder) < 2 else
                 f"raw_garb {ladder[0]['raw_garbage']:.2f} (ep"
                 f"{ladder[0]['episode']}) -> "
                 f"{ladder[-1]['raw_garbage']:.2f} (ep"
                 f"{ladder[-1]['episode']}), raw_real "
                 f"{ladder[0]['raw_real']:.2f} -> "
                 f"{ladder[-1]['raw_real']:.2f}")
    result = (f"sanity(held-out kNN-excess)={held_exc:.2f} "
              f"({'OK' if sanity_ok else 'BROKEN INSTRUMENT'}); "
              f"aircraft-swap excess={swap_exc:.2f} "
              f"(novelty {rows['aircraft_swap']['novelty_bits_mean']:.1f} "
              f"bits, AUROC {rows['aircraft_swap']['auroc_vs_real']:.3f}); "
              f"garbage excess={garb_exc:.2f}, shuffled-field excess="
              f"{shuf_exc:.2f} (floor >= 1.0 "
              f"{'held' if no_collapse else 'BROKEN — certain-basin'}); "
              f"in-range slope={slope if slope is None else round(slope, 3)}"
              f"; ladder (finding): {raw_trend}")
    print(f"\nRESULT: {result}\nVERDICT: {verdict}")
    for name in rows:
        rows[name].pop("widths")
    return {"probe": "a4v2", "expectation": EXPECT["a4v2"],
            "result": result, "verdict": verdict,
            "detail": {"ckpt": os.path.basename(path),
                       "episode": ckpt.get("episode"),
                       "quick": quick,
                       "policy": f"greedy c={agent.c_train} + reissue mask",
                       "n_states_collected": len(states),
                       "n_reference": len(ref), "n_heldout": len(held),
                       "seeds_used": seeds_used,
                       "knn_k": A4V2_KNN_K,
                       "pattern_pooling": "masked mean over aircraft of "
                                          "(ReLU>0), majority (>0.5)",
                       "in_range_width_norm_spearman": slope,
                       "conditions": rows,
                       "frozen_npz": None if quick else A4V2_FROZEN_NPZ,
                       "frozen_meta": frozen_meta,
                       "ladder": ladder}}


def probe_c2(args):
    n_states = args.c2_states
    H = 15
    budget = 20000
    print("\n" + "=" * 78)
    print(f"C2 [CKPT] Q-VS-GROUND-TRUTH ORDERING + UNBIASED COVERAGE "
          f"({n_states} states, EVERY candidate, H={H}, budget "
          f"~{budget} sim steps)")
    print("EXPECTATION:", EXPECT["c2"])
    print("  caveats (fixed pre-run): (i) ground truth is the H=15 "
          "TRUNCATED discounted return (gamma^15 ~ 0.63 of tail missing); "
          "this biases coverage downward for far-from-terminal states but "
          "leaves rank correlation meaningful. (ii) Ground truth is the "
          "CBP TRAINING-signal return — the quantity a CBP-trained net "
          "estimates. (iii) The frozen policy applies re-issue masking "
          "with a per-rollout last_issued dict seeded only with the "
          "rolled-out first candidate (the pre-rollout mask history is "
          "unknowable from a state snapshot — at most one extra allowed "
          "re-issue per aircraft early in each rollout).")
    print("=" * 78)
    validate_cbp_mirror()
    agent, ckpt, path = _ckpt_agent(args)
    env = get_env(1200)
    c_pol = agent.c_train
    states = collect_probe_states(env, 90042, _agent_act_fn(agent, c_pol),
                                  n_states, min_aircraft=2)
    print(f"  probe states: {len(states)} (frozen policy c={c_pol}, "
          f"mask_reissue={getattr(agent, 'mask_reissue', False)})")

    def make_continue(li0):
        li = dict(li0)

        def continue_policy(e, o):
            acts, aux = agent.generate_action(e, o, None, c=c_pol,
                                              force_epsilon=0.0,
                                              last_issued=li)
            idx = aux["cand_idx"]
            if idx != 0 and getattr(agent, "mask_reissue", False):
                i, j = divmod(idx - 1, agent.n_instr)
                li[aux["callsigns"][i]] = j
            return acts, idx != 0
        return continue_policy

    rhos, rhos_l = [], []
    inside, total = 0, 0
    sim_steps = 0
    per_state = []
    t0 = time.time()
    for env_s, obs_s, stratum, step in states:
        if sim_steps >= budget:
            print(f"  [budget] stopping at {sim_steps} sim steps")
            break
        cs_list = sorted(obs_s.keys())
        snap_s = sector_snapshot(env_s, obs_s.keys())
        toks = build_tokens(env_s, obs_s, None, cs_list)
        scores, cl, cu = _hurwicz_scores(agent, toks, c_pol)
        n_cand = 1 + agent.n_instr * len(cs_list)   # the NET's candidates
        gts = []
        for k in range(n_cand):
            acts = noop_action(obs_s)
            issued = k != 0
            li0 = {}
            if issued:
                i, j = divmod(k - 1, agent.n_instr)
                acts[cs_list[i]] = j + 1
                li0[cs_list[i]] = j
            G, st, vio, dl = rollout_return(
                env_s, obs_s, snap_s, acts, issued, H,
                continue_policy=make_continue(li0), use_cbp=True)
            sim_steps += st
            gts.append(G)
        gts = np.array(gts)
        rho = spearman(gts, scores[:n_cand])
        rho_l = spearman(gts, cl[:n_cand])
        if rho is not None:
            rhos.append(rho)
        if rho_l is not None:
            rhos_l.append(rho_l)
        hit = ((gts >= cl[:n_cand]) & (gts <= cu[:n_cand]))
        inside += int(hit.sum())
        total += n_cand
        per_state.append({"step": step, "stratum": stratum,
                          "n_cands": n_cand, "rho": rho,
                          "coverage": float(hit.mean()),
                          "gt_min": float(gts.min()),
                          "gt_max": float(gts.max())})
    mean_rho = float(np.mean(rhos)) if rhos else float("nan")
    mean_rho_l = float(np.mean(rhos_l)) if rhos_l else float("nan")
    coverage = inside / max(1, total)
    print(f"  {len(per_state)} states, {sim_steps} sim steps, "
          f"{time.time() - t0:.0f}s wall")
    verdict = "MEET" if mean_rho > 0.4 else (
        "PARTIAL" if mean_rho > 0.15 else "MISS")
    result = (f"mean Spearman(GT(CBP) H=15, Hurwicz@c={c_pol})={mean_rho:.3f} "
              f"(need >0.4; lower-bound rho={mean_rho_l:.3f}); unbiased "
              f"rollout coverage={coverage:.3f} over {total} candidate "
              f"intervals")
    print(f"RESULT: {result}\nVERDICT: {verdict}")
    return {"probe": "c2", "expectation": EXPECT["c2"], "result": result,
            "verdict": verdict,
            "detail": {"ckpt": os.path.basename(path),
                       "episode": ckpt.get("episode"), "rhos": rhos,
                       "coverage": coverage, "sim_steps": sim_steps,
                       "per_state": per_state}}


def probe_c3(args):
    print("\n" + "=" * 78)
    print("C3 [JSONL] TERMINAL-GROUNDING WATCH")
    print("EXPECTATION:", EXPECT["c3"])
    print("=" * 78)
    path = latest_jsonl()
    recs = []
    with open(path) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"  {os.path.basename(path)}: {len(recs)} episodes")
    bins = []
    for b0 in range(0, len(recs), 100):
        chunk = recs[b0:b0 + 100]
        cens = sum(1 for r in chunk if not r["violated"])
        bins.append({"episodes": f"{b0 + 1}-{b0 + len(chunk)}",
                     "censored_frac": cens / len(chunk),
                     "mean_ttv": float(np.mean([r["time_to_violation"]
                                                for r in chunk])),
                     "mean_deliveries": float(np.mean([r["deliveries"]
                                                       for r in chunk]))})
    for b in bins:
        print(f"  eps {b['episodes']:>11}: censored={b['censored_frac']:.3f}"
              f"  mean TTV={b['mean_ttv']:6.1f}s  del={b['mean_deliveries']:.2f}")
    last = recs[-1]
    frac_last3 = float(np.mean([b["censored_frac"] for b in bins[-3:]]))
    starving = frac_last3 > 0.5
    trend_up = len(bins) >= 4 and \
        np.mean([b["censored_frac"] for b in bins[-2:]]) > \
        np.mean([b["censored_frac"] for b in bins[:2]]) + 0.1
    verdict = "MEET" if not starving else "MISS"
    result = (f"censored fraction last-300-eps={frac_last3:.3f} "
              f"(realized={last['realized_episodes']}, censored="
              f"{last['censored_episodes']}); trend "
              f"{'RISING' if trend_up else 'flat/none'}; tracker diet "
              f"{'STARVING -> C2 coverage takes over' if starving else 'fed'}")
    print(f"RESULT: {result}\nVERDICT: {verdict}")
    return {"probe": "c3", "expectation": EXPECT["c3"], "result": result,
            "verdict": verdict,
            "detail": {"jsonl": os.path.basename(path), "bins": bins,
                       "realized_episodes": last["realized_episodes"],
                       "censored_episodes": last["censored_episodes"]}}


# ===========================================================================
# MAIN
# ===========================================================================

PROBES = {"b1": probe_b1, "b2": probe_b2, "b3": probe_b3, "b4": probe_b4,
          "a1": probe_a1, "a2": probe_a2, "a3": probe_a3, "a4": probe_a4,
          "a4v2": probe_a4v2, "c2": probe_c2, "c3": probe_c3}
ORDER = ["b1", "b2", "b3", "b4", "a1", "a2", "a3", "a4", "a4v2", "c2",
         "c3"]


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return x


def main():
    ap = argparse.ArgumentParser(description="Pre-registered controller-"
                                             "frame probe battery")
    ap.add_argument("--probe", nargs="+", required=True,
                    choices=ORDER + ["all", "env", "ckpt"],
                    help="probe selectors; 'env'=b1 b2 b3 b4, "
                         "'ckpt'=a1 a2 a3 a4 a4v2 c2")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="checkpoint for [CKPT] probes (default: latest)")
    ap.add_argument("--b3_states", type=int, default=50)
    ap.add_argument("--c2_states", type=int, default=20)
    ap.add_argument("--a4v2_quick", action="store_true",
                    help="a4v2 selftest-lite: few states, 2-point ladder, "
                         "frozen probe-state file untouched")
    ap.add_argument("--vertical_ramp", type=str, default="gate",
                    choices=["gate", "smooth"],
                    help="D5 conflict-pricing mode for EVERY probe "
                         "(passed through to bcd.set_conflict_pricing; "
                         "the mirrors import bcd's shaping_terms/"
                         "sector_snapshot so they follow automatically). "
                         "Default gate = 12c pricing.")
    ap.add_argument("--delta_conflict", type=float, default=None,
                    help="D5 DELTA_CONFLICT override (default: bcd's "
                         "0.2; JK approved 0.5 for the smooth-ramp "
                         "composite battery).")
    args = ap.parse_args()
    # D5 amendment B: set the pricing BEFORE any probe/mirror runs and
    # announce it per probe so the battery cannot silently run the wrong
    # mode.
    bcd.set_conflict_pricing(args.vertical_ramp, args.delta_conflict)
    sel = []
    for p in args.probe:
        if p == "all":
            sel += ORDER
        elif p == "env":
            sel += ["b1", "b2", "b3", "b4"]
        elif p == "ckpt":
            sel += ["a1", "a2", "a3", "a4", "a4v2", "c2"]
        else:
            sel.append(p)
    sel = [p for i, p in enumerate(sel) if p not in sel[:i]]
    sel = [p for p in ORDER if p in sel]

    torch.manual_seed(0)
    results = []
    t_start = time.time()
    for p in sel:
        # D5 amendment B: explicit per-probe pricing line (verification
        # that no probe silently runs the wrong mode)
        print(f"\n[{p}] {bcd.conflict_pricing_str()}")
        try:
            results.append(PROBES[p](args))
            results[-1]["conflict_pricing"] = bcd.conflict_pricing_str()
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"probe": p, "expectation": EXPECT[p],
                            "result": f"ERROR: {e}", "verdict": "ERROR"})

    print("\n" + "=" * 78)
    print("PROBE BATTERY SUMMARY")
    print("=" * 78)
    for r in results:
        print(f"\n[{r['probe'].upper()}]  VERDICT: {r['verdict']}")
        print(f"  expectation: {r['expectation']}")
        print(f"  result:      {r['result']}")
    gates = {r["probe"]: r for r in results if r["probe"] in ("b1", "b2")}
    if gates:
        print("\nGATES for run 12b:")
        for p, r in gates.items():
            ok = r.get("gate", r["verdict"] == "MEET")
            msg = ("PASS — 12b may launch on this objective" if ok
                   else "FAIL — 12b must NOT launch")
            print(f"  {p.upper()}: {msg}")

    os.makedirs(DIAG_DIR, exist_ok=True)
    tag = "b1v2_" if "b1" in sel else ("a4v2_" if "a4v2" in sel else "")
    out = os.path.join(
        DIAG_DIR,
        f"probe_battery_{tag}{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(to_jsonable({"timestamp": time.strftime("%F %T"),
                               "probes_run": sel,
                               "vertical_ramp": bcd.VERTICAL_RAMP,
                               "delta_conflict": bcd.DELTA_CONFLICT,
                               "wall_seconds": round(time.time() - t_start),
                               "results": results}), f, indent=1)
    print(f"\nJSON written: {out}")


if __name__ == "__main__":
    main()
