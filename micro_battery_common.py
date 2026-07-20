"""
Run-13 micro battery — shared harness (2026-07-19)
==================================================

Shared machinery for the M2 / M2-XL / M3 / M4 / M5 micro tests of
RUN13_FIX_C_DESIGN.md Section 3 (post-audit v2, amendments A5/A6/A7/A15).
The per-test modules are micro_battery_m2.py, micro_battery_m2xl.py,
micro_battery_m3.py, micro_battery_m4.py, micro_battery_m5.py.

Faithfulness contract (same as micro_level_allocation.py): scenario envs
go through diagnose_controller.make_custom_density_env; scripted episodes
are priced by bcd.run_episode itself (ScriptedShim via run_scripted);
training IS bcd.run_episode(train=True); ground truth reuses
diagnose_controller.rollout_return / run_scripted. NOTHING here edits or
monkey-patches bluebird_controller_dqn / diagnose_controller /
micro_level_allocation — imports only.

HarnessAgent: a ControllerAgent subclass adding two HARNESS-SIDE features
the battery needs without touching bcd:
  (1) occupied-step counting (A7: the M5 command-rate denominator is
      steps with >= 1 aircraft in obs; bcd's step_hook does not carry an
      obs count, so occupancy is counted here in generate_action, which
      run_episode calls exactly once per step — documented deviation from
      the doc's literal "via step_hook");
  (2) an optional CANDIDATE-LEVEL instruction-type mask (M3: verticals
      masked) applied to greedy selection, epsilon sampling AND the
      Double-DQN bootstrap argmax, so masked candidates neither get
      selected nor carry bootstrap targets. The mask lives entirely in
      this subclass; bcd's code paths are unchanged.

--cf_replay: accepted by every training arm but only passed through when
bcd.run_episode has grown the parameter (the fix-C stage-1 build, a
separate work item). Until then the flag aborts loudly — the rho bars
that need CF-grounded untaken candidates are BLOCKED, not silently run.
"""

import copy
import inspect
import json
import os
import random
import time

import numpy as np
import torch

import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import (
    GAMMA, N_INSTR, INSTR_NAMES, KIN_FEATS, CMD_COST,
    ControllerAgent, run_episode, sector_snapshot, save_checkpoint,
    build_tokens, build_reissue_mask,
)
from bluebird_interval_dqn import detect_violation, haversine_nm, \
    SEC_PER_STEP
from diagnose_controller import (
    make_custom_density_env, ScriptedShim, NoopPolicy, run_scripted,
    collect_probe_states, rollout_return, spearman, noop_action,
    to_jsonable,
)
from micro_level_allocation import (
    SingleActionPolicy, ScriptPolicy, build_probe_set, rho_eval,
    CLIMB, DESCEND,
)

BATTERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "checkpoints", "micro_battery")

# env action ints (candidate instruction types j are these minus 1)
L10, R10, ROUTE_PARALLEL = 1, 2, 3
LATERAL_ACTS = (L10, R10, ROUTE_PARALLEL)
VERTICAL_TYPES = (3, 4)          # instruction types j for climb10/descend10


# ===========================================================================
# Pre-registered bar printing
# ===========================================================================

_OPS = {"<": lambda v, t: v < t, "<=": lambda v, t: v <= t,
        ">": lambda v, t: v > t, ">=": lambda v, t: v >= t,
        "==": lambda v, t: v == t}


def bar(name, value, op, thresh, note="", null_band=None):
    """Print one pre-registered bar with its threshold and MEET/MISS;
    returns the boolean. NaN values always MISS (never silently pass).
    null_band (ROUND3 D1): dict with mean/sd/p2.5/p97.5 from the
    30-init untrained null — printed BESIDE the bar so no reading is
    ever shown without the band that decides whether it means anything."""
    ok = (value == value) and _OPS[op](value, thresh)
    v = f"{value:.3f}" if isinstance(value, float) else str(value)
    nb = ""
    if null_band is not None:
        nb = (f"  [null 30-init: mean {null_band['mean']:+.3f} "
              f"sd {null_band['sd']:.3f} band "
              f"[{null_band['p2.5']:+.3f}, {null_band['p97.5']:+.3f}]"
              + (" — INSIDE NULL BAND, no verdict"
                 if (value == value and
                     null_band['p2.5'] <= value <= null_band['p97.5'])
                 else "") + "]")
    print(f"  BAR {name}: {v} (pre-registered: need {op} {thresh})"
          f" -> {'MEET' if ok else 'MISS'}"
          + (f"  [{note}]" if note else "") + nb)
    return bool(ok)


# ===========================================================================
# ROUND3 D2 — config echo (folded into the D1 fingerprint; one mechanism)
# ===========================================================================

# Names the round-3 bcd build MAY grow (W_rec payment, NOOP tie-break
# tolerance). Probed by getattr so the echo reports them the moment they
# land, and reports "absent" until then — never a silent gap.
_BCD_FUTURE_GLOBALS = ("W_REC", "W_rec", "NOOP_TOLERANCE",
                       "NOOP_TIE_TOLERANCE", "TIE_BREAK_TOLERANCE")
_AGENT_TOL_ATTRS = ("noop_tolerance", "noop_tie_tolerance",
                    "tie_break_tolerance", "noop_tol")


def bcd_config_echo(args=None):
    """Every cf / pricing / nstep / floor / W_rec flag as one dict
    (ROUND3_REVIEW amendment D2). Echoed into every summary JSON the
    battery writes (write_summary injects it) and folded into the D1
    probe-set fingerprint so a pricing or flag change hard-fails stale
    GT tables instead of silently re-basing a reading."""
    echo = {
        # pricing (live module state, not the arg — announce() may have
        # already applied the CLI override)
        "vertical_ramp": bcd.VERTICAL_RAMP,
        "delta_conflict": bcd.DELTA_CONFLICT,
        "gamma": GAMMA,
        "cmd_cost": CMD_COST,
        "delivery_bonus": bcd.DELIVERY_BONUS,
        "violation_penalty": bcd.VIOLATION_PENALTY,
        # cf machinery (module floors are live state — cf_state_floor
        # CLI override mutates bcd.CF_STATE_FLOOR)
        "cf_state_floor_effective": bcd.CF_STATE_FLOOR,
        "cf_k_floor_effective": bcd.CF_K_FLOOR,
        "cf_age_cap": bcd.CF_AGE_CAP,
        "cf_boost": bcd.CF_BOOST,
        "cf_max_frac": bcd.CF_MAX_FRAC,
        # round-3 flags that live in bcd once the other build lands;
        # "absent" until then (never omitted, never guessed)
        **{f"bcd.{n}": (getattr(bcd, n) if hasattr(bcd, n) else "absent")
           for n in _BCD_FUTURE_GLOBALS},
    }
    if args is not None:
        for name in ("nstep", "cbp", "cf_replay", "cf_k",
                     "cf_budget_rate", "cf_pure_mc", "cf_state_floor",
                     "bootstrap_support", "explore_bonus",
                     "explore_bonus_scale", "explore_bonus_halflife",
                     "c_train", "warmup_steps", "lr", "agent_seed",
                     "tracker_tripwire", "episodes", "scenario_seed"):
            echo[name] = getattr(args, name, "absent")
    return echo


_CONFIG_ECHO = None     # set by announce(); read by write_summary()


def detect_noop_tolerance(agent):
    """Locate the NOOP tie-break tolerance once bcd ships it (agent
    attribute or module global). Returns (holder_kind, name, value) or
    None. The C-instruments lock does not depend on the exact name —
    this probe list is the dual-mode reporting hook's best effort and
    reports 'absent' loudly when nothing matches."""
    for n in _AGENT_TOL_ATTRS:
        if hasattr(agent, n):
            return ("agent", n, getattr(agent, n))
    for n in ("NOOP_TOLERANCE", "NOOP_TIE_TOLERANCE",
              "TIE_BREAK_TOLERANCE"):
        if hasattr(bcd, n):
            return ("bcd", n, getattr(bcd, n))
    return None


def greedy_eval_dual_mode(env, agent, seed, cbp=True):
    """ROUND3 re-review C-instruments lock (c): command-RATE bars cannot
    be de-biased from logs (the trajectory diverges after the first
    converted selection), so when the NOOP tie-break tolerance is live
    the greedy eval episode is run in BOTH modes — tolerance-on (the
    policy-level rate) and tolerance-off (the number the C attribution
    prediction is scored on). Until bcd ships the tolerance this
    degrades to a single episode with tolerance='absent'."""
    tol = detect_noop_tolerance(agent)
    out = {"tolerance": (None if tol is None
                         else {"holder": tol[0], "name": tol[1],
                               "value": tol[2]}),
           "on": greedy_eval_episode(env, agent, seed, cbp=cbp),
           "off": None}
    if tol is not None and tol[2] not in (None, False, 0, 0.0):
        holder = agent if tol[0] == "agent" else bcd
        setattr(holder, tol[1], 0.0)
        try:
            out["off"] = greedy_eval_episode(env, agent, seed, cbp=cbp)
        finally:
            setattr(holder, tol[1], tol[2])
    return out


def dual_mode_cmd_rate_bar(name, dual, op, thresh, null_band=None):
    """Score a command-rate bar on the TOLERANCE-OFF number (per the
    re-review's C lock); report the tolerance-on number beside it."""
    def rate(st):
        return st["commands"] / max(1, st["steps"])
    if dual["off"] is None:
        note = ("tolerance absent in bcd — single mode"
                if dual["tolerance"] is None else
                f"tolerance {dual['tolerance']['name']}="
                f"{dual['tolerance']['value']} inactive — single mode")
        return bar(name, rate(dual["on"]), op, thresh, note=note,
                   null_band=null_band)
    ok = bar(name + " [TOLERANCE-OFF, scored]", rate(dual["off"]), op,
             thresh, null_band=null_band)
    print(f"    (policy-level, tolerance-ON: "
          f"{rate(dual['on']):.3f} — reported, not scored)")
    return ok


# ===========================================================================
# Harness agent (occupied-step counter + candidate-level instruction mask)
# ===========================================================================

class HarnessAgent(ControllerAgent):
    """ControllerAgent + battery-side occupancy counting and an optional
    instruction-type mask (see module docstring). masked_instr is a tuple
    of instruction TYPES j in 0..n_instr-1 (candidate layout: column
    1 + n_instr*i + j). With masked_instr=() behavior is intended to be
    identical to ControllerAgent (generate_action mirrors bcd's body
    statement-for-statement, same global-`random` draws)."""

    def __init__(self, *a, masked_instr=(), **kw):
        super().__init__(*a, **kw)
        self.masked_instr = tuple(masked_instr)
        self.ep_occupied = 0     # steps this episode with >= 1 ac in obs
        self.ep_steps = 0

    def begin_episode(self):
        self.ep_occupied = 0
        self.ep_steps = 0

    # ---- candidate-layout mask helpers --------------------------------
    def instr_mask_vec(self, n_cand):
        m = np.zeros(n_cand, dtype=bool)
        for j in self.masked_instr:
            m[1 + j::self.n_instr] = True
        return m

    def _instr_mask_cols(self, n_cols, device):
        m = torch.zeros(n_cols, dtype=torch.bool, device=device)
        for j in self.masked_instr:
            m[torch.arange(1 + j, n_cols, self.n_instr, device=device)] = True
        return m

    # ---- selection (mirror of bcd generate_action + mask + counters) --
    def generate_action(self, env, obs_dict, info_dict, c=None,
                        force_epsilon=None, adaptive=False,
                        last_issued=None, stratum=None, rng=random):
        # rng mirrors bcd's A12.3 CF isolation (CF continuations pass
        # agent.cf_rng). Occupancy counters are LIVE-step statistics
        # (M5's cmd-rate denominator): only the live stream counts.
        if rng is random:
            self.ep_steps += 1
            if obs_dict:
                self.ep_occupied += 1
        cs_list = sorted(obs_dict.keys())
        tokens = build_tokens(env, obs_dict, info_dict, cs_list)
        n_cand = 1 + self.n_instr * len(cs_list)
        rmask = None
        if self.mask_reissue and last_issued:
            rmask = build_reissue_mask(cs_list, last_issued, self.n_instr)
        if self.masked_instr and n_cand > 1:
            im = self.instr_mask_vec(n_cand)
            rmask = im if rmask is None else (rmask | im)
        idx, mean_width, c_used = self.select_candidate(
            tokens, c=c, adaptive=adaptive, reissue_mask=rmask,
            bonus_stratum=stratum)
        eps = self._epsilon(force_epsilon)
        if rng.random() < eps:
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
            actions[cs_list[i]] = j + 1
        aux = {"cand_idx": idx, "tokens": tokens,
               "callsigns": tuple(cs_list), "mean_width": mean_width,
               "c_used": c_used}
        return actions, aux

    # ---- bootstrap argmax: masked types never carry targets -----------
    def double_dqn_argmax(self, ocl, ocu, nvalid, reissue_mask, c):
        if self.masked_instr:
            cm = self._instr_mask_cols(ocl.shape[1], ocl.device)
            reissue_mask = cm if reissue_mask is None \
                else (reissue_mask | cm)
        return ControllerAgent.double_dqn_argmax(ocl, ocu, nvalid,
                                                 reissue_mask, c)

    def restricted_support_argmax(self, ocl, ocu, nvalid, reissue_mask, c,
                                  next_as):
        if self.masked_instr:
            cm = self._instr_mask_cols(ocl.shape[1], ocl.device)
            reissue_mask = cm if reissue_mask is None \
                else (reissue_mask | cm)
        return ControllerAgent.restricted_support_argmax(
            ocl, ocu, nvalid, reissue_mask, c, next_as)


def make_agent(env, scenario_seed, args, masked_instr=()):
    """Seeded HarnessAgent with the micro_level_allocation.train_micro
    construction (same hyperparameters and announcement line)."""
    torch.manual_seed(args.agent_seed)
    np.random.seed(args.agent_seed)
    random.seed(args.agent_seed)
    obs, info = env.reset(seed=scenario_seed)
    if obs:
        obs_dim = int(next(iter(obs.values())).shape[0])
    else:  # M5 single starter can begin BEFORE_ENTRY with empty obs
        for _ in range(int(getattr(env, "maxstep", 100))):
            obs = env.step({cs: 0 for cs in obs})[0]
            if obs:
                obs_dim = int(next(iter(obs.values())).shape[0])
                break
        else:
            raise AssertionError("obs never populated — no aircraft?")
    token_dim = obs_dim + KIN_FEATS
    n_env_actions = int(env.get_action_parser().get_total_num_actions())
    assert n_env_actions == 1 + N_INSTR
    agent = HarnessAgent(
        token_dim=token_dim, n_instr=N_INSTR, lr=args.lr, gamma=GAMMA,
        c_train=args.c_train, warmup_steps=args.warmup_steps,
        warmup_epsilon=0.5, buffer_size=20000, batch_size=64,
        device="cpu", bootstrap_support=args.bootstrap_support,
        explore_bonus=args.explore_bonus,
        explore_bonus_scale=args.explore_bonus_scale,
        explore_bonus_halflife=args.explore_bonus_halflife,
        tracker_tripwire=args.tracker_tripwire,
        masked_instr=masked_instr, **cf_agent_kwargs(args))
    print(f"  agent: token_dim {token_dim}, n_instr {N_INSTR}, "
          f"{agent.param_count()} params, c_train {args.c_train}, "
          f"warmup {args.warmup_steps} steps, cbp="
          f"{'ON' if args.cbp else 'OFF'}, nstep={args.nstep}, "
          f"masked_instr={[INSTR_NAMES[j] for j in masked_instr] or 'none'}"
          f", {bcd.conflict_pricing_str()}")
    return agent


def cf_replay_kwargs(args):
    """CF-replay landed as an AGENT-level switch (ControllerAgent
    cf_replay=..., read by run_episode via getattr), not a run_episode
    parameter — wire-check corrected 19 July after both builds merged.
    run_episode therefore needs no extra kwargs; agent construction
    (cf_agent_kwargs below) carries the flag. Kept as the loud-abort
    guard: never silently train without CF when it was asked for."""
    if not getattr(args, "cf_replay", False):
        return {}
    if "cf_replay" not in inspect.signature(
            bcd.ControllerAgent.__init__).parameters:
        raise SystemExit(
            "--cf_replay requested but bcd.ControllerAgent does not "
            "accept cf_replay — CF-branch machinery missing from this "
            "checkout. The rho bars that need CF-grounded untaken "
            "candidates are BLOCKED until it lands.")
    return {}


def cf_agent_kwargs(args):
    """Agent-constructor kwargs for --cf_replay (the actual switch)."""
    if not getattr(args, "cf_replay", False):
        return {}
    cf_replay_kwargs(args)   # loud-abort check
    kw = {"cf_replay": True,
          "cf_seed": 10_000 + getattr(args, "agent_seed", 42)}
    for name in ("cf_k", "cf_budget_rate"):
        if getattr(args, name, None) is not None:
            kw[name] = getattr(args, name)
    if getattr(args, "cf_pure_mc", False):
        kw["cf_pure_mc"] = True
        print("  [cf] EXPLORATORY: pure-MC cf targets (no bootstrap "
              "tail) — non-pre-registered arm")
    if getattr(args, "cf_state_floor", None) is not None:
        bcd.CF_STATE_FLOOR = float(args.cf_state_floor)
        print(f"  [cf] EXPLORATORY: CF_STATE_FLOOR overridden to "
              f"{bcd.CF_STATE_FLOOR} (pre-registered 0.05; "
              f"JK-authorized 19 July) — non-pre-registered arm")
    return kw


# ===========================================================================
# Scenario tracing (multi-aircraft generalization of micro noop_trace)
# ===========================================================================

def noop_trace_multi(env, seed):
    """All-NOOP rollout recording per-step PAIRWISE geometry for every
    IN_SECTOR pair (read-only measurement; pricing not needed). Returns
    (rows, violated, kind, vstep, involved); each row: {step, n_obs,
    in_sector: [cs...], fls: {cs: fl}, pairs: {(a, b): (d_nm, dfl)}}."""
    obs, info = env.reset(seed=seed)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    rows = []
    violated, kind, vstep, involved = False, None, None, []
    for step in range(maxstep):
        snap = sector_snapshot(env, obs.keys())
        css = sorted(snap)
        pairs = {}
        for i in range(len(css)):
            for j in range(i + 1, len(css)):
                a, b = snap[css[i]], snap[css[j]]
                pairs[(css[i], css[j])] = (
                    haversine_nm(a.lat, a.lon, b.lat, b.lon),
                    abs(a.fl - b.fl))
        rows.append({"step": step, "n_obs": len(obs), "in_sector": css,
                     "fls": {cs: snap[cs].fl for cs in css},
                     "pairs": pairs})
        obs, _r, _d, _t, info = env.step({cs: 0 for cs in obs})
        v, k, inv = detect_violation(info)
        if v:
            violated, kind, vstep, involved = True, k, step, list(inv)
            break
    return rows, violated, kind, vstep, involved


def pair_series(rows, pair):
    """[(step, d_nm, dfl)] for one (a, b) callsign pair across a trace."""
    key = tuple(sorted(pair))
    out = []
    for r in rows:
        if key in r["pairs"]:
            d, dfl = r["pairs"][key]
            out.append((r["step"], d, dfl))
    return out


def segments_spanned(steps, lo, hi, n_seg=5):
    """Number of distinct equal-width segments of [lo, hi] hit by the
    given step numbers (the A6 'distinct trajectory segments' count)."""
    if hi <= lo:
        return 1 if steps else 0
    return len({min(n_seg - 1, int((s - lo) / (hi - lo + 1e-9) * n_seg))
                for s in steps})


# ===========================================================================
# Probe set with episode-capped GT horizon (A10 + basis consistency)
# ===========================================================================

def build_probe_set_capped(env, seed, n_states, h, min_aircraft=2):
    """micro_level_allocation.build_probe_set (frozen states on the NOOP
    trajectory; GT = CBP-priced NOOP-continuation rollout_return) with ONE
    change: each state's GT horizon is capped at the REMAINING EPISODE
    length, h_s = min(h, maxstep - step). Rationale (found on the M2
    scenario, 19 Jul): an uncapped window rolls PAST the episode's time
    limit and prices post-episode events (e.g. a sector excursion at step
    ~109 of a 100-step episode) that the training-time returns are
    censored away from — GT(NOOP) drifts to -36 on a NOOP-clean scenario
    and the B4 NOOP-level bar becomes unfair by construction. Capping
    makes the GT basis match the training truncation exactly; the A10
    CPA-matched rule is still satisfied whenever the scenario's decisive
    event lies inside the episode (asserted by the M2 gate sweep)."""
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    states = collect_probe_states(
        env, seed, lambda e, o, i: (noop_action(o), False),
        n_states, min_aircraft=min_aircraft)
    probe = []
    t0 = time.time()
    for env_s, obs_s, stratum, step in states:
        cs_list = sorted(obs_s.keys())
        snap_s = sector_snapshot(env_s, obs_s.keys())
        toks = build_tokens(env_s, obs_s, None, cs_list)
        n_cand = 1 + N_INSTR * len(cs_list)
        h_s = max(1, min(h, maxstep - step))
        gts = []
        for k in range(n_cand):
            acts = noop_action(obs_s)
            issued = k != 0
            if issued:
                i, j = divmod(k - 1, N_INSTR)
                acts[cs_list[i]] = j + 1
            G, _st, _vio, _dl = rollout_return(
                env_s, obs_s, snap_s, acts, issued, h_s,
                continue_policy=None, use_cbp=True)
            gts.append(G)
        gts = np.asarray(gts)
        probe.append({"step": step, "tokens": toks, "cs_list": cs_list,
                      "n_cand": n_cand, "gt": gts, "h": h_s,
                      "stratum": stratum,
                      "gt_spread": float(gts.max() - gts.min())})
    hot = sum(1 for p in probe if p["gt_spread"] > 5.0)
    print(f"  probe set: {len(probe)} frozen states (GT h={h} capped at "
          f"episode end, CBP signal, NOOP continuation), {hot} "
          f"outcome-swinging (spread > 5), built in {time.time() - t0:.0f}s")
    return probe


# ===========================================================================
# Net-side probe-state evaluation (frozen states; greedy, epsilon 0)
# ===========================================================================

def probe_argmax(agent, probe_state, c):
    """Greedy Hurwicz argmax on one frozen probe state (exact-tie rule:
    ties involving NOOP resolve to NOOP; no re-issue context at a frozen
    state). C-INSTRUMENTS LOCK (ROUND3 re-review): this reads
    candidate_q DIRECTLY and must NEVER inherit selection-path
    modifications — in particular do NOT mirror the NOOP tie-break
    TOLERANCE or the count bonus into this read. probe_argmax,
    noop_argmax_fraction, rho_eval/rho_eval_subset and the M2-XL
    state_read are locked bit-identical under any tolerance value by
    test_c_instruments_lock.py."""
    cl, cu = agent.candidate_q(probe_state["tokens"])
    n = probe_state["n_cand"]
    scores = (cl + c * (cu - cl))[:n]
    best = scores.max()
    return 0 if scores[0] == best else int(scores.argmax())


def noop_argmax_fraction(agent, probe, c):
    """Fraction of frozen probe states whose greedy argmax is NOOP."""
    hits = [probe_argmax(agent, p, c) == 0 for p in probe]
    return float(np.mean(hits)) if hits else float("nan")


def noop_level_hits(agent, probe, c, tol=2.0):
    """A6 bar 4: fraction of probe states with
    |Hurwicz(NOOP) - GT(NOOP)| < tol, plus the per-state deviations."""
    devs = []
    for p in probe:
        cl, cu = agent.candidate_q(p["tokens"])
        h_noop = float(cl[0] + c * (cu[0] - cl[0]))
        devs.append(abs(h_noop - float(p["gt"][0])))
    frac = float(np.mean([d < tol for d in devs])) if devs else float("nan")
    return frac, devs


def rho_eval_subset(agent, probe, c, cand_keep):
    """rho_eval restricted to a candidate subset (M3: the lateral-only
    trained support). cand_keep(k) -> bool over candidate indices.
    Same reporting structure as micro_level_allocation.rho_eval."""
    rhos, rhos_hot = [], []
    inside, total = 0, 0
    for p in probe:
        keep = np.array([cand_keep(k) for k in range(p["n_cand"])])
        cl, cu = agent.candidate_q(p["tokens"])
        scores = (cl + c * (cu - cl))[:p["n_cand"]][keep]
        gt = p["gt"][keep]
        rho = spearman(gt, scores)
        if rho is not None:
            rhos.append(rho)
            if float(gt.max() - gt.min()) > 5.0:
                rhos_hot.append(rho)
        hit = (gt >= cl[:p["n_cand"]][keep]) & (gt <= cu[:p["n_cand"]][keep])
        inside += int(hit.sum())
        total += int(keep.sum())
    return {"rho": float(np.mean(rhos)) if rhos else float("nan"),
            "rho_hot": float(np.mean(rhos_hot)) if rhos_hot
            else float("nan"),
            "n_states": len(rhos), "n_hot": len(rhos_hot),
            "gt_coverage": inside / max(1, total)}


# ===========================================================================
# Generic training arm (train_micro pattern, trimmed)
# ===========================================================================

def run_training_arm(env, scenario_seed, agent, args, label,
                     rho_fn=None, out_dir=None):
    """Training loop copied from micro_level_allocation.train_micro
    (bcd.run_episode(train=True), cbp + objective v2, per-episode JSONL,
    periodic rho + checkpoint). Returns the history dict the per-test
    verdict blocks read. rho_fn(agent) -> dict, evaluated at ep 0, every
    --rho_every and at the end."""
    cfkw = cf_replay_kwargs(args)
    out_dir = out_dir or BATTERY_DIR
    os.makedirs(out_dir, exist_ok=True)
    # pid in the tag: two arms launched in the same second collided on
    # identical paths and clobbered each other's jsonl + checkpoints
    # (results-audit T1, 20 July — verified shared inode via lsof)
    tag = f"{time.strftime('%Y%m%d_%H%M%S')}_p{os.getpid()}"
    log_path = os.path.join(out_dir, f"{label}_seed{scenario_seed}_{tag}.jsonl")
    log = open(log_path, "w")
    print(f"  logging to {log_path}")

    rho_curve = []
    if rho_fn is not None:
        r0 = rho_fn(agent)
        rho_curve.append({"episode": 0, **r0})
        print(f"  rho@ep0 (untrained): {r0['rho']:+.3f} "
              f"(hot {r0['rho_hot']:+.3f}, n={r0['n_states']}/{r0['n_hot']})")

    hist = []
    t_start = time.time()
    for ep in range(1, args.episodes + 1):
        issues = []

        def hook(d, issues=issues):
            if d["issued"]:
                issues.append((d["step"], d["cand_idx"]))

        agent.begin_episode()
        stats = run_episode(env, agent, seed=scenario_seed, train=True,
                            nstep=args.nstep, cbp=args.cbp,
                            objective_v2=True, step_hook=hook, **cfkw)
        mix = {name: 0 for name in INSTR_NAMES}
        for _s, k in issues:
            mix[INSTR_NAMES[(k - 1) % agent.n_instr]] += 1
        rec = {"episode": ep, "clean": not stats["violated"],
               "violated": stats["violated"],
               "violation_kind": stats["violation_kind"],
               "steps": stats["steps"],
               "occupied_steps": agent.ep_occupied,
               # CF-replay telemetry (wire-check 19 July: cf landed as an
               # agent-level switch; bcd's own JSONL block lives in its
               # train loop, so the battery logs the counters itself)
               **({"cf": {"pool": len(agent.buffer.cf_buf),
                          "states_selected": agent.cf_states_selected,
                          "skipped_budget": agent.cf_skipped_budget,
                          "bank": round(agent.cf_budget.bank, 1),
                          "spent_steps": agent.cf_budget.spent_steps,
                          "spent_clones": agent.cf_budget.spent_clones}}
                  if getattr(agent, "cf_replay", False) else {}),
               "ep_return": stats["ep_return"],
               "commands": stats["commands"],
               "deliveries": stats["deliveries"],
               "mix": mix, "issues": issues[:50],
               "mean_width": stats["mean_width"],
               "epsilon": round(agent._epsilon(), 4),
               "t": round(agent.t, 4)}
        hist.append(rec)
        log.write(json.dumps(rec) + "\n")
        log.flush()

        ckpt_due = (ep % args.rho_every == 0 or ep == args.episodes)
        if rho_fn is not None and ckpt_due:
            r = rho_eval_and_report(rho_fn, agent, ep, hist, t_start)
            rho_curve.append({"episode": ep, **r})
            save_checkpoint(agent,
                            os.path.join(out_dir,
                                         f"{label}_seed{scenario_seed}_"
                                         f"{tag}_ep{ep}.pt"),
                            ep, extra={"micro_scenario_seed": scenario_seed,
                                       "battery_label": label,
                                       "config_echo": bcd_config_echo(args)})
        elif ckpt_due:
            # ROUND3 D2 fix: the rho_fn=None path (M4 round 1 AND round
            # 2) saved NO checkpoints, so the T8 forensics had nothing
            # net-side to recompute. Checkpoints now save on the same
            # cadence regardless of rho_fn.
            path = os.path.join(out_dir, f"{label}_seed{scenario_seed}_"
                                         f"{tag}_ep{ep}.pt")
            save_checkpoint(agent, path, ep,
                            extra={"micro_scenario_seed": scenario_seed,
                                   "battery_label": label,
                                   "config_echo": bcd_config_echo(args)})
            trail = [h["clean"] for h in hist[-100:]]
            print(f"  ep {ep:4d} | clean(last100)={np.mean(trail):.2f} "
                  f"| checkpoint saved (no rho_fn) "
                  f"| {time.time() - t_start:.0f}s")
        elif ep % 10 == 0:
            trail = [h["clean"] for h in hist[-100:]]
            print(f"  ep {ep:4d} | {'clean' if rec['clean'] else 'VIOL'} "
                  f"| clean(last100)={np.mean(trail):.2f} "
                  f"| G={stats['ep_return']:7.2f} cmd={stats['commands']:2d}"
                  f" deliv={stats['deliveries']}"
                  f" | eps={agent._epsilon():.2f} "
                  f"| {time.time() - t_start:.0f}s")
    log.close()
    return {"hist": hist, "rho_curve": rho_curve, "tag": tag,
            "log_path": log_path,
            "wall_seconds": round(time.time() - t_start)}


def rho_eval_and_report(rho_fn, agent, ep, hist, t_start):
    r = rho_fn(agent)
    trail = [h["clean"] for h in hist[-100:]]
    cmds = [h["commands"] / max(1, h["steps"]) for h in hist[-100:]]
    print(f"  ep {ep:4d} | clean(last100)={np.mean(trail):.2f} "
          f"| rho={r['rho']:+.3f} hot={r['rho_hot']:+.3f} "
          f"| cmd/step(last100)={np.mean(cmds):.3f} "
          f"| eps={agent._epsilon():.2f} | {time.time() - t_start:.0f}s")
    return r


def greedy_eval_episode(env, agent, seed, cbp=True):
    """One greedy (train=False, epsilon 0) episode through bcd.run_episode
    recording the issued-command sequence [(step, cand_idx, callsign)]."""
    issues = []
    cs_at_step = {}

    class _Recorder:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, k):
            return getattr(self.inner, k)

        def generate_action(self, env_, obs_dict, info_dict, **kw):
            acts, aux = self.inner.generate_action(env_, obs_dict,
                                                   info_dict, **kw)
            cs_at_step[len(cs_at_step)] = aux["callsigns"]
            return acts, aux

    rec = _Recorder(agent)

    def hook(d):
        if d["issued"]:
            css = cs_at_step.get(d["step"], ())
            i, _j = divmod(d["cand_idx"] - 1, agent.n_instr)
            target = css[i] if i < len(css) else None
            issues.append((d["step"], d["cand_idx"], target))

    agent.begin_episode()
    stats = run_episode(env, rec, seed=seed, train=False, cbp=cbp,
                        objective_v2=True, step_hook=hook)
    stats["issue_list"] = issues
    stats["occupied_steps"] = agent.ep_occupied
    return stats


# ===========================================================================
# Shared argparse
# ===========================================================================

def base_argparser(ap):
    """Common battery knobs added onto a module's ArgumentParser (the
    per-test modules add their own scenario knobs)."""
    ap.add_argument("--gates", action="store_true",
                    help="scenario finder + scripted gates")
    ap.add_argument("--train", action="store_true",
                    help="training arm (use --episodes; <= 50 for smoke)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--scenario_seed", type=int, default=None,
                    help="skip the scan and use this seed")
    ap.add_argument("--agent_seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--c_train", type=float, default=0.5)
    ap.add_argument("--warmup_steps", type=int, default=1500)
    ap.add_argument("--cbp", action="store_true", default=True)
    ap.add_argument("--no-cbp", dest="cbp", action="store_false")
    ap.add_argument("--bootstrap_support", type=str, default="full",
                    choices=["full", "taken_noop"])
    ap.add_argument("--explore_bonus", type=str, default="off",
                    choices=["off", "count"])
    ap.add_argument("--explore_bonus_scale", type=float, default=0.5)
    ap.add_argument("--explore_bonus_halflife", type=float, default=2000.0)
    ap.add_argument("--tracker_tripwire", action="store_true", default=True)
    ap.add_argument("--rho_every", type=int, default=25)
    ap.add_argument("--probe_states", type=int, default=20)
    ap.add_argument("--h", type=int, default=None,
                    help="GT rollout horizon (default: CPA-matched per "
                         "A10, computed from the found scenario)")
    ap.add_argument("--nstep", type=int, default=36,
                    help="n-step window (run-13 D5 ruling: 36; bcd code "
                         "default is 12)")
    ap.add_argument("--cf_state_floor", type=float, default=None,
                    help="EXPLORATORY (JK-authorized 19 July, M2 arm): "
                         "override bcd.CF_STATE_FLOOR — the probability "
                         "of CF-probing SAFE-stratum states. The "
                         "pre-registered value is 0.05; deviations must "
                         "be labeled non-pre-registered in reports.")
    ap.add_argument("--cf_k", type=int, default=None)
    ap.add_argument("--cf_pure_mc", action="store_true", default=False,
                    help="EXPLORATORY (19 July M2 diagnosis): cf rows "
                         "train on the pure H_cf return, no bootstrap "
                         "tail (breaks the doom-tail contamination on "
                         "clean scenarios; truncation-biased where true "
                         "tails are nonzero).")
    ap.add_argument("--cf_budget_rate", type=float, default=None,
                    help="CF budget accrual (step-equivalents per live "
                         "step). Micro arms use 20 (the certified micro "
                         "coverage point); None = bcd's throttled "
                         "default.")
    ap.add_argument("--cf_replay", action="store_true", default=False,
                    help="pass cf_replay=True to bcd.run_episode once the "
                         "stage-1 CF machinery has landed (aborts if not)")
    ap.add_argument("--vertical_ramp", type=str, default="gate",
                    choices=["gate", "smooth"])
    ap.add_argument("--delta_conflict", type=float, default=None)
    return ap


def announce(title, args):
    global _CONFIG_ECHO
    bcd.set_conflict_pricing(args.vertical_ramp, args.delta_conflict)
    _CONFIG_ECHO = bcd_config_echo(args)   # ROUND3 D2: captured once,
    #                                        injected into every summary
    print("=" * 74)
    print(title)
    print(f"[battery] {bcd.conflict_pricing_str()}, nstep={args.nstep}, "
          f"cf_replay={'ON' if args.cf_replay else 'OFF'}")
    print("=" * 74)


def write_summary(out_dir, name, payload):
    os.makedirs(out_dir, exist_ok=True)
    # ROUND3 D2: every summary JSON carries the full flag echo. Pricing
    # fields are re-read at write time (they are live module state), the
    # arg-side fields come from announce()'s capture.
    if "config_echo" not in payload:
        echo = dict(_CONFIG_ECHO) if _CONFIG_ECHO else bcd_config_echo()
        echo.update(bcd_config_echo())     # refresh live bcd state
        payload = {**payload, "config_echo": echo}
    path = os.path.join(out_dir,
                        f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(to_jsonable(payload), f, indent=1)
    print(f"  summary written: {path}")
    return path
