"""Diagnostic data-gathering campaign on the interval-MCTS DESIGN (P1-P5 of
MCTS_ARCHITECTURE.md).  Produces the data that settles the five candidate
foundational problems; it draws NO design conclusions.

READ-ONLY with respect to the agent: bluebird_interval_mcts.py is imported
and instrumented purely via subclassing and runtime monkeypatching.  Nothing
in the repository is edited.

Probes
------
D1 (P1, objective mismatch): one fully instrumented seed-10043 episode
    (TTC + macro-turns config = current defaults).  Per decision, per planned
    aircraft: root interval per action, chosen action, and per-simulation
    rehearsal traces (violation kind / depth / discounted penalty, both for
    the planned aircraft and in the background).  Emits the "knowing choices"
    table: decisions where EVERY visited root option contained a violation in
    rehearsal.
D2 (P2, frozen default future): at saved pre-excursion snapshots, re-run the
    joint decision with the rehearsal default policy switched from
    "everyone freezes" (NOOP) to "everyone flies their route"
    (simple_heading_route_parallel, injected into the deepcopied env's action
    parser -- it is NOT in the planner's action_config).  Compare root
    intervals, chosen actions, and rehearsed violations.
D3 (P3, horizon): same snapshots, horizon 15/25/40 at sims 24/32/40; when
    does the excursion first appear in rehearsal, does the chosen action
    change, and what does the longer sight cost in wall time.
D4 (P4, attention): during the D1 episode, mini-searches (8 sims, horizon 8)
    for aircraft the gate excluded (and, for comparability, for the planned
    ones too): root intervals -> decision-relevance metrics vs the TTC-gate
    ranking.
D5 (P5, interval semantics): hybrid pass (leaf net best_run6.pt, mean-bound
    backup) over the same snapshots; envelope widths vs credal-mean widths,
    correlation of widths / chosen actions / Hurwicz gaps.

Snapshots are NOT pickled (the env contains lambdas); instead D1 records the
exact executed joint action at every live step, and D2/D3/D5 rebuild any
snapshot by replaying those actions through a fresh env (verified exact:
max divergence 0.0 nm over mixed-action replays; all-NOOP fast-forward is
NOT used anywhere).  deepcopy((env, obs, info)) keeps the info dict's
simulator reference tied to the copied env.

Usage
-----
    nice -n 15 .venv/bin/python diagnose_mcts_design.py --all
    nice -n 15 .venv/bin/python diagnose_mcts_design.py --d1 --d4
    nice -n 15 .venv/bin/python diagnose_mcts_design.py --d2 --d3 --d5   # reuses saved D1 actions
    nice -n 15 .venv/bin/python diagnose_mcts_design.py --all --quick    # small smoke of the machinery

Outputs (checkpoints/bluebird/diagnostics/design_campaign/):
    d1_seed<seed>.json ... d5_seed<seed>.json, campaign_<stamp>.log
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bluebird_interval_mcts as bim
from bluebird_interval_mcts import (
    NOOP,
    IntervalMCTSAgent,
    IntervalNode,
    _exit_potential,
    _status,
    _tracker_callsigns,
    aircraft_kinematics,
    aircraft_states,
    cpa_nm_s,
    find_violations,
    haversine_nm,
    make_env,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "checkpoints", "bluebird", "diagnostics",
                       "design_campaign")
LEAF_CKPT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "checkpoints", "bluebird", "best_run6.pt")


# ============================================================================
# Small utilities
# ============================================================================

class Tee:
    def __init__(self, path):
        self.f = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.f.write(s)
        self.stdout.write(s)

    def flush(self):
        self.f.flush()
        self.stdout.flush()

    def reconfigure(self, **kw):   # bluebird_interval_dqn reconfigures stdout
        try:
            self.stdout.reconfigure(**kw)
        except Exception:
            pass

    def isatty(self):
        return False

    def close(self):
        sys.stdout = self.stdout
        self.f.close()


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return str(o)
    return str(o)


def _sanitize(obj):
    """Replace inf/nan floats so json stays strictly valid."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(_sanitize(obj), f, indent=1, default=_json_default)
    print(f"  [io] wrote {path} ({os.path.getsize(path)/1e6:.2f} MB)")


class RngGuard:
    """Save/restore the GLOBAL random and numpy RNG states around probe code
    so probes can never perturb the live episode's stochastic stream (the sim
    is believed to carry its own RNG -- deepcopy clones it -- but this makes
    the guarantee unconditional)."""

    def __enter__(self):
        self.rs = random.getstate()
        self.ns = np.random.get_state()
        return self

    def __exit__(self, *exc):
        random.setstate(self.rs)
        np.random.set_state(self.ns)
        return False


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# ============================================================================
# Violation classification (kind-aware; the agent itself only sees a bool)
# ============================================================================

def classify_planned_violation(info: dict, cs: str) -> Optional[dict]:
    """Kind of violation the PLANNED aircraft is in, from a sim info dict.
    Mirrors bluebird_interval_mcts.violation_involving but reports kind."""
    if _status(info, cs) == "OUT_SECTOR":
        return {"kind": "excursion", "with": None, "sep_nm": None}
    if _status(info, cs) != "IN_SECTOR":
        return None
    in_sector = [c for c in _tracker_callsigns(info)
                 if _status(info, c) == "IN_SECTOR"]
    states = aircraft_states(info, in_sector)
    if cs not in states:
        return None
    la, lo, fa = states[cs]
    best = None
    for other, (lb, lob, fb) in states.items():
        if other == cs or abs(fa - fb) >= bim.LOS_VERTICAL_FL:
            continue
        d = haversine_nm(la, lo, lb, lob)
        if d < bim.LOS_LATERAL_NM and (best is None or d < best["sep_nm"]):
            best = {"kind": "LoS", "with": other, "sep_nm": round(d, 3)}
    return best


def background_violations(info: dict, cs: str) -> List[str]:
    """All violations in the sim state that do NOT involve the planned
    aircraft (these carry no gradient for it -- see planning-reward doc)."""
    return [v for v in find_violations(info) if cs not in v]


# ============================================================================
# InstrumentedAgent: behaviour-identical wrapper capture (D1/D3/D4/D5)
# ============================================================================

class InstrumentedAgent(IntervalMCTSAgent):
    """Subclass that records everything while calling super() for ALL
    planning logic, so behaviour (incl. rng stream) is bit-identical to
    IntervalMCTSAgent."""

    def __init__(self, *a, record_bg: bool = True, **kw):
        super().__init__(*a, **kw)
        self.record_bg = record_bg
        self.decision_records: List[dict] = []   # one per generate_action
        self._live_step: Optional[int] = None
        self._cur_decision: Optional[dict] = None
        self._cur_plan: Optional[dict] = None
        self._steps_cur: Optional[List[dict]] = None
        self._last_root: Optional[IntervalNode] = None

    # -- driver hooks --------------------------------------------------
    def begin_decision(self, live_step: int) -> dict:
        self._cur_decision = {"step": live_step, "planned": []}
        self.decision_records.append(self._cur_decision)
        return self._cur_decision

    def preset_actions(self, all_actions, action_names) -> None:
        """Bind the planner action set explicitly (needed when the probe env's
        parser has extra injected actions the planner must NOT search)."""
        self.all_actions = tuple(all_actions)
        self.action_names = dict(action_names)
        self._actions_bound = True

    # -- recording helpers ----------------------------------------------
    def _begin_sim(self, root: IntervalNode):
        self._steps_cur = []
        return (root.visit_count, root.lo_sum, root.up_sum)

    def _end_sim(self, root: IntervalNode, prev):
        g_l = root.lo_sum - prev[1]
        g_u = root.up_sum - prev[2]
        steps = self._steps_cur or []
        viol = None
        bg = None
        for st in steps:
            if viol is None and st["violated"]:
                viol = {"kind": st["vkind"], "with": st["vwith"],
                        "sep_nm": st["vsep"], "depth": st["depth"],
                        "disc_pen": round(
                            (self.gamma ** st["depth"])
                            * self.violation_penalty, 4)}
            if bg is None and st["bg"]:
                bg = {"descs": st["bg"], "depth": st["depth"]}
            if viol is not None and bg is not None:
                break
        last = steps[-1] if steps else None
        rec = {
            "a0": steps[0]["a"] if steps else None,
            "acts": [st["a"] for st in steps],
            "g_l": round(g_l, 4), "g_u": round(g_u, 4),
            "viol": viol, "bg": bg,
            "term": (None if last is None or not last["done"] else
                     ("viol" if last["violated"] else
                      ("trunc" if last["trunc"] else "exit"))),
            "n": len(steps),
        }
        if self._cur_plan is not None:
            self._cur_plan["sims"].append(rec)
        self._steps_cur = None
        return rec

    def _record_step(self, planned_cs, depth, action, r, violated, done,
                     obs, term, trunc, info):
        v = classify_planned_violation(info, planned_cs) if violated else None
        st = {
            "depth": depth, "a": int(action), "r": round(float(r), 5),
            "violated": bool(violated), "done": bool(done),
            "trunc": bool(trunc.get(planned_cs, False)),
            "vkind": v["kind"] if v else None,
            "vwith": v["with"] if v else None,
            "vsep": v["sep_nm"] if v else None,
            "bg": (background_violations(info, planned_cs)
                   if self.record_bg else []),
        }
        if self._steps_cur is not None:
            self._steps_cur.append(st)

    def root_table(self, root: IntervalNode) -> Dict[int, dict]:
        tab = {}
        for a, ch in root.children.items():
            if ch.visit_count == 0:
                continue
            tab[int(a)] = {
                "lower": round(ch.lower, 4), "upper": round(ch.upper, 4),
                "mean": round(ch.mean, 4), "visits": ch.visit_count,
                "width": round(ch.width, 4),
                "score_act": round(
                    ch.lower + self.c_act * (ch.upper - ch.lower), 4),
            }
        return tab

    # -- instrumented planner entry points --------------------------------
    def _plan_single(self, live_env, root_callsigns, planned_cs, frozen):
        self._cur_plan = {"cs": planned_cs, "frozen": {k: int(v) for k, v
                                                       in frozen.items()},
                          "sims": []}
        self._last_root = None
        t0 = time.perf_counter()
        action, widths, chosen_env = super()._plan_single(
            live_env, root_callsigns, planned_cs, frozen)
        self._cur_plan["wall_s"] = round(time.perf_counter() - t0, 3)
        root = self._last_root
        self._cur_plan["root"] = self.root_table(root) if root else {}
        self._cur_plan["chosen"] = int(action)
        self._cur_plan["chosen_env"] = (
            [round(chosen_env[0], 4), round(chosen_env[1], 4)]
            if chosen_env else None)
        if self._cur_decision is not None:
            self._cur_decision["planned"].append(self._cur_plan)
        self._last_plan = self._cur_plan
        self._cur_plan = None
        return action, widths, chosen_env

    def _simulate(self, root, env, callsigns, planned_cs, frozen):
        self._last_root = root
        prev = self._begin_sim(root)
        super()._simulate(root, env, callsigns, planned_cs, frozen)
        self._end_sim(root, prev)

    def _step_sim(self, env, callsigns, planned_cs, planned_action, depth,
                  frozen):
        cap = {}
        orig_step = env.step

        def wrapped(actions):
            out = orig_step(actions)
            cap["out"] = out
            return out

        env.step = wrapped
        try:
            r, done, violated, next_cs, pobs = super()._step_sim(
                env, callsigns, planned_cs, planned_action, depth, frozen)
        finally:
            try:
                del env.step
            except AttributeError:
                pass
        obs, rew, term, trunc, info = cap["out"]
        self._record_step(planned_cs, depth, planned_action, r, violated,
                          done, obs, term, trunc, info)
        return r, done, violated, next_cs, pobs


# ============================================================================
# Route-default rehearsal agent (D2). The two policy hooks REQUIRE copying
# _simulate/_step_sim bodies from bluebird_interval_mcts (verbatim except the
# marked lines); mode ("noop","noop") is validated to reproduce the base
# agent exactly before any counterfactual is trusted.
# ============================================================================

RP_FALLBACKS = {"n": 0}   # safe route-parallel wrapper fallback counter


def patch_registry_safe_rp():
    """Wrap the route-parallel action factory so per-aircraft edge cases
    (missing route data etc.) degrade to no-command instead of crashing the
    whole rehearsal step.  Runtime patch only."""
    from bluebird_gymnasium.actions import registry_actions
    orig = registry_actions["simple_heading_route_parallel"]
    if getattr(orig, "_is_safe_rp", False):
        return

    def safe_rp(callsign, gym_env, value=1, agent="Agent"):
        try:
            return orig(callsign, gym_env, value, agent)
        except Exception:
            RP_FALLBACKS["n"] += 1
            return None

    safe_rp._is_safe_rp = True
    registry_actions._loaded["simple_heading_route_parallel"] = safe_rp


def enable_route_parallel(env) -> int:
    """Add simple_heading_route_parallel__1 to a (snapshot copy of an) env's
    action parser and return its action int.  The live episode env is never
    touched -- only replay-rebuilt snapshot copies."""
    p = env.get_action_parser()
    for idx, spec in p.action_formatter_map.items():
        if spec.startswith("simple_heading_route_parallel"):
            return idx
    idx = max(p.action_formatter_map) + 1
    p._action_formatter_map[idx] = "simple_heading_route_parallel__1"
    p.num_actions_per_aircraft += 1
    p.total_num_actions += 1
    p.actions_heading_parallel.append(idx)
    return idx


class RouteDefaultAgent(InstrumentedAgent):
    """rollout_mode: planned aircraft's beyond-tree policy ("noop"|"route").
    others_mode: undecided aircraft's policy every rehearsal step
    ("noop"|"route").  Already-decided (frozen) aircraft keep their one-shot
    command at depth 0 then NOOP in BOTH modes (their command is an intent
    the counterfactual must not overwrite)."""

    def __init__(self, *a, rp_idx: int, rollout_mode: str = "route",
                 others_mode: str = "route", **kw):
        super().__init__(*a, **kw)
        self.rp_idx = rp_idx
        self.rollout_mode = rollout_mode
        self.others_mode = others_mode

    # ---- copied from IntervalMCTSAgent._simulate; MODIFIED line marked
    def _simulate(self, root, env, callsigns, planned_cs, frozen):
        self.bind_action_space(env)
        self._last_root = root
        prev = self._begin_sim(root)
        node = root
        path = [root]
        g_l = g_u = 0.0
        disc = 1.0
        depth, done = 0, False
        planned_obs = None
        actions = self.all_actions

        while depth < self.horizon:
            if node is root:
                max_children = len(actions)
            else:
                max_children = max(1, int(
                    self.pw_k * max(1, node.visit_count) ** self.pw_alpha))
            unexpanded = [a for a in actions if a not in node.children]
            if unexpanded and len(node.children) < min(len(actions),
                                                       max_children):
                action = self.rng.choice(unexpanded)
                expanding = True
            else:
                action = self._select_action(node)
                expanding = False

            r, done, violated, callsigns, planned_obs = self._step_sim(
                env, callsigns, planned_cs, action, depth, frozen)
            g_l += disc * r
            g_u += disc * r
            if violated:
                g_l += disc * self.violation_penalty
                g_u += disc * self.violation_penalty
            disc *= self.gamma
            depth += 1

            child = node.children.get(action)
            if child is None:
                child = IntervalNode(mode=self.backup_mode)
                node.children[action] = child
            path.append(child)
            node = child

            if done or violated:
                done = True
                break
            if expanding:
                break

        if not done:
            while depth < self.horizon:
                # MODIFIED: rollout default is route-parallel, not NOOP
                ra = self.rp_idx if self.rollout_mode == "route" else NOOP
                r, done, violated, callsigns, planned_obs = self._step_sim(
                    env, callsigns, planned_cs, ra, depth, frozen)
                g_l += disc * r
                g_u += disc * r
                if violated:
                    g_l += disc * self.violation_penalty
                    g_u += disc * self.violation_penalty
                disc *= self.gamma
                depth += 1
                if done or violated:
                    break

        if self.leaf_net is not None and not done and planned_obs is not None:
            q_l, q_u = self._leaf_interval(planned_obs)
            g_l += disc * q_l
            g_u += disc * q_u

        for nd in path:
            nd.update(g_l, g_u)
        self._end_sim(root, prev)

    # ---- copied from IntervalMCTSAgent._step_sim; MODIFIED line marked
    def _step_sim(self, env, callsigns, planned_cs, planned_action, depth,
                  frozen):
        actions = {}
        for cs in callsigns:
            if cs == planned_cs:
                actions[cs] = planned_action
            elif cs in frozen:
                actions[cs] = frozen[cs] if depth == 0 else NOOP
            else:
                # MODIFIED: undecided others fly their route, not NOOP
                actions[cs] = (self.rp_idx if self.others_mode == "route"
                               else NOOP)

        phi = (_exit_potential(env, planned_cs)
               if self.progress_coeff else None)
        obs, rew, term, trunc, info = env.step(actions)
        next_callsigns = list(obs.keys())

        r = float(rew.get(planned_cs, 0.0))
        truncated = bool(trunc.get(planned_cs, False))
        done = (
            planned_cs not in obs
            or bool(term.get(planned_cs, False))
            or truncated
        )
        violated = bim.violation_involving(info, planned_cs)
        if (done and not truncated and not violated
                and _status(info, planned_cs) != "OUT_SECTOR"):
            r += self.exit_bonus
        if self.progress_coeff and not done and phi is not None:
            phi_next = _exit_potential(env, planned_cs)
            if phi_next is not None:
                r += self.progress_coeff * (self.gamma * phi_next - phi)
        self._record_step(planned_cs, depth, planned_action, r, violated,
                          done, obs, term, trunc, info)
        return r, done, violated, next_callsigns, obs.get(planned_cs)


# ============================================================================
# Gate mirror (recompute generate_action's attention decision for recording;
# validated each step against the observed _plan_single order)
# ============================================================================

def mirror_gate(agent: IntervalMCTSAgent, obs: dict, info: dict) -> dict:
    callsigns = list(obs.keys())
    states = aircraft_states(info, callsigns)
    kin = aircraft_kinematics(info, callsigns)
    risk, engaged = {}, set()
    for cs in callsigns:
        if cs not in states:
            continue
        la, lo, fa = states[cs]
        dmin, ttc_min = math.inf, math.inf
        trigger = False
        for other, (lb, lob, fb) in states.items():
            if other == cs or abs(fa - fb) >= agent.alert_fl:
                continue
            d = haversine_nm(la, lo, lb, lob)
            dmin = min(dmin, d)
            if d < agent.alert_radius_nm:
                trigger = True
            t_cpa, d_cpa = cpa_nm_s(kin.get(cs), kin.get(other))
            if (t_cpa is not None and d_cpa < agent.cpa_dist_nm
                    and t_cpa < agent.cpa_time_s):
                trigger = True
                ttc_min = min(ttc_min, t_cpa)
        risk[cs] = (ttc_min, dmin)
        if trigger:
            engaged.add(cs)
    midman = {cs for cs in agent.active_maneuvers if cs in obs}
    decided = {cs for cs in callsigns
               if (cs not in engaged and cs not in midman)
               or _status(info, cs) == "BEFORE_ENTRY"}
    candidates = [cs for cs in callsigns if cs not in decided]
    rk = lambda cs: risk.get(cs, (math.inf, math.inf))   # noqa: E731
    to_plan = (sorted((c for c in candidates if c in engaged), key=rk)
               + sorted((c for c in candidates if c not in engaged), key=rk))
    dropped = to_plan[agent.max_planned:]
    to_plan = to_plan[: agent.max_planned]
    return {
        "risk": {cs: [None if math.isinf(r[0]) else round(r[0], 1),
                      None if math.isinf(r[1]) else round(r[1], 2)]
                 for cs, r in risk.items()},
        "engaged": sorted(engaged),
        "midman": sorted(midman),
        "to_plan": to_plan,
        "cap_dropped": dropped,
        "statuses": {cs: _status(info, cs) for cs in callsigns},
    }


def tracked_geometry(env, cs: str) -> Optional[dict]:
    try:
        d = env.get_tracked_aircraft_data(cs)
    except Exception:
        return None
    if d is None:
        return None
    out = {}
    try:
        out["heading"] = None if d.heading is None else round(float(d.heading), 1)
        out["dist_to_exit_nm"] = (None if d.track_dist_to_exit_cr is None
                                  else round(float(d.track_dist_to_exit_cr), 2))
        if d.position is not None and d.sector_exit_pos is not None:
            out["bearing_to_exit"] = round(
                float(d.position.bearing_to(d.sector_exit_pos)), 1)
        else:
            out["bearing_to_exit"] = None
    except Exception:
        return out or None
    return out


# ============================================================================
# D4: attention probes (mini-searches for gate-excluded aircraft)
# ============================================================================

def relevance_metrics(tab: Dict[int, dict], c_act: float) -> dict:
    """Decision-relevance of a root interval table."""
    if not tab:
        return {"spread": None, "hurwicz_gap": None, "pick": None,
                "pick_nonnoop": None}
    ups = [v["upper"] for v in tab.values()]
    los = [v["lower"] for v in tab.values()]
    scores = sorted(((v["score_act"], a) for a, v in tab.items()),
                    reverse=True)
    # NOOP-first tie-break, mirroring the root decision rule
    best = max(tab.items(),
               key=lambda kv: (kv[1]["score_act"], kv[0] == NOOP))
    pick = best[0]
    gap = (scores[0][0] - scores[1][0]) if len(scores) > 1 else None
    return {
        "spread": round(max(ups) - max(los), 4),   # max upper - max lower
        "hurwicz_gap": None if gap is None else round(gap, 4),
        "pick": int(pick),
        "pick_nonnoop": bool(pick != NOOP),
        "noop_vs_best_gap": round(
            scores[0][0] - tab[NOOP]["score_act"], 4) if NOOP in tab else None,
    }


def d4_probe(live_env, obs, info, decided: Dict[str, int], cs: str,
             base_agent: IntervalMCTSAgent, seed: int, step: int,
             sims: int, horizon: int) -> dict:
    """Mini interval search for one aircraft, freezing everyone else to the
    actually-decided joint action.  Runs entirely on deepcopies."""
    probe = InstrumentedAgent(
        n_simulations=sims, horizon=horizon, gamma=base_agent.gamma,
        c_search=base_agent.c_search, c_act=base_agent.c_act,
        c_visit=base_agent.c_visit,
        violation_penalty=base_agent.violation_penalty,
        exit_bonus=base_agent.exit_bonus,
        progress_coeff=base_agent.progress_coeff,
        rng=random.Random((seed * 1000003 + step * 131
                           + zlib.crc32(cs.encode())) & 0x7FFFFFFF),
        record_bg=False,
    )
    probe.preset_actions(base_agent.all_actions, base_agent.action_names)
    probe.begin_decision(step)
    frozen = {c: a for c, a in decided.items() if c != cs}
    callsigns = list(obs.keys())
    t0 = time.perf_counter()
    action, widths, chosen_env = probe._plan_single(
        live_env, callsigns, cs, frozen)
    wall = time.perf_counter() - t0
    plan = probe.decision_records[-1]["planned"][-1]
    tab = plan["root"]
    viol_sims = sum(1 for s in plan["sims"] if s["viol"])
    return {
        "cs": cs, "root": tab, "chosen": int(action),
        "metrics": relevance_metrics(tab, base_agent.c_act),
        "viol_sims": viol_sims, "n_sims": len(plan["sims"]),
        "viol_kinds": sorted({s["viol"]["kind"] for s in plan["sims"]
                              if s["viol"]}),
        "wall_s": round(wall, 3),
    }


# ============================================================================
# D1: the instrumented episode
# ============================================================================

def run_d1(args) -> dict:
    print("=" * 78)
    print(f"[D1] instrumented episode | seed {args.seed} | sims {args.sims} "
          f"| horizon {args.horizon} | duration {args.duration}s "
          f"| d4_probes={'on' if args.with_d4 else 'off'}")
    print("=" * 78)
    # Seeding exactly as bluebird_interval_mcts.main()
    random.seed(args.seed)
    np.random.seed(args.seed)
    agent = InstrumentedAgent(
        n_simulations=args.sims, horizon=args.horizon,
        rng=random.Random(args.seed),
    )
    env = make_env(scenario_duration=args.duration, macro_turns=True)
    obs, info = env.reset(seed=args.seed)
    agent.reset_episode()

    d1 = {
        "config": {"seed": args.seed, "sims": args.sims,
                   "horizon": args.horizon, "duration": args.duration,
                   "gamma": agent.gamma, "c_act": agent.c_act,
                   "c_search": agent.c_search, "c_visit": agent.c_visit,
                   "violation_penalty": agent.violation_penalty,
                   "exit_bonus": agent.exit_bonus,
                   "progress_coeff": agent.progress_coeff,
                   "macro_turns": True, "encoder_cls": "extra_minimal",
                   "k_nearest": 2,
                   "d4": {"enabled": bool(args.with_d4),
                          "sims": args.d4_sims, "horizon": args.d4_horizon,
                          "dense_from": args.d4_from, "dense_to": args.d4_to,
                          "sparse_every": args.d4_sparse_every}},
        "actions_log": [],          # executed joint action per live step
        "decisions": [],            # per-step instrumentation
        "episode": {},
    }

    ep_return = 0.0
    first_violation = None
    step = 0
    t_ep = time.time()

    while True:
        pre_man = {cs: int(a) for cs, a in agent.active_maneuvers.items()
                   if cs in obs}
        gate = mirror_gate(agent, obs, info)
        dec = agent.begin_decision(step)
        t0 = time.perf_counter()
        action = agent.generate_action(env, obs, info)
        wall = time.perf_counter() - t0

        observed_plan_order = [p["cs"] for p in dec["planned"]]
        dec.update({
            "n_ac": len(obs),
            "gate": gate,
            "gate_mirror_ok": observed_plan_order == gate["to_plan"],
            "observed_plan_order": observed_plan_order,
            "pre_maneuvers": pre_man,
            "post_maneuvers": {cs: int(a) for cs, a
                               in agent.active_maneuvers.items()},
            "decided": {cs: int(a) for cs, a in action.items()},
            "wall_s": round(wall, 3),
            "air02": tracked_geometry(env, "AIR-02"),
        })
        if not dec["gate_mirror_ok"]:
            print(f"  [warn] gate mirror mismatch at step {step}: "
                  f"mirror {gate['to_plan']} vs observed "
                  f"{observed_plan_order}")

        # ---- D4 attention probes (gate-excluded + planned, on deepcopies)
        if args.with_d4:
            dense = args.d4_from <= step <= args.d4_to
            due = dense or (step % args.d4_sparse_every == 0)
            if due:
                with RngGuard():
                    probes = []
                    targets = [cs for cs in obs
                               if gate["statuses"].get(cs) == "IN_SECTOR"]
                    # excluded first (AIR-02 top priority within them),
                    # then planned (for a common ranking)
                    targets.sort(key=lambda cs: (cs in gate["to_plan"],
                                                 cs != "AIR-02"))
                    for cs in targets[: args.d4_cap]:
                        pr = d4_probe(env, obs, info, dec["decided"], cs,
                                      agent, args.seed, step,
                                      args.d4_sims, args.d4_horizon)
                        pr["gate_planned"] = cs in gate["to_plan"]
                        pr["gate_engaged"] = cs in gate["engaged"]
                        probes.append(pr)
                    dec["d4_probes"] = probes

        d1["decisions"].append(dec)
        d1["actions_log"].append({cs: int(a) for cs, a in action.items()})

        obs, rew, term, trunc, info = env.step(action)
        step += 1
        ep_return += float(sum(rew.values()))

        if first_violation is None:
            viol = find_violations(info)
            if viol:
                first_violation = {"step": step,
                                   "time_s": step * bim.SEC_PER_STEP,
                                   "desc": "; ".join(viol)}
                print(f"  [episode] FIRST VIOLATION at step {step} "
                      f"({step * bim.SEC_PER_STEP} s): {first_violation['desc']}")

        if step % 10 == 0:
            print(f"  step {step:3d} | n_ac {len(obs):2d} | "
                  f"decision {wall*1000:8.1f} ms | ep_return {ep_return:9.2f} "
                  f"| elapsed {time.time()-t_ep:6.0f} s")

        if not obs or all(term.values()) or all(trunc.values()):
            break

    d1["episode"] = {
        "steps": step,
        "episode_return": round(ep_return, 2),
        "first_violation": first_violation,
        "wall_s": round(time.time() - t_ep, 1),
        "reproduces_prior_fact": bool(
            first_violation and first_violation["step"] == 64
            and "AIR-02" in first_violation["desc"]
            and "excursion" in first_violation["desc"]),
    }
    print(f"\n[D1] episode done in {d1['episode']['wall_s']:.0f} s; "
          f"first violation: {first_violation}")
    print(f"[D1] reproduces prior fact (excursion AIR-02 @ step 64): "
          f"{d1['episode']['reproduces_prior_fact']}")

    d1["knowing_choices"] = knowing_choices(d1)
    d1["air02_history"] = air02_history(d1)
    save_json(os.path.join(args.out_dir, f"d1_seed{args.seed}.json"), d1)
    print_d1_summary(d1)
    return d1


def group_sims_by_root_action(plan: dict) -> Dict[int, List[dict]]:
    by = {}
    for s in plan["sims"]:
        if s["a0"] is None:
            continue
        by.setdefault(int(s["a0"]), []).append(s)
    return by


def action_violation_summary(sims: List[dict]) -> dict:
    v = [s for s in sims if s["viol"]]
    kinds = {}
    for s in v:
        kinds.setdefault(s["viol"]["kind"], []).append(s["viol"]["depth"])
    return {
        "n_sims": len(sims), "n_viol": len(v),
        "kinds": {k: {"n": len(d), "min_depth": min(d),
                      "min_disc_pen": round(
                          max((0.97 ** dd) for dd in d) * -50.0, 3)}
                  for k, d in kinds.items()},
        "bg_sims": sum(1 for s in sims if s["bg"]),
    }


def knowing_choices(d1: dict) -> List[dict]:
    """Decisions where EVERY visited root option contained a planned-aircraft
    violation in at least one rehearsal (the planner knowingly picked among
    violating futures)."""
    rows = []
    for dec in d1["decisions"]:
        for plan in dec["planned"]:
            by = group_sims_by_root_action(plan)
            visited = [a for a in plan["root"]]
            if not visited:
                continue
            per_a = {int(a): action_violation_summary(by.get(int(a), []))
                     for a in visited}
            if all(per_a[int(a)]["n_viol"] > 0 for a in visited):
                ch = int(plan["chosen"])
                rows.append({
                    "step": dec["step"], "cs": plan["cs"], "chosen": ch,
                    "chosen_summary": per_a.get(ch),
                    "per_action": per_a,
                    "root": plan["root"],
                })
    return rows


def air02_history(d1: dict) -> List[dict]:
    hist = []
    for dec in d1["decisions"]:
        a = dec["decided"].get("AIR-02")
        if a is None:
            continue
        planned = next((p for p in dec["planned"] if p["cs"] == "AIR-02"),
                       None)
        hist.append({
            "step": dec["step"],
            "action": int(a),
            "searched": planned is not None,
            "gate_engaged": "AIR-02" in dec["gate"]["engaged"],
            "midman": "AIR-02" in dec["gate"]["midman"],
            "status": dec["gate"]["statuses"].get("AIR-02"),
            "geom": dec.get("air02"),
            "chosen_env": planned["chosen_env"] if planned else None,
        })
    return hist


def print_d1_summary(d1: dict) -> None:
    print("\n[D1] --- summary ---")
    kc = d1["knowing_choices"]
    print(f"  knowing choices (all visited options violating): {len(kc)}")
    for row in kc:
        ks = ", ".join(f"{k}:{v['n']}@d>={v['min_depth']}"
                       for k, v in row["chosen_summary"]["kinds"].items())
        print(f"    step {row['step']:3d} {row['cs']}: chose a={row['chosen']}"
              f" ({ks})")
    n_dec = sum(len(d["planned"]) for d in d1["decisions"])
    n_viol_reh = sum(1 for d in d1["decisions"] for p in d["planned"]
                     if any(s["viol"] for s in p["sims"]))
    print(f"  searches: {n_dec}; searches with >=1 violating rehearsal: "
          f"{n_viol_reh}")
    mm = [d["step"] for d in d1["decisions"] if not d.get("gate_mirror_ok",
                                                          True)]
    print(f"  gate mirror mismatches: {len(mm)} {mm[:10]}")


# ============================================================================
# Snapshot rebuild by exact action replay (never all-NOOP fast-forward)
# ============================================================================

def rebuild_snapshots(actions_log: List[Dict[str, int]], steps: List[int],
                      seed: int, duration: int,
                      encoder_cls: str = "extra_minimal",
                      k_nearest: int = 2) -> Dict[int, tuple]:
    """Replay the recorded joint actions through a fresh env and deepcopy
    (env, obs, info) together at each requested decision step, so the info
    dict's simulator reference stays tied to the copied env."""
    want = sorted(set(steps))
    env = make_env(scenario_duration=duration, encoder_cls=encoder_cls,
                   k_nearest=k_nearest, macro_turns=True)
    obs, info = env.reset(seed=seed)
    out = {}
    t0 = time.time()
    for t in range(max(want) + 1):
        if t in want:
            out[t] = copy.deepcopy((env, obs, info))
        if t >= len(actions_log):
            break
        act = {cs: int(a) for cs, a in actions_log[t].items() if cs in obs}
        obs, rew, term, trunc, info = env.step(act)
        if not obs:
            break
    print(f"  [snapshots] rebuilt {sorted(out)} (encoder={encoder_cls}, "
          f"k={k_nearest}) in {time.time()-t0:.1f} s")
    missing = [t for t in want if t not in out]
    if missing:
        print(f"  [snapshots] WARNING: could not reach steps {missing}")
    return out


def load_d1(args) -> dict:
    path = os.path.join(args.out_dir, f"d1_seed{args.seed}.json")
    with open(path) as f:
        d1 = json.load(f)
    print(f"  [io] loaded {path} ({len(d1['actions_log'])} recorded steps)")
    return d1


def restore_probe_agent(agent: IntervalMCTSAgent, d1: dict, step: int,
                        env) -> None:
    """Give a probe agent the live agent's episode state at `step`
    (pre-decision maneuver tracking + bound action set)."""
    dec = next(d for d in d1["decisions"] if d["step"] == step)
    agent.active_maneuvers = {cs: int(a) for cs, a
                              in dec["pre_maneuvers"].items()}
    agent._last_env_timestep = getattr(env, "timestep", None)


def plan_compact(plan: dict) -> dict:
    """Compact per-planned-aircraft record for probe comparisons."""
    sims = plan["sims"]
    viol = [s for s in sims if s["viol"]]
    exc = [s for s in viol if s["viol"]["kind"] == "excursion"]
    los = [s for s in viol if s["viol"]["kind"] == "LoS"]
    bg = [s for s in sims if s["bg"]]
    return {
        "cs": plan["cs"], "chosen": plan["chosen"],
        "chosen_env": plan["chosen_env"], "root": plan["root"],
        "wall_s": plan.get("wall_s"),
        "n_sims": len(sims), "n_viol": len(viol),
        "n_excursion": len(exc), "n_los": len(los), "n_bg": len(bg),
        "min_excursion_depth": min((s["viol"]["depth"] for s in exc),
                                   default=None),
        "min_los_depth": min((s["viol"]["depth"] for s in los),
                             default=None),
        "min_bg_depth": min((s["bg"]["depth"] for s in bg), default=None),
        "bg_descs": sorted({d for s in bg for d in s["bg"]["descs"]})[:6],
        "chosen_action_viol": action_violation_summary(
            group_sims_by_root_action(plan).get(int(plan["chosen"]), [])),
    }


def run_joint_probe(agent: InstrumentedAgent, snap, step: int) -> dict:
    """One full joint decision (generate_action) on a snapshot."""
    env, obs, info = snap
    agent.begin_decision(step)
    t0 = time.perf_counter()
    action = agent.generate_action(env, obs, info)
    wall = time.perf_counter() - t0
    dec = agent.decision_records[-1]
    return {
        "step": step,
        "decided": {cs: int(a) for cs, a in action.items()},
        "plan_order": [p["cs"] for p in dec["planned"]],
        "planned": [plan_compact(p) for p in dec["planned"]],
        "wall_s": round(wall, 3),
    }


# ============================================================================
# D2: frozen default future vs route-following default
# ============================================================================

def run_d2(d1: dict, args) -> dict:
    print("=" * 78)
    print(f"[D2] rehearsal default policy counterfactual | snapshots "
          f"{args.snapshot_steps}")
    print("=" * 78)
    patch_registry_safe_rp()
    snaps = rebuild_snapshots(d1["actions_log"], args.snapshot_steps,
                              args.seed, args.duration)
    # derive the true action set from a pristine snapshot (pre-injection)
    any_env = snaps[sorted(snaps)[0]][0]
    acts, names = bim.derive_action_space(any_env)

    d2 = {"config": {"snapshot_steps": sorted(snaps), "sims": args.sims,
                     "horizon": args.horizon,
                     "modes": ["noop", "route"],
                     "frozen_aircraft_policy": "one-shot command at depth 0, "
                                               "then NOOP (both modes)"},
          "validation": None, "snapshots": []}

    for i, step in enumerate(sorted(snaps)):
        print(f"\n[D2] snapshot step {step}")
        env0, obs0, info0 = snaps[step]
        rp_idx = enable_route_parallel(env0)   # snapshot copy only
        rec = {"step": step, "rp_idx": rp_idx, "modes": {}}

        # one-time self-test: mode noop/noop must equal the base agent
        if i == 0:
            with RngGuard():
                snapA = copy.deepcopy((env0, obs0, info0))
                snapB = copy.deepcopy((env0, obs0, info0))
                base = InstrumentedAgent(n_simulations=args.sims,
                                         horizon=args.horizon,
                                         rng=random.Random(1234))
                base.preset_actions(acts, names)
                restore_probe_agent(base, d1, step, snapA[0])
                ref = run_joint_probe(base, snapA, step)
                nn = RouteDefaultAgent(n_simulations=args.sims,
                                       horizon=args.horizon,
                                       rp_idx=rp_idx, rollout_mode="noop",
                                       others_mode="noop",
                                       rng=random.Random(1234))
                nn.preset_actions(acts, names)
                restore_probe_agent(nn, d1, step, snapB[0])
                chk = run_joint_probe(nn, snapB, step)
                same = (ref["decided"] == chk["decided"]
                        and [p["root"] for p in ref["planned"]]
                        == [p["root"] for p in chk["planned"]])
                d2["validation"] = {
                    "step": step,
                    "copied_impl_matches_base_under_noop": bool(same)}
                print(f"  [D2] copied-implementation self-test "
                      f"(noop==base): {'PASS' if same else 'FAIL'}")

        for mode in ("noop", "route"):
            with RngGuard():
                snap = copy.deepcopy((env0, obs0, info0))
                ag = RouteDefaultAgent(
                    n_simulations=args.sims, horizon=args.horizon,
                    rp_idx=rp_idx,
                    rollout_mode=mode, others_mode=mode,
                    rng=random.Random(args.seed * 7919 + step))
                ag.preset_actions(acts, names)
                restore_probe_agent(ag, d1, step, snap[0])
                fb0 = RP_FALLBACKS["n"]
                rec["modes"][mode] = run_joint_probe(ag, snap, step)
                rec["modes"][mode]["rp_fallbacks"] = RP_FALLBACKS["n"] - fb0
            r = rec["modes"][mode]
            print(f"  mode={mode:5s}: decided "
                  f"{ {c: a for c, a in r['decided'].items() if a != 0} } "
                  f"| planned {r['plan_order']}")
            for p in r["planned"]:
                print(f"    {p['cs']}: chosen {p['chosen']} env "
                      f"{p['chosen_env']} viol {p['n_viol']}/{p['n_sims']} "
                      f"(exc {p['n_excursion']}, los {p['n_los']}, "
                      f"bg {p['n_bg']})")
        # what actually happened in D1 at this step
        d1dec = next(d for d in d1["decisions"] if d["step"] == step)
        rec["d1_actual"] = {
            "decided": d1dec["decided"],
            "plan_order": d1dec["observed_plan_order"],
            "planned": [plan_compact(p) for p in d1dec["planned"]],
        }
        rec["command_differs"] = {
            cs: {"noop": rec["modes"]["noop"]["decided"].get(cs),
                 "route": rec["modes"]["route"]["decided"].get(cs)}
            for cs in rec["modes"]["noop"]["decided"]
            if rec["modes"]["noop"]["decided"].get(cs)
            != rec["modes"]["route"]["decided"].get(cs)}
        d2["snapshots"].append(rec)

    save_json(os.path.join(args.out_dir, f"d2_seed{args.seed}.json"), d2)
    return d2


# ============================================================================
# D3: horizon sensitivity
# ============================================================================

def run_d3(d1: dict, args) -> dict:
    print("=" * 78)
    print(f"[D3] horizon sensitivity {args.d3_horizons} at sims "
          f"{args.d3_sims} | snapshots {args.snapshot_steps}")
    print("=" * 78)
    snaps = rebuild_snapshots(d1["actions_log"], args.snapshot_steps,
                              args.seed, args.duration)
    any_env = snaps[sorted(snaps)[0]][0]
    acts, names = bim.derive_action_space(any_env)
    d3 = {"config": {"horizons": args.d3_horizons, "sims": args.d3_sims,
                     "snapshot_steps": sorted(snaps)},
          "snapshots": []}
    for step in sorted(snaps):
        env0, obs0, info0 = snaps[step]
        rec = {"step": step, "configs": {}}
        print(f"\n[D3] snapshot step {step}")
        for H, S in zip(args.d3_horizons, args.d3_sims):
            with RngGuard():
                snap = copy.deepcopy((env0, obs0, info0))
                ag = InstrumentedAgent(
                    n_simulations=S, horizon=H,
                    rng=random.Random(args.seed * 104729 + step * 13 + H))
                ag.preset_actions(acts, names)
                restore_probe_agent(ag, d1, step, snap[0])
                rec["configs"][f"h{H}_s{S}"] = run_joint_probe(ag, snap, step)
            r = rec["configs"][f"h{H}_s{S}"]
            nz = {c: a for c, a in r["decided"].items() if a != 0}
            exc = {p["cs"]: p["min_excursion_depth"] for p in r["planned"]
                   if p["min_excursion_depth"] is not None}
            print(f"  h={H:2d} s={S:2d} wall {r['wall_s']:7.1f}s | "
                  f"non-NOOP {nz} | excursion visible {exc or '-'}")
        d3["snapshots"].append(rec)
    save_json(os.path.join(args.out_dir, f"d3_seed{args.seed}.json"), d3)
    return d3


# ============================================================================
# D5: interval semantics (envelope vs hybrid mean-bound backup)
# ============================================================================

def run_d5(d1: dict, args) -> dict:
    print("=" * 78)
    print(f"[D5] hybrid (leaf {os.path.basename(args.leaf_value)}) vs "
          f"envelope | snapshots {args.snapshot_steps}")
    print("=" * 78)
    import torch
    ck = torch.load(args.leaf_value, map_location="cpu", weights_only=False)
    enc, k = ck.get("encoder_cls", "extra_minimal"), ck.get("k", 2)
    print(f"  [D5] checkpoint metadata: encoder_cls={enc} k={k} "
          f"episode={ck.get('episode')} n_actions={ck.get('n_actions')}")
    # Rebuild the SAME sim states, but with the leaf net's observation
    # encoder (dynamics verified identical across encoders).
    snaps = rebuild_snapshots(d1["actions_log"], args.snapshot_steps,
                              args.seed, args.duration,
                              encoder_cls=enc, k_nearest=k)
    any_env = snaps[sorted(snaps)[0]][0]
    acts, names = bim.derive_action_space(any_env)

    d5 = {"config": {"leaf_value": args.leaf_value, "encoder_cls": enc,
                     "k_nearest": k, "sims": args.sims,
                     "horizon": args.horizon,
                     "snapshot_steps": sorted(snaps)},
          "snapshots": [], "pairing": [], "summary": {}}

    for step in sorted(snaps):
        with RngGuard():
            snap = copy.deepcopy(snaps[step])
            ag = InstrumentedAgent(
                n_simulations=args.sims, horizon=args.horizon,
                leaf_value_path=args.leaf_value,
                rng=random.Random(args.seed * 31337 + step))
            ag.preset_actions(acts, names)
            restore_probe_agent(ag, d1, step, snap[0])
            rec = run_joint_probe(ag, snap, step)
        d1dec = next(d for d in d1["decisions"] if d["step"] == step)
        rec["d1_actual"] = {
            "decided": d1dec["decided"],
            "planned": [plan_compact(p) for p in d1dec["planned"]],
        }
        d5["snapshots"].append(rec)
        print(f"  step {step}: hybrid planned "
              f"{[p['cs'] for p in rec['planned']]}, wall {rec['wall_s']}s")

    # Pair (step, cs, action) across the two interval objects
    pairs = []
    for rec in d5["snapshots"]:
        env_by_cs = {p["cs"]: p for p in rec["d1_actual"]["planned"]}
        for hp in rec["planned"]:
            ep = env_by_cs.get(hp["cs"])
            if ep is None:
                continue
            for a, hv in hp["root"].items():
                ev = ep["root"].get(str(a)) or ep["root"].get(a)
                if ev is None:
                    continue
                pairs.append({
                    "step": rec["step"], "cs": hp["cs"], "a": int(a),
                    "env_lower": ev["lower"], "env_upper": ev["upper"],
                    "env_width": ev["width"],
                    "hyb_lower": hv["lower"], "hyb_upper": hv["upper"],
                    "hyb_width": hv["width"],
                    "env_score": ev["score_act"], "hyb_score": hv["score_act"],
                })
        for hp in rec["planned"]:
            ep = env_by_cs.get(hp["cs"])
            if ep is not None:
                d5["pairing"].append({
                    "step": rec["step"], "cs": hp["cs"],
                    "env_chosen": ep["chosen"], "hyb_chosen": hp["chosen"],
                    "agree": ep["chosen"] == hp["chosen"],
                    "env_gap": relevance_metrics(
                        {int(k2): v for k2, v in ep["root"].items()},
                        0.1)["hurwicz_gap"],
                    "hyb_gap": relevance_metrics(
                        {int(k2): v for k2, v in hp["root"].items()},
                        0.1)["hurwicz_gap"],
                })
    d5["paired_actions"] = pairs
    ws = [(p["env_width"], p["hyb_width"]) for p in pairs]
    gaps = [(q["env_gap"], q["hyb_gap"]) for q in d5["pairing"]
            if q["env_gap"] is not None and q["hyb_gap"] is not None]
    d5["summary"] = {
        "n_paired_actions": len(pairs),
        "width_corr": pearson([w[0] for w in ws], [w[1] for w in ws]),
        "mean_env_width": (round(float(np.mean([w[0] for w in ws])), 4)
                           if ws else None),
        "mean_hyb_width": (round(float(np.mean([w[1] for w in ws])), 4)
                           if ws else None),
        "hurwicz_gap_corr": pearson([g[0] for g in gaps],
                                    [g[1] for g in gaps]),
        "chosen_agree_frac": (round(float(np.mean(
            [q["agree"] for q in d5["pairing"]])), 3)
            if d5["pairing"] else None),
        "n_paired_decisions": len(d5["pairing"]),
    }
    print(f"  [D5] summary: {d5['summary']}")
    save_json(os.path.join(args.out_dir, f"d5_seed{args.seed}.json"), d5)
    return d5


# ============================================================================
# D4 standalone summary (data collected during D1)
# ============================================================================

def summarize_d4(d1: dict, args) -> dict:
    print("=" * 78)
    print("[D4] attention: TTC gate vs interval decision-relevance")
    print("=" * 78)
    rows = []
    for dec in d1["decisions"]:
        probes = dec.get("d4_probes")
        if not probes:
            continue
        ranked = sorted(
            (p for p in probes if p["metrics"]["spread"] is not None),
            key=lambda p: -p["metrics"]["spread"])
        rel_rank = [p["cs"] for p in ranked]
        gate_rank = dec["gate"]["to_plan"]
        k = len(gate_rank)
        topk = set(rel_rank[:k]) if k else set()
        overlap = len(topk & set(gate_rank)) / k if k else None
        rows.append({
            "step": dec["step"],
            "gate_to_plan": gate_rank,
            "relevance_rank": rel_rank,
            "topk_overlap": overlap,
            "air02": next((p for p in probes if p["cs"] == "AIR-02"), None),
        })
    ol = [r["topk_overlap"] for r in rows if r["topk_overlap"] is not None]
    picks = []
    for dec in d1["decisions"]:
        for p in dec.get("d4_probes", []):
            if not p["gate_planned"]:
                picks.append(p["metrics"]["pick_nonnoop"])
    air02_excluded_wouldact = []
    for r in rows:
        p = r["air02"]
        if p and p["cs"] == "AIR-02" and not p["gate_planned"]:
            air02_excluded_wouldact.append(
                {"step": r["step"], "pick": p["metrics"]["pick"],
                 "spread": p["metrics"]["spread"],
                 "viol_sims": p["viol_sims"]})
    out = {
        "per_step": rows,
        "mean_topk_overlap": (round(float(np.mean(ol)), 3) if ol else None),
        "excluded_nonnoop_frac": (round(float(np.mean(picks)), 3)
                                  if picks else None),
        "n_excluded_probes": len(picks),
        "air02_excluded_steps": air02_excluded_wouldact,
    }
    print(f"  steps probed: {len(rows)}; mean top-k overlap "
          f"(gate vs interval-relevance): {out['mean_topk_overlap']}")
    print(f"  gate-excluded probes: {out['n_excluded_probes']}; "
          f"would act (non-NOOP pick): {out['excluded_nonnoop_frac']}")
    print(f"  AIR-02 while excluded: {len(air02_excluded_wouldact)} probes")
    save_json(os.path.join(args.out_dir, f"d4_seed{args.seed}.json"), out)
    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Design-level diagnostic campaign for "
                    "bluebird_interval_mcts (P1-P5)")
    ap.add_argument("--d1", action="store_true", help="instrumented episode")
    ap.add_argument("--d2", action="store_true", help="default-future probes")
    ap.add_argument("--d3", action="store_true", help="horizon probes")
    ap.add_argument("--d4", action="store_true",
                    help="attention probes (during D1; implies --d1 if no "
                         "saved D1 data exists)")
    ap.add_argument("--d5", action="store_true", help="hybrid interval probes")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="tiny end-to-end smoke (seed 42, 180 s)")
    ap.add_argument("--seed", type=int, default=10043)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--snapshot-steps", type=int, nargs="*",
                    default=[45, 48, 51, 54, 57, 60])
    ap.add_argument("--d3-horizons", type=int, nargs="*", default=[15, 25, 40])
    ap.add_argument("--d3-sims", type=int, nargs="*", default=[24, 32, 40])
    ap.add_argument("--d4-sims", type=int, default=8)
    ap.add_argument("--d4-horizon", type=int, default=8)
    ap.add_argument("--d4-from", type=int, default=30,
                    help="dense D4 probing window start (every step)")
    ap.add_argument("--d4-to", type=int, default=72)
    ap.add_argument("--d4-sparse-every", type=int, default=5)
    ap.add_argument("--d4-cap", type=int, default=14,
                    help="max mini-searches per decision")
    ap.add_argument("--leaf-value", type=str, default=LEAF_CKPT_DEFAULT)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    if args.all:
        args.d1 = args.d2 = args.d3 = args.d4 = args.d5 = True
    if not any([args.d1, args.d2, args.d3, args.d4, args.d5]):
        ap.error("select at least one of --d1..--d5 or --all")

    if args.quick:
        args.seed = 42 if args.seed == 10043 else args.seed
        args.duration = 180
        args.sims, args.horizon = 8, 8
        args.snapshot_steps = [10, 16]
        args.d3_horizons, args.d3_sims = [8, 12], [8, 10]
        args.d4_from, args.d4_to, args.d4_cap = 0, 30, 6
        args.d4_sims, args.d4_horizon = 6, 6

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tee = Tee(os.path.join(args.out_dir, f"campaign_{stamp}.log"))
    sys.stdout = tee
    print(f"design campaign | {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
          f"seed {args.seed} | quick={args.quick}")
    print(f"selected: d1={args.d1} d2={args.d2} d3={args.d3} d4={args.d4} "
          f"d5={args.d5}")
    t_all = time.time()

    d1 = None
    args.with_d4 = args.d4
    need_episode = args.d1 or args.d4
    if need_episode:
        d1_path = os.path.join(args.out_dir, f"d1_seed{args.seed}.json")
        if args.d4 and not args.d1 and os.path.exists(d1_path):
            d1 = load_d1(args)
            if not any("d4_probes" in d for d in d1["decisions"]):
                print("  [D4] saved D1 has no probes -> re-running episode "
                      "with probes")
                d1 = run_d1(args)
        else:
            d1 = run_d1(args)

    if any([args.d2, args.d3, args.d5]) and d1 is None:
        d1 = load_d1(args)

    if args.d4 and d1 is not None:
        summarize_d4(d1, args)
    if args.d2:
        run_d2(d1, args)
    if args.d3:
        run_d3(d1, args)
    if args.d5:
        run_d5(d1, args)

    print(f"\ncampaign wall time: {time.time()-t_all:.0f} s")
    tee.close()


if __name__ == "__main__":
    main()
