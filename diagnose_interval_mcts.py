"""
Step-by-step diagnostic suite for the interval MCTS in bluebird_interval_mcts.py.

Motivation: the suspicion that "interval MCTS was never truly functioning
properly" (including its design ancestor IP-ML/nim_interval_mcts.py).  Every
stage of the planner is therefore verified INDEPENDENTLY, with no assumptions:
the backup arithmetic is re-derived by hand from recorded simulations, the
tree invariants are re-checked from first principles, selection is unit-tested
on hand-built nodes, and the violation penalty is shown to actually reach the
root envelope on a real pre-violation state.

Checks
------
  1  BACKUP MATH vs HAND COMPUTATION   record every simulation (action path,
     per-step rewards, violation flags, total g) on a tiny fixed setup and
     re-derive every g and every node's (visit_count, lower, upper, ret_sum)
     by hand.  FAIL on any mismatch > 1e-9.
  2  ENVELOPE INVARIANTS               real _plan_single with 24 sims:
     lower <= mean <= upper on every node, visit-count flow conservation
     (parent visits == sum(child visits) + sims terminating at parent),
     sum(root child visits) == n_simulations.
  3  EXPANSION                          root expands all 3 actions within the
     first 3 sims (root is PW-exempt); every internal tree step obeys the
     progressive-widening gate cap = max(1, floor(k * N^alpha)); no root
     action starved (< 2 visits) at 24 sims.
  4  SELECTION SANITY                   unit tests on synthetic IntervalNode
     children: Hurwicz ordering, UCB bonus lifting a rare arm, root pick at
     c_act=0 = max lower bound with NOOP winning exact ties.
  5  DETERMINIZED CLONE FIDELITY        deepcopy(env) tracks the live env
     exactly under identical actions, diverges under different actions, and
     generate_action leaves the live env bit-identical (state + RNG future).
  6  PENALTY REACHABILITY               from the real pre-violation state of
     seed 10043 (all-NOOP LoS AIR-00/AIR-02 at step 77), fast-forward to step
     65 and plan AIR-00: at least one simulation must record violated=True
     with g including the discounted -50, and the root envelope must reflect
     it.  FAIL = the planner is blind to the violation.
  7  SEED-10043 AUTOPSY  (--heavy)      instrumented decision replay of steps
     55..77 (gating, root envelopes, chosen actions) plus a beam search over
     open-loop AIR-00 action sequences: was the LoS avoidable at all?
  8  DISCOUNT ANALYSIS                  pure math: -50*gamma^t vs accumulated
     shaping-reward differences over the horizon ("penalty neutered by
     discounting" hypothesis).

Usage
-----
    nice -n 15 .venv/bin/python diagnose_interval_mcts.py            # all light checks
    nice -n 15 .venv/bin/python diagnose_interval_mcts.py --check 1 4 8
    nice -n 15 .venv/bin/python diagnose_interval_mcts.py --heavy    # adds check 7 (LONG)

Outputs (log + JSON summary) go to checkpoints/bluebird/diagnostics/.
The live env instances used here are private to this script; no training
process or training checkpoint is touched.
"""

import argparse
import contextlib
import copy
import inspect
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import bluebird_interval_mcts as M
from bluebird_interval_mcts import (
    ALL_ACTIONS, LEFT_10, NOOP, RIGHT_10,
    IntervalMCTSAgent, IntervalNode,
    find_violations, make_env, violation_involving, _status,
)

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "checkpoints" / "bluebird" / "diagnostics"

ACTION_NAMES = {NOOP: "NOOP", LEFT_10: "LEFT", RIGHT_10: "RIGHT"}
TOL = 1e-9

# ============================================================================
# Reporting plumbing
# ============================================================================

class Tee:
    """Duplicate stdout to a log file."""
    def __init__(self, path):
        self._f = open(path, "w")
        self._stdout = sys.stdout

    def write(self, s):
        self._stdout.write(s)
        self._f.write(s)

    def flush(self):
        self._stdout.flush()
        self._f.flush()

    def close(self):
        self._f.close()


RESULTS = []  # (check_no, name, status, evidence-lines)


def emit(check_no, name, status, evidence):
    RESULTS.append({"check": check_no, "name": name, "status": status,
                    "evidence": evidence})
    print(f"\n  >>> [{check_no}] {name}: {status}")


def section(no, title):
    print("\n" + "=" * 78)
    print(f"[{no}] {title}")
    print("=" * 78)


def act_name(a):
    return ACTION_NAMES.get(a, str(a))


def path_str(actions):
    return "root" + "".join(f"->{act_name(a)}" for a in actions)


# ============================================================================
# Shared helpers
# ============================================================================

def env_at(seed, n_steps):
    """Fresh env at `seed`, all-NOOP fast-forwarded n_steps. Returns
    (env, obs, info).  ~10-30 ms per step -- cheap."""
    env = make_env()
    obs, info = env.reset(seed=seed)
    for _ in range(n_steps):
        obs, rew, term, trunc, info = env.step({cs: NOOP for cs in obs})
    return env, obs, info


def snapshot(info):
    """Value snapshot (floats copied NOW) of every tracked aircraft's
    kinematic state, from the sim object referenced by info."""
    sim = info["simulator_environment"]
    snap = {}
    for cs, ac in sorted(sim.aircraft.items()):
        snap[cs] = tuple(float(getattr(ac, f)) for f in ("lat", "lon", "fl")
                         if hasattr(ac, f))
        for extra in ("hdg", "gs"):
            if hasattr(ac, extra):
                try:
                    snap[cs] += (float(getattr(ac, extra)),)
                except (TypeError, ValueError):
                    pass
    return snap


@contextlib.contextmanager
def capture_created_nodes():
    """Monkeypatch module-level IntervalNode so every node created inside
    bluebird_interval_mcts (root included) is recorded.  created[0] is the
    root of the next _plan_single call."""
    created = []
    orig = M.IntervalNode

    class SpyNode(orig):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created.append(self)

    M.IntervalNode = SpyNode
    try:
        yield created
    finally:
        M.IntervalNode = orig


@contextlib.contextmanager
def capture_updates(store):
    """Monkeypatch IntervalNode.update to record (node id, g) in call order.
    Patched on the BASE class so SpyNode subclasses are covered too."""
    orig = IntervalNode.update

    def patched(self, g):
        store.append((id(self), g))
        return orig(self, g)

    IntervalNode.update = patched
    try:
        yield
    finally:
        IntervalNode.update = orig


class InstrumentedAgent(IntervalMCTSAgent):
    """Records, for every simulation, the full step trace
    (action, depth, r, done, violated) and every IntervalNode.update call
    (node id, g).  The recording wrapper does NOT alter control flow."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.sim_traces = []
        self._cur = None

    def _step_sim(self, env, callsigns, planned_cs, planned_action, depth, frozen):
        r, done, violated, next_cs = super()._step_sim(
            env, callsigns, planned_cs, planned_action, depth, frozen)
        if self._cur is not None:
            self._cur["steps"].append(
                {"action": planned_action, "depth": depth, "r": r,
                 "done": done, "violated": violated})
        return r, done, violated, next_cs

    def _simulate(self, root, env, callsigns, planned_cs, frozen):
        self._cur = {"steps": [], "updates": []}
        with capture_updates(self._cur["updates"]):
            super()._simulate(root, env, callsigns, planned_cs, frozen)
        self.sim_traces.append(self._cur)
        self._cur = None


def hand_g(steps, gamma, penalty):
    """Re-derive the total discounted return exactly as _simulate builds it:
    disc starts at 1 and is multiplied AFTER each step, in step order, so
    this is bit-compatible with the production arithmetic."""
    g, disc = 0.0, 1.0
    for s in steps:
        g += disc * s["r"]
        if s["violated"]:
            g += disc * penalty
        disc *= gamma
    return g


class ShadowNode:
    """Independent re-implementation of the node statistics for cross-check."""
    __slots__ = ("visit", "ret", "lo", "hi", "children")

    def __init__(self):
        self.visit, self.ret = 0, 0.0
        self.lo, self.hi = math.inf, -math.inf
        self.children = {}

    def update(self, g):
        self.visit += 1
        self.ret += g
        self.lo = min(self.lo, g)
        self.hi = max(self.hi, g)


def collect_real(root):
    """{action-path tuple: node} for every node in a real IntervalNode tree."""
    out = {}
    stack = [((), root)]
    while stack:
        p, nd = stack.pop()
        out[p] = nd
        for a, ch in nd.children.items():
            stack.append((p + (a,), ch))
    return out


def collect_shadow(root):
    out = {}
    stack = [((), root)]
    while stack:
        p, nd = stack.pop()
        out[p] = nd
        for a, ch in nd.children.items():
            stack.append((p + (a,), ch))
    return out


def run_instrumented_search(agent, live_env, obs, planned_cs, frozen=None):
    """Drive the exact _plan_single simulation loop (deepcopy per sim,
    _simulate into a shared root) while KEEPING the root, so node statistics
    can be audited afterwards.  Mirrors _plan_single lines 363-367."""
    frozen = dict(frozen or {})
    root = IntervalNode()
    callsigns = list(obs.keys())
    for _ in range(agent.n_simulations):
        sim_env = copy.deepcopy(live_env)
        agent._simulate(root, sim_env, list(callsigns), planned_cs, frozen)
    return root


def riskiest_callsign(agent, obs, info):
    """Replicate generate_action's risk ordering; return the riskiest
    aircraft (smallest min-separation to a vertically-close neighbour)."""
    callsigns = list(obs.keys())
    states = M.aircraft_states(info, callsigns)
    risk = {}
    for cs in callsigns:
        if cs not in states:
            continue
        la, lo, fa = states[cs]
        dmin = math.inf
        for other, (lb, lob, fb) in states.items():
            if other == cs or abs(fa - fb) >= agent.alert_fl:
                continue
            dmin = min(dmin, M.haversine_nm(la, lo, lb, lob))
        risk[cs] = dmin
    ranked = sorted(risk, key=lambda cs: risk.get(cs, math.inf))
    return ranked[0] if ranked else callsigns[0], risk


# ============================================================================
# CHECK 1: backup math vs hand computation
# ============================================================================

def check1():
    section(1, "BACKUP MATH vs HAND COMPUTATION")
    print("Setup: seed-42 env fast-forwarded 30 steps; n_simulations=6, "
          "horizon=3;\nevery simulation's steps and every IntervalNode.update "
          "recorded; g and all\nnode stats re-derived by hand (tolerance 1e-9).")

    env, obs, info = env_at(42, 30)
    print(f"State: {len(obs)} aircraft: {sorted(obs.keys())}")
    planned_cs, risk = riskiest_callsign(
        IntervalMCTSAgent(), obs, info)
    # Exercise the frozen-action path too (does not change backup math):
    frozen_cs = next((c for c in sorted(obs) if c != planned_cs), None)
    frozen = {frozen_cs: RIGHT_10} if frozen_cs else {}
    print(f"Planned aircraft: {planned_cs} (risk {risk.get(planned_cs, math.inf):.2f} nm); "
          f"frozen: { {c: act_name(a) for c, a in frozen.items()} }")

    agent = InstrumentedAgent(n_simulations=6, horizon=3,
                              rng=random.Random(42))
    root = run_instrumented_search(agent, env, obs, planned_cs, frozen)

    evidence, failures = [], []
    shadow_root = ShadowNode()

    for i, tr in enumerate(agent.sim_traces):
        steps, updates = tr["steps"], tr["updates"]
        # depths must be contiguous 0..n-1 (disc bookkeeping is lockstep)
        depths = [s["depth"] for s in steps]
        if depths != list(range(len(steps))):
            failures.append(f"sim {i}: non-contiguous depths {depths}")
        gs = {g for _, g in updates}
        if len(gs) != 1:
            failures.append(f"sim {i}: path nodes updated with UNEQUAL g: {gs}")
            continue
        g_rec = next(iter(gs))
        g_hand = hand_g(steps, agent.gamma, agent.violation_penalty)
        dg = abs(g_rec - g_hand)
        n_tree = len(updates) - 1          # path = root + one node per tree step
        tree_actions = [s["action"] for s in steps[:n_tree]]
        viol = any(s["violated"] for s in steps)
        line = (f"sim {i}: path {path_str(tree_actions):<28s} "
                f"tree_steps {n_tree} total_steps {len(steps)} "
                f"violated {str(viol):<5s} g_rec {g_rec:+.6f} "
                f"g_hand {g_hand:+.6f} |diff| {dg:.2e}")
        print("  " + line)
        for s in steps:
            print(f"      t={s['depth']} a={act_name(s['action']):<5s} "
                  f"r={s['r']:+.6f} violated={s['violated']} done={s['done']}")
        evidence.append(line)
        if dg > TOL:
            failures.append(f"sim {i}: g mismatch {dg:.3e} (> 1e-9)")
        if n_tree < 1 or n_tree > len(steps):
            failures.append(f"sim {i}: implausible tree-step count {n_tree} "
                            f"vs {len(steps)} recorded steps")
        # shadow backup with the HAND-computed g
        node = shadow_root
        node.update(g_hand)
        for a in tree_actions:
            node = node.children.setdefault(a, ShadowNode())
            node.update(g_hand)

    # ---- compare real tree against the shadow tree, node by node
    real = collect_real(root)
    shad = collect_shadow(shadow_root)
    if set(real) != set(shad):
        failures.append(f"tree topology differs: real {sorted(real)} "
                        f"vs shadow {sorted(shad)}")
    print(f"\n  Node-by-node comparison ({len(real)} real nodes, "
          f"{len(shad)} shadow nodes):")
    for p in sorted(set(real) | set(shad)):
        rn, sn = real.get(p), shad.get(p)
        if rn is None or sn is None:
            print(f"    {path_str(p):<30s} MISSING on one side "
                  f"(real={rn is not None}, shadow={sn is not None})")
            continue
        dv = rn.visit_count - sn.visit
        dl = abs(rn.lower - sn.lo)
        du = abs(rn.upper - sn.hi)
        dr = abs(rn.ret_sum - sn.ret)
        line = (f"{path_str(p):<30s} N {rn.visit_count}=={sn.visit} "
                f"lower {rn.lower:+.6f} (d={dl:.1e}) "
                f"upper {rn.upper:+.6f} (d={du:.1e}) "
                f"ret_sum {rn.ret_sum:+.6f} (d={dr:.1e})")
        print("    " + line)
        evidence.append(line)
        if dv != 0:
            failures.append(f"{path_str(p)}: visit_count real {rn.visit_count} "
                            f"!= hand {sn.visit}")
        for nm, d in (("lower", dl), ("upper", du), ("ret_sum", dr)):
            if d > TOL:
                failures.append(f"{path_str(p)}: {nm} mismatch {d:.3e} (> 1e-9)")

    status = "FAIL" if failures else "PASS"
    for f in failures:
        print("  FAIL: " + f)
    evidence = failures + evidence if failures else \
        [f"{len(agent.sim_traces)} sims, {len(real)} nodes: all g and all "
         f"node stats match hand computation to <= 1e-9"] + evidence
    emit(1, "backup math vs hand computation", status, evidence[:40])
    return status


# ============================================================================
# CHECK 2: envelope invariants on a real planning call
# ============================================================================

def check2():
    section(2, "ENVELOPE INVARIANTS (real _plan_single, 24 sims)")
    print("Invariant derivation from the code: every simulation updates every\n"
          "node on its root->leaf path exactly once with the same g.  Hence:\n"
          "  (a) lower <= mean <= upper on every node (min/mean/max of the\n"
          "      same multiset of returns);\n"
          "  (b) visit flow: node.visit == sum(child.visit) + (#sims that\n"
          "      TERMINATED at node), so node.visit >= sum(child.visit);\n"
          "  (c) root.visit == n_simulations, and -- because with horizon>=1\n"
          "      the tree phase always takes at least one step from the root\n"
          "      -- sum(root child visits) == n_simulations exactly.")

    env, obs, info = env_at(42, 30)
    agent = IntervalMCTSAgent(n_simulations=24, rng=random.Random(42))
    planned_cs, _ = riskiest_callsign(agent, obs, info)

    with capture_created_nodes() as created:
        action, widths = agent._plan_single(env, list(obs.keys()), planned_cs, {})
    root = created[0]
    nodes = collect_real(root)
    print(f"\nPlanned {planned_cs}; chosen action {act_name(action)}; "
          f"{len(nodes)} nodes in tree")

    evidence, failures = [], []
    n_term_total = 0
    for p in sorted(nodes):
        nd = nodes[p]
        kid_sum = sum(c.visit_count for c in nd.children.values())
        n_term = nd.visit_count - kid_sum
        n_term_total += n_term
        line = (f"{path_str(p):<34s} N={nd.visit_count:<3d} "
                f"[{nd.lower:+9.4f}, {nd.upper:+9.4f}] mean {nd.mean:+9.4f} "
                f"kid_sum={kid_sum} term_here={n_term}")
        print("  " + line)
        evidence.append(line)
        if nd.visit_count < 1:
            failures.append(f"{path_str(p)}: node in tree with 0 visits")
        if not (nd.lower <= nd.mean + 1e-12 and nd.mean <= nd.upper + 1e-12):
            failures.append(f"{path_str(p)}: lower<=mean<=upper violated: "
                            f"{nd.lower} / {nd.mean} / {nd.upper}")
        if nd.lower > nd.upper:
            failures.append(f"{path_str(p)}: lower > upper")
        if n_term < 0:
            failures.append(f"{path_str(p)}: visit flow broken "
                            f"(N={nd.visit_count} < children sum {kid_sum})")
        if not (math.isfinite(nd.ret_sum) and math.isfinite(nd.lower)
                and math.isfinite(nd.upper)):
            failures.append(f"{path_str(p)}: non-finite statistic")

    root_kid_sum = sum(c.visit_count for c in root.children.values())
    print(f"\n  root.visit_count = {root.visit_count} "
          f"(n_simulations = {agent.n_simulations})")
    print(f"  sum(root child visits) = {root_kid_sum}")
    print(f"  total terminations across tree = {n_term_total} "
          f"(should equal n_simulations: each sim terminates at exactly one node)")
    if root.visit_count != agent.n_simulations:
        failures.append(f"root.visit_count {root.visit_count} != "
                        f"n_simulations {agent.n_simulations}")
    if root_kid_sum != agent.n_simulations:
        # spec: document any off-by-one with evidence
        failures.append(f"sum(root child visits) {root_kid_sum} != "
                        f"n_simulations {agent.n_simulations}")
    if n_term_total != agent.n_simulations:
        failures.append(f"termination count {n_term_total} != "
                        f"n_simulations {agent.n_simulations}")

    status = "FAIL" if failures else "PASS"
    for f in failures:
        print("  FAIL: " + f)
    head = [f"root N={root.visit_count}, root-children sum={root_kid_sum}, "
            f"terminations={n_term_total}, n_sims={agent.n_simulations}; "
            f"{len(nodes)} nodes all satisfy lower<=mean<=upper and visit flow"]
    emit(2, "envelope invariants", status, failures + head + evidence[:20])
    return status


# ============================================================================
# CHECK 3: expansion / progressive widening
# ============================================================================

def check3():
    section(3, "EXPANSION: root PW exemption, internal PW schedule, starvation")
    print("Method: instrumented run (24 sims, horizon 6), then REPLAY the\n"
          "recorded tree paths through a shadow tree, checking at every tree\n"
          "step that an expansion happened IFF the PW gate allowed one:\n"
          "  root:      cap = 3 (exempt);\n"
          "  internal:  cap = max(1, floor(pw_k * max(1,N)^pw_alpha)), N = the\n"
          "             node's visit count at selection time (pre-backup).")

    env, obs, info = env_at(42, 30)
    agent = InstrumentedAgent(n_simulations=24, horizon=6,
                              rng=random.Random(42))
    planned_cs, _ = riskiest_callsign(agent, obs, info)
    root = run_instrumented_search(agent, env, obs, planned_cs)

    evidence, failures = [], []
    shadow_root = ShadowNode()
    expansions = []          # (sim, path-to-node, node_visits, n_children, action)
    root_children_by_sim = []

    for i, tr in enumerate(agent.sim_traces):
        steps = tr["steps"]
        n_tree = len(tr["updates"]) - 1
        tree_actions = [s["action"] for s in steps[:n_tree]]
        # gate replay against pre-backup shadow state
        node, at_path = shadow_root, ()
        for a in tree_actions:
            if node is shadow_root:
                cap = len(ALL_ACTIONS)
            else:
                cap = max(1, int(agent.pw_k *
                                 max(1, node.visit) ** agent.pw_alpha))
            unexpanded = [x for x in ALL_ACTIONS if x not in node.children]
            allowed = bool(unexpanded) and \
                len(node.children) < min(len(ALL_ACTIONS), cap)
            expanded = a not in node.children
            if expanded != allowed:
                failures.append(
                    f"sim {i} at {path_str(at_path)}: expansion={expanded} but "
                    f"PW gate allowed={allowed} (N={node.visit}, "
                    f"children={len(node.children)}, cap={cap})")
            if expanded:
                expansions.append((i, at_path, node.visit,
                                   len(node.children), a))
            node = node.children.setdefault(a, ShadowNode())
            at_path += (a,)
        # backup into shadow (visit counts only matter here)
        g = tr["updates"][0][1]
        nd = shadow_root
        nd.update(g)
        for a in tree_actions:
            nd = nd.children[a]
            nd.update(g)
        root_children_by_sim.append(len(shadow_root.children))

    print("\n  Expansion events (sim, node, node_visits_at_expansion, "
          "children_before, new_action):")
    for i, p, nv, nc, a in expansions:
        line = (f"sim {i:2d}  node {path_str(p):<26s} N={nv:<3d} "
                f"children_before={nc} expands {act_name(a)}")
        print("    " + line)
        evidence.append(line)

    print(f"\n  Root children after sims 1/2/3: "
          f"{root_children_by_sim[:3]} (expect [1, 2, 3])")
    if root_children_by_sim[:3] != [1, 2, 3]:
        failures.append(f"root did NOT expand all {len(ALL_ACTIONS)} actions in "
                        f"the first 3 sims: {root_children_by_sim[:3]}")

    print("  Root child visits at 24 sims (starvation check, need >= 2 each):")
    for a in ALL_ACTIONS:
        ch = root.children.get(a)
        v = ch.visit_count if ch else 0
        line = (f"root child {act_name(a):<5s}: visits={v:<3d} "
                + (f"[{ch.lower:+.4f}, {ch.upper:+.4f}]" if ch else "(never created)"))
        print("    " + line)
        evidence.append(line)
        if v < 2:
            failures.append(f"root action {act_name(a)} starved: "
                            f"{v} visits at 24 sims")

    internal = [e for e in expansions if e[1] != ()]
    print(f"\n  {len(expansions)} expansions total, {len(internal)} at internal "
          f"nodes -- every one matched the PW gate replay"
          if not failures else "")
    status = "FAIL" if failures else "PASS"
    for f in failures:
        print("  FAIL: " + f)
    head = [f"{len(expansions)} expansions ({len(internal)} internal) all "
            f"consistent with PW schedule; root fully expanded by sim 3; "
            f"no starvation at 24 sims"]
    emit(3, "expansion / progressive widening", status,
         failures + head + evidence[:25])
    return status


# ============================================================================
# CHECK 4: selection sanity (no env)
# ============================================================================

def _mknode(lo, hi, visits, ret=None):
    nd = IntervalNode()
    nd.visit_count = visits
    nd.lower, nd.upper = lo, hi
    nd.ret_sum = ret if ret is not None else visits * (lo + hi) / 2.0
    return nd


class _StubSimAgent(IntervalMCTSAgent):
    """Agent whose _simulate only installs preset children -- lets us drive
    the REAL root-decision code in _plan_single without any environment."""
    def __init__(self, preset, **kw):
        super().__init__(**kw)
        self._preset = preset

    def _simulate(self, root, env, callsigns, planned_cs, frozen):
        if not root.children:
            root.children.update(self._preset)
        root.update(0.0)


def _root_pick(preset, c_act):
    agent = _StubSimAgent(preset, n_simulations=1, c_act=c_act)
    action, _ = agent._plan_single(None, ["X"], "X", {})
    return action


def check4():
    section(4, "SELECTION SANITY (synthetic nodes, unit tests)")
    evidence, failures = [], []

    def expect(label, got, want):
        ok = got == want
        line = f"{label}: got {act_name(got)}, expected {act_name(want)} -> " \
               f"{'ok' if ok else 'WRONG'}"
        print("  " + line)
        evidence.append(line)
        if not ok:
            failures.append(line)

    # (a) in-tree Hurwicz ordering, equal visits (equal UCB bonus)
    parent = IntervalNode(); parent.visit_count = 20
    parent.children = {NOOP: _mknode(0.0, 1.0, 10),      # Hurwicz .8: 0.80
                       LEFT_10: _mknode(0.5, 0.6, 10)}   # Hurwicz .8: 0.58
    ag = IntervalMCTSAgent(c_search=0.8, c_visit=3.0)
    expect("(a1) Hurwicz 0.80 vs 0.58, equal visits", ag._select_action(parent), NOOP)
    parent.children = {NOOP: _mknode(0.5, 0.6, 10),
                       LEFT_10: _mknode(0.0, 1.0, 10)}
    expect("(a2) same, arms swapped", ag._select_action(parent), LEFT_10)
    ag0 = IntervalMCTSAgent(c_search=0.8, c_visit=0.0)   # pure Hurwicz
    parent.children = {NOOP: _mknode(0.2, 0.4, 3),       # 0.36
                       LEFT_10: _mknode(0.1, 0.9, 7),    # 0.74
                       RIGHT_10: _mknode(0.3, 0.5, 5)}   # 0.46
    expect("(a3) pure Hurwicz (c_visit=0), 3 arms", ag0._select_action(parent), LEFT_10)

    # (b) UCB bonus lifts a rarely-visited child
    parent = IntervalNode(); parent.visit_count = 51
    parent.children = {NOOP: _mknode(1.0, 1.0, 50),      # hurwicz 1.0, bonus ~0.84
                       LEFT_10: _mknode(0.0, 0.0, 1)}    # hurwicz 0.0, bonus ~5.95
    expect("(b1) rare arm (1 visit) beats good arm (50 visits) via UCB",
           ag._select_action(parent), LEFT_10)
    parent.visit_count = 5000
    parent.children = {NOOP: _mknode(1.0, 1.0, 4999),
                       LEFT_10: _mknode(0.0, 0.0, 1)}
    expect("(b2) still true at 5000 visits (bonus grows with log N)",
           ag._select_action(parent), LEFT_10)
    nz = IntervalNode(); nz.visit_count = 10
    nz.children = {NOOP: _mknode(0.0, 0.0, 9), LEFT_10: _mknode(0.0, 0.0, 0)}
    expect("(b3) unvisited child returned immediately",
           ag._select_action(nz), LEFT_10)

    # (c) root pick via the REAL _plan_single decision code, c_act = 0
    expect("(c1) c_act=0 picks max LOWER bound (ignores width)",
           _root_pick({NOOP: _mknode(0.4, 10.0, 5),
                       LEFT_10: _mknode(0.5, 0.5, 5)}, c_act=0.0), LEFT_10)
    expect("(c2) exact tie on score -> NOOP wins (NOOP-first, strict >)",
           _root_pick({LEFT_10: _mknode(0.5, 0.5, 5),
                       NOOP: _mknode(0.5, 0.5, 5),
                       RIGHT_10: _mknode(0.2, 0.9, 5)}, c_act=0.0), NOOP)
    expect("(c3) three-way exact tie -> NOOP",
           _root_pick({RIGHT_10: _mknode(1.0, 1.0, 5),
                       LEFT_10: _mknode(1.0, 1.0, 5),
                       NOOP: _mknode(1.0, 1.0, 5)}, c_act=0.0), NOOP)
    expect("(c4) c_act=0.1 trades a little width in",
           _root_pick({NOOP: _mknode(0.50, 0.50, 5),      # 0.500
                       LEFT_10: _mknode(0.45, 1.45, 5)},  # 0.55
                      c_act=0.1), LEFT_10)
    expect("(c5) zero-visit child ignored at root",
           _root_pick({NOOP: _mknode(0.1, 0.1, 5),
                       LEFT_10: _mknode(99.0, 99.0, 0)}, c_act=0.0), NOOP)

    status = "FAIL" if failures else "PASS"
    emit(4, "selection sanity (unit tests)", status,
         failures + evidence)
    return status


# ============================================================================
# CHECK 5: determinized clone fidelity
# ============================================================================

def check5():
    section(5, "DETERMINIZED CLONE FIDELITY (deepcopy as generative model)")
    evidence, failures = [], []

    env, obs, info = env_at(42, 30)

    # (a) identical actions -> identical trajectories, 10 steps
    clone = copy.deepcopy(env)
    obs_l, obs_c = obs, obs
    same = True
    for t in range(10):
        acts_l = {cs: NOOP for cs in obs_l}
        acts_c = {cs: NOOP for cs in obs_c}
        obs_l, _, _, _, info_l = env.step(acts_l)
        obs_c, _, _, _, info_c = clone.step(acts_c)
        if snapshot(info_l) != snapshot(info_c):
            same = False
            failures.append(f"(a) live vs clone diverged at step {t+1} "
                            f"under identical actions")
            break
    line = f"(a) 10 identical-action steps: live == clone -> {same}"
    print("  " + line); evidence.append(line)

    # (b) different actions -> divergence (fresh state, two clones)
    env2, obs2, info2 = env_at(42, 30)
    target = next((cs for cs in sorted(obs2)
                   if _status(info2, cs) == "IN_SECTOR"), None)
    ca, cb = copy.deepcopy(env2), copy.deepcopy(env2)
    oa = ob = obs2
    diverged_at = None
    for t in range(5):
        oa, _, _, _, ia = ca.step({cs: NOOP for cs in oa})
        ob, _, _, _, ib = cb.step(
            {cs: (LEFT_10 if cs == target else NOOP) for cs in ob})
        if snapshot(ia) != snapshot(ib):
            diverged_at = t + 1
            break
    line = (f"(b) clone A all-NOOP vs clone B LEFT_10 on {target} "
            f"(IN_SECTOR): diverged at step {diverged_at}")
    print("  " + line); evidence.append(line)
    if diverged_at is None:
        failures.append("(b) clones did NOT diverge under different actions "
                        "in 5 steps -- actions may not reach the simulator")

    # (c) generate_action leaves the live env untouched (state AND RNG future)
    env3, obs3, info3 = env_at(42, 30)
    ref = copy.deepcopy(env3)                    # pristine reference
    snap_before = snapshot(info3)
    agent = IntervalMCTSAgent(n_simulations=6, horizon=5,
                              alert_radius_nm=1e9, alert_fl=1e9,
                              max_planned=2, rng=random.Random(42))
    action = agent.generate_action(env3, obs3, info3)
    snap_after = snapshot(info3)                 # same live objects, re-read
    ok_state = snap_before == snap_after
    line = f"(c1) aircraft states identical before/after generate_action: {ok_state}"
    print("  " + line); evidence.append(line)
    print(f"       (planned {agent.last_n_planned} aircraft, action "
          f"{ {c: act_name(a) for c, a in action.items()} })")
    if not ok_state:
        failures.append("(c1) generate_action mutated live aircraft state")
    # RNG / hidden-state probe: live env and pristine reference must agree
    # on the FUTURE (spawns included) when stepped identically post-planning.
    ol, orf = obs3, obs3
    ok_future = True
    for t in range(5):
        ol, _, _, _, il = env3.step({cs: NOOP for cs in ol})
        orf, _, _, _, ir = ref.step({cs: NOOP for cs in orf})
        if snapshot(il) != snapshot(ir) or set(ol) != set(orf):
            ok_future = False
            failures.append(f"(c2) live env future diverged from pristine "
                            f"reference at step {t+1} after generate_action "
                            f"-- hidden state (e.g. RNG) was perturbed")
            break
    line = f"(c2) 5-step future identical to pristine reference: {ok_future}"
    print("  " + line); evidence.append(line)

    status = "FAIL" if failures else "PASS"
    for f in failures:
        print("  FAIL: " + f)
    emit(5, "determinized clone fidelity", status, failures + evidence)
    return status


# ============================================================================
# CHECK 6: penalty reachability on the real seed-10043 pre-violation state
# ============================================================================

def check6():
    section(6, "PENALTY REACHABILITY (seed 10043, LoS AIR-00/AIR-02 @ step 77)")
    print("Fast-forward a fresh seed-10043 env all-NOOP to step 65 (the\n"
          "all-NOOP trajectory reproduces the logged LoS at step 77), then run\n"
          "an instrumented search for AIR-00 with horizon 15 / 24 sims.  The\n"
          "violation lies ~12 steps ahead: simulations MUST see it.")

    evidence, failures = [], []
    t0 = time.perf_counter()
    env, obs, info = env_at(10043, 65)
    print(f"  fast-forward to step 65: {time.perf_counter()-t0:.1f} s, "
          f"{len(obs)} aircraft")
    for cs in ("AIR-00", "AIR-02"):
        line = f"  {cs}: status={_status(info, cs)!r}, in obs={cs in obs}"
        print(line); evidence.append(line.strip())
    if "AIR-00" not in obs:
        emit(6, "penalty reachability", "FAIL",
             ["AIR-00 not in obs at step 65 -- scenario assumption broken"])
        return "FAIL"

    agent = InstrumentedAgent(n_simulations=24, horizon=15,
                              rng=random.Random(10043))
    t0 = time.perf_counter()
    root = run_instrumented_search(agent, env, obs, "AIR-00")
    print(f"  24 sims x horizon 15: {time.perf_counter()-t0:.1f} s")

    viol_sims = []
    for i, tr in enumerate(agent.sim_traces):
        steps = tr["steps"]
        v = [s for s in steps if s["violated"]]
        g_rec = tr["updates"][0][1]
        if v:
            d = v[0]["depth"]
            pen_term = (agent.gamma ** d) * agent.violation_penalty
            g_hand = hand_g(steps, agent.gamma, agent.violation_penalty)
            g_no_pen = hand_g(steps, agent.gamma, 0.0)
            n_tree = len(tr["updates"]) - 1
            first_a = steps[0]["action"] if n_tree >= 1 else None
            viol_sims.append((i, d, first_a, g_rec, g_hand, g_no_pen, pen_term))

    print(f"\n  {len(viol_sims)}/24 simulations recorded violated=True:")
    for i, d, a0, g_rec, g_hand, g_np, pen in viol_sims:
        line = (f"sim {i:2d}: violation at depth {d:2d}, root action "
                f"{act_name(a0):<5s} g={g_rec:+9.4f} (hand {g_hand:+9.4f}), "
                f"shaping-only part {g_np:+8.4f}, penalty term "
                f"{pen:+9.4f} = -50*gamma^{d}")
        print("    " + line); evidence.append(line)
        if abs(g_rec - g_hand) > TOL:
            failures.append(f"sim {i}: recorded g differs from hand g by "
                            f"{abs(g_rec - g_hand):.2e}")
        if abs((g_rec - g_np) - pen) > 1e-6:
            failures.append(f"sim {i}: g does NOT contain the -50*gamma^t "
                            f"term (g - shaping = {g_rec - g_np:.4f}, "
                            f"expected {pen:.4f})")

    print("\n  Root envelope per action:")
    min_lower = math.inf
    for a in ALL_ACTIONS:
        ch = root.children.get(a)
        if ch is None or ch.visit_count == 0:
            line = f"root child {act_name(a):<5s}: (unvisited)"
        else:
            min_lower = min(min_lower, ch.lower)
            line = (f"root child {act_name(a):<5s}: N={ch.visit_count:<3d} "
                    f"[{ch.lower:+9.4f}, {ch.upper:+9.4f}] mean {ch.mean:+9.4f} "
                    f"width {ch.width:8.4f}")
        print("    " + line); evidence.append(line)

    if not viol_sims:
        failures.append("NO simulation ever recorded violated=True from the "
                        "real pre-violation state: the planner is BLIND to "
                        "the upcoming LoS")
    else:
        worst_viol_g = min(v[3] for v in viol_sims)
        if min_lower > worst_viol_g + TOL:
            failures.append(f"root envelope does not reflect the penalty: "
                            f"min child lower {min_lower:.4f} > worst violated "
                            f"g {worst_viol_g:.4f}")
        else:
            line = (f"root envelope reflects penalty: min child lower "
                    f"{min_lower:+.4f} <= worst violated-sim g "
                    f"{worst_viol_g:+.4f}")
            print("  " + line); evidence.insert(0, line)

    status = "FAIL" if failures else "PASS"
    for f in failures:
        print("  FAIL: " + f)
    emit(6, "penalty reachability", status, failures + evidence[:30])
    return status


# ============================================================================
# CHECK 7 (--heavy): seed-10043 autopsy.  IMPLEMENTED, NOT RUN BY DEFAULT.
# ============================================================================

class AutopsyAgent(IntervalMCTSAgent):
    """Full agent that additionally captures, per planned aircraft, the root
    children envelopes and chosen action, plus the gating decisions."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.decision_log = []   # per generate_action call
        self._cur_call = None

    def generate_action(self, env, observation_dict, info_dict):
        self._cur_call = {"planned": [], "gated_noop": [], "risk": {}}
        # replicate risk computation for the log (read-only)
        states = M.aircraft_states(info_dict, list(observation_dict.keys()))
        for cs in observation_dict:
            if cs not in states:
                continue
            la, lo, fa = states[cs]
            dmin = math.inf
            for other, (lb, lob, fb) in states.items():
                if other == cs or abs(fa - fb) >= self.alert_fl:
                    continue
                dmin = min(dmin, M.haversine_nm(la, lo, lb, lob))
            self._cur_call["risk"][cs] = dmin
        out = super().generate_action(env, observation_dict, info_dict)
        self._cur_call["gated_noop"] = [
            cs for cs in observation_dict
            if cs not in [p["cs"] for p in self._cur_call["planned"]]]
        self._cur_call["actions"] = dict(out)
        self.decision_log.append(self._cur_call)
        self._cur_call = None
        return out

    def _plan_single(self, live_env, root_callsigns, planned_cs, frozen):
        with capture_created_nodes() as created:
            action, widths = super()._plan_single(
                live_env, root_callsigns, planned_cs, frozen)
        root = created[0]
        rec = {"cs": planned_cs, "chosen": action,
               "root": {a: (c.visit_count, c.lower, c.mean, c.upper)
                        for a, c in root.children.items()}}
        if self._cur_call is not None:
            self._cur_call["planned"].append(rec)
        return action, widths


def check7(beam_width=32, autopsy_from=55, autopsy_to=80, ff_to=65):
    section(7, "SEED-10043 AUTOPSY (--heavy)")
    evidence, failures = [], []

    # ---- Part A: instrumented decision replay, steps 1..autopsy_to
    print(f"Part A: instrumented re-run of the decision sequence "
          f"(logging from step {autopsy_from}).")
    print("NOTE: this replays the same policy/seed as the logged run; exact\n"
          "per-step agreement with the original log is not guaranteed (the\n"
          "original run's rng stream interleaved differently), but gating and\n"
          "envelope behaviour is representative.\n")
    random.seed(10043); np.random.seed(10043)
    env = make_env()
    obs, info = env.reset(seed=10043)
    agent = AutopsyAgent(n_simulations=24, horizon=15,
                         rng=random.Random(10043))
    first_viol = None
    for step in range(1, autopsy_to + 1):
        t0 = time.perf_counter()
        action = agent.generate_action(env, obs, info)
        dt = time.perf_counter() - t0
        dl = agent.decision_log[-1]
        obs, rew, term, trunc, info = env.step(action)
        viol = find_violations(info)
        if step >= autopsy_from or viol:
            planned_css = [p["cs"] for p in dl["planned"]]
            risky = {cs: round(r, 1) for cs, r in sorted(
                dl["risk"].items(), key=lambda kv: kv[1])[:4] if r < 1e8}
            print(f"  step {step:3d} | n_ac {len(obs):2d} | {dt*1000:8.0f} ms | "
                  f"planned {planned_css} | closest {risky}")
            for cs in ("AIR-00", "AIR-02"):
                if cs in dl["risk"] and cs not in planned_css:
                    r = dl["risk"][cs]
                    why = ("outside alert radius" if r > agent.alert_radius_nm
                           else "budget cap / BEFORE_ENTRY")
                    print(f"           {cs} NOT searched (risk {r:.1f} nm: {why})")
                    evidence.append(f"step {step}: {cs} gated out "
                                    f"(risk {r:.1f} nm, {why})")
            for p in dl["planned"]:
                envs = {act_name(a): f"N={n} [{lo:+.2f},{up:+.2f}]"
                        for a, (n, lo, mn, up) in sorted(p["root"].items())}
                print(f"           {p['cs']} -> {act_name(p['chosen'])}  {envs}")
                evidence.append(f"step {step}: {p['cs']} -> "
                                f"{act_name(p['chosen'])} {envs}")
        if viol and first_viol is None:
            first_viol = (step, viol)
            line = f"Part A first violation: step {step}: {'; '.join(viol)}"
            print("  " + line); evidence.append(line)
            break
        if not obs:
            break
    if first_viol is None:
        line = f"Part A: no violation up to step {autopsy_to} in this replay"
        print("  " + line); evidence.append(line)

    # ---- Part B: was the LoS avoidable at all?  Beam search over open-loop
    # AIR-00 action sequences in the determinized sim from step ff_to.
    print(f"\nPart B: open-loop avoidability from all-NOOP step {ff_to} "
          f"(beam width {beam_width}, others NOOP).")
    base_env, base_obs, base_info = env_at(10043, ff_to)
    k = autopsy_to - ff_to
    # beam entries: (neg worst_margin, env, obs, actions, alive)
    beam = [{"env": base_env, "obs": base_obs, "seq": [],
             "worst_sep": math.inf, "alive": True}]
    best_clean = None
    for level in range(k):
        cand = []
        for st in beam:
            if not st["alive"]:
                continue
            for a in ALL_ACTIONS:
                e = copy.deepcopy(st["env"])
                acts = {cs: (a if cs == "AIR-00" else NOOP)
                        for cs in st["obs"]}
                o2, _, _, _, i2 = e.step(acts)
                violated = violation_involving(i2, "AIR-00")
                # AIR-00's min lateral sep to vertically-close IN_SECTOR traffic
                sep = math.inf
                css = [c for c in M._tracker_callsigns(i2)
                       if _status(i2, c) == "IN_SECTOR"]
                stt = M.aircraft_states(i2, css)
                if "AIR-00" in stt:
                    la, lo, fa = stt["AIR-00"]
                    for c, (lb, lob, fb) in stt.items():
                        if c != "AIR-00" and abs(fa - fb) < M.LOS_VERTICAL_FL:
                            sep = min(sep, M.haversine_nm(la, lo, lb, lob))
                cand.append({"env": e, "obs": o2, "seq": st["seq"] + [a],
                             "worst_sep": min(st["worst_sep"], sep),
                             "alive": not violated})
        alive = [c for c in cand if c["alive"]]
        if not alive:
            line = (f"Part B: ALL {len(cand)} beam branches violated by "
                    f"step {ff_to + level + 1} -- LoS UNAVOIDABLE (within "
                    f"beam) for AIR-00 alone")
            print("  " + line); evidence.append(line)
            beam = []
            break
        alive.sort(key=lambda c: -c["worst_sep"])
        beam = alive[:beam_width]
        print(f"    level {level+1:2d} (env step {ff_to+level+1}): "
              f"{len(alive)}/{len(cand)} alive, best worst-sep "
              f"{beam[0]['worst_sep']:.2f} nm")
    if beam:
        best_clean = beam[0]
        line = (f"Part B: AVOIDABLE -- open-loop sequence "
                f"{[act_name(a) for a in best_clean['seq']]} keeps AIR-00 "
                f"violation-free to step {autopsy_to} "
                f"(worst sep {best_clean['worst_sep']:.2f} nm)")
        print("  " + line); evidence.append(line)

    # Verdict logic: informational autopsy -- WARN if the planner failed on
    # an avoidable conflict, PASS if unavoidable or replay stayed clean.
    if best_clean is not None and first_viol is not None:
        status = "WARN"
        evidence.insert(0, "the LoS was avoidable in the determinized sim, "
                           "yet the planner's decision sequence still hit a "
                           "violation -- see per-step envelopes above")
    else:
        status = "PASS"
    emit(7, "seed-10043 autopsy (heavy)", status, evidence[:40])
    return status


# ============================================================================
# CHECK 8: discount analysis (pure math)
# ============================================================================

def check8():
    section(8, "DISCOUNT ANALYSIS: is -50*gamma^t neutered by discounting?")
    sig = inspect.signature(IntervalMCTSAgent.__init__)
    gamma = sig.parameters["gamma"].default
    pen = abs(sig.parameters["violation_penalty"].default)
    H = sig.parameters["horizon"].default
    print(f"File defaults: gamma={gamma}, |penalty|={pen}, horizon={H}")
    print("Comparison target: the maximum accumulated shaping-return\n"
          "DIFFERENCE between two action sequences over the horizon,\n"
          "  D(H, rb) = sum_(u=0..H-1) gamma^u * (2*rb),   rb = per-step |r| bound\n"
          "(two sequences can differ by at most 2*rb per step).\n")

    rbounds = (0.5, 1.0, 2.0)
    D = {rb: sum((gamma ** u) * 2 * rb for u in range(H)) for rb in rbounds}
    hdr = "   t   |pen|*g^t " + "".join(f"   D(H,rb={rb:>3})" for rb in rbounds)
    print(hdr)
    evidence = [f"gamma={gamma} |penalty|={pen} horizon={H}",
                "D(H,rb) = max accumulated shaping difference over horizon: "
                + ", ".join(f"rb={rb}: {D[rb]:.2f}" for rb in rbounds)]
    print("  ('<' : D < discounted penalty, i.e. penalty dominates; "
          "'>' : shaping can outweigh)")
    for t in range(0, 16):
        p = pen * gamma ** t
        marks = "".join(
            f"   {D[rb]:8.2f}{'<' if p > D[rb] else '>'}" for rb in rbounds)
        note = "  <- penalty dominates ALL bounds" \
            if all(p > D[rb] for rb in rbounds) else ""
        print(f"  {t:3d}   {p:9.3f}{marks}{note}")
    crossings = {}
    for rb in rbounds:
        t = 0
        while pen * gamma ** t >= D[rb] and t < 10000:
            t += 1
        crossings[rb] = t
        line = (f"penalty falls below D(H, rb={rb}) at depth t={t} "
                f"({'BEYOND' if t > H - 1 else 'WITHIN'} the horizon of {H})")
        print("  " + line); evidence.append(line)

    within = [rb for rb in rbounds if crossings[rb] <= H - 1]
    if within:
        status = "WARN"
        evidence.insert(0, f"for per-step |r| bound(s) {within} the discounted "
                           f"penalty can be outweighed by shaping WITHIN the "
                           f"horizon -- 'penalty neutered by discounting' is "
                           f"plausible at those reward scales")
    else:
        status = "PASS"
        evidence.insert(0, f"at gamma={gamma} the discounted -{pen} exceeds the "
                           f"worst-case accumulated shaping difference at EVERY "
                           f"depth t<=H-1={H-1} (even for |r|<=2: "
                           f"{pen*gamma**(H-1):.1f} vs {D[2.0]:.1f}); the "
                           f"seed-10043 failure is NOT explained by discounting "
                           f"alone")
    print("\n  Note: this bounds the WORST case.  If the violation is "
          "unavoidable under\n  every root action, the penalty enters every "
          "child's lower bound equally\n  and carries no decision gradient -- "
          "that is a reachability question\n  (checks 6/7), not a discounting "
          "one.")
    emit(8, "discount analysis", status, evidence)
    return status


# ============================================================================
# MAIN
# ============================================================================

LIGHT_CHECKS = {1: check1, 2: check2, 3: check3, 4: check4,
                5: check5, 6: check6, 8: check8}
HEAVY_CHECKS = {7: check7}


def main():
    ap = argparse.ArgumentParser(
        description="Diagnostic suite for bluebird_interval_mcts.py")
    ap.add_argument("--check", type=int, nargs="*", default=None,
                    help="run only these check numbers (default: all light)")
    ap.add_argument("--heavy", action="store_true",
                    help="include heavy checks (7: seed-10043 autopsy; LONG "
                         "-- do not run while other jobs need the machine)")
    args = ap.parse_args()

    # Be a good citizen towards the concurrent DQN training run.
    try:
        cur = os.nice(0)
        if cur < 15:
            os.nice(15 - cur)
    except OSError:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"diagnose_{stamp}.log"
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        print("Interval MCTS diagnostic suite")
        print(f"target : {M.__file__}")
        print(f"python : {sys.executable}")
        print(f"date   : {datetime.now().isoformat(timespec='seconds')}")
        print(f"niceness: {os.nice(0)}")
        print("\nLINEAGE NOTE (semantic drift vs IP-ML/nim_interval_mcts.py, "
              "reported as a\nfinding, not a bug): the ancestor backs up "
              "RUNNING MEANS of network-predicted\ninterval bounds "
              "(backed_lower/upper += (q - backed)/n, with Nim-specific\n"
              "perspective negation-and-swap); this file backs up a MIN/MAX "
              "ENVELOPE of\nscalar Monte-Carlo returns.  Same Hurwicz selection "
              "rule, different interval\nsemantics: credal-mean interval (Nim) "
              "vs support envelope (Bluebird).")

        if args.check:
            todo = args.check
        else:
            todo = sorted(LIGHT_CHECKS) + (sorted(HEAVY_CHECKS) if args.heavy else [])
        t0 = time.perf_counter()
        for n in todo:
            fn = LIGHT_CHECKS.get(n) or HEAVY_CHECKS.get(n)
            if fn is None:
                print(f"\n[{n}] unknown check number, skipping")
                continue
            if n in HEAVY_CHECKS and not args.heavy and args.check is None:
                continue
            tc = time.perf_counter()
            try:
                fn()
            except Exception as e:  # a crash is itself a diagnostic result
                import traceback
                traceback.print_exc()
                emit(n, f"check {n} crashed", "FAIL", [f"{type(e).__name__}: {e}"])
            print(f"  (check {n} wall time: {time.perf_counter()-tc:.1f} s)")

        # ---- summary
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        order = {"FAIL": 0, "WARN": 1, "PASS": 2}
        for r in RESULTS:
            print(f"  [{r['check']}] {r['status']:<4s} {r['name']}")
        worst = min((r["status"] for r in RESULTS),
                    key=lambda s: order[s], default="PASS")
        n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
        n_warn = sum(r["status"] == "WARN" for r in RESULTS)
        print(f"\n  overall: {worst}  ({n_fail} FAIL, {n_warn} WARN, "
              f"{len(RESULTS)-n_fail-n_warn} PASS; "
              f"total {time.perf_counter()-t0:.1f} s)")
        if worst == "PASS":
            print("  verdict: every independently verified stage of the "
                  "interval MCTS behaved\n  exactly as specified.")
        json_path = OUT_DIR / f"diagnose_{stamp}.json"
        with open(json_path, "w") as f:
            json.dump({"timestamp": stamp, "target": M.__file__,
                       "results": RESULTS, "overall": worst}, f, indent=2)
        print(f"\n  log : {log_path}\n  json: {json_path}")
    finally:
        sys.stdout = tee._stdout
        tee.close()


if __name__ == "__main__":
    main()
