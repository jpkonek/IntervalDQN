"""
Interval MCTS v2 for BluebirdATC — the base-policy inversion.

Implements the LOCKED spec in MCTS_V2_SPEC.md (5 July 2026). Evidence base:
MCTS_ARCHITECTURE.md + checkpoints/bluebird/diagnostics/design_campaign/
SYNTHESIS.md (D1-D5, seed 10043). v1 (bluebird_interval_mcts.py) is kept
unchanged as the ablation baseline; shared helpers are imported from it.

The inversion
-------------
The trained interval DQN (adaptive width-conditioned c) flies EVERY aircraft
by default, live. The planner is an override: each live step it selects up to
``max_planned`` aircraft (attention below) and replaces their base-policy
action with an MCTS-searched one. The same DQN controller is deployed live
and rehearsed in imagination (Q3 consistency): during a rehearsal every
non-planned aircraft is flown by the base policy, computed from the cloned
env's own observation dicts at the simulated states.

Attention (spec section 2)
--------------------------
Candidates per step, slots ordered (a) then (b) then (c), budget max_planned:
  (a) geometric floor — the v1 TTC/CPA gate (cpa < 8 nm within 300 s, or
      proximity < 15 nm), riskiest first. Physics, not model opinion.
  (b) interval relevance — ONE batched net pass over all aircraft;
      relevance = width of the adaptive-c-chosen action's interval,
      + AMBIGUITY_BONUS if the argmax at c=0 differs from the argmax at c=1
      (decision ambiguity). Top-k (k = max_planned) by relevance join.
      Ranking use only (ordinal bar — no cardinal threshold).
  (c) random audit — with prob p_audit (default 0.10/step) one aircraft from
      outside (a) union (b) is added. Interpretation choice (documented): if
      the budget is already full, the audit aircraft PREEMPTS the last
      relevance slot (never a floor slot — physics outranks insurance);
      a literal (a)>(b)>(c) ordering would squeeze the audit out on every
      busy step and the insurance channel would never fire.

Rehearsal (no rollouts)
-----------------------
A simulation walks the tree by annealed-c selection, expands ONE child,
evaluates the new state with the DQN's interval, and backs the sample up the
path. Tree nodes cache their (deepcopied) simulator state, so a simulation
costs exactly one deepcopy + one env.step + one net pass; the Bluebird sim is
deterministic within an episode (deepcopy clones RNG state) and the imagined
base policy is greedy, so each (node, action) edge has a unique child state
and caching loses nothing.

Node values, priors and backup (spec section 5; JK's rule)
----------------------------------------------------------
Node statistics live on the RETURN-FROM-ROOT scale (siblings share their
root-to-parent reward prefix, so sibling comparisons equal Q comparisons —
v1 convention). Each simulation backs up the interval

    [g_pre(leaf) + gamma^d * Q_l(s_leaf, a*),
     g_pre(leaf) + gamma^d * Q_u(s_leaf, a*)]

into every node on the path as RUNNING MEANS of lower and upper (the
nim-ancestor rule; --backup envelope retains v1's [min, max] for ablation).
a* is the adaptive-c-chosen action at the leaf state — the same rule the
base policy acts by, so the leaf value is "the value if the deployed
controller takes over here". Terminal simulations back up the realized
return, ungrounded by any bootstrap.

CHILD-PRIOR CHOICE (documented, per spec "a child is born with the network's
interval for its state as prior"): the prior is implemented as the child's
FIRST BACKUP SAMPLE — at expansion the child's stats are initialised with
g_pre + gamma^d * [Q_l, Q_u](child state), i.e. visit_count 1 with the net
interval as the virtual first sample, on the return-from-root scale. Before
a child exists at all, its action is scored during selection by the parent
state's net interval: score = g_pre(parent) + gamma^depth * Q_u(s, a) —
exactly the annealed-c rule at n = 0 (c(0) = c0 = 1.0), so expansion order
falls out of the same selection formula instead of a separate mechanism.

Selection (spec section 6)
--------------------------
    score(a) = l + c(n) * (u - l),  c(n) = c_act + (c0 - c_act)/sqrt(1 + n)
with c0 = 1.0 and n the child's visit count. No UCB term. The ROOT commits
at c_act (default 0.1) with the NOOP-preferring exact-tie break.

Shared fate (spec section 4)
----------------------------
Any violation observed during a rehearsal (find_violations nonempty) ends the
rehearsal at that step: penalty -50 if the planned aircraft is in the
involved set, -10 otherwise, applied at the step's discount.

Environment / base checkpoint coupling
--------------------------------------
The base policy ACTS live, so the env is built with the checkpoint's exact
training configuration (encoder, k, action_config incl. route_parallel,
centreline_coeff) via bluebird_interval_dqn.make_env — action indices then
match the net's heads by construction. This drops v1's macro turns
(deliberate deviation: a 5-action macro env would misalign the 4-action
net's outputs; the route_parallel clearance is the recovery tool instead).

Usage
-----
    python bluebird_interval_mcts_v2.py --smoke
    python bluebird_interval_mcts_v2.py --run --seed 10043 --sims 32
    python bluebird_interval_mcts_v2.py --base-only --seed 10043
    python bluebird_interval_mcts_v2.py --baseline --seed 10043
    python bluebird_interval_mcts_v2.py --aggregate out.json in1.json in2.json
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import bluebird_interval_dqn as bid
from bluebird_interval_mcts import (
    NOOP,
    SEC_PER_STEP,
    IntervalNode,
    _exit_potential,
    _status,
    aircraft_kinematics,
    aircraft_states,
    cpa_nm_s,
    derive_action_space,
    find_violations,
    violation_involving,
)

sys.stdout.reconfigure(line_buffering=True)

# Spec section 4 constants (checkpoint's own training penalty is also 50).
OWN_VIOLATION_PENALTY = -50.0
OTHER_VIOLATION_PENALTY = -10.0
# Relevance bonus when the c=0 and c=1 argmax actions differ (decision
# ambiguity). Only the ORDER of relevance matters (ordinal use), so any
# constant larger than every achievable width works.
AMBIGUITY_BONUS = 1e6

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", "bluebird", "best_run6.pt")


# ============================================================================
# TREE NODE
# ============================================================================

class V2Node:
    """One tree node. Caches the cloned simulator at its state so expansion
    from it costs one deepcopy + one step (see module docstring).

    stats   : IntervalNode (v1) — running-mean (or envelope) bounds over
              backed-up return-from-root samples.
    env     : cloned env AT this node's state (None once fully expanded, or
              terminal; the ROOT holds a reference to the LIVE env, which is
              never stepped — expansion deepcopies first).
    obs     : env obs dict at this state (drives the imagined base policy and
              the planned aircraft's net priors).
    g_pre   : discounted shaped return from the root to this state
              (incl. any shared-fate penalty at the arrival step).
    q_l/q_u : the net's per-action interval bounds for the planned aircraft
              at this state (None if terminal).
    """

    __slots__ = ("stats", "env", "obs", "depth", "g_pre", "terminal",
                 "q_l", "q_u", "children", "base_actions")

    def __init__(self, mode, env, obs, depth, g_pre, terminal, q_l, q_u):
        self.stats = IntervalNode(mode=mode)
        self.env = env
        self.obs = obs
        self.depth = depth
        self.g_pre = g_pre
        self.terminal = terminal
        self.q_l = q_l
        self.q_u = q_u
        self.children: Dict[int, "V2Node"] = {}
        self.base_actions: Optional[Dict[str, int]] = None


# ============================================================================
# AGENT
# ============================================================================

class IntervalMCTSv2Agent:
    """Base-policy-inverted interval MCTS planner (MCTS_V2_SPEC.md)."""

    def __init__(
        self,
        ckpt_path: str = DEFAULT_CKPT,
        n_simulations: int = 32,
        c_act: float = 0.1,
        c0: float = 1.0,
        backup: str = "mean",           # "mean" (spec) | "envelope" (ablation)
        p_audit: float = 0.10,
        max_planned: int = 4,
        alert_radius_nm: float = 15.0,  # geometric floor: proximity trigger
        alert_fl: float = 20.0,         # vertical gate for "neighbour"
        cpa_dist_nm: float = 8.0,       # geometric floor: CPA distance
        cpa_time_s: float = 300.0,      # geometric floor: CPA lookahead
        rng: Optional[random.Random] = None,
        verbose: bool = True,
    ):
        assert backup in ("mean", "envelope")
        self.dqn, self.ckpt = bid.load_agent(ckpt_path, device="cpu")
        self.ckpt_path = ckpt_path
        # Q3 consistency: rehearsal shaping == the checkpoint's training
        # shaping, so path rewards and leaf values live on one scale.
        self.gamma = float(self.ckpt.get("gamma", self.dqn.gamma))
        self.exit_bonus = float(self.ckpt.get("exit_bonus", 10.0))
        self.progress_coeff = float(self.ckpt.get("progress_coeff", 0.05))

        self.n_simulations = n_simulations
        self.c_act = c_act
        self.c0 = c0
        self.backup = backup
        self.p_audit = p_audit
        self.max_planned = max_planned
        self.alert_radius_nm = alert_radius_nm
        self.alert_fl = alert_fl
        self.cpa_dist_nm = cpa_dist_nm
        self.cpa_time_s = cpa_time_s
        self.rng = rng or random.Random(0)
        self.verbose = verbose

        self.n_actions = int(self.dqn.n_actions)
        self.all_actions: Tuple[int, ...] = tuple(range(self.n_actions))
        self.action_names: Dict[int, str] = {a: str(a) for a in self.all_actions}
        self._actions_bound = False

        # Diagnostics, refreshed by generate_action
        self.last_attended: List[dict] = []   # searched aircraft this step
        self.last_base_choices: Dict[str, Tuple[float, float]] = {}
        self.last_root_widths: List[float] = []

        if verbose:
            print(f"  [v2] base+leaf checkpoint {ckpt_path} "
                  f"(ep {self.ckpt.get('episode', '?')}, "
                  f"{self.n_actions} actions, gamma {self.gamma}, "
                  f"encoder {self.ckpt.get('encoder_cls')}, "
                  f"k {self.ckpt.get('k')}, "
                  f"route_parallel {self.ckpt.get('route_parallel')})")

    # ------------------------------------------------------------------
    # Env construction / binding
    # ------------------------------------------------------------------

    def make_episode_env(self, scenario_duration: int = 600):
        """Env with the CHECKPOINT's training configuration (the base policy
        acts live, so obs encoding and action indices must match the net)."""
        return bid.make_env(
            scenario_duration=scenario_duration,
            k_nearest=int(self.ckpt.get("k", 3)),
            route_parallel=bool(self.ckpt.get("route_parallel", False)),
            centreline_coeff=float(self.ckpt.get("centreline_coeff", 0.2)),
            encoder_cls=self.ckpt.get("encoder_cls", "extra_minimal"),
        )

    def bind_action_space(self, env) -> None:
        if self._actions_bound or env is None:
            return
        actions, names = derive_action_space(env)
        if len(actions) != self.n_actions:
            raise RuntimeError(
                f"env exposes {len(actions)} actions but the checkpoint "
                f"trained {self.n_actions} — env/checkpoint mismatch")
        self.all_actions, self.action_names = actions, names
        self._actions_bound = True
        if self.verbose:
            print("  [actions] " + ", ".join(
                f"{a}={names[a]}" for a in actions))

    def reset_episode(self) -> None:
        self.last_attended = []
        self.last_base_choices = {}
        self.last_root_widths = []

    # ------------------------------------------------------------------
    # Base-policy plumbing
    # ------------------------------------------------------------------

    def calibrate(self, seed: int, duration: int = 600) -> float:
        """Calibrate adaptive-c w_mid on one base-policy probe episode
        (mirrors bluebird_interval_dqn.evaluate's calibration block)."""
        env = self.make_episode_env(duration)
        obs, _ = env.reset(seed=seed)
        probe = []
        for _ in range(int(getattr(env, "maxstep", duration // SEC_PER_STEP))):
            if not obs:
                break
            probe.append(dict(obs))
            a, _ = self.dqn.generate_action(obs, c=0.0, force_epsilon=0.0)
            obs, _, _, _, _ = env.step(a)
        env.close()
        w_mid = self.dqn.calibrate_w_mid(probe)
        if self.verbose:
            print(f"  [adaptive-c] w_mid calibrated to {w_mid:.3f} "
                  f"({len(probe)} probe steps, seed {seed})")
        return w_mid

    def _ensure_calibrated(self, obs_dict: dict) -> None:
        """Lazy fallback: calibrate on the first live obs batch if no probe
        episode ran (single-step median — coarser than the probe)."""
        if not hasattr(self.dqn, "adaptive_w_mid") and obs_dict:
            w_mid = self.dqn.calibrate_w_mid([obs_dict])
            if self.verbose:
                print(f"  [adaptive-c] LAZY w_mid calibration on first obs "
                      f"batch: {w_mid:.3f}")

    def _net(self, vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batched net pass: [n, state_dim] -> (lower, upper) [n, A]."""
        with torch.no_grad():
            s = torch.from_numpy(np.asarray(vecs, dtype=np.float32))
            lo, up = self.dqn.q_net(s)
        return lo.numpy(), up.numpy()

    def _adaptive_c_scalar(self, mean_width: float) -> float:
        return float(bid.IntervalDQNAgent.adaptive_c(
            torch.tensor([mean_width]),
            w_mid=getattr(self.dqn, "adaptive_w_mid", 4.0)))

    def _leaf(self, obs_vec) -> Tuple[np.ndarray, np.ndarray, int]:
        """Net interval at one state + the adaptive-c-chosen action there —
        the SAME rule the base policy acts by (spec: deploy and rehearse the
        same controller). Returns (q_l[A], q_u[A], a_star)."""
        lo, up = self._net(np.asarray(obs_vec)[None])
        lo, up = lo[0], up[0]
        c = self._adaptive_c_scalar(float((up - lo).mean()))
        a_star = int(np.argmax(lo + c * (up - lo)))
        return lo, up, a_star

    # ------------------------------------------------------------------
    # Attention (spec section 2)
    # ------------------------------------------------------------------

    def _geometric_floor(self, callsigns, states, kin):
        """v1 TTC/CPA gate. Returns (engaged set, risk dict for ordering)."""
        risk: Dict[str, Tuple[float, float]] = {}
        engaged = set()
        for cs in callsigns:
            if cs not in states:
                continue
            la, lo, fa = states[cs]
            dmin, ttc_min = math.inf, math.inf
            trigger = False
            for other, (lb, lob, fb) in states.items():
                if other == cs or abs(fa - fb) >= self.alert_fl:
                    continue
                d = bid.haversine_nm(la, lo, lb, lob)
                dmin = min(dmin, d)
                if d < self.alert_radius_nm:
                    trigger = True
                t_cpa, d_cpa = cpa_nm_s(kin.get(cs), kin.get(other))
                if (t_cpa is not None and d_cpa < self.cpa_dist_nm
                        and t_cpa < self.cpa_time_s):
                    trigger = True
                    ttc_min = min(ttc_min, t_cpa)
            risk[cs] = (ttc_min, dmin)
            if trigger:
                engaged.add(cs)
        return engaged, risk

    def _attention(self, obs: dict, info: dict) -> List[Tuple[str, str]]:
        """Slot selection: [(callsign, source)] with source in
        {"floor", "relevance", "audit"}, at most max_planned entries."""
        # BEFORE_ENTRY aircraft: env drops their actions — never candidates.
        # They still count as neighbours for the geometric floor (v1 parity).
        all_cs = list(obs.keys())
        eligible = [cs for cs in all_cs
                    if _status(info, cs) != "BEFORE_ENTRY"]
        if not eligible:
            return []
        states = aircraft_states(info, all_cs)
        kin = aircraft_kinematics(info, all_cs)

        engaged, risk = self._geometric_floor(all_cs, states, kin)
        rk = lambda cs: risk.get(cs, (math.inf, math.inf))  # noqa: E731
        floor = sorted((cs for cs in eligible if cs in engaged), key=rk)

        # Interval relevance: one batched pass over ALL eligible aircraft.
        vecs = np.stack([obs[cs] for cs in eligible])
        lo, up = self._net(vecs)
        widths = up - lo
        mean_w = widths.mean(axis=1)
        c_ad = bid.IntervalDQNAgent.adaptive_c(
            torch.from_numpy(mean_w.astype(np.float32)),
            w_mid=getattr(self.dqn, "adaptive_w_mid", 4.0)).numpy()[:, None]
        chosen = np.argmax(lo + c_ad * (up - lo), axis=1)
        amb = np.argmax(lo, axis=1) != np.argmax(up, axis=1)
        relevance = {}
        self.last_base_choices = {}
        for i, cs in enumerate(eligible):
            a = int(chosen[i])
            relevance[cs] = float(widths[i, a]) + (
                AMBIGUITY_BONUS if bool(amb[i]) else 0.0)
            self.last_base_choices[cs] = (float(lo[i, a]), float(up[i, a]))

        rel_sorted = sorted((cs for cs in eligible if cs not in engaged),
                            key=lambda cs: -relevance[cs])
        top_rel = rel_sorted[: self.max_planned]

        slots = ([(cs, "floor") for cs in floor]
                 + [(cs, "relevance") for cs in top_rel])[: self.max_planned]

        # Random audit: one aircraft from outside (a) union (b). If the
        # budget is full it preempts the LAST relevance slot; floor slots
        # (physics) are never preempted — see module docstring.
        if self.rng.random() < self.p_audit:
            candidates = {cs for cs, _ in slots} | set(floor) | set(top_rel)
            pool = [cs for cs in eligible if cs not in candidates]
            if pool:
                audit_cs = self.rng.choice(pool)
                if len(slots) < self.max_planned:
                    slots.append((audit_cs, "audit"))
                else:
                    rel_idx = [i for i, (_, src) in enumerate(slots)
                               if src == "relevance"]
                    if rel_idx:
                        slots[rel_idx[-1]] = (audit_cs, "audit")
        return slots

    # ------------------------------------------------------------------
    # Live interface
    # ------------------------------------------------------------------

    def generate_action(self, env, observation_dict: dict,
                        info_dict: dict) -> Dict[str, int]:
        """Base-policy actions for ALL aircraft (one batched pass), then
        MCTS overrides for the attended ones (sequential, floor-riskiest
        first). The live env is never stepped — only deepcopied."""
        self.bind_action_space(env)
        self._ensure_calibrated(observation_dict)
        self.last_attended = []
        self.last_root_widths = []
        if not observation_dict:
            self.last_base_choices = {}
            return {}

        base_actions, _ = self.dqn.generate_action(
            observation_dict, adaptive=True, force_epsilon=0.0)
        slots = self._attention(observation_dict, info_dict)

        decided = dict(base_actions)
        overrides: Dict[str, int] = {}
        for cs, source in slots:
            action, interval, widths = self._search(
                env, observation_dict, cs, overrides)
            overrides[cs] = action
            decided[cs] = action
            self.last_root_widths.extend(widths)
            self.last_attended.append({
                "cs": cs, "source": source,
                "base_action": int(base_actions[cs]),
                "action": int(action),
                "lo": interval[0], "up": interval[1],
            })
            # attended aircraft report the searched interval, not the raw net
            self.last_base_choices.pop(cs, None)
        return decided

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _annealed_c(self, n: int) -> float:
        return self.c_act + (self.c0 - self.c_act) / math.sqrt(1.0 + n)

    def _search(self, live_env, obs: dict, planned_cs: str,
                overrides: Dict[str, int]):
        """Interval MCTS for one aircraft; returns
        (action, (lower, upper) of the chosen root child, root widths)."""
        q_l, q_u = self._net(np.asarray(obs[planned_cs])[None])
        root = V2Node(self.backup, live_env, dict(obs), 0, 0.0, False,
                      q_l[0], q_u[0])
        for _ in range(self.n_simulations):
            self._simulate(root, planned_cs, overrides)

        # Root commit: pessimistic Hurwicz at c_act, NOOP-preferring
        # exact-tie break (NOOP scored first, strict > afterwards).
        best_action, best_score = NOOP, -math.inf
        for a in sorted(root.children, key=lambda x: x != NOOP):
            st = root.children[a].stats
            if st.visit_count == 0:
                continue
            score = st.lower + self.c_act * (st.upper - st.lower)
            if score > best_score:
                best_score, best_action = score, a
        widths = [ch.stats.width for ch in root.children.values()
                  if ch.stats.visit_count > 0]
        ch = root.children.get(best_action)
        interval = ((ch.stats.lower, ch.stats.upper)
                    if ch is not None and ch.stats.visit_count > 0
                    else (-math.inf, math.inf))
        return best_action, interval, widths

    def _simulate(self, root: V2Node, planned_cs: str,
                  overrides: Dict[str, int]) -> None:
        """One simulation: annealed-c walk, expand one child, evaluate it
        with the net interval, back the sample up the path (running means)."""
        node, path = root, [root.stats]
        while True:
            if node.terminal:
                g_l = g_u = node.g_pre     # grounded terminal revisit
                break
            disc = self.gamma ** node.depth
            best_a, best_s, best_child = NOOP, -math.inf, None
            for a in sorted(self.all_actions, key=lambda x: x != NOOP):
                child = node.children.get(a)
                if child is not None:
                    st = child.stats
                    c = self._annealed_c(st.visit_count)
                    s = st.lower + c * (st.upper - st.lower)
                else:
                    # unvisited: net prior at THIS state, annealed c at
                    # n = 0 -> c0 = 1.0 -> upper bound, rescaled to the
                    # return-from-root scale
                    s = node.g_pre + disc * float(node.q_u[a])
                if s > best_s:
                    best_s, best_a, best_child = s, a, child
            if best_child is None:
                child, (g_l, g_u) = self._expand(
                    node, best_a, planned_cs, overrides)
                path.append(child.stats)
                break
            node = best_child
            path.append(node.stats)
        for st in path:
            st.update(g_l, g_u)

    def _expand(self, node: V2Node, action: int, planned_cs: str,
                overrides: Dict[str, int]):
        """Create the child of `node` under `action`: one deepcopy + one env
        step with (tree action for the planned aircraft, committed overrides
        at depth 0, imagined base policy for everyone else), shared-fate
        check, then the net-interval evaluation of the new state."""
        if node.base_actions is None:
            node.base_actions, _ = self.dqn.generate_action(
                node.obs, adaptive=True, force_epsilon=0.0)
        acts = {}
        for cs in node.obs:
            if cs == planned_cs:
                acts[cs] = action
            elif node.depth == 0 and cs in overrides:
                acts[cs] = overrides[cs]
            else:
                acts[cs] = node.base_actions[cs]

        env2 = copy.deepcopy(node.env)
        phi = (_exit_potential(env2, planned_cs)
               if self.progress_coeff else None)
        obs2, rew, term, trunc, info2 = env2.step(acts)

        r = float(rew.get(planned_cs, 0.0))
        truncated = bool(trunc.get(planned_cs, False))
        done = (planned_cs not in obs2
                or bool(term.get(planned_cs, False)) or truncated)

        # Shared fate (spec 4): ANY violation ends the rehearsal here.
        violations = find_violations(info2)
        pen = 0.0
        if violations:
            pen = (OWN_VIOLATION_PENALTY
                   if violation_involving(info2, planned_cs)
                   else OTHER_VIOLATION_PENALTY)
        # Outcome anchoring, exactly as in DQN training / v1 planning.
        if (done and not truncated and not violations
                and _status(info2, planned_cs) != "OUT_SECTOR"):
            r += self.exit_bonus
        if (self.progress_coeff and not done and not violations
                and phi is not None):
            phi2 = _exit_potential(env2, planned_cs)
            if phi2 is not None:
                r += self.progress_coeff * (self.gamma * phi2 - phi)

        disc = self.gamma ** node.depth
        g_pre = node.g_pre + disc * (r + pen)
        terminal = done or bool(violations)

        if terminal:
            child = V2Node(self.backup, None, None, node.depth + 1, g_pre,
                           True, None, None)
            g_l = g_u = g_pre                     # grounded, no bootstrap
        else:
            q_l, q_u, a_star = self._leaf(obs2[planned_cs])
            child = V2Node(self.backup, env2, obs2, node.depth + 1, g_pre,
                           False, q_l, q_u)
            d2 = self.gamma ** child.depth
            g_l = g_pre + d2 * float(q_l[a_star])
            g_u = g_pre + d2 * float(q_u[a_star])

        node.children[action] = child
        if len(node.children) == self.n_actions:
            node.env = None    # fully expanded: env snapshot no longer needed
        return child, (g_l, g_u)


# ============================================================================
# EPISODE RUNNER + REALIZED H-STEP COVERAGE INSTRUMENT
# ============================================================================

def run_episode(
    agent: IntervalMCTSv2Agent,
    seed: int = 42,
    scenario_duration: int = 600,
    mode: str = "planner",             # "planner" | "base" | "noop"
    verbose: bool = True,
    cov_h: int = 15,
) -> dict:
    """One full episode. All three modes use the SAME env construction (the
    checkpoint's), so scenarios are identical across the comparison table.

    Realized H-step coverage instrument (ported from v1, adapted):
    every decision (planner root interval for attended aircraft; the net's
    chosen-action interval for base-flown aircraft) is scored against the
    realized discounted shaped return from its step, under v2's OWN
    accounting (progress shaping, exit bonus, shared-fate penalties).
    Mean-bound comparability caveat: predictions include a discounted leaf/
    Q bootstrap tail, so ONLY decisions whose aircraft reached a terminal
    (violation, shared-fate stop, or clean exit) within the H-step window
    are scored ("terminal-window-only"); the rest are censored.
    """
    assert mode in ("planner", "base", "noop")
    env = agent.make_episode_env(scenario_duration)
    obs, info = env.reset(seed=seed)
    agent.reset_episode()
    if mode != "noop" and not hasattr(agent.dqn, "adaptive_w_mid"):
        agent.calibrate(seed - 1, duration=scenario_duration)

    gamma = agent.gamma
    ep_return = 0.0
    latencies: List[float] = []
    widths: List[float] = []
    decisions: List[dict] = []         # coverage instrument records
    attention_log: List[dict] = []
    step_log: List[dict] = []          # per-step accounting for resolution
    first_violation_step = None
    first_violation_desc = None
    step = 0

    while True:
        t0 = time.perf_counter()
        if mode == "noop":
            action = {cs: NOOP for cs in obs}
        elif mode == "base":
            agent.bind_action_space(env)
            action, _ = agent.dqn.generate_action(
                obs, adaptive=True, force_epsilon=0.0)
            if obs:
                lo, up = agent._net(np.stack(list(obs.values())))
                for i, cs in enumerate(obs):
                    a = action[cs]
                    decisions.append({
                        "cs": cs, "step": step, "source": "base",
                        "lo": float(lo[i, a]), "up": float(up[i, a]),
                        "n_ac": len(obs)})
        else:
            action = agent.generate_action(env, obs, info)
            for rec in agent.last_attended:
                d = dict(rec, step=step, n_ac=len(obs), source="planner",
                         slot=rec["source"])
                decisions.append(d)
                attention_log.append(d)
                if verbose:
                    nm = agent.action_names
                    print(f"  [attend] step {step:3d} {rec['cs']} "
                          f"({rec['source']}) -> "
                          f"{nm.get(rec['action'], rec['action'])} "
                          f"(base {nm.get(rec['base_action'], rec['base_action'])}) "
                          f"[{rec['lo']:.2f}, {rec['up']:.2f}]")
            for cs, (lo, up) in agent.last_base_choices.items():
                decisions.append({"cs": cs, "step": step, "source": "base",
                                  "lo": lo, "up": up, "n_ac": len(obs)})
            widths.extend(agent.last_root_widths)
        latencies.append(time.perf_counter() - t0)

        prev_cs = list(obs.keys())
        phi_prev = ({cs: _exit_potential(env, cs) for cs in prev_cs}
                    if agent.progress_coeff else {})
        obs, rew, term, trunc, info = env.step(action)
        step += 1
        ep_return += float(sum(rew.values()))

        violations = find_violations(info)
        involved = ({cs for cs in prev_cs if violation_involving(info, cs)}
                    if violations else set())
        step_log.append({
            "rew": {cs: float(rew.get(cs, 0.0)) for cs in prev_cs},
            "viol": bool(violations),
            "involved": involved,
            "done": {cs for cs in prev_cs
                     if cs not in obs or bool(term.get(cs, False))
                     or bool(trunc.get(cs, False))},
            "trunc": {cs for cs in prev_cs if bool(trunc.get(cs, False))},
            "out": {cs for cs in prev_cs if _status(info, cs) == "OUT_SECTOR"},
            "phi": phi_prev,
            "phi_next": {cs: _exit_potential(env, cs) for cs in prev_cs
                         if cs in obs},
        })

        if first_violation_step is None and violations:
            first_violation_step = step
            first_violation_desc = "; ".join(violations)

        if verbose and step % 10 == 0:
            w = f"{np.mean(widths):.3f}" if widths else "n/a"
            print(f"  step {step:3d} | n_ac {len(obs):2d} | "
                  f"decision {latencies[-1]*1000:8.1f} ms | "
                  f"ep_return {ep_return:9.2f} | mean root width {w}")

        if not obs or all(term.values()) or all(trunc.values()):
            break

    env.close()

    # ---- Resolve the coverage instrument (terminal-window-only) ----
    n_hit = n_scored = n_censored = 0
    for d in decisions:
        cs, k = d["cs"], d["step"]
        G, resolved = 0.0, False
        for off in range(cov_h):
            j = k + off
            if j >= len(step_log):
                break
            rec = step_log[j]
            if cs not in rec["rew"]:
                break                     # aircraft gone before this step
            r = rec["rew"][cs]
            if rec["viol"]:
                r += (OWN_VIOLATION_PENALTY if cs in rec["involved"]
                      else OTHER_VIOLATION_PENALTY)
                G += gamma ** off * r
                resolved = True
                break
            if cs in rec["done"]:
                if cs not in rec["trunc"] and cs not in rec["out"]:
                    r += agent.exit_bonus
                G += gamma ** off * r
                resolved = cs not in rec["trunc"]   # truncation = censored
                break
            phi, phin = rec["phi"].get(cs), rec["phi_next"].get(cs)
            if agent.progress_coeff and phi is not None and phin is not None:
                r += agent.progress_coeff * (gamma * phin - phi)
            G += gamma ** off * r
        d["resolved"] = resolved
        if resolved:
            d["G"] = G
            d["hit"] = bool(d["lo"] - 1e-9 <= G <= d["up"] + 1e-9)
            n_scored += 1
            n_hit += int(d["hit"])
        else:
            d["G"], d["hit"] = None, None
            n_censored += 1

    attn_by_source: Dict[str, int] = {}
    for rec in attention_log:
        attn_by_source[rec["slot"]] = attn_by_source.get(rec["slot"], 0) + 1

    ttv = (float(first_violation_step * SEC_PER_STEP)
           if first_violation_step is not None else float(scenario_duration))
    return {
        "mode": mode,
        "seed": seed,
        "steps": step,
        "episode_return": ep_return,
        "first_violation_step": first_violation_step,
        "first_violation_desc": first_violation_desc,
        "time_to_violation": ttv,
        "violated": first_violation_step is not None,
        "mean_latency_s": float(np.mean(latencies)),
        "max_latency_s": float(np.max(latencies)),
        "mean_root_width": float(np.mean(widths)) if widths else 0.0,
        "n_searches": len(attention_log),
        "attention_by_source": attn_by_source,
        "realized_h_coverage": (n_hit / n_scored) if n_scored else None,
        "scored_decisions": n_scored,
        "censored_decisions": n_censored,
        "attention_log": attention_log,
        "decisions": decisions,
    }


# ============================================================================
# BUCKETED CALIBRATION (spec validation battery, feeds the audit-rate call)
# ============================================================================

def _density_bucket(n_ac: int) -> str:
    return "<=8" if n_ac <= 8 else ("9-16" if n_ac <= 16 else ">16")


def _bucket_stats(records: List[dict]) -> dict:
    n = len(records)
    hits = sum(1 for r in records if r["hit"])
    return {"n": n, "coverage": (hits / n) if n else None,
            "mean_width": (float(np.mean([r["up"] - r["lo"]
                                          for r in records])) if n else None)}


def build_calibration(decisions: List[dict]) -> dict:
    """Coverage of realized H-step returns bucketed by traffic density
    (aircraft count at decision time) and predicted-width tercile, per
    decision source (planner root intervals vs raw net intervals)."""
    out = {"n_decisions": len(decisions)}
    for source in ("planner", "base"):
        scored = [d for d in decisions
                  if d["source"] == source and d.get("resolved")]
        entry = {"overall": _bucket_stats(scored), "by_density": {},
                 "by_width_tercile": {}}
        for b in ("<=8", "9-16", ">16"):
            entry["by_density"][b] = _bucket_stats(
                [d for d in scored if _density_bucket(d["n_ac"]) == b])
        if len(scored) >= 3:
            ws = np.array([d["up"] - d["lo"] for d in scored])
            t1, t2 = np.percentile(ws, [100 / 3, 200 / 3])
            terciles = {"narrow": [], "mid": [], "wide": []}
            for d, w in zip(scored, ws):
                key = "narrow" if w <= t1 else ("mid" if w <= t2 else "wide")
                terciles[key].append(d)
            entry["tercile_edges"] = [float(t1), float(t2)]
            for key, rs in terciles.items():
                entry["by_width_tercile"][key] = _bucket_stats(rs)
        out[source] = entry
    return out


# ============================================================================
# REPORTING / MAIN
# ============================================================================

def print_stats(name: str, stats: dict) -> None:
    print(f"\n--- {name} ---")
    if stats["first_violation_step"] is None:
        print(f"  time-to-first-violation : clean episode "
              f"(censored at {stats['time_to_violation']:.0f} s)")
    else:
        print(f"  time-to-first-violation : {stats['time_to_violation']:.0f} s"
              f" (step {stats['first_violation_step']})")
        print(f"  first violation         : {stats['first_violation_desc']}")
    print(f"  episode return          : {stats['episode_return']:.2f}")
    print(f"  steps                   : {stats['steps']}")
    print(f"  decision latency        : mean {stats['mean_latency_s']*1000:.1f} ms, "
          f"max {stats['max_latency_s']*1000:.1f} ms")
    print(f"  searches run            : {stats['n_searches']} "
          f"{stats['attention_by_source']}")
    print(f"  mean root interval width: {stats['mean_root_width']:.4f}")
    if stats.get("realized_h_coverage") is not None:
        print(f"  realized H-step coverage: {stats['realized_h_coverage']:.3f}"
              f" ({stats['scored_decisions']} scored, "
              f"{stats['censored_decisions']} censored; "
              f"terminal-window-only)")


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def save_results(path: str, stats: dict, agent_cfg: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_sanitize({"config": agent_cfg, **stats}), f)
    print(f"  results written: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interval MCTS v2 (base-policy inversion) on BluebirdATC")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="120 s episode, tiny budget, + NOOP comparison")
    mode.add_argument("--run", action="store_true", help="full planner episode")
    mode.add_argument("--base-only", action="store_true",
                      help="pure DQN base policy, no planner")
    mode.add_argument("--baseline", action="store_true", help="all-NOOP episode")
    mode.add_argument("--aggregate", nargs="+", metavar="RESULTS_JSON",
                      help="aggregate decision records from result JSONs "
                           "into the bucketed-calibration JSON (--calib-out)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                        help="base-policy + leaf interval-DQN checkpoint")
    parser.add_argument("--sims", type=int, default=32,
                        help="simulations per search (cheap: 1 deepcopy + "
                             "1 env step + 1 net pass each)")
    parser.add_argument("--backup", choices=["mean", "envelope"],
                        default="mean",
                        help="running-mean bounds (spec) or min/max envelope "
                             "(ablation)")
    parser.add_argument("--c-act", type=float, default=0.1,
                        help="Hurwicz c at the root commit; annealing floor")
    parser.add_argument("--c0", type=float, default=1.0,
                        help="annealed-c start value at n=0")
    parser.add_argument("--p-audit", type=float, default=0.10,
                        help="per-step probability of the random audit slot")
    parser.add_argument("--max-planned", type=int, default=4,
                        help="attention budget: searches per live step")
    parser.add_argument("--alert-radius", type=float, default=15.0)
    parser.add_argument("--alert-fl", type=float, default=20.0)
    parser.add_argument("--cpa-dist", type=float, default=8.0)
    parser.add_argument("--cpa-time", type=float, default=300.0)
    parser.add_argument("--cov-h", type=int, default=15,
                        help="H for the realized-coverage instrument window")
    parser.add_argument("--json-out", type=str, default=None,
                        help="write full stats+decisions JSON here")
    parser.add_argument("--calib-out", type=str,
                        default="mcts_v2_results/calibration.json",
                        help="bucketed-calibration output for --aggregate")
    args = parser.parse_args()

    if args.aggregate:
        decisions = []
        for p in args.aggregate:
            with open(p) as f:
                decisions.extend(json.load(f)["decisions"])
        calib = build_calibration(decisions)
        os.makedirs(os.path.dirname(os.path.abspath(args.calib_out)),
                    exist_ok=True)
        with open(args.calib_out, "w") as f:
            json.dump(_json_sanitize(calib), f, indent=2)
        print(json.dumps(_json_sanitize(calib), indent=2))
        print(f"calibration written: {args.calib_out}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    smoke = args.smoke
    duration = 120 if smoke else args.duration
    agent = IntervalMCTSv2Agent(
        ckpt_path=args.ckpt,
        n_simulations=8 if smoke else args.sims,
        c_act=args.c_act, c0=args.c0, backup=args.backup,
        p_audit=args.p_audit,
        max_planned=2 if smoke else args.max_planned,
        alert_radius_nm=args.alert_radius, alert_fl=args.alert_fl,
        cpa_dist_nm=args.cpa_dist, cpa_time_s=args.cpa_time,
        rng=random.Random(args.seed),
    )
    agent_cfg = {
        "ckpt": args.ckpt, "sims": agent.n_simulations,
        "backup": args.backup, "c_act": args.c_act, "c0": args.c0,
        "p_audit": args.p_audit, "max_planned": agent.max_planned,
        "cov_h": args.cov_h, "duration": duration, "seed": args.seed,
    }

    if args.baseline:
        run_mode, label = "noop", "NOOP baseline"
    elif args.base_only:
        run_mode, label = "base", "Base policy only (adaptive-c DQN)"
    else:
        run_mode, label = "planner", ("Interval MCTS v2 (smoke)" if smoke
                                      else "Interval MCTS v2")

    print(f"{label} | seed {args.seed} | duration {duration}s | "
          f"sims {agent.n_simulations} | backup {args.backup} | "
          f"c_act {args.c_act} | p_audit {args.p_audit} | "
          f"max_planned {agent.max_planned}")
    t0 = time.time()
    stats = run_episode(agent, seed=args.seed, scenario_duration=duration,
                        mode=run_mode, cov_h=args.cov_h)
    print(f"\nEpisode wall time: {time.time()-t0:.1f} s")
    print_stats(label, stats)

    if args.json_out:
        save_results(args.json_out, stats, agent_cfg)

    if smoke:
        print("\nRunning all-NOOP baseline for comparison...")
        base = run_episode(agent, seed=args.seed, scenario_duration=duration,
                           mode="noop", verbose=False, cov_h=args.cov_h)
        print_stats("NOOP baseline (smoke)", base)
        ok = stats["steps"] > 0 and stats["n_searches"] > 0
        print(f"\nVerdict: v2 TTV {stats['time_to_violation']:.0f} s vs "
              f"NOOP {base['time_to_violation']:.0f} s")
        print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")


if __name__ == "__main__":
    main()
