"""
Interval MCTS agent for the BluebirdATC air-traffic-control gymnasium environment.

Model-free interval Monte-Carlo Tree Search using the REAL simulator (via
``copy.deepcopy`` of the live environment) as the generative model.  This is
the model-free sibling of the learned-model interval MCTS in
IP-ML/nim_interval_mcts.py: no dynamics network, no value heads -- node value
intervals come from the dispersion of Monte-Carlo rollout returns.

Interval-statistics scheme (the ONE scheme used everywhere)
-----------------------------------------------------------
Each tree node maintains a *running envelope* over the total discounted
returns of every simulation that passed through it:

    lower  = running MIN of backed-up returns
    upper  = running MAX of backed-up returns
    mean   = running average (reported, used for tie-breaks)

All simulations start from the same root state, so siblings share the exact
root-to-parent reward prefix; comparing siblings by their total-return
envelopes is therefore equivalent to comparing their Q-value envelopes.  The
[min, max] envelope is the natural model-free credal set: it brackets the set
of returns achievable under the *subtree policy uncertainty* (which action
sequence the planner commits to below this node) plus any environment
stochasticity.  Upper = best return found so far (optimism drives search),
lower = worst return found so far (pessimism drives safe execution).

Action selection is Hurwicz over these intervals, exactly as in the lineage:

    score(a) = lower(a) + c * (upper(a) - lower(a))            (+ UCB bonus)

* Internal (in-tree) selection uses ``c_search`` (default 0.8, optimistic:
  chase the best return seen in a subtree) plus a visit-count exploration
  bonus ``c_visit * sqrt(log N_parent / N_child)``.  The bonus is essential:
  the Bluebird simulator is (near-)deterministic within an episode, so with a
  fixed rollout policy interval widths can collapse to zero on straight-line
  subtrees -- expected and fine, but width alone would then starve arms.
* At the ROOT, the actually-executed action is chosen with a separate,
  pessimistic ``c_act`` (default 0.1) and NO exploration bonus: execute the
  action whose near-worst-case return is best (robustness -- the competition
  metric is time-to-first-violation).

Determinized planning
---------------------
``copy.deepcopy(env)`` clones the simulator INCLUDING its RNG state, so
future aircraft spawns are IDENTICAL across planning branches.  This is
intentional (determinized / hindsight planning on a single determinization);
it removes spawn noise from the interval statistics so that interval width
reflects genuine subtree outcome dispersion.

Joint-action factorisation
--------------------------
N aircraft x 3 actions is exponential jointly.  At each live env step we plan
SEQUENTIALLY per aircraft, ordered by risk (min pairwise lateral separation
among vertically-proximate neighbours, ascending -- riskiest first):

  * aircraft with no neighbour within ``alert_radius_nm`` (default 15 nm)
    AND ``alert_fl`` flight levels get NOOP without any search (saves budget);
  * at most ``max_planned`` aircraft are searched per decision (budget cap;
    the remainder get NOOP);
  * while planning aircraft i, already-decided aircraft are frozen to their
    chosen action on the FIRST simulated step (a heading change is a one-shot
    instruction) and NOOP afterwards; undecided aircraft fly NOOP throughout.

Planning reward
---------------
Return for planning = per-aircraft reward for the planned aircraft from the
env's reward dict, discounted by ``gamma``, PLUS a large terminal penalty
(default -50) at the first step where the planned aircraft is involved in a
loss of separation (lateral < 5 nm AND vertical < 10 FL, haversine on the
true simulator state) or leaves the sector (pos_status == OUT_SECTOR).  The
penalty is applied at discount gamma^t, so EARLY violations are penalised
more sharply than late ones -- matching the time-to-first-violation metric.
(Violations between two *other* aircraft do not enter aircraft i's return:
they are identical across i's actions and carry no gradient for i.)

Usage
-----
    python bluebird_interval_mcts.py --smoke
    python bluebird_interval_mcts.py --run --seed 42 --sims 24 --horizon 15 --c-act 0.1
    python bluebird_interval_mcts.py --baseline --seed 42
"""

import argparse
import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from bluebird_gymnasium.envs import InfiniteEnv
from bluebird_gymnasium.envs.infinite import ScenarioName

# Action encoding (verified for this action_config)
NOOP, LEFT_10, RIGHT_10 = 0, 1, 2
ALL_ACTIONS = (NOOP, LEFT_10, RIGHT_10)

# ICAO separation minima used for violation detection
LOS_LATERAL_NM = 5.0
LOS_VERTICAL_FL = 10.0

EARTH_RADIUS_NM = 3440.065


# ============================================================================
# ENVIRONMENT CONSTRUCTION
# ============================================================================

def make_env(scenario_duration: int = 600,
             centreline_coeff: float = 0.2,
             encoder_cls: str = "extra_minimal",
             k_nearest: int = 2) -> InfiniteEnv:
    """Build the target BluebirdATC environment (verified configuration).

    centreline_coeff 0.2 matches bluebird_interval_dqn's outcome-anchored
    reward regime (run 4 onward), keeping episode returns comparable
    across the two agents. Pass 1.0 to reproduce the run-2/3 regime.

    encoder_cls / k_nearest only shape the observation vectors; planning
    itself is model-free on the simulator state. They matter in hybrid
    mode, where the leaf net reads horizon observations: they must match
    the DQN checkpoint's training encoder (main() reads them from the
    checkpoint metadata — DESIGN_REVISIONS item 5).
    """
    cfg = InfiniteEnv.get_default_env_config()
    cfg.state_repr_config = {"encoder_cls": encoder_cls,
                             "k_nearest_aircraft": k_nearest}
    cfg.action_config = {"simple_heading_left": [10], "simple_heading_right": [10]}
    cfg.reward_config = {
        "fns": [
            "position_status_const",
            "lateral_centreline_distance_shaped",
            "safety_simple_avoidance_exp",
        ],
        "coeffs": [1.0, centreline_coeff, 1.2],
    }
    cfg.scenario_config["scenario_name"] = ScenarioName.sector_xplus
    cfg.view_config["type"] = "decentralized"
    cfg.view_config["decentralized_params"] = {}
    cfg.scenario_duration = scenario_duration
    return InfiniteEnv(config=cfg)


# ============================================================================
# GEOMETRY / VIOLATION DETECTION
# ============================================================================

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_NM * math.asin(min(1.0, math.sqrt(a)))


def aircraft_states(info: dict, callsigns) -> Dict[str, Tuple[float, float, float]]:
    """Extract (lat, lon, fl) for the given callsigns from the sim state."""
    sim = info["simulator_environment"]
    return {
        cs: (sim.aircraft[cs].lat, sim.aircraft[cs].lon, sim.aircraft[cs].fl)
        for cs in callsigns
        if cs in sim.aircraft
    }


def _status(info: dict, cs: str) -> str:
    entry = info.get(cs)
    return str(entry.get("pos_status", "")) if isinstance(entry, dict) else ""


def _exit_potential(env, cs):
    """Minus the along-track distance (nm) to the aircraft's exit, or None.
    Potential for progress shaping (see IntervalMCTSAgent.progress_coeff)."""
    try:
        d = env.get_tracked_aircraft_data(cs)
    except Exception:
        return None
    if d is None or d.track_dist_to_exit_cr is None:
        return None
    return -float(d.track_dist_to_exit_cr)


def _tracker_callsigns(info: dict) -> List[str]:
    """All aircraft the env is still tracking (per-callsign info entries).

    This deliberately includes OUT_SECTOR aircraft, which the obs dict
    filters out — deriving callsigns from obs would make sector excursions
    invisible to violation checks (the aircraft vanishes from obs on the
    very step it excurses).
    """
    return [cs for cs, d in info.items()
            if cs != "simulator_environment" and isinstance(d, dict)]


def find_violations(info: dict) -> List[str]:
    """Return human-readable violations among all tracked aircraft.

    Semantics match bluebird_interval_dqn.detect_violation exactly:
      - sector excursion: pos_status == OUT_SECTOR
      - loss of separation: lateral < 5 nm AND vertical < 10 FL, counted
        only when BOTH aircraft are IN_SECTOR (aircraft still BEFORE_ENTRY
        are another sector's problem)
    """
    out = []
    css = _tracker_callsigns(info)
    for cs in css:
        if _status(info, cs) == "OUT_SECTOR":
            out.append(f"excursion {cs}")
    in_sector = [cs for cs in css if _status(info, cs) == "IN_SECTOR"]
    states = aircraft_states(info, in_sector)
    ins = list(states)
    for i in range(len(ins)):
        for j in range(i + 1, len(ins)):
            a, b = ins[i], ins[j]
            la, lo, fa = states[a]
            lb, lob, fb = states[b]
            if abs(fa - fb) >= LOS_VERTICAL_FL:
                continue
            d = haversine_nm(la, lo, lb, lob)
            if d < LOS_LATERAL_NM:
                out.append(f"LoS {a}/{b} ({d:.2f} nm)")
    return out


def violation_involving(info: dict, planned_cs: str) -> bool:
    """True if the planned aircraft is out of sector or in LoS with anyone.

    The OUT_SECTOR check runs on tracker info BEFORE any position lookup:
    an excursed aircraft may already be gone from obs (and soon from the
    sim's aircraft dict), but its pos_status persists in info for several
    steps — this is what makes the planner's excursion penalty reachable.
    """
    if _status(info, planned_cs) == "OUT_SECTOR":
        return True
    if _status(info, planned_cs) != "IN_SECTOR":
        return False
    in_sector = [cs for cs in _tracker_callsigns(info)
                 if _status(info, cs) == "IN_SECTOR"]
    states = aircraft_states(info, in_sector)
    if planned_cs not in states:
        return False
    la, lo, fa = states[planned_cs]
    for cs, (lb, lob, fb) in states.items():
        if cs == planned_cs or abs(fa - fb) >= LOS_VERTICAL_FL:
            continue
        if haversine_nm(la, lo, lb, lob) < LOS_LATERAL_NM:
            return True
    return False


# ============================================================================
# INTERVAL MCTS (model-free, single-aircraft factored)
# ============================================================================

@dataclass
class IntervalNode:
    """Tree node holding interval statistics over backed-up returns.

    Two backup modes (set per-agent, passed at construction):

    - "envelope" (model-free default): each simulation backs up one scalar
      return g; the node interval is the running [min, max] envelope. The
      credal object is "range over explored continuations". Envelopes never
      tighten with visits and are visit-count biased (documented).
    - "mean" (hybrid, exactly analogous to interval DQN and to the Nim-era
      nim_interval_mcts backup): each simulation backs up an INTERVAL
      [g_l, g_u] (path rewards plus the interval-DQN leaf bootstrap at the
      horizon); the node keeps separate running MEANS of lower and upper.
      Intervals tighten with visits and are comparable across visit counts.
    """
    mode: str = "envelope"
    visit_count: int = 0
    lo_sum: float = 0.0
    up_sum: float = 0.0
    lo_min: float = math.inf
    up_max: float = -math.inf
    children: Dict[int, "IntervalNode"] = field(default_factory=dict)

    def update(self, g_l: float, g_u: float = None) -> None:
        if g_u is None:
            g_u = g_l
        self.visit_count += 1
        self.lo_sum += g_l
        self.up_sum += g_u
        self.lo_min = min(self.lo_min, g_l)
        self.up_max = max(self.up_max, g_u)

    @property
    def lower(self) -> float:
        if self.visit_count == 0:
            return math.inf
        return (self.lo_sum / self.visit_count if self.mode == "mean"
                else self.lo_min)

    @property
    def upper(self) -> float:
        if self.visit_count == 0:
            return -math.inf
        return (self.up_sum / self.visit_count if self.mode == "mean"
                else self.up_max)

    @property
    def mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return (self.lo_sum + self.up_sum) / (2 * self.visit_count)

    @property
    def width(self) -> float:
        return (self.upper - self.lower) if self.visit_count else 0.0


class IntervalMCTSAgent:
    """Sequential per-aircraft interval MCTS planner over deepcopies of the
    live simulator.

    Competition interface: ``generate_action(env, observation_dict, info_dict)
    -> dict[callsign, int]``.  The live env is NEVER stepped during planning;
    every simulation runs on a fresh ``copy.deepcopy(env)``.
    """

    def __init__(
        self,
        n_simulations: int = 24,
        horizon: int = 15,
        gamma: float = 0.97,
        c_search: float = 0.8,          # optimistic Hurwicz c inside the tree
        c_act: float = 0.1,             # pessimistic Hurwicz c at the root
        c_visit: float = 3.0,           # UCB visit-count bonus coefficient
        pw_k: float = 1.0,              # progressive widening: max_children =
        pw_alpha: float = 0.5,          #   max(1, k * N^alpha), capped at 3
        violation_penalty: float = -50.0,
        exit_bonus: float = 10.0,       # planning reward for a clean exit inside
                                        # the rollout; without it, exiting ends the
                                        # income stream early and the planner has a
                                        # mild ANTI-exit incentive (outcome anchoring,
                                        # mirroring bluebird_interval_dqn run 4+)
        progress_coeff: float = 0.05,   # potential-based progress shaping
                                        # (gamma*phi' - phi with phi = -dist to
                                        # exit), mirroring bluebird_interval_dqn
        alert_radius_nm: float = 15.0,  # neighbours farther than this: NOOP, no search
        alert_fl: float = 20.0,         # vertical gate for "neighbour" (level flights here)
        max_planned: int = 4,           # budget cap: riskiest-first searches per decision
        leaf_value_path: Optional[str] = None,  # interval-DQN checkpoint for
                                        # hybrid leaf bootstrap (switches the
                                        # backup to running-mean bounds)
        leaf_c: Optional[float] = None, # Hurwicz c for the leaf state's value
                                        # (default: c_search)
        rng: Optional[random.Random] = None,
    ):
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.gamma = gamma
        self.c_search = c_search
        self.c_act = c_act
        self.c_visit = c_visit
        self.pw_k = pw_k
        self.pw_alpha = pw_alpha
        self.violation_penalty = violation_penalty
        self.exit_bonus = exit_bonus
        self.progress_coeff = progress_coeff
        self.alert_radius_nm = alert_radius_nm
        self.alert_fl = alert_fl
        self.max_planned = max_planned
        self.rng = rng or random.Random(0)

        # Diagnostics (read after each generate_action call)
        self.last_root_widths: List[float] = []   # root child interval widths
        self.last_root_choices: Dict[str, Tuple[float, float]] = {}
        self.last_n_planned: int = 0

        # Hybrid mode: interval-DQN leaf evaluator + running-mean backup
        # (exactly analogous to the DQN update and the nim_interval_mcts
        # ancestor). Without it: model-free [min, max] envelope backup.
        self.leaf_net = None
        self.leaf_c = leaf_c if leaf_c is not None else c_search
        self.backup_mode = "envelope"
        if leaf_value_path is not None:
            import torch  # noqa: local import — torch optional in model-free mode
            import bluebird_interval_dqn as bid
            self._torch = torch
            dqn_agent, ckpt = bid.load_agent(leaf_value_path, device="cpu")
            self.leaf_net = dqn_agent.q_net.eval()
            self.backup_mode = "mean"
            ckpt_gamma = ckpt.get("gamma")
            if ckpt_gamma is not None and abs(ckpt_gamma - gamma) > 1e-9:
                print(f"  [leaf-value] WARNING: checkpoint gamma {ckpt_gamma} "
                      f"!= planner gamma {gamma}; bootstrap scale will be "
                      f"inconsistent")
            print(f"  [leaf-value] hybrid backup: mean bounds, leaf net from "
                  f"{leaf_value_path} (ep {ckpt.get('episode', '?')}, "
                  f"{ckpt.get('n_actions', '?')} actions), leaf_c={self.leaf_c}")

    def _leaf_interval(self, obs_vec):
        """Interval-DQN value of a horizon state: the net's [Q_l, Q_u] for
        its Hurwicz-greedy action at leaf_c. The net may know actions the
        planner does not execute (e.g. route_parallel); that is fine — this
        is 'value if the DQN policy took over from here'."""
        torch = self._torch
        with torch.no_grad():
            s = torch.from_numpy(
                np.asarray(obs_vec, dtype=np.float32)).unsqueeze(0)
            lo, up = self.leaf_net(s)
            a = int((lo + self.leaf_c * (up - lo)).argmax(dim=1))
            return float(lo[0, a]), float(up[0, a])

    # ------------------------------------------------------------------
    # Competition interface
    # ------------------------------------------------------------------

    def generate_action(self, env, observation_dict: dict, info_dict: dict) -> Dict[str, int]:
        """Plan a joint action for all controllable aircraft."""
        self.last_root_widths = []
        self.last_root_choices = {}
        self.last_n_planned = 0

        callsigns = list(observation_dict.keys())
        if not callsigns:
            return {}

        states = aircraft_states(info_dict, callsigns)

        # --- Risk assessment: min lateral separation to any vertically-close
        # neighbour.  Aircraft with no such neighbour inside the alert radius
        # get NOOP for free.
        risk: Dict[str, float] = {}
        for cs in callsigns:
            if cs not in states:
                continue
            la, lo, fa = states[cs]
            dmin = math.inf
            for other, (lb, lob, fb) in states.items():
                if other == cs or abs(fa - fb) >= self.alert_fl:
                    continue
                dmin = min(dmin, haversine_nm(la, lo, lb, lob))
            risk[cs] = dmin

        decided: Dict[str, int] = {
            cs: NOOP for cs in callsigns
            if risk.get(cs, math.inf) > self.alert_radius_nm
            # BEFORE_ENTRY aircraft: the env silently drops their actions and
            # zeroes their rewards — searching them burns budget for a no-op
            or _status(info_dict, cs) == "BEFORE_ENTRY"
        }
        to_plan = sorted(
            (cs for cs in callsigns if cs not in decided),
            key=lambda cs: risk.get(cs, math.inf),
        )
        # Budget cap: anything beyond max_planned flies NOOP this step.
        for cs in to_plan[self.max_planned:]:
            decided[cs] = NOOP
        to_plan = to_plan[: self.max_planned]

        # --- Sequential per-aircraft search, riskiest first.
        for cs in to_plan:
            action, widths, chosen_env = self._plan_single(
                env, callsigns, cs, dict(decided))
            decided[cs] = action
            self.last_root_widths.extend(widths)
            if chosen_env is not None:
                self.last_root_choices[cs] = chosen_env
            self.last_n_planned += 1

        return {cs: decided.get(cs, NOOP) for cs in callsigns}

    # ------------------------------------------------------------------
    # Single-aircraft interval MCTS
    # ------------------------------------------------------------------

    def _plan_single(
        self, live_env, root_callsigns: List[str], planned_cs: str,
        frozen: Dict[str, int],
    ) -> Tuple[int, List[float], Optional[Tuple[float, float]]]:
        """Run interval MCTS for one aircraft; return (action, root widths)."""
        root = IntervalNode(mode=self.backup_mode)

        for _ in range(self.n_simulations):
            sim_env = copy.deepcopy(live_env)   # generative model = real sim
            self._simulate(root, sim_env, list(root_callsigns), planned_cs, frozen)

        # Root decision: pessimistic Hurwicz (c_act), no exploration bonus.
        # NOOP is evaluated first with strict > comparison, so exact ties
        # (common under deterministic rollouts) resolve to "do nothing"
        # rather than an arbitrary heading command.
        best_action, best_score = NOOP, -math.inf
        for a, child in sorted(root.children.items(),
                               key=lambda kv: kv[0] != NOOP):
            if child.visit_count == 0:
                continue
            score = child.lower + self.c_act * (child.upper - child.lower)
            if score > best_score:
                best_score, best_action = score, a

        widths = [c.width for c in root.children.values() if c.visit_count > 0]
        chosen = root.children.get(best_action)
        chosen_env = ((chosen.lower, chosen.upper)
                      if chosen is not None and chosen.visit_count > 0
                      else None)
        return best_action, widths, chosen_env

    def _simulate(
        self, root: IntervalNode, env, callsigns: List[str],
        planned_cs: str, frozen: Dict[str, int],
    ) -> None:
        """One simulation: select/expand down the tree, NOOP rollout, backup.

        Model-free mode backs up a scalar return g (g_l == g_u). Hybrid mode
        (leaf_net set) adds the interval-DQN leaf bootstrap at the horizon,
        so each simulation backs up an interval [g_l, g_u] — the exact tree
        analogue of the DQN's interval Bellman target.
        """
        node = root
        path = [root]
        g_l = g_u = 0.0
        disc = 1.0
        depth, done = 0, False
        planned_obs = None

        # --- Selection / expansion (tree phase)
        while depth < self.horizon:
            # Progressive widening gate (3 actions, so mostly a formality --
            # kept for lineage consistency and to stagger early expansion).
            # The ROOT is exempt: with only 3 actions, all root children must
            # exist before selection or small budgets never try the 3rd action.
            if node is root:
                max_children = len(ALL_ACTIONS)
            else:
                max_children = max(1, int(self.pw_k * max(1, node.visit_count) ** self.pw_alpha))
            unexpanded = [a for a in ALL_ACTIONS if a not in node.children]
            if unexpanded and len(node.children) < min(len(ALL_ACTIONS), max_children):
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
                break  # newly expanded leaf -> switch to rollout

        # --- Rollout phase: planned aircraft holds heading (NOOP policy)
        if not done:
            while depth < self.horizon:
                r, done, violated, callsigns, planned_obs = self._step_sim(
                    env, callsigns, planned_cs, NOOP, depth, frozen)
                g_l += disc * r
                g_u += disc * r
                if violated:
                    g_l += disc * self.violation_penalty
                    g_u += disc * self.violation_penalty
                disc *= self.gamma
                depth += 1
                if done or violated:
                    break

        # --- Hybrid leaf bootstrap: horizon reached without a terminal ->
        # complete the return with the interval network's value at s_H,
        # exactly as the DQN's Bellman target completes r + gamma*[L', U'].
        # Terminal simulations stay grounded in the actual outcome.
        if self.leaf_net is not None and not done and planned_obs is not None:
            q_l, q_u = self._leaf_interval(planned_obs)
            g_l += disc * q_l
            g_u += disc * q_u

        # --- Backup into every node on the path
        for nd in path:
            nd.update(g_l, g_u)

    def _select_action(self, node: IntervalNode) -> int:
        """In-tree selection: optimistic Hurwicz + visit-count bonus."""
        best_action, best_score = NOOP, -math.inf
        parent_visits = max(1, node.visit_count)
        for a, child in node.children.items():
            if child.visit_count == 0:
                return a
            hurwicz = child.lower + self.c_search * (child.upper - child.lower)
            bonus = self.c_visit * math.sqrt(math.log(parent_visits) / child.visit_count)
            score = hurwicz + bonus
            if score > best_score:
                best_score, best_action = score, a
        return best_action

    def _step_sim(
        self, env, callsigns: List[str], planned_cs: str,
        planned_action: int, depth: int, frozen: Dict[str, int],
    ) -> Tuple[float, bool, bool, List[str], Optional[np.ndarray]]:
        """Step a deepcopied sim one tick.

        Frozen (already-decided) aircraft apply their chosen heading change on
        the first simulated step only (one-shot instruction), then NOOP.
        Undecided aircraft fly NOOP.  Returns (reward, done, violated,
        next_callsigns) for the planned aircraft.
        """
        actions = {}
        for cs in callsigns:
            if cs == planned_cs:
                actions[cs] = planned_action
            elif depth == 0 and cs in frozen:
                actions[cs] = frozen[cs]
            else:
                actions[cs] = NOOP

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
        violated = violation_involving(info, planned_cs)
        # Outcome anchoring: a clean exit inside the rollout earns the exit
        # bonus (guards: not the sim time limit, not an excursion — those are
        # violations, and a same-step OUT_SECTOR status never collects).
        if (done and not truncated and not violated
                and _status(info, planned_cs) != "OUT_SECTOR"):
            r += self.exit_bonus
        # Progress shaping on non-terminal steps, mirroring the DQN: pay for
        # approaching the exit, refund on retreat (potential-based, so it
        # cannot change which action is optimal — it densifies the signal).
        if self.progress_coeff and not done and phi is not None:
            phi_next = _exit_potential(env, planned_cs)
            if phi_next is not None:
                r += self.progress_coeff * (self.gamma * phi_next - phi)
        return r, done, violated, next_callsigns, obs.get(planned_cs)


# ============================================================================
# EPISODE RUNNER
# ============================================================================

SEC_PER_STEP = 6  # env default; matches bluebird_interval_dqn.SEC_PER_STEP


def run_episode(
    agent: Optional[IntervalMCTSAgent],
    seed: int = 42,
    scenario_duration: int = 600,
    verbose: bool = True,
    encoder_cls: str = "extra_minimal",
    k_nearest: int = 2,
) -> dict:
    """Run one full episode; agent=None means the all-NOOP baseline.

    encoder_cls / k_nearest must match the leaf net's training encoder in
    hybrid mode (main() derives them from the checkpoint metadata).

    Returns a stats dict: time-to-first-violation (headline metric), episode
    return, decision latencies, mean root interval widths.
    """
    env = make_env(scenario_duration=scenario_duration,
                   encoder_cls=encoder_cls, k_nearest=k_nearest)
    obs, info = env.reset(seed=seed)

    ep_return = 0.0
    latencies: List[float] = []
    widths: List[float] = []
    n_planned_total = 0
    first_violation_step: Optional[int] = None
    first_violation_time: Optional[float] = None
    first_violation_desc: Optional[str] = None
    step = 0

    # Realized-coverage instrumentation (agent runs only): for each searched
    # decision, the chosen root child's envelope predicts an H-step shaped
    # return; we accumulate each aircraft's live shaped rewards under the
    # planner's own accounting and score the envelopes post-episode.
    decision_records = []          # (cs, step_idx, lower, upper)
    shaped = {}                    # cs -> {step_idx: shaped reward}
    stream_end = {}                # cs -> terminal step_idx (inclusive)

    while True:
        t0 = time.perf_counter()
        if agent is not None:
            action = agent.generate_action(env, obs, info)
            widths.extend(agent.last_root_widths)
            n_planned_total += agent.last_n_planned
            for cs, (lo, up) in agent.last_root_choices.items():
                decision_records.append((cs, step, lo, up))
        else:
            action = {cs: NOOP for cs in obs}
        latencies.append(time.perf_counter() - t0)

        prev_obs_cs = list(obs.keys())
        phi_prev = ({cs: _exit_potential(env, cs) for cs in prev_obs_cs}
                    if agent is not None and agent.progress_coeff else {})
        obs, rew, term, trunc, info = env.step(action)
        step += 1
        ep_return += float(sum(rew.values()))

        if agent is not None:
            # live shaped rewards under the planner's own accounting
            for cs in prev_obs_cs:
                if cs in stream_end:
                    continue
                r = float(rew.get(cs, 0.0))
                truncated = bool(trunc.get(cs, False))
                done_cs = (cs not in obs or bool(term.get(cs, False))
                           or truncated)
                if violation_involving(info, cs):
                    r += agent.violation_penalty
                    done_cs = True
                elif (done_cs and not truncated
                        and _status(info, cs) != "OUT_SECTOR"):
                    r += agent.exit_bonus
                if agent.progress_coeff and not done_cs:
                    phi, phi_next = phi_prev.get(cs), _exit_potential(env, cs)
                    if phi is not None and phi_next is not None:
                        r += agent.progress_coeff * (
                            agent.gamma * phi_next - phi)
                shaped.setdefault(cs, {})[step - 1] = r
                if done_cs:
                    stream_end[cs] = step - 1

        if first_violation_step is None:
            viol = find_violations(info)
            if viol:
                first_violation_step = step
                # same convention as bluebird_interval_dqn: steps * 6 s
                first_violation_time = float(step * SEC_PER_STEP)
                first_violation_desc = "; ".join(viol)

        if verbose and step % 10 == 0:
            w = f"{np.mean(widths):.3f}" if widths else "n/a"
            print(f"  step {step:3d} | n_ac {len(obs):2d} | "
                  f"decision {latencies[-1]*1000:7.1f} ms | "
                  f"ep_return {ep_return:9.2f} | mean root width {w}")

        if not obs or all(term.values()) or all(trunc.values()):
            break

    # Score realized H-step shaped returns against the chosen envelopes.
    # A decision is scored only if its H-step window resolved: the aircraft
    # terminated inside it, or all H live steps exist. Otherwise censored
    # (episode ended first) — same discipline as the DQN realized tracker.
    n_hit = n_scored = n_censored = 0
    if agent is not None:
        H = agent.horizon
        for cs, k, lo, up in decision_records:
            stream = shaped.get(cs, {})
            end = stream_end.get(cs)
            last = min(k + H - 1, end if end is not None else k + H - 1)
            idxs = list(range(k, last + 1))
            if any(j not in stream for j in idxs):
                n_censored += 1
                continue
            if end is None and len(idxs) < H:
                n_censored += 1
                continue
            if (agent.backup_mode == "mean"
                    and (end is None or end > k + H - 1)):
                # hybrid envelopes include a gamma^H leaf-bootstrap tail;
                # a tail-less realized H-step return is only comparable
                # when the aircraft actually terminated inside the window
                n_censored += 1
                continue
            G = 0.0
            for j in reversed(idxs):
                G = stream[j] + agent.gamma * G
            n_scored += 1
            n_hit += int(lo - 1e-9 <= G <= up + 1e-9)

    return {
        "steps": step,
        "episode_return": ep_return,
        "realized_h_coverage": (n_hit / n_scored) if n_scored else None,
        "scored_decisions": n_scored,
        "censored_decisions": n_censored,
        "first_violation_step": first_violation_step,
        "first_violation_time_s": first_violation_time,
        # censored at the scenario duration when clean, so means aggregate
        # correctly and match bluebird_interval_dqn's time_to_violation
        "time_to_violation": (first_violation_time
                              if first_violation_time is not None
                              else float(scenario_duration)),
        "violated": first_violation_step is not None,
        "first_violation_desc": first_violation_desc,
        "mean_latency_s": float(np.mean(latencies)),
        "max_latency_s": float(np.max(latencies)),
        "mean_root_width": float(np.mean(widths)) if widths else 0.0,
        "n_searches": n_planned_total,
    }


def print_stats(name: str, stats: dict) -> None:
    print(f"\n--- {name} ---")
    if stats["first_violation_step"] is None:
        print(f"  time-to-first-violation : clean episode "
              f"(censored at {stats['time_to_violation']:.0f} s)")
    else:
        print(f"  time-to-first-violation : {stats['time_to_violation']:.0f} s "
              f"(step {stats['first_violation_step']})")
        print(f"  first violation         : {stats['first_violation_desc']}")
    print(f"  episode return          : {stats['episode_return']:.2f}")
    print(f"  steps                   : {stats['steps']}")
    print(f"  decision latency        : mean {stats['mean_latency_s']*1000:.1f} ms, "
          f"max {stats['max_latency_s']*1000:.1f} ms")
    print(f"  MCTS searches run       : {stats['n_searches']}")
    print(f"  mean root interval width: {stats['mean_root_width']:.4f}")
    if stats.get("realized_h_coverage") is not None:
        print(f"  realized H-step coverage: {stats['realized_h_coverage']:.3f} "
              f"({stats['scored_decisions']} scored, "
              f"{stats['censored_decisions']} censored)")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="short episode (120 s), tiny budget, plus NOOP baseline")
    mode.add_argument("--run", action="store_true", help="full episode with stats")
    mode.add_argument("--baseline", action="store_true", help="all-NOOP episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sims", type=int, default=24, help="simulations per search")
    parser.add_argument("--horizon", type=int, default=15, help="rollout horizon (env steps)")
    parser.add_argument("--leaf-value", type=str, default=None,
                        help="interval-DQN checkpoint for the hybrid leaf "
                             "bootstrap; switches backup to running-mean "
                             "bounds (exact interval-DQN analogy)")
    parser.add_argument("--leaf-c", type=float, default=None,
                        help="Hurwicz c for leaf state values "
                             "(default: c_search)")
    parser.add_argument("--progress-coeff", type=float, default=0.05,
                        help="potential-based progress shaping coefficient "
                             "(pay for approaching the exit; 0 disables)")
    parser.add_argument("--exit-bonus", type=float, default=10.0,
                        help="planning reward for a clean exit inside the rollout "
                             "(outcome anchoring; 0 restores the old objective)")
    parser.add_argument("--c-act", type=float, default=0.1,
                        help="Hurwicz c for the executed root action (pessimistic)")
    parser.add_argument("--c-search", type=float, default=0.8,
                        help="Hurwicz c for in-tree selection (optimistic)")
    parser.add_argument("--c-visit", type=float, default=3.0, help="UCB visit bonus coeff")
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--alert-radius", type=float, default=15.0,
                        help="nm; aircraft with no neighbour inside get NOOP unsearched")
    parser.add_argument("--alert-fl", type=float, default=20.0,
                        help="FL; vertical gate for counting a neighbour")
    parser.add_argument("--max-planned", type=int, default=4,
                        help="max aircraft searched per decision (budget cap)")
    parser.add_argument("--duration", type=int, default=600, help="scenario duration (s)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Hybrid mode: the leaf net reads the horizon observations, so the env
    # MUST be built with the encoder the checkpoint was trained on
    # (DESIGN_REVISIONS item 5). Model-free mode keeps the historic default.
    encoder_cls, k_nearest = "extra_minimal", 2
    if args.leaf_value is not None:
        import torch  # local import — torch optional in model-free mode
        ckpt_meta = torch.load(args.leaf_value, map_location="cpu",
                               weights_only=False)
        encoder_cls = ckpt_meta.get("encoder_cls", "extra_minimal")
        k_nearest = ckpt_meta.get("k", 2)
        print(f"[leaf-value] env reconstructed from checkpoint metadata: "
              f"encoder_cls={encoder_cls}, k_nearest={k_nearest} "
              f"({args.leaf_value})")

    if args.baseline:
        print(f"All-NOOP baseline | seed {args.seed} | duration {args.duration}s")
        stats = run_episode(None, seed=args.seed, scenario_duration=args.duration)
        print_stats("NOOP baseline", stats)
        return

    if args.smoke:
        # Tiny budget; alert gates opened wide so the search machinery is
        # actually exercised (seed-42 traffic is well separated early on,
        # so the production gates would skip every search in 120 s).
        print("SMOKE TEST: 120 s episode, 8 sims, horizon 8, max_planned 2")
        agent = IntervalMCTSAgent(
            n_simulations=8, horizon=8, gamma=args.gamma,
            c_search=args.c_search, c_act=args.c_act, c_visit=args.c_visit,
        exit_bonus=args.exit_bonus, progress_coeff=args.progress_coeff,
        leaf_value_path=args.leaf_value, leaf_c=args.leaf_c,
            alert_radius_nm=1e9, alert_fl=1e9, max_planned=2,
            rng=random.Random(args.seed),
        )
        t0 = time.time()
        stats = run_episode(agent, seed=args.seed, scenario_duration=120,
                            encoder_cls=encoder_cls, k_nearest=k_nearest)
        print(f"\nSmoke episode wall time: {time.time()-t0:.1f} s")
        print_stats("Interval MCTS (smoke)", stats)

        print("\nRunning all-NOOP baseline for comparison...")
        base = run_episode(None, seed=args.seed, scenario_duration=120, verbose=False)
        print_stats("NOOP baseline (smoke)", base)

        def ttv(s):
            return s["first_violation_step"] if s["first_violation_step"] is not None else math.inf
        if ttv(stats) > ttv(base):
            verdict = "MCTS beats baseline on time-to-first-violation"
        elif ttv(stats) == ttv(base):
            verdict = ("tie on time-to-first-violation"
                       + (" (neither violated)" if ttv(stats) == math.inf else ""))
        else:
            verdict = "MCTS WORSE than baseline on time-to-first-violation"
        print(f"\nVerdict: {verdict}")
        print("SMOKE TEST PASSED" if stats["steps"] > 0 else "SMOKE TEST FAILED")
        return

    # --run
    print(f"Interval MCTS | seed {args.seed} | sims {args.sims} | horizon {args.horizon} | "
          f"c_act {args.c_act} | c_search {args.c_search} | duration {args.duration}s")
    agent = IntervalMCTSAgent(
        n_simulations=args.sims, horizon=args.horizon, gamma=args.gamma,
        c_search=args.c_search, c_act=args.c_act, c_visit=args.c_visit,
        exit_bonus=args.exit_bonus, progress_coeff=args.progress_coeff,
        leaf_value_path=args.leaf_value, leaf_c=args.leaf_c,
        alert_radius_nm=args.alert_radius, alert_fl=args.alert_fl,
        max_planned=args.max_planned, rng=random.Random(args.seed),
    )
    t0 = time.time()
    stats = run_episode(agent, seed=args.seed, scenario_duration=args.duration,
                        encoder_cls=encoder_cls, k_nearest=k_nearest)
    print(f"\nEpisode wall time: {time.time()-t0:.1f} s")
    print_stats("Interval MCTS", stats)


if __name__ == "__main__":
    main()
