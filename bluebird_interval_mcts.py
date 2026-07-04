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

def make_env(scenario_duration: int = 600) -> InfiniteEnv:
    """Build the target BluebirdATC environment (verified configuration)."""
    cfg = InfiniteEnv.get_default_env_config()
    cfg.state_repr_config = {"encoder_cls": "extra_minimal", "k_nearest_aircraft": 2}
    cfg.action_config = {"simple_heading_left": [10], "simple_heading_right": [10]}
    cfg.reward_config = {
        "fns": [
            "position_status_const",
            "lateral_centreline_distance_shaped",
            "safety_simple_avoidance_exp",
        ],
        "coeffs": [1.0, 1.0, 1.2],
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
    """Tree node holding a running [min, max] envelope over backed-up returns."""
    visit_count: int = 0
    ret_sum: float = 0.0
    lower: float = math.inf     # running min of returns through this node
    upper: float = -math.inf    # running max of returns through this node
    children: Dict[int, "IntervalNode"] = field(default_factory=dict)

    def update(self, g: float) -> None:
        self.visit_count += 1
        self.ret_sum += g
        self.lower = min(self.lower, g)
        self.upper = max(self.upper, g)

    @property
    def mean(self) -> float:
        return self.ret_sum / self.visit_count if self.visit_count else 0.0

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
        alert_radius_nm: float = 15.0,  # neighbours farther than this: NOOP, no search
        alert_fl: float = 20.0,         # vertical gate for "neighbour" (level flights here)
        max_planned: int = 4,           # budget cap: riskiest-first searches per decision
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
        self.alert_radius_nm = alert_radius_nm
        self.alert_fl = alert_fl
        self.max_planned = max_planned
        self.rng = rng or random.Random(0)

        # Diagnostics (read after each generate_action call)
        self.last_root_widths: List[float] = []   # root child interval widths
        self.last_n_planned: int = 0

    # ------------------------------------------------------------------
    # Competition interface
    # ------------------------------------------------------------------

    def generate_action(self, env, observation_dict: dict, info_dict: dict) -> Dict[str, int]:
        """Plan a joint action for all controllable aircraft."""
        self.last_root_widths = []
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
            action, widths = self._plan_single(env, callsigns, cs, dict(decided))
            decided[cs] = action
            self.last_root_widths.extend(widths)
            self.last_n_planned += 1

        return {cs: decided.get(cs, NOOP) for cs in callsigns}

    # ------------------------------------------------------------------
    # Single-aircraft interval MCTS
    # ------------------------------------------------------------------

    def _plan_single(
        self, live_env, root_callsigns: List[str], planned_cs: str,
        frozen: Dict[str, int],
    ) -> Tuple[int, List[float]]:
        """Run interval MCTS for one aircraft; return (action, root widths)."""
        root = IntervalNode()

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
        return best_action, widths

    def _simulate(
        self, root: IntervalNode, env, callsigns: List[str],
        planned_cs: str, frozen: Dict[str, int],
    ) -> None:
        """One simulation: select/expand down the tree, NOOP rollout, backup."""
        node = root
        path = [root]
        g, disc = 0.0, 1.0
        depth, done = 0, False

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

            r, done, violated, callsigns = self._step_sim(
                env, callsigns, planned_cs, action, depth, frozen)
            g += disc * r
            if violated:
                g += disc * self.violation_penalty
            disc *= self.gamma
            depth += 1

            child = node.children.get(action)
            if child is None:
                child = IntervalNode()
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
                r, done, violated, callsigns = self._step_sim(
                    env, callsigns, planned_cs, NOOP, depth, frozen)
                g += disc * r
                if violated:
                    g += disc * self.violation_penalty
                disc *= self.gamma
                depth += 1
                if done or violated:
                    break

        # --- Backup: total discounted return into every node on the path
        for nd in path:
            nd.update(g)

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
    ) -> Tuple[float, bool, bool, List[str]]:
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

        obs, rew, term, trunc, info = env.step(actions)
        next_callsigns = list(obs.keys())

        r = float(rew.get(planned_cs, 0.0))
        done = (
            planned_cs not in obs
            or bool(term.get(planned_cs, False))
            or bool(trunc.get(planned_cs, False))
        )
        violated = violation_involving(info, planned_cs)
        return r, done, violated, next_callsigns


# ============================================================================
# EPISODE RUNNER
# ============================================================================

SEC_PER_STEP = 6  # env default; matches bluebird_interval_dqn.SEC_PER_STEP


def run_episode(
    agent: Optional[IntervalMCTSAgent],
    seed: int = 42,
    scenario_duration: int = 600,
    verbose: bool = True,
) -> dict:
    """Run one full episode; agent=None means the all-NOOP baseline.

    Returns a stats dict: time-to-first-violation (headline metric), episode
    return, decision latencies, mean root interval widths.
    """
    env = make_env(scenario_duration=scenario_duration)
    obs, info = env.reset(seed=seed)

    ep_return = 0.0
    latencies: List[float] = []
    widths: List[float] = []
    n_planned_total = 0
    first_violation_step: Optional[int] = None
    first_violation_time: Optional[float] = None
    first_violation_desc: Optional[str] = None
    step = 0

    while True:
        t0 = time.perf_counter()
        if agent is not None:
            action = agent.generate_action(env, obs, info)
            widths.extend(agent.last_root_widths)
            n_planned_total += agent.last_n_planned
        else:
            action = {cs: NOOP for cs in obs}
        latencies.append(time.perf_counter() - t0)

        obs, rew, term, trunc, info = env.step(action)
        step += 1
        ep_return += float(sum(rew.values()))

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

    return {
        "steps": step,
        "episode_return": ep_return,
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
            alert_radius_nm=1e9, alert_fl=1e9, max_planned=2,
            rng=random.Random(args.seed),
        )
        t0 = time.time()
        stats = run_episode(agent, seed=args.seed, scenario_duration=120)
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
        alert_radius_nm=args.alert_radius, alert_fl=args.alert_fl,
        max_planned=args.max_planned, rng=random.Random(args.seed),
    )
    t0 = time.time()
    stats = run_episode(agent, seed=args.seed, scenario_duration=args.duration)
    print(f"\nEpisode wall time: {time.time()-t0:.1f} s")
    print_stats("Interval MCTS", stats)


if __name__ == "__main__":
    main()
