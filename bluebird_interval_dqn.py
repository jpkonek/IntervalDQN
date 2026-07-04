"""
Parameter-Shared Interval DQN on BluebirdATC (Flight School benchmark)
======================================================================

Multi-agent adaptation of the LunarLander interval DQN (see
lunarlander_interval_dqn.py) to the BluebirdATC air-traffic-control
gymnasium environment (InfiniteEnv, X-Plus sector, decentralized view).

Setup:
  ONE shared IntervalQNetwork; every aircraft in the sector is an
  independent "agent" that queries the same network on its own local
  observation (extra_minimal encoder: centreline distance, next-fix
  angle, and per-neighbour relative heading + distance).

Ported faithfully from the LunarLander implementation:
  - Interval Q-network: outputs (lower, delta_raw) per action;
    interval = [lower, lower + softplus(delta_raw) + 1e-6]
  - Hurwicz action selection: Q_c = lower + c * (upper - lower);
    c_train = 0.5 for data collection, evaluate at c in {0.0, 0.2, 0.5, 1.0}
  - Interval loss with sampled Bellman targets: N=5 points uniform across
    [r + gamma*L_next, r + gamma*U_next], Double-DQN target network with
    hard updates
  - CoverageTracker adapting t toward ~85% coverage
  - Width regularization once coverage >= target
  - Per-step training, replay buffer, brief epsilon warmup only

New (multi-agent / ATC specific):
  - Batched per-aircraft action selection (one forward pass per env step)
  - Per-aircraft replay transitions with churn handling: aircraft absent
    at t+1 (archived/deleted) are treated as terminal with zero next-state
  - Violation detector: the env does NOT terminate on loss of separation.
    Competition Flight School score = seconds until first violation
    (loss of separation: lateral < 5 nm AND vertical < 10 FL between an
    in-sector pair; or sector excursion: PositionStatus OUT_SECTOR).
    Training episodes end at first violation; time-to-first-violation is
    the headline eval metric.
  - Violation penalty (default 10.0) subtracted from the reward of the
    aircraft involved in the violation (their transition is terminal).
  - MPS-vs-CPU benchmark for the small MLP; --device auto picks the faster.

Usage:
    python bluebird_interval_dqn.py --smoke
    python bluebird_interval_dqn.py --train --episodes 300 --seed 42
    python bluebird_interval_dqn.py --eval --ckpt checkpoints/bluebird/seed42_final.pt --c 0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import json
import math
import sys
import os
import time
from collections import deque

sys.stdout.reconfigure(line_buffering=True)

# Network / training constants (LunarLander parity where sensible)
HIDDEN = 128
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 100  # Hard target update every N gradient steps
GRAD_CLIP = 1.0

# BluebirdATC constants
SEC_PER_STEP = 6           # simulated seconds per env step
LATERAL_SEP_NM = 5.0       # loss-of-separation lateral threshold
VERTICAL_SEP_FL = 10.0     # loss-of-separation vertical threshold
EARTH_RADIUS_NM = 3440.065

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "checkpoints", "bluebird")


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# COVERAGE TRACKER — adapts the loss parameter t toward target coverage
# ============================================================================

class CoverageTracker:
    """Tracks recent interval coverage (target midpoint inside [l, u]) and
    adapts the interval-loss parameter t toward the target coverage:
    coverage below target -> raise t (penalize misses), above -> lower t
    (allow narrower intervals)."""

    def __init__(self, target=0.85, t_init=0.5, window=2000, min_samples=200):
        self.target = target
        self.t = t_init
        self.hits = deque(maxlen=window)
        self.min_samples = min_samples

    def record(self, hits):
        self.hits.extend(hits)

    @property
    def coverage(self):
        return sum(self.hits) / max(1, len(self.hits))

    def update_t(self):
        if len(self.hits) >= self.min_samples:
            if self.coverage < self.target:
                self.t = min(0.95, self.t + 0.001)
            else:
                self.t = max(0.05, self.t - 0.0005)


# ============================================================================
# INTERVAL Q-NETWORK
# ============================================================================

class IntervalQNetwork(nn.Module):
    """Outputs interval [lower, upper] for each action's Q-value."""

    def __init__(self, state_dim, n_actions, hidden=HIDDEN):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_actions * 2)
        self.n_actions = n_actions

    def forward(self, x):
        h = self.shared(x)
        out = self.head(h)
        out = out.view(-1, self.n_actions, 2)
        lower = out[:, :, 0]
        delta = F.softplus(out[:, :, 1]) + 1e-6
        upper = lower + delta
        return lower, upper


# ============================================================================
# PARAMETER-SHARED INTERVAL DQN AGENT
# ============================================================================

class IntervalDQNAgent:
    N_TARGET_SAMPLES = 5

    def __init__(self, state_dim, n_actions, lr=LR, gamma=GAMMA,
                 hidden=HIDDEN, c_train=0.5, target_coverage=0.85,
                 width_reg=0.01, warmup_steps=1000, warmup_epsilon=0.5,
                 buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 device="cpu"):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.c_train = c_train
        self.width_reg = width_reg
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.step_count = 0

        self.q_net = IntervalQNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net = IntervalQNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        # Adaptive coverage
        self.coverage_tracker = CoverageTracker(target=target_coverage)

        # Warmup with epsilon-greedy (env-step based)
        self.warmup_steps = warmup_steps
        self.warmup_epsilon = warmup_epsilon
        self.total_env_steps = 0

    # convenience passthroughs
    @property
    def t(self):
        return self.coverage_tracker.t

    @property
    def coverage(self):
        return self.coverage_tracker.coverage

    def _epsilon(self, force_epsilon=None):
        if force_epsilon is not None:
            return force_epsilon
        if self.total_env_steps < self.warmup_steps:
            progress = self.total_env_steps / self.warmup_steps
            return self.warmup_epsilon * (1 - progress)
        return 0.0

    def generate_action(self, obs_dict, c=None, force_epsilon=None):
        """Batched Hurwicz action selection for ALL aircraft in obs_dict.

        One forward pass per env step (never per aircraft).
        Returns ({callsign: int}, mean_interval_width).
        """
        if not obs_dict:
            return {}, 0.0

        callsigns = list(obs_dict.keys())
        states = np.stack([obs_dict[cs] for cs in callsigns]).astype(np.float32)
        eps = self._epsilon(force_epsilon)
        use_c = c if c is not None else self.c_train

        with torch.no_grad():
            s = torch.from_numpy(states).to(self.device)
            lower, upper = self.q_net(s)
            q = lower + use_c * (upper - lower)
            greedy = q.argmax(dim=1).cpu().numpy()
            mean_width = (upper - lower).mean().item()

        actions = {}
        for i, cs in enumerate(callsigns):
            if random.random() < eps:
                actions[cs] = random.randint(0, self.n_actions - 1)
            else:
                actions[cs] = int(greedy[i])
        return actions, mean_width

    def interval_loss_sampled(self, lower, upper, target_lower, target_upper):
        N = self.N_TARGET_SAMPLES
        batch_size = lower.shape[0]

        alphas = torch.linspace(0, 1, N, device=lower.device)
        targets = target_lower.unsqueeze(0) + alphas.unsqueeze(1) * (
            target_upper - target_lower).unsqueeze(0)

        # Track coverage against midpoint
        target_mid = (target_lower + target_upper) / 2
        inside_mid = (target_mid >= lower) & (target_mid <= upper)
        self.coverage_tracker.record(
            inside_mid.detach().cpu().numpy().astype(np.float32).tolist())

        t = self.coverage_tracker.t
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

            total_loss += t * outside_penalty + (1 - t) * max_dist_sq

        total_loss /= N

        # Width regularization when coverage exceeds target
        if self.coverage_tracker.coverage > self.coverage_tracker.target:
            width = upper - lower
            total_loss = total_loss + self.width_reg * width ** 2

        return total_loss.mean()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        lower, upper = self.q_net(states)
        lower_a = lower.gather(1, actions.unsqueeze(1)).squeeze(1)
        upper_a = upper.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_lower, next_upper = self.target_net(next_states)
            # Double DQN style: online net selects action
            ol, ou = self.q_net(next_states)
            next_q = ol + self.c_train * (ou - ol)
            next_actions = next_q.argmax(dim=1)

            tgt_lower = next_lower.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            tgt_upper = next_upper.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_l = rewards + self.gamma * tgt_lower * (1 - dones)
            target_u = rewards + self.gamma * tgt_upper * (1 - dones)

        loss = self.interval_loss_sampled(lower_a, upper_a, target_l, target_u)
        if not torch.isfinite(loss):
            # skip the update rather than corrupt weights on a bad batch
            self.optimizer.zero_grad()
            return 0.0

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Adaptive t
        self.coverage_tracker.update_t()

        return loss.item()

    def get_interval_widths(self, states):
        with torch.no_grad():
            s = torch.FloatTensor(np.asarray(states, dtype=np.float32)).to(self.device)
            lower, upper = self.q_net(s)
            widths = (upper - lower).mean(dim=1)
            return widths.mean().item()


# ============================================================================
# ENVIRONMENT + VIOLATION DETECTION
# ============================================================================

def make_env(scenario_duration=600, k_nearest=2, route_parallel=False,
             centreline_coeff=0.2):
    """Build the Flight School InfiniteEnv (X-Plus sector, decentralized).

    route_parallel adds the simple_heading_route_parallel clearance (steer
    along the current route segment) — the natural tool against wrong-exit
    excursions, which dominate the violations seen with heading-only control.

    centreline_coeff scales the on-route income stream. Kept SMALL by design:
    diagnostics on run 2 showed that with a dominant per-step income and
    gamma=0.99 the Bellman fixed point is an unbounded annuity (Q ~ +40 vs
    realized returns ~ -6, realized coverage 0.0). Reward mass belongs on
    outcomes (exit bonus / violation penalty), LunarLander-style, so that
    value bounds fall out of the training dynamics rather than the income.
    """
    from bluebird_gymnasium.envs import InfiniteEnv
    from bluebird_gymnasium.envs.infinite import ScenarioName

    cfg = InfiniteEnv.get_default_env_config()
    cfg.state_repr_config = {"encoder_cls": "extra_minimal",
                             "k_nearest_aircraft": k_nearest}
    cfg.action_config = {"simple_heading_left": [10],
                         "simple_heading_right": [10]}
    if route_parallel:
        cfg.action_config["simple_heading_route_parallel"] = True
    cfg.reward_config = {
        "fns": ["position_status_const",
                "lateral_centreline_distance_shaped",
                "safety_simple_avoidance_exp"],
        "coeffs": [1.0, centreline_coeff, 1.2],
    }
    cfg.scenario_config["scenario_name"] = ScenarioName.sector_xplus
    cfg.view_config["type"] = "decentralized"
    cfg.view_config["decentralized_params"] = {}
    cfg.scenario_duration = scenario_duration
    return InfiniteEnv(config=cfg)


def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * EARTH_RADIUS_NM * math.asin(math.sqrt(a))


def detect_violation(info):
    """Detect a Flight School violation from the per-step info dict.

    The env does NOT terminate on these — we detect them ourselves:
      - sector excursion: any tracked aircraft with PositionStatus
        OUT_SECTOR (left the sector other than through its exit window);
        info[callsign]["pos_status"] carries the status name.
      - loss of separation: an IN_SECTOR pair with lateral distance
        < 5 nm AND |flight level difference| < 10 FL (positions read from
        info["simulator_environment"].aircraft).

    Returns (violated: bool, kind: str | None, involved: list[callsign]).
    """
    in_sector = []
    for cs, d in info.items():
        if cs == "simulator_environment" or not isinstance(d, dict):
            continue
        pos_status = d.get("pos_status")
        if pos_status == "OUT_SECTOR":
            return True, "sector_excursion", [cs]
        if pos_status == "IN_SECTOR":
            in_sector.append(cs)

    sim_env = info.get("simulator_environment")
    if sim_env is None or len(in_sector) < 2:
        return False, None, []

    aircraft = sim_env.aircraft
    for i in range(len(in_sector)):
        ac_i = aircraft.get(in_sector[i])
        if ac_i is None or ac_i.fl is None:
            continue
        for j in range(i + 1, len(in_sector)):
            ac_j = aircraft.get(in_sector[j])
            if ac_j is None or ac_j.fl is None:
                continue
            if abs(ac_i.fl - ac_j.fl) >= VERTICAL_SEP_FL:
                continue
            dist = haversine_nm(ac_i.lat, ac_i.lon, ac_j.lat, ac_j.lon)
            if dist < LATERAL_SEP_NM:
                return True, "loss_of_separation", [in_sector[i], in_sector[j]]

    return False, None, []


# ============================================================================
# DEVICE BENCHMARK — small MLPs often run faster on CPU than MPS; measure.
# ============================================================================

def benchmark_device(state_dim, n_actions, hidden=HIDDEN, batch=BATCH_SIZE,
                     iters=200):
    """Time forward+backward of the interval MLP on CPU and (if available)
    MPS. Returns (best_device, {device: iters_per_sec})."""
    results = {}
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    for dev in devices:
        net = IntervalQNetwork(state_dim, n_actions, hidden).to(dev)
        opt = optim.Adam(net.parameters(), lr=1e-3)
        x = torch.randn(batch, state_dim, device=dev)
        y = torch.randn(batch, n_actions, device=dev)

        def _sync():
            if dev == "mps":
                torch.mps.synchronize()

        for _ in range(10):  # warmup
            lower, upper = net(x)
            loss = ((lower - y) ** 2 + (upper - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        _sync()

        t0 = time.perf_counter()
        for _ in range(iters):
            lower, upper = net(x)
            loss = ((lower - y) ** 2 + (upper - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        _sync()
        results[dev] = iters / (time.perf_counter() - t0)

    best = max(results, key=results.get)
    return best, results


def resolve_device(device_arg, state_dim, n_actions, verbose=True):
    if device_arg != "auto":
        return device_arg
    best, results = benchmark_device(state_dim, n_actions)
    if verbose:
        strs = ", ".join(f"{d}: {r:.0f} it/s" for d, r in results.items())
        print(f"Device benchmark (fwd+bwd, batch {BATCH_SIZE}): {strs} "
              f"-> using {best}")
    return best


# ============================================================================
# EPISODE RUNNER (shared by training and evaluation)
# ============================================================================

def run_episode(env, agent, seed, train=True, c=None, violation_penalty=10.0,
                exit_bonus=10.0):
    """Run one episode; ends at first violation or the time limit.

    exit_bonus anchors reward mass on the OUTCOME (LunarLander-style): a
    terminal +bonus when an aircraft leaves cleanly (exit fix / handoff),
    mirroring the -violation_penalty terminal on excursion/LoS. With the
    centreline income shrunk (see make_env), returns are dominated by
    terminal events, so Bellman bootstrapping is grounded in real outcomes.

    Returns a stats dict including time_to_violation (simulated seconds;
    equals the scenario duration if no violation occurred = censored).
    """
    obs, info = env.reset(seed=seed)
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))

    reward_sum = 0.0
    reward_n = 0
    width_sum = 0.0
    width_n = 0
    loss_sum = 0.0
    loss_n = 0
    n_transitions = 0
    violated = False
    violation_kind = None

    step_i = -1
    for step_i in range(maxstep):
        force_eps = None if train else 0.0
        actions, mean_width = agent.generate_action(
            obs, c=c, force_epsilon=force_eps)
        if actions:
            width_sum += mean_width
            width_n += 1

        next_obs, rew, done, trunc, info = env.step(actions)
        violated, violation_kind, involved = detect_violation(info)

        if train:
            for cs, s in obs.items():
                # skip aircraft still BEFORE_ENTRY: the env ignores their
                # actions and zeroes their rewards, so these transitions
                # only dilute the replay buffer
                d = info.get(cs)
                if isinstance(d, dict) and d.get("pos_status") == "BEFORE_ENTRY":
                    continue
                a = actions[cs]
                r = float(rew.get(cs, 0.0))
                if cs in next_obs:
                    ns = np.asarray(next_obs[cs], dtype=np.float32)
                    # truncation != termination: bootstrap through time limit
                    terminal = (bool(done.get(cs, False))
                                and not bool(trunc.get(cs, False)))
                else:
                    # vanished (exited / archived / deleted): terminal
                    ns = np.zeros_like(s, dtype=np.float32)
                    terminal = True
                if violated and cs in involved:
                    r -= violation_penalty
                    terminal = True
                elif terminal and not (isinstance(d, dict)
                                       and d.get("pos_status") == "OUT_SECTOR"):
                    # clean exit (EXIT_REACHED / outcomm / archived): outcome
                    # bonus. OUT_SECTOR terminals are excursions and never
                    # get it (they are usually caught by the violation branch
                    # above; this guard covers a same-step second excursion).
                    r += exit_bonus
                agent.buffer.push(np.asarray(s, dtype=np.float32), a, r,
                                  ns, float(terminal))
                n_transitions += 1

            agent.total_env_steps += 1
            loss = agent.train_step()
            if loss:
                loss_sum += loss
                loss_n += 1

        reward_sum += float(sum(rew.values()))
        reward_n += max(1, len(rew))
        obs = next_obs

        if violated:
            break

    steps_done = step_i + 1
    time_to_violation = (steps_done * SEC_PER_STEP if violated
                         else maxstep * SEC_PER_STEP)

    return {
        "steps": steps_done,
        "sim_seconds": steps_done * SEC_PER_STEP,
        "violated": violated,
        "violation_kind": violation_kind,
        "time_to_violation": time_to_violation,
        "mean_step_reward": reward_sum / max(1, reward_n),
        "mean_width": width_sum / max(1, width_n),
        "mean_loss": loss_sum / max(1, loss_n),
        "transitions": n_transitions,
    }


# ============================================================================
# CHECKPOINT SAVE/LOAD
# ============================================================================

def ckpt_extra(args):
    """Everything needed to reproduce the training env/targets at eval time."""
    return {"k": args.k,
            "route_parallel": args.route_parallel,
            "gamma": args.gamma,
            "exit_bonus": args.exit_bonus,
            "centreline_coeff": args.centreline_coeff,
            "violation_penalty": args.violation_penalty}


def save_checkpoint(agent, path, episode, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "q_net": agent.q_net.state_dict(),
        "state_dim": agent.state_dim,
        "n_actions": agent.n_actions,
        "c_train": agent.c_train,
        "t": agent.coverage_tracker.t,
        "coverage": agent.coverage,
        "episode": episode,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_agent(path, device="cpu", **kwargs):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    agent = IntervalDQNAgent(
        state_dim=ckpt["state_dim"], n_actions=ckpt["n_actions"],
        c_train=ckpt.get("c_train", 0.5), device=device, **kwargs)
    agent.q_net.load_state_dict(ckpt["q_net"])
    agent.target_net.load_state_dict(ckpt["q_net"])
    agent.coverage_tracker.t = ckpt.get("t", 0.5)
    agent.total_env_steps = 10 ** 9  # past warmup: no epsilon at eval
    agent.q_net.eval()
    return agent, ckpt


# ============================================================================
# TRAINING
# ============================================================================

def run_training(args, smoke=False):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    duration = args.duration
    episodes = args.episodes

    print("=" * 100)
    tag = "SMOKE TEST" if smoke else "TRAINING"
    print(f"BLUEBIRD INTERVAL DQN — {tag} (seed={args.seed}, "
          f"c_train={args.c_train}, duration={duration}s, "
          f"episodes={episodes}, k={args.k})")
    print("=" * 100)

    print("Creating environment...")
    env = make_env(scenario_duration=duration, k_nearest=args.k,
                   route_parallel=args.route_parallel,
                   centreline_coeff=args.centreline_coeff)
    obs, _ = env.reset(seed=args.seed)
    state_dim = int(next(iter(obs.values())).shape[0])
    n_actions = int(env.get_action_parser().get_total_num_actions())
    action_map = env.get_action_parser().action_formatter_map
    print(f"obs dim: {state_dim}, actions ({n_actions}): {action_map}")

    device = resolve_device(args.device, state_dim, n_actions)

    agent = IntervalDQNAgent(
        state_dim=state_dim, n_actions=n_actions, lr=args.lr,
        gamma=args.gamma, c_train=args.c_train,
        target_coverage=args.target_coverage, width_reg=args.width_reg,
        warmup_steps=args.warmup_steps, warmup_epsilon=args.warmup_epsilon,
        buffer_size=args.buffer, batch_size=args.batch, device=device)

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
    for ep in range(episodes):
        ep_seed = args.seed + ep
        t0 = time.time()
        stats = run_episode(env, agent, seed=ep_seed, train=True,
                            violation_penalty=args.violation_penalty,
                            exit_bonus=args.exit_bonus)
        wall = time.time() - t0
        total_steps += stats["steps"]
        eps = agent._epsilon()

        record = {
            "episode": ep + 1,
            "seed": ep_seed,
            **stats,
            "buffer_size": len(agent.buffer),
            "coverage": round(agent.coverage, 4),
            "t": round(agent.t, 4),
            "epsilon": round(eps, 4),
            "c_train": agent.c_train,
            "steps_per_sec": round(stats["steps"] / max(1e-9, wall), 2),
            "wall_time_s": round(wall, 2),
        }
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

        vio = (f"VIOLATION[{stats['violation_kind']}]@{stats['time_to_violation']}s"
               if stats["violated"] else f"clean({stats['time_to_violation']}s)")
        print(f"  Ep {ep+1:4d} | {vio:>38} | R:{stats['mean_step_reward']:7.3f} "
              f"| W:{stats['mean_width']:.3f} Cvg:{agent.coverage:.2f} "
              f"t:{agent.t:.3f} eps:{eps:.3f} | buf:{len(agent.buffer):6d} "
              f"loss:{stats['mean_loss']:.4f} | {record['steps_per_sec']:.1f} st/s")

        if (ep + 1) % args.ckpt_every == 0:
            # run_tag in the filename: concurrent/sequential runs must never
            # overwrite each other's checkpoints (run 3 clobbered run 2's)
            path = os.path.join(CHECKPOINT_DIR,
                                f"{prefix}_seed{args.seed}_{run_tag}_ep{ep+1}.pt")
            save_checkpoint(agent, path, ep + 1, extra=ckpt_extra(args))
            print(f"    Saved checkpoint: {path}")

    final_path = os.path.join(CHECKPOINT_DIR,
                              f"{prefix}_seed{args.seed}_{run_tag}_final.pt")
    save_checkpoint(agent, final_path, episodes, extra=ckpt_extra(args))
    elapsed = time.time() - t_start
    print("-" * 100)
    print(f"Training done: {episodes} episodes, {total_steps} env steps, "
          f"{elapsed:.0f}s ({total_steps / max(1e-9, elapsed):.1f} steps/s "
          f"incl. training). Final checkpoint: {final_path}")

    # ---- Final evaluation across Hurwicz c values ----
    n_eval = args.final_eval_episodes
    if n_eval > 0:
        print()
        print("=" * 100)
        print(f"FINAL EVALUATION ({n_eval} episodes per c, "
              f"duration {duration}s)")
        print("=" * 100)
        for c_val in args.c_eval:
            evaluate(env, agent, c=c_val, n_episodes=n_eval,
                     base_seed=args.seed + 10000)

    env.close()
    log_file.close()
    return agent


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(env, agent, c, n_episodes=5, base_seed=10042, verbose=True):
    """Evaluate at a fixed Hurwicz c over several seeds.
    Headline metric: mean time-to-first-violation (simulated seconds)."""
    ttvs, rewards, widths = [], [], []
    n_violated = 0
    kinds = {}
    for i in range(n_episodes):
        stats = run_episode(env, agent, seed=base_seed + i, train=False, c=c)
        ttvs.append(stats["time_to_violation"])
        rewards.append(stats["mean_step_reward"])
        widths.append(stats["mean_width"])
        if stats["violated"]:
            n_violated += 1
            kinds[stats["violation_kind"]] = (
                kinds.get(stats["violation_kind"], 0) + 1)

    result = {
        "c": c,
        "mean_time_to_violation": float(np.mean(ttvs)),
        "std_time_to_violation": float(np.std(ttvs)),
        "violation_rate": n_violated / n_episodes,
        "violation_kinds": kinds,
        "mean_step_reward": float(np.mean(rewards)),
        "mean_width": float(np.mean(widths)),
        "n_episodes": n_episodes,
    }
    if verbose:
        print(f"  c={c:<4} | time-to-violation: "
              f"{result['mean_time_to_violation']:7.1f}s "
              f"± {result['std_time_to_violation']:6.1f} | "
              f"violated: {n_violated}/{n_episodes} {kinds if kinds else ''} | "
              f"R:{result['mean_step_reward']:7.3f} | "
              f"W:{result['mean_width']:.3f}")
    return result


def run_eval_only(args):
    print(f"Loading checkpoint: {args.ckpt}")
    device = "cpu" if args.device == "auto" else args.device
    agent, ckpt = load_agent(args.ckpt, device=device)
    k = ckpt.get("k", args.k)
    route_parallel = ckpt.get("route_parallel", args.route_parallel)
    # default 1.0: checkpoints predating outcome-anchoring trained with the
    # full-strength centreline income
    centreline_coeff = ckpt.get("centreline_coeff", 1.0)
    print(f"Loaded: obs dim {ckpt['state_dim']}, {ckpt['n_actions']} actions, "
          f"trained {ckpt.get('episode', '?')} episodes, "
          f"t={ckpt.get('t', 0.5):.3f}")

    print(f"Creating environment (duration {args.duration}s, k={k}, "
          f"route_parallel={route_parallel}, "
          f"centreline_coeff={centreline_coeff})...")
    env = make_env(scenario_duration=args.duration, k_nearest=k,
                   route_parallel=route_parallel,
                   centreline_coeff=centreline_coeff)

    # keep eval scenarios disjoint from training episode seeds
    # (training uses seed..seed+episodes-1; +10000 matches the in-training
    # final eval convention)
    eval_seed = args.seed + 10000
    print("=" * 100)
    print(f"EVALUATION (c={args.c}, {args.eval_episodes} episodes, "
          f"base seed {eval_seed})")
    print("=" * 100)
    result = evaluate(env, agent, c=args.c, n_episodes=args.eval_episodes,
                      base_seed=eval_seed)
    env.close()
    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Parameter-shared Interval DQN on BluebirdATC "
                    "(Flight School benchmark)")
    # modes
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny end-to-end run (train + eval) in <5 min")
    parser.add_argument("--train", action="store_true",
                        help="Full training with checkpointing + JSONL log")
    parser.add_argument("--eval", action="store_true",
                        help="Eval-only from a checkpoint at a given c")
    # common
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Training episodes (default: 300; smoke: 2)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Scenario duration in simulated seconds "
                             "(default: 600; smoke: 120). Use 120 for "
                             "faster early training.")
    parser.add_argument("--k", type=int, default=2,
                        help="k nearest aircraft in the observation")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps"],
                        help="auto = benchmark MPS vs CPU and pick faster")
    # agent hyperparameters
    parser.add_argument("--c_train", type=float, default=0.5)
    parser.add_argument("--route_parallel", action="store_true",
                        help="add the simple_heading_route_parallel clearance "
                             "to the action space (4 actions instead of 3)")
    parser.add_argument("--c_eval", type=float, nargs="+",
                        default=[0.0, 0.2, 0.5, 1.0])
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer", type=int, default=BUFFER_SIZE)
    parser.add_argument("--target_coverage", type=float, default=0.85)
    parser.add_argument("--width_reg", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Epsilon warmup env steps (default: 1000; "
                             "smoke: 20)")
    parser.add_argument("--warmup_epsilon", type=float, default=0.5)
    parser.add_argument("--exit_bonus", type=float, default=10.0,
                        help="terminal reward for a clean exit (outcome "
                             "anchoring; set 0 to disable)")
    parser.add_argument("--centreline_coeff", type=float, default=0.2,
                        help="weight of the on-route income shaping term "
                             "(run 2 used 1.0; small keeps returns "
                             "outcome-dominated)")
    parser.add_argument("--violation_penalty", type=float, default=10.0,
                        help="Reward penalty for aircraft involved in a "
                             "violation (their transition is terminal)")
    # bookkeeping
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path for --eval")
    parser.add_argument("--c", type=float, default=0.0,
                        help="Hurwicz c for --eval")
    parser.add_argument("--eval_episodes", type=int, default=5,
                        help="Episodes (seeds) for --eval")
    parser.add_argument("--final_eval_episodes", type=int, default=None,
                        help="Episodes per c in the post-training eval "
                             "(default: 3; smoke: 1; 0 disables)")
    parser.add_argument("--ckpt_every", type=int, default=25)
    args = parser.parse_args()

    if args.smoke:
        args.episodes = args.episodes if args.episodes is not None else 2
        args.duration = args.duration if args.duration is not None else 120
        args.warmup_steps = (args.warmup_steps if args.warmup_steps is not None
                             else 20)
        args.final_eval_episodes = (args.final_eval_episodes
                                    if args.final_eval_episodes is not None
                                    else 1)
        args.buffer = min(args.buffer, 5000)
        args.batch = min(args.batch, 32)
        args.c_eval = [0.0, 0.5]
        run_training(args, smoke=True)
        print("\nSMOKE TEST PASSED")
    elif args.eval:
        if args.ckpt is None:
            parser.error("--eval requires --ckpt PATH")
        args.duration = args.duration if args.duration is not None else 600
        run_eval_only(args)
    else:
        # --train (also the default when no mode flag is given)
        args.episodes = args.episodes if args.episodes is not None else 300
        args.duration = args.duration if args.duration is not None else 600
        args.warmup_steps = (args.warmup_steps if args.warmup_steps is not None
                             else 1000)
        args.final_eval_episodes = (args.final_eval_episodes
                                    if args.final_eval_episodes is not None
                                    else 3)
        run_training(args, smoke=False)
