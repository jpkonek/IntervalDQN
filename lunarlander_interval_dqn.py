"""
Interval DQN vs Standard DQN vs Ensemble DQN on LunarLander with Wind Perturbation
====================================================================================

Proof of concept: credal Q-networks for robust RL.

Setup:
  Train all agents on LunarLander-v3 with NO wind (800 episodes).
  Evaluate under three conditions: no wind, moderate wind, strong wind.

Three agents trained and evaluated:
  1. Standard DQN (Double DQN) — single network, point Q-values, ε-greedy
  2. Interval DQN              — interval Q-values [lower, upper], Hurwicz selection
                                 Training: c_train for exploration
                                 Evaluation: c=0.0 (pessimistic → safe)
  3. Ensemble DQN (N=5)        — N networks, mean - λ*std for conservative action

Architecture: Double DQN with per-step training and hard target updates (every 100 steps).

Key metrics:
  - Mean reward (higher = better)
  - Crash rate (lower = safer — key safety metric)
  - Solve rate (reward >= 200)

Key question: Under unforeseen wind, does the interval DQN (c=0, pessimistic)
produce safer behaviour (fewer crashes) than standard or ensemble DQN?

Usage:
    python lunarlander_interval_dqn.py
    python lunarlander_interval_dqn.py --single --seed 42
    python lunarlander_interval_dqn.py --seeds 42 123 456 789 1337
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
import sys
import os
import warnings
from collections import deque

sys.stdout.reconfigure(line_buffering=True)

# LunarLander constants
STATE_DIM = 8
N_ACTIONS = 4
HIDDEN = 128
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 100  # Hard target update every N steps
GRAD_CLIP = 1.0


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
# STANDARD DQN (Double DQN)
# ============================================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class StandardDQNAgent:
    def __init__(self, lr=LR, gamma=GAMMA, hidden=HIDDEN):
        self.n_actions = N_ACTIONS
        self.gamma = gamma
        self.step_count = 0

        self.q_net = QNetwork(hidden=hidden)
        self.target_net = QNetwork(hidden=hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state).unsqueeze(0))
            return q.argmax(dim=1).item()

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online selects action, target evaluates
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()


# ============================================================================
# INTERVAL DQN — Hurwicz exploration + sampled interval targets
# ============================================================================

class IntervalQNetwork(nn.Module):
    """Outputs interval [lower, upper] for each action's Q-value."""
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=HIDDEN):
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


class IntervalDQNAgent:
    N_TARGET_SAMPLES = 5

    def __init__(self, lr=LR, gamma=GAMMA, hidden=HIDDEN, c_train=0.5,
                 target_coverage=0.85, width_reg=0.01,
                 warmup_steps=5000, warmup_epsilon=0.8):
        self.n_actions = N_ACTIONS
        self.gamma = gamma
        self.c_train = c_train
        self.width_reg = width_reg
        self.step_count = 0

        self.q_net = IntervalQNetwork(hidden=hidden)
        self.target_net = IntervalQNetwork(hidden=hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

        # Adaptive coverage
        self.target_coverage = target_coverage
        self.t = 0.5
        self.coverage_hits = deque(maxlen=2000)

        # Warmup with ε-greedy (step-based, not episode-based)
        self.warmup_steps = warmup_steps
        self.warmup_epsilon = warmup_epsilon
        self.total_env_steps = 0

    @staticmethod
    def adaptive_c(mean_width, c_low=0.0, c_high=0.3, w_mid=4.0, k=2.0):
        """Width-dependent Hurwicz parameter: wide intervals → low c (cautious),
        narrow intervals → higher c (balanced). Sigmoid transition."""
        import math
        t = 1.0 / (1.0 + math.exp(k * (mean_width - w_mid)))
        return c_low + t * (c_high - c_low)

    def select_action(self, state, c=None, force_epsilon=None, adaptive=False):
        if force_epsilon is not None:
            eps = force_epsilon
        elif self.total_env_steps < self.warmup_steps:
            progress = self.total_env_steps / self.warmup_steps
            eps = self.warmup_epsilon * (1 - progress)
        else:
            eps = 0.0

        if random.random() < eps:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            lower, upper = self.q_net(s)

            if adaptive:
                mean_width = (upper - lower).mean().item()
                use_c = self.adaptive_c(mean_width)
            else:
                use_c = c if c is not None else self.c_train

            q = lower + use_c * (upper - lower)
            return q.argmax(dim=1).item()

    def interval_loss_sampled(self, lower, upper, target_lower, target_upper):
        N = self.N_TARGET_SAMPLES
        batch_size = lower.shape[0]

        alphas = torch.linspace(0, 1, N)
        targets = target_lower.unsqueeze(0) + alphas.unsqueeze(1) * (
            target_upper - target_lower).unsqueeze(0)

        # Track coverage against midpoint
        target_mid = (target_lower + target_upper) / 2
        inside_mid = (target_mid >= lower) & (target_mid <= upper)
        for hit in inside_mid.detach().cpu().numpy():
            self.coverage_hits.append(float(hit))

        total_loss = torch.zeros(batch_size)
        for i in range(N):
            t_i = targets[i]
            inside = (t_i >= lower) & (t_i <= upper)
            dist_to_lower = (t_i - lower) ** 2
            dist_to_upper = (t_i - upper) ** 2
            min_dist_sq = torch.min(dist_to_lower, dist_to_upper)
            max_dist_sq = torch.max(dist_to_lower, dist_to_upper)

            outside_penalty = torch.where(
                inside, torch.zeros_like(min_dist_sq), min_dist_sq)

            total_loss += self.t * outside_penalty + (1 - self.t) * max_dist_sq

        total_loss /= N

        # Width regularization when coverage exceeds target
        coverage = sum(self.coverage_hits) / max(1, len(self.coverage_hits))
        if coverage > self.target_coverage:
            width = upper - lower
            total_loss = total_loss + self.width_reg * width ** 2

        return total_loss.mean()

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

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

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Adaptive t
        if len(self.coverage_hits) >= 200:
            coverage = sum(self.coverage_hits) / len(self.coverage_hits)
            if coverage < self.target_coverage:
                self.t = min(0.95, self.t + 0.001)
            else:
                self.t = max(0.05, self.t - 0.0005)

        return loss.item()

    def get_interval_widths(self, states):
        with torch.no_grad():
            s = torch.FloatTensor(states)
            lower, upper = self.q_net(s)
            widths = (upper - lower).mean(dim=1)
            return widths.mean().item()


# ============================================================================
# ENSEMBLE DQN
# ============================================================================

class EnsembleDQNAgent:
    def __init__(self, lr=LR, gamma=GAMMA, hidden=HIDDEN,
                 n_ensemble=5, pessimism=1.0):
        self.n_actions = N_ACTIONS
        self.n_ensemble = n_ensemble
        self.gamma = gamma
        self.pessimism = pessimism
        self.step_count = 0

        self.q_nets = nn.ModuleList([
            QNetwork(hidden=hidden) for _ in range(n_ensemble)])
        self.target_nets = nn.ModuleList([
            QNetwork(hidden=hidden) for _ in range(n_ensemble)])
        for i in range(n_ensemble):
            self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())

        self.optimizers = [
            optim.Adam(self.q_nets[i].parameters(), lr=lr)
            for i in range(n_ensemble)
        ]
        self.buffer = ReplayBuffer()

    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            all_q = torch.stack([net(s) for net in self.q_nets])
            mean_q = all_q.mean(dim=0)
            std_q = all_q.std(dim=0)
            conservative_q = mean_q - self.pessimism * std_q
            return conservative_q.argmax(dim=1).item()

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        total_loss = 0.0
        for i in range(self.n_ensemble):
            q_values = self.q_nets[i](states).gather(
                1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = self.q_nets[i](next_states).argmax(dim=1)
                next_q = self.target_nets[i](next_states).gather(
                    1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + self.gamma * next_q * (1 - dones)

            loss = F.mse_loss(q_values, targets)
            self.optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_nets[i].parameters(), GRAD_CLIP)
            self.optimizers[i].step()
            total_loss += loss.item()

        self.step_count += 1
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            for i in range(self.n_ensemble):
                self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())

        return total_loss / self.n_ensemble


# ============================================================================
# ENVIRONMENT + HELPERS
# ============================================================================

# Toggled by the --noisy CLI flag at __main__. When True, make_env() wraps
# the env with NoisyLunarLanderWrapper using the same sigma/p_drop that v1
# uses, so retrained checkpoints are usable as v1's seed policy (DESIGN_v1 §11).
NOISY_OBSERVATIONS = False
_NOISY_SEED_COUNTER = [0]  # mutable closure for per-env noisy-RNG seeding


def _next_noisy_seed():
    _NOISY_SEED_COUNTER[0] += 1
    return _NOISY_SEED_COUNTER[0]


def make_env(wind_power=0.0, turbulence_power=0.0, gravity=-10.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = gym.make("LunarLander-v3",
                       wind_power=wind_power,
                       turbulence_power=turbulence_power,
                       gravity=gravity)
    if NOISY_OBSERVATIONS:
        # Late import to avoid circular dependency when v1 is not installed.
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "Invterval PP"))
        from noisy_lunarlander import (NoisyLunarLanderWrapper,
                                       DEFAULT_SIGMA, DEFAULT_DROPOUT)
        env = NoisyLunarLanderWrapper(env, sigma_per_dim=DEFAULT_SIGMA,
                                      dropout_prob=DEFAULT_DROPOUT,
                                      seed=_next_noisy_seed())
    return env


# ============================================================================
# CHECKPOINT SAVE/LOAD
# ============================================================================

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "checkpoints")


def _ckpt_suffix():
    return "_noisy" if NOISY_OBSERVATIONS else ""


def save_agents(standard_agent, interval_agent, ensemble_agent, seed, c_train):
    """Save trained agent weights."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    prefix = f"{CHECKPOINT_DIR}/seed{seed}_c{c_train}{_ckpt_suffix()}"
    torch.save(standard_agent.q_net.state_dict(), f"{prefix}_standard.pt")
    torch.save(interval_agent.q_net.state_dict(), f"{prefix}_interval.pt")
    torch.save({
        'q_nets': [net.state_dict() for net in ensemble_agent.q_nets],
    }, f"{prefix}_ensemble.pt")
    print(f"  Saved checkpoints to {CHECKPOINT_DIR}/seed{seed}_c{c_train}{_ckpt_suffix()}_*.pt")


def load_agents(seed, c_train):
    """Load trained agent weights. Returns (standard, interval, ensemble)."""
    prefix = f"{CHECKPOINT_DIR}/seed{seed}_c{c_train}{_ckpt_suffix()}"
    standard_agent = StandardDQNAgent()
    standard_agent.q_net.load_state_dict(torch.load(f"{prefix}_standard.pt",
                                                      weights_only=True))
    interval_agent = IntervalDQNAgent(c_train=c_train)
    interval_agent.q_net.load_state_dict(torch.load(f"{prefix}_interval.pt",
                                                      weights_only=True))
    ensemble_agent = EnsembleDQNAgent(n_ensemble=5)
    ckpt = torch.load(f"{prefix}_ensemble.pt", weights_only=True)
    for i, sd in enumerate(ckpt['q_nets']):
        ensemble_agent.q_nets[i].load_state_dict(sd)
    return standard_agent, interval_agent, ensemble_agent


def episode_outcome(reward):
    if reward >= 200:
        return "solved"
    elif reward <= -100:
        return "crashed"
    else:
        return "partial"


# ============================================================================
# TRAINING (per-step, inside episode)
# ============================================================================

def train_episode_standard(env, agent, epsilon):
    """Train standard/ensemble agent for one episode with per-step updates."""
    state, _ = env.reset()
    total_reward = 0
    for _ in range(1000):
        action = agent.select_action(state, epsilon=epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.push(state, action, reward, next_state, float(done))
        agent.train_step()
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward


def train_episode_interval(env, agent):
    """Train interval agent for one episode with per-step updates."""
    state, _ = env.reset()
    total_reward = 0
    for _ in range(1000):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.push(state, action, reward, next_state, float(done))
        agent.total_env_steps += 1
        agent.train_step()
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_agent(env, agent, n_episodes=30, c=None, adaptive=False):
    """Evaluate without training. Returns detailed stats.

    For IntervalDQNAgent:
      - c=<float>: fixed Hurwicz parameter
      - adaptive=True: width-dependent c (wide intervals → cautious, narrow → balanced)
    """
    rewards = []
    outcomes = {"solved": 0, "crashed": 0, "partial": 0}
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        for _ in range(1000):
            if isinstance(agent, IntervalDQNAgent):
                if adaptive:
                    action = agent.select_action(state, adaptive=True, force_epsilon=0.0)
                elif c is not None:
                    action = agent.select_action(state, c=c, force_epsilon=0.0)
                else:
                    action = agent.select_action(state, force_epsilon=0.0)
            else:
                action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        outcomes[episode_outcome(total_reward)] += 1

    return {
        "mean": np.mean(rewards),
        "std": np.std(rewards),
        "crash_rate": outcomes["crashed"] / n_episodes,
        "solve_rate": outcomes["solved"] / n_episodes,
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(seed=42, c_train=0.5, c_eval_values=None, verbose=True,
                   skip_ensemble=False, train_episodes=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if c_eval_values is None:
        c_eval_values = [0.0, 0.2, 0.5, 1.0]

    # Noisy retraining is a harder POMDP and benefits from extra episodes
    # (DESIGN_v1 §11 retraining note). CLI --episodes overrides.
    if train_episodes is None:
        TRAIN_EPISODES = 1200 if NOISY_OBSERVATIONS else 800
    else:
        TRAIN_EPISODES = int(train_episodes)
    EVAL_EVERY = 50

    # Create agents
    standard_agent = StandardDQNAgent()
    interval_agent = IntervalDQNAgent(c_train=c_train,
                                       warmup_steps=5000, warmup_epsilon=0.8,
                                       width_reg=0.01, target_coverage=0.85)
    ensemble_agent = None if skip_ensemble else EnsembleDQNAgent(n_ensemble=5, pessimism=1.0)

    # Training environments (no wind)
    env_std = make_env()
    env_int = make_env()
    env_ens = None if skip_ensemble else make_env()

    history = []

    if verbose:
        print("=" * 105)
        print(f"LUNARLANDER INTERVAL DQN (seed={seed}, c_train={c_train})")
        print("=" * 105)
        ens_str = " (ensemble SKIPPED)" if skip_ensemble else ""
        print(f"Training: {TRAIN_EPISODES} episodes, no wind, per-step updates, "
              f"Double DQN{ens_str}")
        print(f"Interval: Hurwicz c={c_train} train, eval at c={c_eval_values}")
        print()
        print("TRAINING PHASE")
        print("-" * 105)

    for ep in range(TRAIN_EPISODES):
        epsilon = max(0.01, 1.0 - ep / 400)

        train_episode_standard(env_std, standard_agent, epsilon)
        train_episode_interval(env_int, interval_agent)
        if ensemble_agent:
            train_episode_standard(env_ens, ensemble_agent, epsilon)

        if (ep + 1) % EVAL_EVERY == 0:
            env_eval = make_env()
            eval_std = evaluate_agent(env_eval, standard_agent, n_episodes=10)
            eval_int = {}
            for c_val in c_eval_values:
                eval_int[c_val] = evaluate_agent(
                    env_eval, interval_agent, n_episodes=10, c=c_val)

            sample_states = [env_eval.reset()[0] for _ in range(20)]
            width = interval_agent.get_interval_widths(np.array(sample_states))
            cvg = (sum(interval_agent.coverage_hits) /
                   max(1, len(interval_agent.coverage_hits)))
            env_eval.close()

            record = {
                "ep": ep + 1, "std": eval_std["mean"],
                "width": width, "cvg": cvg, "t": interval_agent.t,
            }
            for c in c_eval_values:
                record[f"int_c{c}"] = eval_int[c]["mean"]
            history.append(record)

            if verbose:
                c_strs = " ".join(
                    [f"c={c}:{eval_int[c]['mean']:7.1f}" for c in c_eval_values])
                print(f"  Ep {ep+1:4d} | Std:{eval_std['mean']:7.1f}  "
                      f"{c_strs} | "
                      f"W:{width:.2f} Cvg:{cvg:.2f} t:{interval_agent.t:.3f} "
                      f"eps:{epsilon:.3f}")

    env_std.close()
    env_int.close()
    if env_ens:
        env_ens.close()

    # ---- Final evaluation under perturbation conditions ----
    if verbose:
        print()
        print("=" * 105)
        print("FINAL EVALUATION (30 episodes per condition)")
        print("=" * 105)

    results = {"seed": seed, "c_train": c_train, "history": history}

    for cond_name, grav, wind_p, turb_p in EVAL_CONDITIONS:
        env_eval = make_env(wind_power=wind_p, turbulence_power=turb_p, gravity=grav)

        eval_std = evaluate_agent(env_eval, standard_agent, n_episodes=30)
        eval_int = {}
        for c_val in c_eval_values:
            eval_int[c_val] = evaluate_agent(
                env_eval, interval_agent, n_episodes=30, c=c_val)
        env_eval.close()

        results[f"{cond_name}_std"] = eval_std
        for c_val in c_eval_values:
            results[f"{cond_name}_int_c{c_val}"] = eval_int[c_val]

        if verbose:
            print(f"\n  {cond_name.upper()} (g={grav}, w={wind_p}, t={turb_p}):")
            print(f"    {'Agent':>25} | {'Reward':>8} {'±Std':>7} | "
                  f"{'Crash%':>7} {'Solve%':>7}")
            print(f"    {'-' * 63}")
            print(f"    {'Standard DQN':>25} | {eval_std['mean']:8.1f} "
                  f"{eval_std['std']:7.1f} | "
                  f"{eval_std['crash_rate']*100:6.1f}% "
                  f"{eval_std['solve_rate']*100:6.1f}%")
            for c_val in c_eval_values:
                ei = eval_int[c_val]
                print(f"    {f'Interval c={c_val}':>25} | {ei['mean']:8.1f} "
                      f"{ei['std']:7.1f} | "
                      f"{ei['crash_rate']*100:6.1f}% "
                      f"{ei['solve_rate']*100:6.1f}%")

    if verbose:
        env_diag = make_env()
        sample_states = [env_diag.reset()[0] for _ in range(50)]
        width = interval_agent.get_interval_widths(np.array(sample_states))
        cvg = (sum(interval_agent.coverage_hits) /
               max(1, len(interval_agent.coverage_hits)))
        env_diag.close()
        print(f"\n  Interval diagnostics: width={width:.3f}, cvg={cvg:.2f}, "
              f"t={interval_agent.t:.3f}")

    # Save checkpoints (create dummy ensemble if skipped)
    if ensemble_agent is None:
        ensemble_agent = EnsembleDQNAgent(n_ensemble=5)
    save_agents(standard_agent, interval_agent, ensemble_agent, seed, c_train)

    return results


# ============================================================================
# EVAL-ONLY MODE — load checkpoints, evaluate under many perturbations
# ============================================================================

EVAL_CONDITIONS = [
    # (name, gravity, wind_power, turbulence_power)
    ("normal",           -10.0,  0,  0.0),
    ("wind_10",          -10.0, 10,  1.0),
    ("wind_20",          -10.0, 20,  2.0),
    ("wind_30",          -10.0, 30,  3.0),
    ("wind_40",          -10.0, 40,  4.0),
    ("wind_50",          -10.0, 50,  5.0),
    ("low_grav",          -3.0,  0,  0.0),
    ("low_grav_wind",     -3.0, 20,  2.0),
    ("high_grav",        -11.9,  0,  0.0),
    ("high_grav_wind",   -11.9, 20,  2.0),
    ("combo_extreme",    -11.9, 40,  4.0),
]


def run_eval_only(seed=42, c_train=0.5, c_eval_values=None, n_episodes=30):
    """Load trained agents and evaluate under many perturbation conditions."""
    if c_eval_values is None:
        c_eval_values = [0.0, 0.2, 0.5]

    print(f"Loading checkpoints for seed={seed}, c_train={c_train}...")
    standard_agent, interval_agent, ensemble_agent = load_agents(seed, c_train)
    standard_agent.q_net.eval()
    interval_agent.q_net.eval()
    for net in ensemble_agent.q_nets:
        net.eval()
    print("Loaded.\n")

    print("=" * 110)
    print(f"PERTURBATION EVALUATION (seed={seed}, {n_episodes} episodes per condition)")
    print("=" * 110)

    all_results = {}

    for cond_name, grav, wind_p, turb_p in EVAL_CONDITIONS:
        env = make_env(wind_power=wind_p, turbulence_power=turb_p, gravity=grav)

        eval_std = evaluate_agent(env, standard_agent, n_episodes=n_episodes)
        eval_int = {}
        for c_val in c_eval_values:
            eval_int[c_val] = evaluate_agent(
                env, interval_agent, n_episodes=n_episodes, c=c_val)
        eval_ens = evaluate_agent(env, ensemble_agent, n_episodes=n_episodes)
        env.close()

        all_results[cond_name] = {
            "std": eval_std, "ens": eval_ens,
            **{f"int_c{c}": eval_int[c] for c in c_eval_values}
        }

        print(f"\n  {cond_name.upper()} (g={grav}, w={wind_p}, t={turb_p}):")
        print(f"    {'Agent':>25} | {'Reward':>8} {'±Std':>7} | "
              f"{'Crash%':>7} {'Solve%':>7}")
        print(f"    {'-' * 63}")
        print(f"    {'Standard DQN':>25} | {eval_std['mean']:8.1f} "
              f"{eval_std['std']:7.1f} | "
              f"{eval_std['crash_rate']*100:6.1f}% "
              f"{eval_std['solve_rate']*100:6.1f}%")
        for c_val in c_eval_values:
            ei = eval_int[c_val]
            print(f"    {f'Interval c={c_val}':>25} | {ei['mean']:8.1f} "
                  f"{ei['std']:7.1f} | "
                  f"{ei['crash_rate']*100:6.1f}% "
                  f"{ei['solve_rate']*100:6.1f}%")
        print(f"    {'Ensemble (N=5)':>25} | {eval_ens['mean']:8.1f} "
              f"{eval_ens['std']:7.1f} | "
              f"{eval_ens['crash_rate']*100:6.1f}% "
              f"{eval_ens['solve_rate']*100:6.1f}%")

    return all_results


def run_multi_seed(seeds=None, c_train=0.5, c_eval_values=None):
    if seeds is None:
        seeds = [42, 123, 456]
    if c_eval_values is None:
        c_eval_values = [0.0, 0.2, 0.5, 1.0]

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'#' * 105}")
        print(f"# SEED {seed} ({i+1}/{len(seeds)})")
        print(f"{'#' * 105}")
        res = run_experiment(seed=seed, c_train=c_train,
                            c_eval_values=c_eval_values, verbose=True)
        all_results.append(res)

    valid = [r for r in all_results if "no_wind_std" in r]
    if not valid:
        print("No valid results!")
        return

    # Aggregate
    print(f"\n{'=' * 105}")
    print(f"AGGREGATE ACROSS {len(valid)} SEEDS (c_train={c_train})")
    print(f"{'=' * 105}")

    WIND_CONDS = [
        ("no_wind", "No Wind"),
        ("moderate", "Moderate Wind (10,1)"),
        ("strong", "Strong Wind (20,2)"),
    ]

    for cond_key, cond_label in WIND_CONDS:
        print(f"\n  {cond_label}:")
        print(f"    {'Agent':>25} | {'Reward':>10} | {'Crash%':>10} | {'Solve%':>10}")
        print(f"    {'-' * 63}")

        for agent_label, key_fn in [
            ("Standard DQN", lambda r: r[f"{cond_key}_std"]),
        ] + [
            (f"Interval c={c}", lambda r, c=c: r[f"{cond_key}_int_c{c}"])
            for c in c_eval_values
        ] + [
            ("Ensemble (N=5)", lambda r: r[f"{cond_key}_ens"]),
        ]:
            rewards = [key_fn(r)["mean"] for r in valid]
            crashes = [key_fn(r)["crash_rate"] * 100 for r in valid]
            solves = [key_fn(r)["solve_rate"] * 100 for r in valid]
            print(f"    {agent_label:>25} | "
                  f"{np.mean(rewards):6.1f}±{np.std(rewards):4.1f} | "
                  f"{np.mean(crashes):5.1f}±{np.std(crashes):4.1f}% | "
                  f"{np.mean(solves):5.1f}±{np.std(solves):4.1f}%")

    # Key comparison
    print(f"\n  KEY METRIC — Crash rate under strong wind:")
    std_crashes = [r["strong_std"]["crash_rate"] * 100 for r in valid]
    print(f"    Standard DQN:       {np.mean(std_crashes):5.1f}% ± {np.std(std_crashes):.1f}%")

    for c_val in c_eval_values:
        int_crashes = [r[f"strong_int_c{c_val}"]["crash_rate"] * 100 for r in valid]
        reduction = np.mean(std_crashes) - np.mean(int_crashes)
        print(f"    Interval c={c_val}:      {np.mean(int_crashes):5.1f}% ± "
              f"{np.std(int_crashes):.1f}%  "
              f"({'↓' if reduction > 0 else '↑'}{abs(reduction):.1f}pp vs std)")

    ens_crashes = [r["strong_ens"]["crash_rate"] * 100 for r in valid]
    reduction = np.mean(std_crashes) - np.mean(ens_crashes)
    print(f"    Ensemble (N=5):     {np.mean(ens_crashes):5.1f}% ± "
          f"{np.std(ens_crashes):.1f}%  "
          f"({'↓' if reduction > 0 else '↑'}{abs(reduction):.1f}pp vs std)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load saved checkpoints and evaluate under perturbations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--c_train", type=float, default=0.5)
    parser.add_argument("--c_eval", type=float, nargs="+", default=None)
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Skip ensemble training (faster)")
    parser.add_argument("--n_eval", type=int, default=30,
                        help="Episodes per eval condition (default: 30)")
    parser.add_argument("--noisy", action="store_true",
                        help="Wrap envs with NoisyLunarLanderWrapper (DESIGN_v1 §11). "
                             "Saves checkpoints with _noisy suffix.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override training episode budget (default: 800 / 1200 noisy).")
    args = parser.parse_args()

    if args.noisy:
        globals()["NOISY_OBSERVATIONS"] = True

    if args.eval_only:
        run_eval_only(seed=args.seed, c_train=args.c_train,
                     c_eval_values=args.c_eval, n_episodes=args.n_eval)
    elif args.single:
        run_experiment(seed=args.seed, c_train=args.c_train,
                      c_eval_values=args.c_eval, verbose=True,
                      skip_ensemble=args.no_ensemble,
                      train_episodes=args.episodes)
    else:
        run_multi_seed(seeds=args.seeds, c_train=args.c_train,
                      c_eval_values=args.c_eval)
