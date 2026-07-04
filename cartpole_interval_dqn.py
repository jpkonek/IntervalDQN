"""
Interval DQN vs Standard DQN vs Ensemble DQN on CartPole with Dynamics Shift
=============================================================================

Proof of concept: credal Q-networks for robust RL.

Setup:
  Phase 1 (episodes 1-300):   Standard CartPole (gravity=9.8)
  Phase 2 (episodes 301-500): Shifted CartPole (gravity=15.0, +53%)

Three agents trained and evaluated:
  1. Standard DQN        — single network, point Q-values, ε-greedy exploration
  2. Interval DQN        — single network, interval Q-values [lower, upper]
                           Training: Hurwicz with c_train (optimistic → explore)
                           Evaluation: Hurwicz with c_eval (pessimistic → safe)
                           No ε-greedy needed — intervals drive exploration directly
  3. Ensemble DQN (N=5)  — N networks, uses mean - λ*std for conservative action

Key design choices:
  1. Sampled interval targets — instead of collapsing the next-state interval to
     a midpoint, we sample N points across [r + γ·L_next, r + γ·U_next] and
     average the interval loss. This propagates width through Bellman backup.
  2. Hurwicz for BOTH data collection AND evaluation — c_train (high) drives
     optimistic exploration; c_eval (low) drives pessimistic safety.
  3. Single model, multiple behaviours — same network evaluated at different c.

Key question: Under dynamics shift, does the interval DQN degrade less than
standard DQN, and comparably to the ensemble at 1/5th inference cost?

Usage:
    python cartpole_interval_dqn.py
    python cartpole_interval_dqn.py --single --seed 42
    python cartpole_interval_dqn.py --c_train 0.8 --c_eval 0.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
import sys
from collections import deque

sys.stdout.reconfigure(line_buffering=True)


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# STANDARD DQN
# ============================================================================

class StandardQNetwork(nn.Module):
    def __init__(self, state_dim=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class StandardDQNAgent:
    def __init__(self, state_dim=4, n_actions=2, lr=1e-3, gamma=0.99,
                 tau=0.005, hidden=128):
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau

        self.q_net = StandardQNetwork(state_dim, n_actions, hidden)
        self.target_net = StandardQNetwork(state_dim, n_actions, hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state).unsqueeze(0))
            return q.argmax(dim=1).item()

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return loss.item()


# ============================================================================
# INTERVAL DQN — Hurwicz exploration + sampled interval targets
# ============================================================================

class IntervalQNetwork(nn.Module):
    """Outputs interval [lower, upper] for each action's Q-value."""
    def __init__(self, state_dim=4, n_actions=2, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
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
    # Number of points to sample across the target interval
    N_TARGET_SAMPLES = 5

    def __init__(self, state_dim=4, n_actions=2, lr=1e-3, gamma=0.99,
                 tau=0.005, hidden=128, c_train=1.0, target_coverage=0.85,
                 warmup_episodes=20, warmup_epsilon=0.5):
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.c_train = c_train  # Hurwicz c for data collection (high = optimistic)

        self.q_net = IntervalQNetwork(state_dim, n_actions, hidden)
        self.target_net = IntervalQNetwork(state_dim, n_actions, hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

        # Adaptive coverage tracking
        self.target_coverage = target_coverage
        self.t = 0.5
        self.coverage_hits = deque(maxlen=1000)

        # Small warmup with ε-greedy before intervals are calibrated
        self.warmup_episodes = warmup_episodes
        self.warmup_epsilon = warmup_epsilon
        self.episodes_seen = 0

    def select_action(self, state, c=None, force_epsilon=None):
        """
        Select action using Hurwicz criterion.
        During warmup, mix with ε-greedy to bootstrap interval calibration.
        """
        # Warmup: small amount of random exploration
        if force_epsilon is not None:
            eps = force_epsilon
        elif self.episodes_seen < self.warmup_episodes:
            # Linear decay from warmup_epsilon to 0 over warmup period
            progress = self.episodes_seen / self.warmup_episodes
            eps = self.warmup_epsilon * (1 - progress)
        else:
            eps = 0.0

        if random.random() < eps:
            return random.randint(0, self.n_actions - 1)

        use_c = c if c is not None else self.c_train
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            lower, upper = self.q_net(s)
            q = lower + use_c * (upper - lower)
            return q.argmax(dim=1).item()

    def interval_loss_sampled(self, lower, upper, target_lower, target_upper):
        """
        Interval loss with sampled targets across [target_lower, target_upper].

        Instead of collapsing the target interval to a single point, we sample
        N points uniformly across it and average the interval loss. This
        propagates width backwards through the Bellman chain.

        Coverage is tracked against the midpoint for adaptive t stability.
        """
        N = self.N_TARGET_SAMPLES
        batch_size = lower.shape[0]

        # Sample N points uniformly across the target interval
        # alphas: (N,) values in [0, 1]
        alphas = torch.linspace(0, 1, N)
        # targets: (N, batch_size)
        targets = target_lower.unsqueeze(0) + alphas.unsqueeze(1) * (
            target_upper - target_lower).unsqueeze(0)

        # Track coverage against midpoint (for adaptive t stability)
        target_mid = (target_lower + target_upper) / 2
        inside_mid = (target_mid >= lower) & (target_mid <= upper)
        for hit in inside_mid.detach().cpu().numpy():
            self.coverage_hits.append(float(hit))

        # Compute interval loss for each sampled target and average
        total_loss = torch.zeros(batch_size)
        for i in range(N):
            t_i = targets[i]  # (batch_size,)

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
            total_loss = total_loss + 0.1 * width ** 2

        return total_loss.mean()

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current interval Q-values for chosen actions
        lower, upper = self.q_net(states)
        lower_a = lower.gather(1, actions.unsqueeze(1)).squeeze(1)
        upper_a = upper.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: use target network's intervals
        with torch.no_grad():
            next_lower, next_upper = self.target_net(next_states)

            # Action selection for target: use c_train (same policy as data collection)
            next_q = next_lower + self.c_train * (next_upper - next_lower)
            next_actions = next_q.argmax(dim=1)

            # Get target intervals for selected actions
            tgt_lower = next_lower.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            tgt_upper = next_upper.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Bellman interval targets
            target_l = rewards + self.gamma * tgt_lower * (1 - dones)
            target_u = rewards + self.gamma * tgt_upper * (1 - dones)

        # Sampled interval loss
        loss = self.interval_loss_sampled(lower_a, upper_a, target_l, target_u)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Soft update target network
        for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # Adaptive t
        if len(self.coverage_hits) >= 100:
            coverage = sum(self.coverage_hits) / len(self.coverage_hits)
            if coverage < self.target_coverage:
                self.t = min(0.95, self.t + 0.002)
            else:
                self.t = max(0.05, self.t - 0.001)

        return loss.item()

    def get_interval_widths(self, states):
        """Get mean interval width for a batch of states."""
        with torch.no_grad():
            s = torch.FloatTensor(states)
            lower, upper = self.q_net(s)
            widths = (upper - lower).mean(dim=1)
            return widths.mean().item()


# ============================================================================
# ENSEMBLE DQN
# ============================================================================

class EnsembleDQNAgent:
    """Ensemble of N DQN networks for uncertainty estimation."""
    def __init__(self, state_dim=4, n_actions=2, lr=1e-3, gamma=0.99,
                 tau=0.005, hidden=128, n_ensemble=5, pessimism=1.0):
        self.n_actions = n_actions
        self.n_ensemble = n_ensemble
        self.gamma = gamma
        self.tau = tau
        self.pessimism = pessimism

        self.q_nets = nn.ModuleList([
            StandardQNetwork(state_dim, n_actions, hidden)
            for _ in range(n_ensemble)
        ])
        self.target_nets = nn.ModuleList([
            StandardQNetwork(state_dim, n_actions, hidden)
            for _ in range(n_ensemble)
        ])
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

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        total_loss = 0.0
        for i in range(self.n_ensemble):
            q_values = self.q_nets[i](states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self.target_nets[i](next_states).max(dim=1)[0]
                targets = rewards + self.gamma * next_q * (1 - dones)

            loss = F.mse_loss(q_values, targets)
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()

            for p, tp in zip(self.q_nets[i].parameters(), self.target_nets[i].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            total_loss += loss.item()

        return total_loss / self.n_ensemble

    def get_ensemble_std(self, states):
        with torch.no_grad():
            s = torch.FloatTensor(states)
            all_q = torch.stack([net(s) for net in self.q_nets])
            return all_q.std(dim=0).mean().item()


# ============================================================================
# ENVIRONMENT
# ============================================================================

def make_cartpole(gravity=9.8):
    env = gym.make("CartPole-v1")
    env.unwrapped.gravity = gravity
    return env


# ============================================================================
# TRAINING + EVALUATION
# ============================================================================

def run_episode_standard(env, agent, epsilon=0.0, train=True, max_steps=500):
    """Run one episode for standard or ensemble agent (ε-greedy)."""
    state, _ = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        action = agent.select_action(state, epsilon=epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if train:
            agent.buffer.push(state, action, reward, next_state, float(done))

        state = next_state
        total_reward += reward
        if done:
            break

    return total_reward


def run_episode_interval(env, agent, train=True, max_steps=500, c=None):
    """Run one episode for interval agent (Hurwicz exploration, no ε-greedy)."""
    state, _ = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        action = agent.select_action(state, c=c)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if train:
            agent.buffer.push(state, action, reward, next_state, float(done))

        state = next_state
        total_reward += reward
        if done:
            break

    return total_reward


def evaluate_agent(env, agent, n_episodes=20, c=None):
    """Evaluate without exploration."""
    rewards = []
    for _ in range(n_episodes):
        if c is not None and isinstance(agent, IntervalDQNAgent):
            r = run_episode_interval(env, agent, train=False, c=c)
        else:
            r = run_episode_standard(env, agent, epsilon=0.0, train=False)
        rewards.append(r)
    return np.mean(rewards), np.std(rewards)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(seed=42, c_train=1.0, c_eval_values=None, verbose=True):
    """
    Run the full experiment.

    Args:
        c_train: Hurwicz c for data collection (high = optimistic exploration)
        c_eval_values: list of c values to evaluate at (default: [0.0, 0.2, 0.5, 1.0])
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if c_eval_values is None:
        c_eval_values = [0.0, 0.2, 0.5, 1.0]

    # Hyperparameters
    PHASE1_EPISODES = 300
    PHASE2_EPISODES = 200
    TOTAL_EPISODES = PHASE1_EPISODES + PHASE2_EPISODES
    EVAL_EVERY = 10
    TRAIN_STEPS_PER_EP = 10
    BATCH_SIZE = 64
    GRAVITY_NORMAL = 9.8
    GRAVITY_SHIFTED = 15.0

    # Create agents
    standard_agent = StandardDQNAgent(lr=1e-3)
    interval_agent = IntervalDQNAgent(lr=1e-3, c_train=c_train,
                                       warmup_episodes=30, warmup_epsilon=0.3)
    ensemble_agent = EnsembleDQNAgent(lr=1e-3, n_ensemble=5, pessimism=1.0)

    # Separate environments
    env_std = make_cartpole(GRAVITY_NORMAL)
    env_int = make_cartpole(GRAVITY_NORMAL)
    env_ens = make_cartpole(GRAVITY_NORMAL)
    env_eval = make_cartpole(GRAVITY_NORMAL)

    history = []

    if verbose:
        print("=" * 95)
        print(f"CARTPOLE INTERVAL DQN (seed={seed}, c_train={c_train})")
        print("=" * 95)
        print(f"Phase 1 (ep 1-{PHASE1_EPISODES}):   gravity={GRAVITY_NORMAL}")
        print(f"Phase 2 (ep {PHASE1_EPISODES+1}-{TOTAL_EPISODES}): "
              f"gravity={GRAVITY_SHIFTED} (+{(GRAVITY_SHIFTED/GRAVITY_NORMAL-1)*100:.0f}%)")
        print(f"Interval agent: Hurwicz c={c_train} for exploration, "
              f"eval at c={c_eval_values}")
        print(f"Standard/Ensemble: ε-greedy exploration")
        print(f"Targets: sampled across next-state interval "
              f"({IntervalDQNAgent.N_TARGET_SAMPLES} points)")
        print()

        # Header
        c_headers = " ".join([f"{'c='+str(c):>7}" for c in c_eval_values])
        print(f"{'Ep':>4} {'Phase':>7} | {'Std':>7} {c_headers} {'Ens':>7} | "
              f"{'Width':>6} {'Cvg':>5} {'t':>5}")
        print("-" * (55 + 8 * len(c_eval_values)))

    for ep in range(TOTAL_EPISODES):
        # ---- Dynamics shift ----
        if ep == PHASE1_EPISODES:
            if verbose:
                print(f"\n{'!' * 80}")
                print(f"  DYNAMICS SHIFT: gravity {GRAVITY_NORMAL} → {GRAVITY_SHIFTED}")
                print(f"{'!' * 80}\n")
            for e in [env_std, env_int, env_ens, env_eval]:
                e.unwrapped.gravity = GRAVITY_SHIFTED

        # ---- Epsilon for standard/ensemble (same schedule) ----
        if ep < PHASE1_EPISODES:
            progress = ep / PHASE1_EPISODES
        else:
            # Re-explore after shift
            ep_in_phase2 = ep - PHASE1_EPISODES
            progress = ep_in_phase2 / PHASE2_EPISODES
        epsilon = max(0.05, 1.0 - progress * 2.0)  # faster decay

        # ---- Collect episodes ----
        run_episode_standard(env_std, standard_agent, epsilon=epsilon, train=True)
        # Interval agent: Hurwicz exploration (c_train), no ε-greedy
        run_episode_interval(env_int, interval_agent, train=True)
        interval_agent.episodes_seen += 1
        run_episode_standard(env_ens, ensemble_agent, epsilon=epsilon, train=True)

        # ---- Train ----
        for _ in range(TRAIN_STEPS_PER_EP):
            standard_agent.train_step(BATCH_SIZE)
            interval_agent.train_step(BATCH_SIZE)
            ensemble_agent.train_step(BATCH_SIZE)

        # ---- Evaluate periodically ----
        if (ep + 1) % EVAL_EVERY == 0:
            eval_std, _ = evaluate_agent(env_eval, standard_agent, n_episodes=10)

            eval_int = {}
            for c_val in c_eval_values:
                ev, _ = evaluate_agent(env_eval, interval_agent, n_episodes=10, c=c_val)
                eval_int[c_val] = ev

            eval_ens, _ = evaluate_agent(env_eval, ensemble_agent, n_episodes=10)

            # Measure interval widths
            sample_states = [env_eval.reset()[0] for _ in range(20)]
            width = interval_agent.get_interval_widths(np.array(sample_states))
            coverage = (sum(interval_agent.coverage_hits) /
                       max(1, len(interval_agent.coverage_hits)))

            phase = "normal" if ep < PHASE1_EPISODES else "shifted"
            record = {
                "episode": ep + 1,
                "phase": phase,
                "eval_std": eval_std,
                "eval_ens": eval_ens,
                "width": width,
                "coverage": coverage,
                "t": interval_agent.t,
                "epsilon": epsilon,
            }
            for c_val in c_eval_values:
                record[f"eval_int_c{c_val}"] = eval_int[c_val]
            history.append(record)

            if verbose:
                c_vals_str = " ".join([f"{eval_int[c]:7.1f}" for c in c_eval_values])
                print(f"{ep+1:4d} {phase:>7} | "
                      f"{eval_std:7.1f} {c_vals_str} {eval_ens:7.1f} | "
                      f"{width:6.2f} {coverage:5.2f} {interval_agent.t:5.3f}")

    # ---- Cleanup ----
    for e in [env_std, env_int, env_ens, env_eval]:
        e.close()

    # ---- Summary ----
    phase1 = [h for h in history if h["phase"] == "normal"]
    phase2 = [h for h in history if h["phase"] == "shifted"]
    results = {"seed": seed, "c_train": c_train, "history": history}

    if phase1 and phase2:
        p1_late = phase1[-5:]
        p2_early = phase2[:5]
        p2_late = phase2[-5:]

        def avg(records, key):
            return np.mean([h[key] for h in records])

        results["p1_std"] = avg(p1_late, "eval_std")
        results["p1_ens"] = avg(p1_late, "eval_ens")
        results["p2e_std"] = avg(p2_early, "eval_std")
        results["p2e_ens"] = avg(p2_early, "eval_ens")
        results["p2l_std"] = avg(p2_late, "eval_std")
        results["p2l_ens"] = avg(p2_late, "eval_ens")

        for c_val in c_eval_values:
            key = f"eval_int_c{c_val}"
            results[f"p1_int_c{c_val}"] = avg(p1_late, key)
            results[f"p2e_int_c{c_val}"] = avg(p2_early, key)
            results[f"p2l_int_c{c_val}"] = avg(p2_late, key)

        results["p1_width"] = avg(p1_late, "width")
        results["p2e_width"] = avg(p2_early, "width")
        results["p2l_width"] = avg(p2_late, "width")

        if verbose:
            print(f"\n{'=' * 80}")
            print("SUMMARY")
            print(f"{'=' * 80}")

            print(f"\n  Phase 1 (normal, last 5 evals):")
            print(f"    Standard DQN:    {results['p1_std']:7.1f}")
            for c_val in c_eval_values:
                print(f"    Interval c={c_val}:   {results[f'p1_int_c{c_val}']:7.1f}")
            print(f"    Ensemble (N=5):  {results['p1_ens']:7.1f}")
            print(f"    Interval width:  {results['p1_width']:.3f}")

            print(f"\n  Phase 2 (shifted, first 5 evals — immediate impact):")
            drop_std = results['p1_std'] - results['p2e_std']
            print(f"    Standard DQN:    {results['p2e_std']:7.1f}  (drop: {drop_std:+.1f})")
            for c_val in c_eval_values:
                drop = results[f'p1_int_c{c_val}'] - results[f'p2e_int_c{c_val}']
                print(f"    Interval c={c_val}:   {results[f'p2e_int_c{c_val}']:7.1f}  "
                      f"(drop: {drop:+.1f})")
            drop_ens = results['p1_ens'] - results['p2e_ens']
            print(f"    Ensemble (N=5):  {results['p2e_ens']:7.1f}  (drop: {drop_ens:+.1f})")
            print(f"    Interval width:  {results['p2e_width']:.3f}")

            print(f"\n  Phase 2 (shifted, last 5 evals — after adaptation):")
            print(f"    Standard DQN:    {results['p2l_std']:7.1f}")
            for c_val in c_eval_values:
                print(f"    Interval c={c_val}:   {results[f'p2l_int_c{c_val}']:7.1f}")
            print(f"    Ensemble (N=5):  {results['p2l_ens']:7.1f}")
            print(f"    Interval width:  {results['p2l_width']:.3f}")

            wr = results["p2e_width"] / max(results["p1_width"], 1e-6)
            print(f"\n  Width ratio (post-shift / pre-shift): {wr:.2f}x")

    return results


def run_multi_seed(seeds=None, c_train=1.0, c_eval_values=None):
    if seeds is None:
        seeds = [42, 123, 456]
    if c_eval_values is None:
        c_eval_values = [0.0, 0.2, 0.5, 1.0]

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'#' * 80}")
        print(f"# SEED {seed} ({i+1}/{len(seeds)})")
        print(f"{'#' * 80}")
        res = run_experiment(seed=seed, c_train=c_train,
                            c_eval_values=c_eval_values, verbose=True)
        all_results.append(res)

    valid = [r for r in all_results if "p1_std" in r]
    if not valid:
        print("No valid results!")
        return

    print(f"\n{'=' * 80}")
    print(f"AGGREGATE ACROSS {len(valid)} SEEDS (c_train={c_train})")
    print(f"{'=' * 80}")

    def report(label, key):
        vals = [r[key] for r in valid]
        print(f"  {label:>30}: {np.mean(vals):7.1f} ± {np.std(vals):5.1f}")

    for phase_label, prefix in [("Phase 1 (normal)", "p1"),
                                 ("Phase 2 early (shifted)", "p2e"),
                                 ("Phase 2 late (adapted)", "p2l")]:
        print(f"\n  {phase_label}:")
        report("Standard DQN", f"{prefix}_std")
        for c_val in c_eval_values:
            report(f"Interval c={c_val}", f"{prefix}_int_c{c_val}")
        report("Ensemble (N=5)", f"{prefix}_ens")

    print(f"\n  Interval width:")
    for label, key in [("Phase 1", "p1_width"), ("Phase 2 early", "p2e_width"),
                        ("Phase 2 late", "p2l_width")]:
        vals = [r[key] for r in valid]
        print(f"    {label:>20}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    ratios = [r["p2e_width"] / max(r["p1_width"], 1e-6) for r in valid]
    print(f"    {'Width ratio (post/pre)':>20}: {np.mean(ratios):.2f}x ± {np.std(ratios):.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true",
                        help="Run single seed with full output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--c_train", type=float, default=1.0,
                        help="Hurwicz c for data collection (default: 1.0 = optimistic)")
    parser.add_argument("--c_eval", type=float, nargs="+", default=None,
                        help="Hurwicz c values for evaluation (default: 0.0 0.2 0.5 1.0)")
    args = parser.parse_args()

    if args.single:
        run_experiment(seed=args.seed, c_train=args.c_train,
                      c_eval_values=args.c_eval, verbose=True)
    else:
        run_multi_seed(seeds=args.seeds, c_train=args.c_train,
                      c_eval_values=args.c_eval)
