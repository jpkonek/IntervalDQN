"""
Interval DQN vs Standard DQN vs Ensemble DQN on MountainCar
============================================================

Tests whether interval-valued Q-networks enable principled exploration
in sparse-reward environments via the Hurwicz criterion.

Why MountainCar:
  - Sparse reward: -1 per step, 0 at goal. Random policies never reach the flag.
  - Standard DQN with ε-greedy famously struggles: learns "everything is equally
    bad" because it rarely sees the goal reward.
  - Interval DQN with Hurwicz c=1 (optimistic) should preferentially select
    actions with wide intervals (uncertain/unexplored), driving directed
    exploration toward the flag.
  - This tests the EXPLORATION benefit of intervals, not just decision-time caution.

Design:
  1. Interval DQN: Hurwicz c_train for data collection, eval at multiple c values.
     Sampled interval targets propagate width through Bellman backup.
     No ε-greedy — intervals drive exploration directly.
  2. Standard DQN: ε-greedy exploration (the known-hard baseline).
  3. Ensemble DQN (N=5): UCB-style optimistic exploration (mean + λ*std)
     for fair comparison — this is what intervals should match or beat.

Key metrics:
  - Episodes to first solve (reach flag)
  - Fraction of episodes solved over training
  - Final evaluation reward (higher = faster to flag = fewer steps)

Optional Phase 2: dynamics shift (reduce car power) after learning.

Usage:
    python mountaincar_interval_dqn.py
    python mountaincar_interval_dqn.py --single --seed 42
    python mountaincar_interval_dqn.py --c_train 0.8
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
    def __init__(self, capacity=100000):
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
# STANDARD DQN
# ============================================================================

class StandardQNetwork(nn.Module):
    def __init__(self, state_dim=2, n_actions=3, hidden=128):
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
    def __init__(self, state_dim=2, n_actions=3, lr=1e-3, gamma=0.99,
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
# INTERVAL DQN
# ============================================================================

class IntervalQNetwork(nn.Module):
    """Outputs interval [lower, upper] for each action's Q-value.

    Option A: Wide initialisation. The delta_raw bias is set so that
    softplus(delta_raw) ≈ init_width at the start of training. This means
    intervals begin wide (uncertain about everything) and narrow only where
    training data provides consistent Bellman targets. States/actions that
    are never visited retain wide intervals, which Hurwicz c=1 will
    preferentially select — driving exploration toward the unknown.
    """
    def __init__(self, state_dim=2, n_actions=3, hidden=128, init_width=50.0):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_actions * 2)
        self.n_actions = n_actions

        # Initialise delta_raw bias so softplus(bias) ≈ init_width
        # softplus(x) ≈ x for large x, so bias ≈ init_width
        # Only set bias for the delta_raw outputs (odd indices)
        with torch.no_grad():
            bias = self.head.bias.view(n_actions, 2)
            bias[:, 1] = init_width  # delta_raw bias → wide intervals
            # Also zero out the head weights for delta columns to reduce
            # input-dependence initially — let bias dominate
            weight = self.head.weight.view(n_actions, 2, hidden)
            weight[:, 1, :] *= 0.1  # small weights for delta, bias-dominated

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

    def __init__(self, state_dim=2, n_actions=3, lr=1e-3, gamma=0.99,
                 tau=0.005, hidden=128, c_train=1.0, target_coverage=0.85,
                 warmup_episodes=50, warmup_epsilon=0.3, init_width=50.0):
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.c_train = c_train

        self.q_net = IntervalQNetwork(state_dim, n_actions, hidden,
                                       init_width=init_width)
        self.target_net = IntervalQNetwork(state_dim, n_actions, hidden,
                                            init_width=init_width)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

        # Adaptive coverage
        self.target_coverage = target_coverage
        self.t = 0.5
        self.coverage_hits = deque(maxlen=2000)

        # Warmup
        self.warmup_episodes = warmup_episodes
        self.warmup_epsilon = warmup_epsilon
        self.episodes_seen = 0

    def select_action(self, state, c=None):
        """Select action using Hurwicz criterion."""
        # Warmup: small ε-greedy to bootstrap
        if self.episodes_seen < self.warmup_episodes:
            progress = self.episodes_seen / self.warmup_episodes
            eps = self.warmup_epsilon * (1 - progress)
            if random.random() < eps:
                return random.randint(0, self.n_actions - 1)

        use_c = c if c is not None else self.c_train
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            lower, upper = self.q_net(s)
            q = lower + use_c * (upper - lower)
            return q.argmax(dim=1).item()

    def interval_loss_sampled(self, lower, upper, target_lower, target_upper):
        """Interval loss with sampled targets across [target_lower, target_upper]."""
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
        # Gentle — we want width to persist in unvisited regions
        coverage = sum(self.coverage_hits) / max(1, len(self.coverage_hits))
        if coverage > self.target_coverage:
            width = upper - lower
            total_loss = total_loss + 0.005 * width ** 2

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

        lower, upper = self.q_net(states)
        lower_a = lower.gather(1, actions.unsqueeze(1)).squeeze(1)
        upper_a = upper.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_lower, next_upper = self.target_net(next_states)
            next_q = next_lower + self.c_train * (next_upper - next_lower)
            next_actions = next_q.argmax(dim=1)

            tgt_lower = next_lower.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            tgt_upper = next_upper.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_l = rewards + self.gamma * tgt_lower * (1 - dones)
            target_u = rewards + self.gamma * tgt_upper * (1 - dones)

        loss = self.interval_loss_sampled(lower_a, upper_a, target_l, target_u)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # Adaptive t
        if len(self.coverage_hits) >= 200:
            coverage = sum(self.coverage_hits) / len(self.coverage_hits)
            if coverage < self.target_coverage:
                self.t = min(0.95, self.t + 0.002)
            else:
                self.t = max(0.05, self.t - 0.001)

        return loss.item()

    def get_interval_widths(self, states):
        with torch.no_grad():
            s = torch.FloatTensor(states)
            lower, upper = self.q_net(s)
            widths = (upper - lower).mean(dim=1)
            return widths.mean().item()

    def get_interval_detail(self, state):
        """Get per-action interval detail for a single state."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            lower, upper = self.q_net(s)
            return lower.squeeze(0).numpy(), upper.squeeze(0).numpy()


# ============================================================================
# ENSEMBLE DQN — with optimistic exploration for fair comparison
# ============================================================================

class EnsembleDQNAgent:
    """Ensemble of N DQN networks with UCB-style optimistic exploration."""
    def __init__(self, state_dim=2, n_actions=3, lr=1e-3, gamma=0.99,
                 tau=0.005, hidden=128, n_ensemble=5,
                 optimism_train=1.0, pessimism_eval=1.0):
        self.n_actions = n_actions
        self.n_ensemble = n_ensemble
        self.gamma = gamma
        self.tau = tau
        self.optimism_train = optimism_train  # mean + λ*std for exploration
        self.pessimism_eval = pessimism_eval  # mean - λ*std for evaluation

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

    def select_action(self, state, mode="train"):
        """
        mode="train": optimistic (mean + λ*std) for exploration
        mode="eval":  pessimistic (mean - λ*std) for safety
        mode="greedy": mean Q-value
        """
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            all_q = torch.stack([net(s) for net in self.q_nets])
            mean_q = all_q.mean(dim=0)
            std_q = all_q.std(dim=0)

            if mode == "train":
                q = mean_q + self.optimism_train * std_q
            elif mode == "eval":
                q = mean_q - self.pessimism_eval * std_q
            else:
                q = mean_q
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
# EPISODE RUNNERS
# ============================================================================

def run_episode(env, agent, mode="train", c=None, epsilon=None, max_steps=200):
    """
    Run one episode.
    For interval agent: uses c (Hurwicz parameter)
    For standard agent: uses epsilon (ε-greedy)
    For ensemble agent: uses mode ("train"=optimistic, "eval"=pessimistic)
    """
    state, _ = env.reset()
    total_reward = 0
    solved = False

    for step in range(max_steps):
        # Action selection
        if isinstance(agent, IntervalDQNAgent):
            action = agent.select_action(state, c=c)
        elif isinstance(agent, EnsembleDQNAgent):
            action = agent.select_action(state, mode=mode)
        else:
            eps = epsilon if epsilon is not None else 0.0
            action = agent.select_action(state, epsilon=eps)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if mode == "train":
            agent.buffer.push(state, action, reward, next_state, float(done))

        state = next_state
        total_reward += reward

        if terminated:  # reached the flag (not just truncated at max steps)
            solved = True

        if done:
            break

    return total_reward, solved, step + 1


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(seed=42, c_train=1.0, c_eval_values=None,
                   total_episodes=1000, eval_every=20, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if c_eval_values is None:
        c_eval_values = [0.0, 0.2, 0.5, 1.0]

    TRAIN_STEPS_PER_EP = 16
    BATCH_SIZE = 64

    # Agents
    standard_agent = StandardDQNAgent(state_dim=2, n_actions=3, lr=1e-3)
    interval_agent = IntervalDQNAgent(state_dim=2, n_actions=3, lr=5e-4,
                                       c_train=c_train, warmup_episodes=50,
                                       warmup_epsilon=0.3, init_width=50.0)
    ensemble_agent = EnsembleDQNAgent(state_dim=2, n_actions=3, lr=1e-3,
                                       n_ensemble=5, optimism_train=1.0,
                                       pessimism_eval=1.0)

    # Environments
    env_std = gym.make("MountainCar-v0")
    env_int = gym.make("MountainCar-v0")
    env_ens = gym.make("MountainCar-v0")
    env_eval = gym.make("MountainCar-v0")

    history = []

    # Track first solve
    first_solve = {"standard": None, "interval": None, "ensemble": None}

    # Rolling solve rates
    std_solves = deque(maxlen=50)
    int_solves = deque(maxlen=50)
    ens_solves = deque(maxlen=50)

    if verbose:
        print("=" * 100)
        print(f"MOUNTAINCAR INTERVAL DQN (seed={seed}, c_train={c_train})")
        print("=" * 100)
        print(f"Interval: Hurwicz c={c_train} for exploration, eval at c={c_eval_values}")
        print(f"Standard: ε-greedy (decaying)")
        print(f"Ensemble: optimistic (mean + std) for exploration, pessimistic for eval")
        print()

        c_headers = " ".join([f"{'c='+str(c):>7}" for c in c_eval_values])
        print(f"{'Ep':>5} | {'ε':>4} | {'Std':>5} {'StdR%':>5} | "
              f"{c_headers} {'IntR%':>5} | "
              f"{'Ens':>5} {'EnsR%':>5} | "
              f"{'Wvis':>6} {'Wunv':>6} {'Ratio':>5} "
              f"{'Cvg':>5} {'t':>5}")
        print("-" * (78 + 8 * len(c_eval_values)))

    for ep in range(total_episodes):
        # ---- Epsilon for standard DQN ----
        # Aggressive early exploration, slow decay
        epsilon = max(0.05, 1.0 - ep / (total_episodes * 0.5))

        # ---- Collect episodes ----
        r_std, solved_std, _ = run_episode(env_std, standard_agent,
                                            mode="train", epsilon=epsilon)
        r_int, solved_int, _ = run_episode(env_int, interval_agent, mode="train")
        interval_agent.episodes_seen += 1
        r_ens, solved_ens, _ = run_episode(env_ens, ensemble_agent, mode="train")

        # Track solves
        std_solves.append(solved_std)
        int_solves.append(solved_int)
        ens_solves.append(solved_ens)

        if solved_std and first_solve["standard"] is None:
            first_solve["standard"] = ep + 1
        if solved_int and first_solve["interval"] is None:
            first_solve["interval"] = ep + 1
        if solved_ens and first_solve["ensemble"] is None:
            first_solve["ensemble"] = ep + 1

        # ---- Train ----
        for _ in range(TRAIN_STEPS_PER_EP):
            standard_agent.train_step(BATCH_SIZE)
            interval_agent.train_step(BATCH_SIZE)
            ensemble_agent.train_step(BATCH_SIZE)

        # ---- Evaluate ----
        if (ep + 1) % eval_every == 0:
            n_eval = 10

            # Standard: greedy eval
            std_eval_rewards = []
            for _ in range(n_eval):
                r, _, _ = run_episode(env_eval, standard_agent,
                                       mode="eval", epsilon=0.0)
                std_eval_rewards.append(r)
            eval_std = np.mean(std_eval_rewards)

            # Interval: eval at multiple c values
            eval_int = {}
            for c_val in c_eval_values:
                int_eval_rewards = []
                for _ in range(n_eval):
                    r, _, _ = run_episode(env_eval, interval_agent,
                                           mode="eval", c=c_val)
                    int_eval_rewards.append(r)
                eval_int[c_val] = np.mean(int_eval_rewards)

            # Ensemble: pessimistic eval
            ens_eval_rewards = []
            for _ in range(n_eval):
                r, _, _ = run_episode(env_eval, ensemble_agent, mode="eval")
                ens_eval_rewards.append(r)
            eval_ens = np.mean(ens_eval_rewards)

            # Interval widths — measure both visited (valley) and unvisited (flag)
            sample_states = np.array([env_eval.reset()[0] for _ in range(30)])
            width = interval_agent.get_interval_widths(sample_states)
            visited_states = np.array([[-0.5, 0.0], [-0.45, 0.01], [-0.55, -0.01]])
            unvisited_states = np.array([[0.3, 0.05], [0.4, 0.06], [0.5, 0.07]])
            w_visited = interval_agent.get_interval_widths(visited_states)
            w_unvisited = interval_agent.get_interval_widths(unvisited_states)
            coverage = (sum(interval_agent.coverage_hits) /
                       max(1, len(interval_agent.coverage_hits)))

            # Rolling solve rates
            std_rate = sum(std_solves) / len(std_solves) if std_solves else 0
            int_rate = sum(int_solves) / len(int_solves) if int_solves else 0
            ens_rate = sum(ens_solves) / len(ens_solves) if ens_solves else 0

            record = {
                "episode": ep + 1,
                "eval_std": eval_std,
                "eval_ens": eval_ens,
                "width": width,
                "coverage": coverage,
                "t": interval_agent.t,
                "epsilon": epsilon,
                "std_solve_rate": std_rate,
                "int_solve_rate": int_rate,
                "ens_solve_rate": ens_rate,
            }
            for c_val in c_eval_values:
                record[f"eval_int_c{c_val}"] = eval_int[c_val]
            history.append(record)

            record["w_visited"] = w_visited
            record["w_unvisited"] = w_unvisited

            if verbose:
                c_vals_str = " ".join([f"{eval_int[c]:7.1f}" for c in c_eval_values])
                print(f"{ep+1:5d} | {epsilon:4.2f} | "
                      f"{eval_std:5.0f} {std_rate:5.0%} | "
                      f"{c_vals_str} {int_rate:5.0%} | "
                      f"{eval_ens:5.0f} {ens_rate:5.0%} | "
                      f"{w_visited:6.1f} {w_unvisited:6.1f} "
                      f"{w_unvisited/max(w_visited,0.01):5.1f}x "
                      f"{coverage:5.2f} {interval_agent.t:5.3f}")

    # Cleanup
    for e in [env_std, env_int, env_ens, env_eval]:
        e.close()

    # ---- Summary ----
    results = {
        "seed": seed,
        "c_train": c_train,
        "history": history,
        "first_solve": first_solve,
    }

    if history:
        last5 = history[-5:]

        def avg(records, key):
            return np.mean([h[key] for h in records])

        results["final_std"] = avg(last5, "eval_std")
        results["final_ens"] = avg(last5, "eval_ens")
        results["final_std_rate"] = avg(last5, "std_solve_rate")
        results["final_int_rate"] = avg(last5, "int_solve_rate")
        results["final_ens_rate"] = avg(last5, "ens_solve_rate")
        results["final_width"] = avg(last5, "width")

        for c_val in c_eval_values:
            results[f"final_int_c{c_val}"] = avg(last5, f"eval_int_c{c_val}")

    if verbose:
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")

        print(f"\n  First solve (episode):")
        print(f"    Standard DQN:    {first_solve['standard'] or 'NEVER'}")
        print(f"    Interval DQN:    {first_solve['interval'] or 'NEVER'}")
        print(f"    Ensemble DQN:    {first_solve['ensemble'] or 'NEVER'}")

        print(f"\n  Final performance (last 5 evals):")
        print(f"    Standard DQN:    reward={results.get('final_std', 0):.1f}, "
              f"solve rate={results.get('final_std_rate', 0):.0%}")
        for c_val in c_eval_values:
            r = results.get(f'final_int_c{c_val}', 0)
            print(f"    Interval c={c_val}:   reward={r:.1f}")
        print(f"    Interval solve:  rate={results.get('final_int_rate', 0):.0%}")
        print(f"    Ensemble DQN:    reward={results.get('final_ens', 0):.1f}, "
              f"solve rate={results.get('final_ens_rate', 0):.0%}")
        print(f"    Interval width:  {results.get('final_width', 0):.3f}")

    return results


def run_multi_seed(seeds=None, c_train=1.0, c_eval_values=None,
                   total_episodes=1000):
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
                            c_eval_values=c_eval_values,
                            total_episodes=total_episodes, verbose=True)
        all_results.append(res)

    valid = [r for r in all_results if "final_std" in r]
    if not valid:
        print("No valid results!")
        return

    print(f"\n{'=' * 80}")
    print(f"AGGREGATE ACROSS {len(valid)} SEEDS (c_train={c_train})")
    print(f"{'=' * 80}")

    # First solve
    print(f"\n  First solve (episode):")
    for agent in ["standard", "interval", "ensemble"]:
        vals = [r["first_solve"][agent] for r in valid if r["first_solve"][agent] is not None]
        never = sum(1 for r in valid if r["first_solve"][agent] is None)
        if vals:
            print(f"    {agent:>10}: mean={np.mean(vals):.0f} ± {np.std(vals):.0f}  "
                  f"(range: {min(vals)}-{max(vals)})  "
                  f"[{len(vals)}/{len(valid)} seeds solved"
                  f"{f', {never} never' if never else ''}]")
        else:
            print(f"    {agent:>10}: NEVER SOLVED in any seed")

    # Final performance
    print(f"\n  Final reward (last 5 evals):")
    def report(label, key):
        vals = [r[key] for r in valid]
        print(f"    {label:>20}: {np.mean(vals):7.1f} ± {np.std(vals):5.1f}")

    report("Standard DQN", "final_std")
    for c_val in c_eval_values:
        report(f"Interval c={c_val}", f"final_int_c{c_val}")
    report("Ensemble DQN", "final_ens")

    # Solve rates
    print(f"\n  Final solve rate (last 5 evals):")
    for label, key in [("Standard", "final_std_rate"),
                        ("Interval", "final_int_rate"),
                        ("Ensemble", "final_ens_rate")]:
        vals = [r[key] for r in valid]
        print(f"    {label:>10}: {np.mean(vals):.0%} ± {np.std(vals):.0%}")

    # Per-seed table
    print(f"\n  Per-seed detail:")
    print(f"  {'Seed':>6} {'1st Std':>7} {'1st Int':>7} {'1st Ens':>7} | "
          f"{'Std':>6} {'Int c=0':>7} {'Ens':>6} | "
          f"{'Std%':>5} {'Int%':>5} {'Ens%':>5}")
    print(f"  {'-'*80}")
    for r in valid:
        fs = r["first_solve"]
        print(f"  {r['seed']:>6} "
              f"{str(fs['standard']) if fs['standard'] else 'never':>7} "
              f"{str(fs['interval']) if fs['interval'] else 'never':>7} "
              f"{str(fs['ensemble']) if fs['ensemble'] else 'never':>7} | "
              f"{r['final_std']:>6.0f} "
              f"{r.get('final_int_c0.0', 0):>7.0f} "
              f"{r['final_ens']:>6.0f} | "
              f"{r['final_std_rate']:>4.0%} "
              f"{r['final_int_rate']:>5.0%} "
              f"{r['final_ens_rate']:>5.0%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--c_train", type=float, default=1.0)
    parser.add_argument("--c_eval", type=float, nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    if args.single:
        run_experiment(seed=args.seed, c_train=args.c_train,
                      c_eval_values=args.c_eval,
                      total_episodes=args.episodes, verbose=True)
    else:
        run_multi_seed(seeds=args.seeds, c_train=args.c_train,
                      c_eval_values=args.c_eval,
                      total_episodes=args.episodes)
