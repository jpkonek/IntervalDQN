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
# RISK STRATA (opt-in via --strat_t) — nearest-neighbour distance buckets
# ============================================================================
# Feature layout of the "relative" encoder with 1 forward fix (BluebirdATC
# bluebird_gymnasium/state_repr/relative.py, RelativeRepresentation): 20 base
# features + 7 forward-fix features + one 9-feature block per neighbour.
# Within each neighbour block: field 3 = distance to that neighbour in
# NM / 50 (clipped at 150 NM); field 7 = controllable flag (-1/+1 for a real
# neighbour, 0.0 marks a zero-padded block = no such neighbour).
# Verified empirically against a live env (2026-07-05 probe): the distance
# field matches pairwise haversine within 0.3 NM on 2513/2513 blocks, and
# padded blocks are all-zero. Blocks are NOT guaranteed nearest-first, so the
# nearest neighbour is the MIN over all non-padded blocks.

N_STRATA = 3
STRAT_BOUNDS_NM = (10.0, 30.0)   # stratum 0: <10 NM, 1: 10-30 NM, 2: rest
REL_BASE_AND_FIX = 27            # 20 base + 7 forward-fix features
REL_NEIGH_FEATS = 9
REL_NEIGH_DIST = 3               # NM / 50 within the block
REL_NEIGH_FLAG = 7               # 0.0 == zero padding (no neighbour)
REL_DIST_SCALE = 50.0


def state_stratum(state):
    """Risk stratum of one relative-encoder observation vector.

    Returns 0 (nearest neighbour <10 NM), 1 (10-30 NM) or
    2 (>30 NM, or no neighbour at all).
    """
    n_blocks = (len(state) - REL_BASE_AND_FIX) // REL_NEIGH_FEATS
    nearest = None
    for i in range(n_blocks):
        base = REL_BASE_AND_FIX + i * REL_NEIGH_FEATS
        if state[base + REL_NEIGH_FLAG] == 0.0:
            continue  # zero-padded block: no i-th neighbour
        d = state[base + REL_NEIGH_DIST] * REL_DIST_SCALE
        if nearest is None or d < nearest:
            nearest = d
    if nearest is None or nearest > STRAT_BOUNDS_NM[1]:
        return 2
    if nearest < STRAT_BOUNDS_NM[0]:
        return 0
    return 1


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Stores n-step windows (state, action, R_n, next_state, disc) where
    R_n is the discounted n-step reward sum and disc is the bootstrap
    discount factor: gamma^m for an m-step bootstrapped window, 0.0 for a
    window that reached a terminal (the target IS the realized return)."""

    def __init__(self, capacity=BUFFER_SIZE, terminal_boost=1.0):
        # terminal_boost > 1 splits storage into a terminal pool (disc == 0,
        # realized-G targets) and a regular pool, and oversamples the
        # terminal pool by ~boost in sample(). Motivation (run-8 forensics):
        # realized outcomes are a few percent of windows, so the coverage
        # pressure toward reality is population-diluted; boosting the
        # realized minority restores its gradient share. boost == 1.0
        # preserves the original single-deque behavior exactly.
        self.terminal_boost = float(terminal_boost)
        if self.terminal_boost > 1.0:
            # terminal pool gets a protected slice of capacity (also gives
            # rare realized samples longer retention)
            term_cap = max(1000, capacity // 5)
            self.term_buf = deque(maxlen=term_cap)
            self.reg_buf = deque(maxlen=capacity - term_cap)
            self.buffer = None
        else:
            self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, disc, stratum=0):
        # stratum: small int risk-stratum id recorded at push time
        # (always 0 when --strat_t is off)
        item = (state, action, reward, next_state, disc, stratum)
        if self.buffer is not None:
            self.buffer.append(item)
        elif disc == 0.0:
            self.term_buf.append(item)
        else:
            self.reg_buf.append(item)

    def sample(self, batch_size, with_strata=False):
        if self.buffer is not None:
            batch = random.sample(self.buffer, batch_size)
        else:
            n_term, n_reg = len(self.term_buf), len(self.reg_buf)
            share = n_term / max(1, n_term + n_reg)
            target = min(0.5, self.terminal_boost * share)
            k_t = min(n_term, int(round(batch_size * target)))
            k_r = batch_size - k_t
            if k_r > n_reg:  # early training: not enough regular samples
                k_r = n_reg
                k_t = min(n_term, batch_size - k_r)
            batch = (random.sample(self.term_buf, k_t) +
                     random.sample(self.reg_buf, k_r))
        states, actions, rewards, next_states, discs, strata = zip(*batch)
        out = (np.array(states), np.array(actions),
               np.array(rewards, dtype=np.float32),
               np.array(next_states), np.array(discs, dtype=np.float32))
        if with_strata:
            return out + (np.array(strata, dtype=np.int64),)
        return out

    def __len__(self):
        if self.buffer is not None:
            return len(self.buffer)
        return len(self.term_buf) + len(self.reg_buf)


class NStepWindower:
    """Per-aircraft n-step replay-window builder over a transition stream.

    Feeds (s, a, R_n, s_next, disc) windows into the replay buffer as soon
    as they are determined (Rainbow-style uncorrected n-step, per
    DESIGN_REVISIONS item 3):

      - full window i (n rewards available, aircraft alive at i+n):
        R_n = sum_{k<n} gamma^k r_{i+k}, s_next = state at i+n,
        disc = gamma^n;
      - terminal at stream index T, window i > T-n: R_n = the exact
        realized discounted tail sum_{k=i..T} gamma^{k-i} r_k,
        s_next = zeros, disc = 0.0 (the bootstrap VANISHES);
      - censored stream end (episode over, aircraft alive, m < n rewards
        after i): bootstrap at the LAST available next state with
        disc = gamma^m (variable-length bootstrap — censored data still
        trains).

    With n = 1 this reproduces the previous 1-step scheme exactly,
    including truncation semantics: a time-limit truncated aircraft is
    NOT terminal (its stream is censored -> bootstrap with disc = gamma),
    while a genuine terminal (clean exit / vanished / violation) gets
    disc = 0. Stored states are the SAME array objects the caller passes.
    """

    def __init__(self, buffer, gamma, n, strat_fn=None):
        self.buffer = buffer
        self.gamma = gamma
        self.n = n
        # optional stratum function: window state -> small int stratum id,
        # recorded at push time (0 for every window when None / flag off)
        self.strat_fn = strat_fn
        self.pending = []    # [(s, a, r, stratum)] awaiting future rewards
        self.last_ns = None  # next state of the most recent transition

    def _window_return(self, i):
        return sum(self.gamma ** k * r
                   for k, (_s, _a, r, _st) in enumerate(self.pending[i:]))

    def add(self, s, a, r, ns, terminal):
        """Record one transition; emit every window it completes."""
        st = self.strat_fn(s) if self.strat_fn is not None else 0
        self.pending.append((s, a, r, st))
        self.last_ns = ns
        if terminal:
            # every pending window sees the terminal inside it: exact
            # realized tail, no bootstrap
            zeros = np.zeros_like(ns)
            for i in range(len(self.pending)):
                s_i, a_i, _, st_i = self.pending[i]
                self.buffer.push(s_i, a_i, self._window_return(i),
                                 zeros, 0.0, st_i)
            self.pending.clear()
        elif len(self.pending) == self.n:
            s_0, a_0, _, st_0 = self.pending[0]
            self.buffer.push(s_0, a_0, self._window_return(0), ns,
                             self.gamma ** self.n, st_0)
            self.pending.pop(0)

    def flush_censored(self):
        """Episode ended with the aircraft still alive: emit the remaining
        windows with a variable-length bootstrap (disc = gamma^m) at the
        last available next state."""
        for i in range(len(self.pending)):
            s_i, a_i, _, st_i = self.pending[i]
            m = len(self.pending) - i
            self.buffer.push(s_i, a_i, self._window_return(i),
                             self.last_ns, self.gamma ** m, st_i)
        self.pending.clear()


# ============================================================================
# COVERAGE TRACKER — adapts the loss parameter t toward target coverage
# ============================================================================

class CoverageTracker:
    """Tracks recent interval coverage (target midpoint inside [l, u]) and
    adapts the interval-loss parameter t toward the target coverage:
    coverage below target -> raise t (penalize misses), above -> lower t
    (allow narrower intervals)."""

    def __init__(self, target=0.85, t_init=0.5, window=2000, min_samples=200,
                 t_max=0.95):
        self.target = target
        self.t = t_init
        # per-tracker cap: with stratified t, the failing (risky) stratum
        # may run a higher cap (e.g. 0.99) to weaken the width-shrink term
        # exactly where coverage fails, without loosening it elsewhere
        self.t_max = t_max
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
                self.t = min(self.t_max, self.t + 0.001)
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
                 device="cpu", strat_t=False, t_cap_risky=0.99,
                 terminal_boost=1.0):
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
        self.buffer = ReplayBuffer(buffer_size, terminal_boost=terminal_boost)

        # Adaptive coverage — the tracker (and hence t) is driven by REALIZED
        # returns of finished aircraft streams (record_realized_stream), not
        # by bootstrapped Bellman targets. Bootstrap coverage is kept as a
        # logging-only comparison metric.
        self.coverage_tracker = CoverageTracker(target=target_coverage)
        # Stratified-t (opt-in, --strat_t): a bank of trackers, one per risk
        # stratum (nearest-neighbour distance bucket, see state_stratum).
        # Each stratum's t is driven only by realized returns of states in
        # that stratum, so high-risk states can keep a high t (coverage
        # pressure) while empty-sky states relax toward narrow intervals.
        # When off, self.trackers aliases the single tracker and every
        # stored stratum id is 0 — behavior is identical to the unstratified
        # scheme.
        self.strat_t = strat_t
        if strat_t:
            # stratum 0 (<10nm, the failing stratum in every measurement)
            # runs a raised cap; safe strata keep the standard 0.95
            self.trackers = [
                CoverageTracker(target=target_coverage,
                                t_max=(t_cap_risky if s == 0 else 0.95))
                for s in range(N_STRATA)]
            self.stratum_fn = state_stratum
        else:
            self.trackers = [self.coverage_tracker]
            self.stratum_fn = None
        self.bootstrap_hits = deque(maxlen=2000)
        self.realized_streams = 0
        self.censored_streams = 0

        # Warmup with epsilon-greedy (env-step based)
        self.warmup_steps = warmup_steps
        self.warmup_epsilon = warmup_epsilon
        self.total_env_steps = 0

    # convenience passthroughs
    @property
    def t(self):
        if self.strat_t:
            # headline scalar: mean per-stratum t (per-stratum values are
            # logged separately)
            return float(np.mean([tr.t for tr in self.trackers]))
        return self.coverage_tracker.t

    @property
    def coverage(self):
        """Realized coverage (drives t). Pooled across strata when
        --strat_t is on."""
        if self.strat_t:
            n = sum(len(tr.hits) for tr in self.trackers)
            return sum(sum(tr.hits) for tr in self.trackers) / max(1, n)
        return self.coverage_tracker.coverage

    @property
    def bootstrap_coverage(self):
        """Coverage of bootstrapped target midpoints (logging only)."""
        return sum(self.bootstrap_hits) / max(1, len(self.bootstrap_hits))

    def record_realized_stream(self, states, actions, rewards):
        """Feed one terminated aircraft's realized returns to the t controller.

        Arguments are the aircraft's full in-sector trajectory in order;
        rewards already carry the terminal exit bonus / violation penalty.
        The realized discounted return-to-go G_k is computed backward and
        each hit (G_k inside the current net's [Q_l, Q_u] at (s_k, a_k))
        goes into the coverage tracker that adapts t. Only finished flights
        reach this method — censored streams have unknown returns.
        """
        if not states:
            return
        G, returns = 0.0, [0.0] * len(rewards)
        for k in range(len(rewards) - 1, -1, -1):
            G = rewards[k] + self.gamma * G
            returns[k] = G
        with torch.no_grad():
            s = torch.FloatTensor(np.asarray(states, dtype=np.float32)).to(self.device)
            a = torch.LongTensor(actions).to(self.device)
            lower, upper = self.q_net(s)
            l_a = lower.gather(1, a.unsqueeze(1)).squeeze(1).cpu().numpy()
            u_a = upper.gather(1, a.unsqueeze(1)).squeeze(1).cpu().numpy()
        g = np.asarray(returns, dtype=np.float32)
        hits = ((g >= l_a) & (g <= u_a)).astype(np.float32)
        if self.strat_t:
            # bucket each (s, a, G) hit into its state's stratum tracker
            for s_k, hit in zip(states, hits):
                self.trackers[self.stratum_fn(s_k)].record([float(hit)])
        else:
            self.coverage_tracker.record(hits.tolist())
        self.realized_streams += 1

    def _epsilon(self, force_epsilon=None):
        if force_epsilon is not None:
            return force_epsilon
        if self.total_env_steps < self.warmup_steps:
            progress = self.total_env_steps / self.warmup_steps
            return self.warmup_epsilon * (1 - progress)
        return 0.0

    @staticmethod
    def adaptive_c(widths, c_low=0.0, c_high=0.3, w_mid=4.0, k=2.0):
        """Width-dependent Hurwicz parameter, ported from the LunarLander
        implementation: wide intervals -> low c (cautious), narrow -> higher
        c (balanced), sigmoid transition. Vectorised: `widths` is a tensor
        of per-aircraft mean interval widths -> per-aircraft c. In the
        multi-aircraft setting this IS a state-conditioned risk attitude:
        aircraft near traffic have wider intervals (verified by the width
        perturbation diagnostic) and act cautiously; aircraft in the clear
        act more optimistically (chase their exits). w_mid must be
        calibrated to the checkpoint's width scale (LunarLander used 4.0;
        run-7 nets live around 12-20) — see calibrate_w_mid().
        """
        t = torch.sigmoid(-k * (widths - w_mid))
        return c_low + t * (c_high - c_low)

    def calibrate_w_mid(self, obs_dicts):
        """Set the adaptive-c midpoint to the median per-aircraft mean width
        over a sample of observation dicts (e.g. one probe episode)."""
        widths = []
        with torch.no_grad():
            for od in obs_dicts:
                if not od:
                    continue
                s = torch.from_numpy(np.stack(list(od.values()))
                                     .astype(np.float32)).to(self.device)
                lower, upper = self.q_net(s)
                widths.extend((upper - lower).mean(dim=1).cpu().tolist())
        self.adaptive_w_mid = float(np.median(widths)) if widths else 4.0
        return self.adaptive_w_mid

    def generate_action(self, obs_dict, c=None, force_epsilon=None,
                        adaptive=False):
        """Batched Hurwicz action selection for ALL aircraft in obs_dict.

        One forward pass per env step (never per aircraft).
        adaptive=True: per-aircraft width-dependent c (overrides `c`);
        requires calibrate_w_mid() first (falls back to 4.0).
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
            if adaptive:
                per_ac_width = (upper - lower).mean(dim=1)
                use_c = self.adaptive_c(
                    per_ac_width,
                    w_mid=getattr(self, "adaptive_w_mid", 4.0)).unsqueeze(1)
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

    def interval_loss_sampled(self, lower, upper, target_lower, target_upper,
                              t_vec=None, width_mask=None):
        """Interval loss over N sampled Bellman targets.

        t_vec: optional per-sample t tensor (shape [B]) for stratified
        coverage control; None -> the single tracker's scalar t (the
        original behavior, bit-identical).
        width_mask: optional per-sample 0/1 tensor gating the width
        regularizer (stratified: active only for samples whose stratum
        coverage exceeds its target); None -> the original global gate.
        """
        N = self.N_TARGET_SAMPLES
        batch_size = lower.shape[0]

        alphas = torch.linspace(0, 1, N, device=lower.device)
        targets = target_lower.unsqueeze(0) + alphas.unsqueeze(1) * (
            target_upper - target_lower).unsqueeze(0)

        # Bootstrap-target coverage: LOGGING ONLY. The t controller runs on
        # realized coverage (record_realized_stream). Run 2 showed the
        # bootstrapped proxy certifying 85% while realized coverage was 0%,
        # so it no longer steers the loss.
        target_mid = (target_lower + target_upper) / 2
        inside_mid = (target_mid >= lower) & (target_mid <= upper)
        self.bootstrap_hits.extend(
            inside_mid.detach().cpu().numpy().astype(np.float32).tolist())

        # scalar t (flag off) or per-sample tensor (broadcasts elementwise)
        t = self.coverage_tracker.t if t_vec is None else t_vec
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
        if width_mask is not None:
            # per-sample gate (stratified): only samples whose stratum has
            # adequate coverage pay the width penalty
            width = upper - lower
            total_loss = total_loss + self.width_reg * width ** 2 * width_mask
        elif self.coverage_tracker.coverage > self.coverage_tracker.target:
            width = upper - lower
            total_loss = total_loss + self.width_reg * width ** 2

        return total_loss.mean()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        if self.strat_t:
            (states, actions, rewards, next_states, discs,
             strata) = self.buffer.sample(self.batch_size, with_strata=True)
        else:
            states, actions, rewards, next_states, discs = self.buffer.sample(
                self.batch_size)
            strata = None
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        discs = torch.FloatTensor(discs).to(self.device)

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

            # rewards holds the discounted n-step sum R_n; disc is the
            # per-sample bootstrap factor (gamma^m, or 0 at terminals —
            # there the target IS the realized return)
            target_l = rewards + discs * tgt_lower
            target_u = rewards + discs * tgt_upper

        if self.strat_t:
            # per-sample t and width gate from each sample's stratum tracker
            t_vec = torch.FloatTensor(
                [self.trackers[si].t for si in strata]).to(self.device)
            width_mask = torch.FloatTensor(
                [1.0 if (self.trackers[si].coverage > self.trackers[si].target)
                 else 0.0 for si in strata]).to(self.device)
            loss = self.interval_loss_sampled(lower_a, upper_a,
                                              target_l, target_u,
                                              t_vec=t_vec,
                                              width_mask=width_mask)
        else:
            loss = self.interval_loss_sampled(lower_a, upper_a,
                                              target_l, target_u)
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

        # Adaptive t (self.trackers is [coverage_tracker] when flag off,
        # so this is the same single update as before)
        for tr in self.trackers:
            tr.update_t()

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
             centreline_coeff=0.2, encoder_cls="extra_minimal"):
    """Build the Flight School InfiniteEnv (X-Plus sector, decentralized).

    encoder_cls selects the observation encoder ("extra_minimal" was the
    run-2..6 default; "relative" adds the exit-relative block and 9
    features per neighbour incl. a controllable_flag that disambiguates
    padding from a collision-range neighbour — DESIGN_REVISIONS item 1).

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
    cfg.state_repr_config = {"encoder_cls": encoder_cls,
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


def exit_potential(env, cs):
    """Potential for progress shaping: minus the along-track distance (nm)
    from the aircraft to its exit. None if the env is not tracking cs.

    Used as textbook potential-based shaping, r += coeff * (gamma*phi' - phi):
    pay a little for getting closer to the exit, take it back for moving
    away. Payments telescope, so total pay over a flight is fixed by entry
    and exit positions — it cannot change the optimal policy, it only makes
    the sparse exit bonus learnable. (The env's own rewards.expeditious
    variants are not used: expeditious_linear rewards the SIGN OPPOSITE of
    its docstring as of bluebird-gymnasium 0.2.0, and expeditious_const pays
    per progressing step, which favours long paths.)
    """
    try:
        d = env.get_tracked_aircraft_data(cs)
    except Exception:
        return None
    if d is None or d.track_dist_to_exit_cr is None:
        return None
    return -float(d.track_dist_to_exit_cr)


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
                exit_bonus=10.0, progress_coeff=0.05, nstep=6, adaptive=False):
    """Run one episode; ends at first violation or the time limit.

    nstep: length of the n-step interval Bellman windows pushed to replay
    (see NStepWindower; nstep=1 reproduces the old 1-step scheme exactly).

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
    # per-aircraft (states, actions, rewards) streams for realized-coverage
    # tracking; resolved on terminal, censored streams are dropped
    streams = {}
    # per-aircraft n-step window builders feeding the replay buffer
    windowers = {}

    step_i = -1
    for step_i in range(maxstep):
        force_eps = None if train else 0.0
        actions, mean_width = agent.generate_action(
            obs, c=c, force_epsilon=force_eps, adaptive=adaptive)
        if actions:
            width_sum += mean_width
            width_n += 1

        # potentials before the step, for progress shaping
        phi_prev = ({cs: exit_potential(env, cs) for cs in obs}
                    if train and progress_coeff else {})

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

                # progress shaping on non-terminal steps (terminal steps
                # carry the outcome reward; skipping them avoids the
                # positive kick a zeroed terminal potential would give to
                # far-from-exit violations)
                if progress_coeff and not terminal:
                    phi = phi_prev.get(cs)
                    phi_next = exit_potential(env, cs)
                    if phi is not None and phi_next is not None:
                        r += progress_coeff * (agent.gamma * phi_next - phi)
                s32 = np.asarray(s, dtype=np.float32)
                if cs not in windowers:
                    windowers[cs] = NStepWindower(
                        agent.buffer, agent.gamma, nstep,
                        strat_fn=getattr(agent, "stratum_fn", None))
                windowers[cs].add(s32, a, r, ns, terminal)
                n_transitions += 1

                st_s, st_a, st_r = streams.setdefault(cs, ([], [], []))
                st_s.append(s32)
                st_a.append(a)
                st_r.append(r)
                if terminal:
                    windowers.pop(cs)  # add() already flushed its windows
                    agent.record_realized_stream(*streams.pop(cs))

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

    if train:
        # censored streams still train: flush their remaining windows with
        # a variable-length bootstrap at the last available next state
        for w in windowers.values():
            w.flush_censored()
        # aircraft still flying at episode end: returns unknown, not counted
        # by the realized tracker
        agent.censored_streams += len(streams)

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
            "encoder_cls": args.encoder,
            "route_parallel": args.route_parallel,
            "gamma": args.gamma,
            "exit_bonus": args.exit_bonus,
            "centreline_coeff": args.centreline_coeff,
            "violation_penalty": args.violation_penalty,
            "progress_coeff": args.progress_coeff,
            "nstep": args.nstep,
            "strat_t": bool(getattr(args, "strat_t", False)),
            "t_cap_risky": float(getattr(args, "t_cap_risky", 0.99)),
            "terminal_boost": float(getattr(args, "terminal_boost", 1.0))}


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
    if getattr(agent, "strat_t", False):
        ckpt["strat_t_values"] = [tr.t for tr in agent.trackers]
        ckpt["strat_coverages"] = [tr.coverage for tr in agent.trackers]
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_agent(path, device="cpu", **kwargs):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    kwargs.setdefault("strat_t", bool(ckpt.get("strat_t", False)))
    agent = IntervalDQNAgent(
        state_dim=ckpt["state_dim"], n_actions=ckpt["n_actions"],
        c_train=ckpt.get("c_train", 0.5), device=device, **kwargs)
    agent.q_net.load_state_dict(ckpt["q_net"])
    agent.target_net.load_state_dict(ckpt["q_net"])
    agent.coverage_tracker.t = ckpt.get("t", 0.5)
    if agent.strat_t:
        for tr, tv in zip(agent.trackers, ckpt.get("strat_t_values", [])):
            tr.t = tv
    # restore the training gamma: silently wrong for resume-training or any
    # realized-return computation via agent.gamma otherwise (run 7 = 0.97,
    # module default = 0.99)
    agent.gamma = ckpt.get("gamma", agent.gamma)
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
          f"episodes={episodes}, encoder={args.encoder}, k={args.k}, "
          f"nstep={args.nstep})")
    print("=" * 100)

    print("Creating environment...")
    env = make_env(scenario_duration=duration, k_nearest=args.k,
                   route_parallel=args.route_parallel,
                   centreline_coeff=args.centreline_coeff,
                   encoder_cls=args.encoder)
    obs, _ = env.reset(seed=args.seed)
    state_dim = int(next(iter(obs.values())).shape[0])
    n_actions = int(env.get_action_parser().get_total_num_actions())
    action_map = env.get_action_parser().action_formatter_map
    print(f"obs dim: {state_dim}, actions ({n_actions}): {action_map}")

    device = resolve_device(args.device, state_dim, n_actions)

    strat_t = bool(getattr(args, "strat_t", False))
    if strat_t:
        # the stratum function reads the relative encoder's neighbour blocks
        if args.encoder != "relative":
            raise SystemExit("--strat_t requires --encoder relative "
                             f"(got {args.encoder!r})")
        if (state_dim - REL_BASE_AND_FIX) % REL_NEIGH_FEATS != 0:
            raise SystemExit(f"--strat_t: obs dim {state_dim} does not match "
                             "the relative-encoder layout (20 base + 7 fix "
                             "+ 9 per neighbour)")
        print(f"Stratified-t ON: {N_STRATA} strata by nearest-neighbour "
              f"distance (<{STRAT_BOUNDS_NM[0]:.0f} NM, "
              f"{STRAT_BOUNDS_NM[0]:.0f}-{STRAT_BOUNDS_NM[1]:.0f} NM, "
              f">{STRAT_BOUNDS_NM[1]:.0f} NM/none)")

    agent = IntervalDQNAgent(
        state_dim=state_dim, n_actions=n_actions, lr=args.lr,
        gamma=args.gamma, c_train=args.c_train,
        target_coverage=args.target_coverage, width_reg=args.width_reg,
        warmup_steps=args.warmup_steps, warmup_epsilon=args.warmup_epsilon,
        buffer_size=args.buffer, batch_size=args.batch, device=device,
        strat_t=strat_t, t_cap_risky=args.t_cap_risky,
        terminal_boost=args.terminal_boost)

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
    lr_decay_ep = int(getattr(args, "lr_decay_ep", 0) or 0)
    for ep in range(episodes):
        if lr_decay_ep and ep == lr_decay_ep:
            # single step-decay damper against late-run value churn
            # (run-7 forensics: mid-run TTV dip = replay-recency churn)
            for g in agent.optimizer.param_groups:
                g["lr"] *= 0.3
            print(f"    [lr decay] episode {ep}: lr -> "
                  f"{agent.optimizer.param_groups[0]['lr']:.2e}")
        ep_seed = args.seed + ep
        t0 = time.time()
        stats = run_episode(env, agent, seed=ep_seed, train=True,
                            violation_penalty=args.violation_penalty,
                            exit_bonus=args.exit_bonus,
                            progress_coeff=args.progress_coeff,
                            nstep=args.nstep)
        wall = time.time() - t0
        total_steps += stats["steps"]
        eps = agent._epsilon()

        record = {
            "episode": ep + 1,
            "seed": ep_seed,
            **stats,
            "buffer_size": len(agent.buffer),
            "coverage": round(agent.coverage, 4),          # realized (drives t)
            "bootstrap_coverage": round(agent.bootstrap_coverage, 4),
            "realized_streams": agent.realized_streams,
            "censored_streams": agent.censored_streams,
            "t": round(agent.t, 4),
            "epsilon": round(eps, 4),
            "c_train": agent.c_train,
            "steps_per_sec": round(stats["steps"] / max(1e-9, wall), 2),
            "wall_time_s": round(wall, 2),
        }
        if strat_t:
            # stable keys: per-stratum realized coverage, t, and sample count
            record["strat_coverage"] = [round(tr.coverage, 4)
                                        for tr in agent.trackers]
            record["strat_t"] = [round(tr.t, 4) for tr in agent.trackers]
            record["strat_hits"] = [len(tr.hits) for tr in agent.trackers]
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

        vio = (f"VIOLATION[{stats['violation_kind']}]@{stats['time_to_violation']}s"
               if stats["violated"] else f"clean({stats['time_to_violation']}s)")
        print(f"  Ep {ep+1:4d} | {vio:>38} | R:{stats['mean_step_reward']:7.3f} "
              f"| W:{stats['mean_width']:.3f} rCvg:{agent.coverage:.2f} "
              f"bCvg:{agent.bootstrap_coverage:.2f} "
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

def evaluate(env, agent, c, n_episodes=5, base_seed=10042, verbose=True,
             adaptive=False):
    """Evaluate at a fixed Hurwicz c (or adaptive width-conditioned c) over
    several seeds. Headline metric: mean time-to-first-violation."""
    if adaptive and not hasattr(agent, "adaptive_w_mid"):
        # calibrate the sigmoid midpoint on one probe episode's observations
        probe_obs = []
        p_obs, _ = env.reset(seed=base_seed - 1)
        for _ in range(int(getattr(env, "maxstep", 100))):
            if not p_obs:
                break
            probe_obs.append(dict(p_obs))
            a, _ = agent.generate_action(p_obs, c=0.0, force_epsilon=0.0)
            p_obs, _, _, _, _ = env.step(a)
        w_mid = agent.calibrate_w_mid(probe_obs)
        if verbose:
            print(f"  adaptive c: w_mid calibrated to {w_mid:.3f} "
                  f"(median per-aircraft width, {len(probe_obs)} probe steps)")
    ttvs, rewards, widths = [], [], []
    n_violated = 0
    kinds = {}
    for i in range(n_episodes):
        stats = run_episode(env, agent, seed=base_seed + i, train=False, c=c,
                            adaptive=adaptive)
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
        c_label = "adap" if adaptive else c
        print(f"  c={c_label:<4} | time-to-violation: "
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
    # checkpoints predating the configurable encoder trained on extra_minimal
    encoder_cls = ckpt.get("encoder_cls", "extra_minimal")
    route_parallel = ckpt.get("route_parallel", args.route_parallel)
    # default 1.0: checkpoints predating outcome-anchoring trained with the
    # full-strength centreline income
    centreline_coeff = ckpt.get("centreline_coeff", 1.0)
    print(f"Loaded: obs dim {ckpt['state_dim']}, {ckpt['n_actions']} actions, "
          f"trained {ckpt.get('episode', '?')} episodes, "
          f"t={ckpt.get('t', 0.5):.3f}")

    print(f"Creating environment (duration {args.duration}s, "
          f"encoder={encoder_cls}, k={k}, "
          f"route_parallel={route_parallel}, "
          f"centreline_coeff={centreline_coeff})...")
    env = make_env(scenario_duration=args.duration, k_nearest=k,
                   route_parallel=route_parallel,
                   centreline_coeff=centreline_coeff,
                   encoder_cls=encoder_cls)

    # keep eval scenarios disjoint from training episode seeds
    # (training uses seed..seed+episodes-1; +10000 matches the in-training
    # final eval convention)
    eval_seed = args.seed + 10000
    print("=" * 100)
    print(f"EVALUATION (c={args.c}, {args.eval_episodes} episodes, "
          f"base seed {eval_seed})")
    print("=" * 100)
    result = evaluate(env, agent, c=args.c, n_episodes=args.eval_episodes,
                      base_seed=eval_seed,
                      adaptive=getattr(args, "adaptive", False))
    env.close()
    return result


# ============================================================================
# SELF-TESTS — n-step window semantics (run with --selftest)
# ============================================================================

class _ListBuffer:
    """Minimal replay-buffer stub capturing pushes for the self-tests."""

    def __init__(self):
        self.items = []

    def push(self, s, a, r, ns, disc, stratum=0):
        self.items.append((s, a, r, ns, disc, stratum))


def _assert_windows(items, expected, label):
    assert len(items) == len(expected), (
        f"{label}: {len(items)} windows, expected {len(expected)}")
    for i, ((s, a, r, ns, disc, _st),
            (es, ea, er, ens, edisc)) in enumerate(zip(items, expected)):
        assert s is es, f"{label} window {i}: state is not the same object"
        assert a == ea, f"{label} window {i}: action {a} != {ea}"
        assert abs(r - er) < 1e-9, f"{label} window {i}: R {r} != {er}"
        assert np.array_equal(ns, ens), f"{label} window {i}: next-state"
        assert abs(disc - edisc) < 1e-12, (
            f"{label} window {i}: disc {disc} != {edisc}")
    print(f"  OK  {label}: {len(items)} windows match hand-computed values")


def _selftest_synthetic_windows():
    """Hand-computed window sums for a synthetic stream (gamma = 0.5,
    rewards 1,2,3,4 with a terminal at stream index T=3):
      - n=6 > stream length: every window is the exact realized tail
        (realized-G collapse — the bootstrap vanishes, disc = 0);
      - n=2: mixed — two bootstrapped full windows, two realized tails;
    plus a censored 5-step stream (no terminal) exercising the
    variable-length gamma^m bootstrap at the last available next state."""
    print("[selftest] synthetic window sums")
    gamma = 0.5
    s = [np.full(2, float(i), dtype=np.float32) for i in range(6)]
    z = np.zeros(2, dtype=np.float32)

    def feed(n, rewards, terminal_at):
        buf = _ListBuffer()
        w = NStepWindower(buf, gamma, n)
        for i, r in enumerate(rewards):
            term = (i == terminal_at)
            w.add(s[i], i, r, z if term else s[i + 1], term)
        if terminal_at is None:
            w.flush_censored()
        return buf.items

    # terminal stream, n=6: exact realized tails G_i, disc 0 everywhere
    # G_0 = 1 + .5*2 + .25*3 + .125*4 = 3.25; G_1 = 4.5; G_2 = 5; G_3 = 4
    _assert_windows(
        feed(6, [1.0, 2.0, 3.0, 4.0], terminal_at=3),
        [(s[0], 0, 3.25, z, 0.0), (s[1], 1, 4.5, z, 0.0),
         (s[2], 2, 5.0, z, 0.0), (s[3], 3, 4.0, z, 0.0)],
        "terminal@3, n=6 (realized-G collapse)")

    # terminal stream, n=2: windows 0,1 bootstrap (gamma^2 = 0.25) at the
    # state 2 ahead; windows 2,3 contain the terminal -> realized tail
    _assert_windows(
        feed(2, [1.0, 2.0, 3.0, 4.0], terminal_at=3),
        [(s[0], 0, 1.0 + 0.5 * 2.0, s[2], 0.25),
         (s[1], 1, 2.0 + 0.5 * 3.0, s[3], 0.25),
         (s[2], 2, 3.0 + 0.5 * 4.0, z, 0.0),
         (s[3], 3, 4.0, z, 0.0)],
        "terminal@3, n=2 (mixed windows)")

    # censored stream (episode over, aircraft alive), n=2: full windows
    # bootstrap normally; the flushed remainder (m=1) bootstraps at the
    # LAST available next state with disc = gamma^1
    _assert_windows(
        feed(2, [1.0, 2.0, 3.0, 4.0, 5.0], terminal_at=None),
        [(s[0], 0, 1.0 + 0.5 * 2.0, s[2], 0.25),
         (s[1], 1, 2.0 + 0.5 * 3.0, s[3], 0.25),
         (s[2], 2, 3.0 + 0.5 * 4.0, s[4], 0.25),
         (s[3], 3, 4.0 + 0.5 * 5.0, s[5], 0.25),
         (s[4], 4, 5.0, s[5], 0.5)],
        "censored, n=2 (variable-length bootstrap)")


def _selftest_n1_equivalence(seed=123):
    """--nstep 1 must reproduce the old 1-step scheme exactly.

    Runs one fixed-seed episode through the REAL pipeline with nstep=1
    while recording the raw transitions (s, a, r, ns, terminal) at the
    exact point the old scheme called buffer.push. The expected old-scheme
    tuples are reconstructed independently from those raws —
        non-terminal (incl. time-limit truncation): bootstrap, disc=gamma
        terminal (exit / vanished / violation):     s_next=0,   disc=0
    — and must equal the windows the new pipeline pushed, in order."""
    print(f"[selftest] n=1 equivalence on a real episode (seed {seed})")
    # 600 s: long enough for clean exits, so the terminal branch is hit too
    env = make_env(scenario_duration=600, k_nearest=2)
    obs, _ = env.reset(seed=seed)
    state_dim = int(next(iter(obs.values())).shape[0])
    n_actions = int(env.get_action_parser().get_total_num_actions())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent = IntervalDQNAgent(state_dim, n_actions, device="cpu",
                             warmup_steps=20, batch_size=32)

    raws = []
    orig_add = NStepWindower.add

    def recording_add(self, s, a, r, ns, terminal):
        raws.append((s, a, r, ns, terminal))
        return orig_add(self, s, a, r, ns, terminal)

    NStepWindower.add = recording_add
    try:
        stats = run_episode(env, agent, seed=seed, train=True, nstep=1)
    finally:
        NStepWindower.add = orig_add
    env.close()

    produced = list(agent.buffer.buffer)
    assert len(produced) == len(raws), (
        f"n=1 must emit exactly one window per transition: "
        f"{len(produced)} windows vs {len(raws)} transitions")
    n_term = 0
    for i, ((s, a, r, ns, term),
            (ps, pa, pr, pns, pdisc, pst)) in enumerate(zip(raws, produced)):
        assert ps is s, f"transition {i}: state is not the same object"
        assert pst == 0, f"transition {i}: stratum must be 0 with flag off"
        assert pa == a and abs(pr - r) < 1e-9, f"transition {i}: (a, r)"
        if term:
            n_term += 1
            assert pdisc == 0.0, f"transition {i}: terminal must have disc=0"
            assert not pns.any(), f"transition {i}: terminal next state != 0"
        else:
            assert abs(pdisc - agent.gamma) < 1e-12, (
                f"transition {i}: non-terminal must bootstrap at disc=gamma")
            assert np.array_equal(pns, ns), f"transition {i}: next-state"
    print(f"  OK  n=1 equivalence: {len(raws)} transitions "
          f"({n_term} terminal, {len(raws) - n_term} bootstrapped) over "
          f"{stats['steps']} env steps — new pipeline == old 1-step scheme")


def _make_strat_state(dists_flags, dim=54):
    """Relative-encoder-shaped vector with the given neighbour
    (distance_nm, flag) pairs; remaining blocks stay zero-padded."""
    s = np.zeros(dim, dtype=np.float32)
    for i, (d, f) in enumerate(dists_flags):
        base = REL_BASE_AND_FIX + i * REL_NEIGH_FEATS
        s[base + REL_NEIGH_DIST] = d / REL_DIST_SCALE
        s[base + REL_NEIGH_FLAG] = f
    return s


def _selftest_stratified_t(seed=321):
    """Stratified-t unit test: stratum function on crafted vectors, stratum
    ids recorded through the windower, and per-stratum t divergence —
    synthetic realized streams with dispersed targets in stratum 0 (mostly
    outside the net's intervals -> t rises) vs on-midpoint targets in
    stratum 2 (inside -> t falls), stratum 1 untouched (t unchanged)."""
    print("[selftest] stratified-t: stratum fn, bucketing, t divergence")

    # 1. stratum function (padding, boundaries, unsorted blocks)
    cases = [
        ([(5.0, 1.0)], 0),
        ([(40.0, 1.0), (8.0, -1.0)], 0),   # min over unsorted blocks
        ([(20.0, -1.0)], 1),
        # boundaries (exact 10/30 NM are fp32-fuzzy after the /50 round-trip,
        # so probe just inside each edge)
        ([(10.5, 1.0)], 1),
        ([(29.5, 1.0)], 1),
        ([(9.5, 1.0)], 0),
        ([(30.5, 1.0)], 2),
        ([], 2),                           # no neighbour at all
        ([(100.0, 1.0)], 2),
        ([(5.0, 0.0)], 2),                 # flag 0 = padding: dist ignored
    ]
    for spec, want in cases:
        got = state_stratum(_make_strat_state(spec))
        assert got == want, f"state_stratum({spec}) = {got}, want {want}"
    print(f"  OK  state_stratum: {len(cases)} crafted vectors")

    # 2. windower stores the window STATE's stratum at push time
    buf = _ListBuffer()
    w = NStepWindower(buf, 0.9, 2, strat_fn=state_stratum)
    sts = [_make_strat_state([(5.0, 1.0)]),
           _make_strat_state([(20.0, 1.0)]),
           _make_strat_state([])]
    z = np.zeros(54, dtype=np.float32)
    w.add(sts[0], 0, 1.0, sts[1], False)
    w.add(sts[1], 1, 1.0, sts[2], False)   # completes window 0
    w.add(sts[2], 2, 1.0, z, True)         # terminal: flushes windows 1, 2
    got = [item[5] for item in buf.items]
    assert got == [0, 1, 2], f"windower strata {got} != [0, 1, 2]"
    print("  OK  windower: per-window stratum ids recorded at push time")

    # 3. bucketing via record_realized_stream + t divergence
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent = IntervalDQNAgent(54, 4, device="cpu", strat_t=True)
    s0 = _make_strat_state([(5.0, 1.0)])
    s2 = _make_strat_state([])
    with torch.no_grad():
        l0, u0 = agent.q_net(torch.from_numpy(s0).unsqueeze(0))
        l2, u2 = agent.q_net(torch.from_numpy(s2).unsqueeze(0))
    mid2 = float((l2[0, 0] + u2[0, 0]) / 2)
    for k in range(300):
        # stratum 0: dispersed targets, always outside the interval
        g0 = (float(u0[0, 0]) + 50.0 + k if k % 2 == 0
              else float(l0[0, 0]) - 50.0 - k)
        agent.record_realized_stream([s0], [0], [g0])
        # stratum 2: target dead-centre, always inside
        agent.record_realized_stream([s2], [0], [mid2])
    assert len(agent.trackers[0].hits) == 300, "stratum-0 bucketing"
    assert len(agent.trackers[2].hits) == 300, "stratum-2 bucketing"
    assert len(agent.trackers[1].hits) == 0, "stratum 1 must stay empty"
    t_before = [tr.t for tr in agent.trackers]
    for _ in range(100):
        for tr in agent.trackers:
            tr.update_t()
    t_after = [tr.t for tr in agent.trackers]
    assert t_after[0] > t_before[0], "low-coverage stratum: t must rise"
    assert t_after[2] < t_before[2], "high-coverage stratum: t must fall"
    assert abs(t_after[1] - t_before[1]) < 1e-12, "empty stratum: t frozen"
    assert t_after[0] > t_after[1] > t_after[2], "t ordering"
    print(f"  OK  t divergence after 100 updates: "
          f"t={[round(t, 4) for t in t_after]} "
          f"cov={[round(tr.coverage, 2) for tr in agent.trackers]} "
          f"n={[len(tr.hits) for tr in agent.trackers]}")


def run_selftest():
    print("=" * 100)
    print("N-STEP WINDOW + STRATIFIED-T SELF-TESTS")
    print("=" * 100)
    _selftest_synthetic_windows()
    _selftest_stratified_t()
    _selftest_n1_equivalence()
    print("SELFTEST PASSED")


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
    parser.add_argument("--selftest", action="store_true",
                        help="Run the n-step window semantics self-tests "
                             "(n=1 equivalence + hand-computed windows)")
    # common
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Training episodes (default: 300; smoke: 2)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Scenario duration in simulated seconds "
                             "(default: 600; smoke: 120). Use 120 for "
                             "faster early training.")
    parser.add_argument("--k", type=int, default=3,
                        help="k nearest aircraft in the observation")
    parser.add_argument("--encoder", type=str, default="relative",
                        help="observation encoder_cls (run 7 default: "
                             "relative; runs 2-6 used extra_minimal)")
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
    parser.add_argument("--lr_decay_ep", type=int, default=0,
                        help="episode at which to step lr down x0.3 once "
                             "(0 disables)")
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer", type=int, default=BUFFER_SIZE)
    parser.add_argument("--target_coverage", type=float, default=0.85)
    parser.add_argument("--width_reg", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Epsilon warmup env steps (default: 1000; "
                             "smoke: 20)")
    parser.add_argument("--warmup_epsilon", type=float, default=0.5)
    parser.add_argument("--progress_coeff", type=float, default=0.05,
                        help="potential-based progress shaping coefficient "
                             "(pay for approaching the exit; 0 disables)")
    parser.add_argument("--exit_bonus", type=float, default=10.0,
                        help="terminal reward for a clean exit (outcome "
                             "anchoring; set 0 to disable)")
    parser.add_argument("--centreline_coeff", type=float, default=0.2,
                        help="weight of the on-route income shaping term "
                             "(run 2 used 1.0; small keeps returns "
                             "outcome-dominated)")
    parser.add_argument("--violation_penalty", type=float, default=50.0,
                        help="Reward penalty for aircraft involved in a "
                             "violation (their transition is terminal). "
                             "Run 7 default 50: violations are "
                             "lexicographically bad, unifying with the "
                             "MCTS objective (runs 2-6 used 10)")
    parser.add_argument("--nstep", type=int, default=6,
                        help="n-step interval Bellman window length "
                             "(1 reproduces the old 1-step targets)")
    parser.add_argument("--terminal_boost", type=float, default=1.0,
                        help="oversample realized-G (terminal) replay windows "
                             "by this factor so real outcomes get gradient "
                             "share (run-8 forensics: they are ~4%% of "
                             "windows); 1.0 = off, run-9 recipe uses 3.0")
    parser.add_argument("--t_cap_risky", type=float, default=0.99,
                        help="t cap for the risky (<10nm) stratum when "
                             "--strat_t is on; other strata keep 0.95")
    parser.add_argument("--strat_t", action="store_true",
                        help="stratified coverage control: one CoverageTracker"
                             " per risk stratum (nearest-neighbour distance "
                             "<10 / 10-30 / >30 NM-or-none from the relative "
                             "encoder), per-sample t in the interval loss. "
                             "OFF by default; off = bit-identical to the "
                             "unstratified scheme")
    # bookkeeping
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path for --eval")
    parser.add_argument("--adaptive", action="store_true",
                        help="eval with width-conditioned per-aircraft c "
                             "(LunarLander adaptive-c rule; overrides --c)")
    parser.add_argument("--c", type=float, default=0.0,
                        help="Hurwicz c for --eval")
    parser.add_argument("--eval_episodes", type=int, default=5,
                        help="Episodes (seeds) for --eval")
    parser.add_argument("--final_eval_episodes", type=int, default=None,
                        help="Episodes per c in the post-training eval "
                             "(default: 3; smoke: 1; 0 disables)")
    parser.add_argument("--ckpt_every", type=int, default=25)
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    elif args.smoke:
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
