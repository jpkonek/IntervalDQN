"""
Ensemble-RPF Interval DQN over the controller frame (run-13 groundwork)
=======================================================================

WHY (the A4 finding, 2026-07-07): the single controller net's interval
width is NOT a trustworthy epistemic signal. The pilot-frame nets' 10-40x
OOD widening was an architectural accident (unbounded ReLU magnitude
extrapolation; LayerNorm deletes it), and the trained controller net maps
garbage input into a learned "certain -50" basin: garbage_A4 width 3.35 vs
real 35.0 (ratio 0.10 — NARROWER on garbage). The honest epistemic
mechanism is an ensemble with randomized prior functions (Osband et al.,
"Randomized Prior Functions for Deep Reinforcement Learning", 2018):
M independently seeded members, each = trainable net + frozen random
prior, epistemic uncertainty = member DISAGREEMENT. The member intervals
form a finite credal set — the imprecise-probability object this research
program is actually about — and the credal hull [min_m l_m, max_m u_m]
is its interval summary.

DESIGN DECISIONS (each one deliberate; see the report accompanying this
module):

1. PRIOR APPLICATION POINT: member output = f_theta(s) + beta * p(s)
   applied to the LOWER head and to delta_raw PRE-softplus:
       l_member    = l_f + beta * l_p
       raw_member  = raw_f + beta * raw_p
       u_member    = l_member + softplus(raw_member) + 1e-6
   This keeps u >= l structurally (softplus ordering survives the prior
   addition — the guardrail is asserted in selftest 4). The raw head
   values are captured EXACTLY via forward hooks on `instr_head` /
   `noop_head` of the UNMODIFIED ControllerQNet (their final Linear
   emits (lower, delta_raw) directly), so nothing from
   bluebird_controller_dqn.py is copied or forked.

2. BETA DEFAULT 3.0: reward scale is terminals +-50, per-step terms
   <~0.1, and returns-to-go typically O(10). An untrained ControllerQNet
   head output is O(1), so beta=3 gives prior perturbations of O(3) on
   both l and raw — large vs per-step signal (members genuinely disagree
   early), small vs the terminal scale (training data can override the
   prior where it has evidence). NOTE, stated honestly: if the trained
   ensemble collapses OOD into the same "certain -50" basin, beta may
   need to be 10-30 so the prior can move Q-values at the |Q| ~ 50
   scale. Configurable; sweep before run 13.

3. REPLAY SHARING: ONE shared replay buffer, NO bootstrap masks; member
   diversity comes from independent init + independent frozen priors +
   independent minibatch draws. This is the configuration the RPF paper
   found sufficient (bootstrapping added little once priors were in);
   it also keeps the buffer memory O(1) in M. Each member samples its
   OWN minibatch per train step.

4. AGGREGATION: credal interval [min_m l_m, max_m u_m] over the member
   intervals (the hull of the credal set), plus TWO disagreement
   scalars, both exposed:
     - disagreement_mid    = mean over candidates of std_m(midpoint_m)
     - disagreement_hurwicz= mean over candidates of std_m(l_m + c(u_m-l_m))
   Action selection = Hurwicz on the CREDAL interval with the base
   agent's machinery (same tie-to-NOOP rule, re-issue masking, epsilon
   warmup, adaptive-c sigmoid via bid.IntervalDQNAgent.adaptive_c —
   reused by import, not copied).

5. PER-MEMBER COVERAGE/t: each member keeps its own stratified
   CoverageTracker state, fed with realized returns against ITS OWN
   intervals (delegation in record_realized_episode) — coverage is a
   property of a member's intervals, not of the ensemble.

6. TRAINING = thin orchestrator: ensemble.train_step() runs one
   ControllerAgent.train_step() per member (unchanged base code, each
   with its own optimizer/target net/trackers) against the shared
   buffer. bcd.run_episode works on the ensemble by duck-typing (same
   generate_action/buffer/train_step/record_realized_episode surface),
   so windowing/CBP/objective code is reused untouched. Compute: per
   net call the RPF wrapper runs 2 forwards (trainable + prior), and
   there are M members, so network compute is ~2M x the single agent;
   env stepping (the CBP-dominated cost) is shared, so the wall-clock
   training multiplier is well below 2M (M4 Pro estimate printed by
   selftest 7).

7. CHECKPOINT: ONE file: per-member trainable / target / PRIOR
   state_dicts + tracker t's + step counts + all constructor metadata;
   load_ensemble restores exactly (selftest 6 asserts bitwise-equal
   outputs after a round trip).

Usage:
    python bluebird_controller_ensemble.py --selftest
    python bluebird_controller_ensemble.py --train --episodes 1500 \
        --duration 1200 --members 4 --beta 3.0
"""

import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bluebird_controller_dqn as bcd
import bluebird_interval_dqn as bid
from bluebird_controller_dqn import (ControllerAgent, ControllerQNet,
                                     ControllerReplayBuffer, build_tokens,
                                     build_reissue_mask, make_controller_env,
                                     pad_state_batch, N_INSTR, GAMMA,
                                     BUFFER_SIZE, BATCH_SIZE, TERMINAL_BOOST,
                                     NSTEP, CBP_LAG, KIN_FEATS)

sys.stdout.reconfigure(line_buffering=True)

N_MEMBERS = 4          # default ensemble size (configurable)
BETA_PRIOR = 3.0       # default prior scale (see design decision 2)

ENSEMBLE_CKPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints", "bluebird_controller_ensemble")


# ===========================================================================
# RAW-HEAD CAPTURE + RPF WRAPPER
# ===========================================================================

def _raw_heads(net, tokens, mask=None):
    """(ac_lower [B,N,I], ac_delta_raw [B,N,I], noop_lower [B],
    noop_delta_raw [B]) of a ControllerQNet, captured PRE-softplus via
    forward hooks on its head submodules. The net's own forward runs
    unmodified (mask plumbing, empty-row attention fix, pooling), so this
    is exact — the final Linear of instr_head/noop_head IS the
    (lower, delta_raw) pair, no numeric inversion involved."""
    cap = {}
    handles = [
        net.instr_head.register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("instr", o)),
        net.noop_head.register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("noop", o)),
    ]
    try:
        net(tokens, mask)
    finally:
        for h in handles:
            h.remove()
    B, N, _ = tokens.shape
    n_instr = net.n_instr
    if "instr" in cap:
        out = cap["instr"].view(B, N, n_instr, 2)
        ac_l, ac_raw = out[..., 0], out[..., 1]
    else:
        # N == 0: ControllerQNet.forward never calls instr_head
        ac_l = tokens.new_zeros(B, 0, n_instr)
        ac_raw = tokens.new_zeros(B, 0, n_instr)
    no = cap["noop"]
    return ac_l, ac_raw, no[:, 0], no[:, 1]


class RPFQNet(nn.Module):
    """Randomized-prior member net: trainable ControllerQNet `f` + FROZEN
    randomly-initialized ControllerQNet `p` of the same architecture.

    Output contract is IDENTICAL to ControllerQNet.forward
    ((ac_l, ac_u, noop_l, noop_u)), so every consumer in
    bluebird_controller_dqn (candidate_q, train_step, Double-DQN targets,
    record_realized_episode) works on it unchanged. The prior is added at
    the raw head level (design decision 1):
        l = l_f + beta * l_p
        u = l + softplus(raw_f + beta * raw_p) + 1e-6   (=> u >= l always)
    beta == 0 reproduces the base net exactly up to 1 ulp on u (the
    softplus kernel dispatches differently for the base net's strided
    head view vs our contiguous sum; measured max 1.2e-7). The prior
    forward runs
    under no_grad; its params have requires_grad=False, so the member's
    optimizer/grad-clip (built over f before wrapping) never touch it.
    """

    def __init__(self, trainable, prior, beta):
        super().__init__()
        self.f = trainable
        self.p = prior
        for q in self.p.parameters():
            q.requires_grad_(False)
        self.p.eval()
        self.beta = float(beta)
        self.n_instr = trainable.n_instr
        self.token_dim = trainable.token_dim

    def forward(self, tokens, mask=None):
        lf, rf, nlf, nrf = _raw_heads(self.f, tokens, mask)
        with torch.no_grad():
            lp, rp, nlp, nrp = _raw_heads(self.p, tokens, mask)
        b = self.beta
        ac_l = lf + b * lp
        ac_u = ac_l + F.softplus(rf + b * rp) + 1e-6
        noop_l = nlf + b * nlp
        noop_u = noop_l + F.softplus(nrf + b * nrp) + 1e-6
        return ac_l, ac_u, noop_l, noop_u


def _param_l1_dist(net_a, net_b):
    """Sum of |a - b| over matching parameters (0.0 iff identical)."""
    with torch.no_grad():
        return sum(float((pa - pb).abs().sum())
                   for pa, pb in zip(net_a.parameters(),
                                     net_b.parameters()))


# ===========================================================================
# ENSEMBLE AGENT — thin orchestrator over M ControllerAgents
# ===========================================================================

class EnsembleControllerAgent:
    """M RPF members + credal aggregation. Duck-type compatible with
    ControllerAgent where bcd.run_episode / bcd.evaluate need it
    (generate_action, buffer, gamma, n_instr, mask_reissue, train_step,
    record_realized_episode, total_env_steps, coverage properties)."""

    def __init__(self, token_dim, n_members=N_MEMBERS, beta=BETA_PRIOR,
                 ensemble_seed=0, device="cpu", **agent_kwargs):
        self.token_dim = token_dim
        self.n_members = int(n_members)
        self.beta = float(beta)
        self.ensemble_seed = int(ensemble_seed)
        self.device = torch.device(device)

        # ONE shared replay buffer (design decision 3); members get a
        # dummy size-1 buffer at construction and are re-pointed at it.
        buffer_size = agent_kwargs.pop("buffer_size", BUFFER_SIZE)
        terminal_boost = agent_kwargs.pop("terminal_boost", TERMINAL_BOOST)
        self.buffer = ControllerReplayBuffer(
            buffer_size, terminal_boost=terminal_boost)

        self.members = []
        for m in range(self.n_members):
            torch.manual_seed(self._member_seed(m, prior=False))
            member = ControllerAgent(token_dim, device=device,
                                     buffer_size=1,
                                     terminal_boost=terminal_boost,
                                     **agent_kwargs)
            member.buffer = self.buffer
            torch.manual_seed(self._member_seed(m, prior=True))
            prior = ControllerQNet(token_dim,
                                   n_instr=member.n_instr).to(self.device)
            # target net shares the SAME frozen prior instance: the
            # periodic target load_state_dict copies prior params onto
            # themselves (no-op), and the target evaluates the same
            # f_target + beta*p composition.
            member.q_net = RPFQNet(member.q_net, prior, self.beta)
            member.target_net = RPFQNet(member.target_net, prior, self.beta)
            self.members.append(member)

        # independence guarantee: every pair of trainable nets AND every
        # pair of priors must differ (selftest 1 re-checks pairwise).
        for a in range(self.n_members):
            for b in range(a + 1, self.n_members):
                assert _param_l1_dist(self.members[a].q_net.f,
                                      self.members[b].q_net.f) > 0.0, \
                    f"members {a},{b}: identical trainable init"
                assert _param_l1_dist(self.members[a].q_net.p,
                                      self.members[b].q_net.p) > 0.0, \
                    f"members {a},{b}: identical priors"

        # mirrored attributes (run_episode / selection surface)
        m0 = self.members[0]
        self.gamma = m0.gamma
        self.n_instr = m0.n_instr
        self.c_train = m0.c_train
        self.mask_reissue = m0.mask_reissue
        self.warmup_steps = m0.warmup_steps
        self.warmup_epsilon = m0.warmup_epsilon
        self.total_env_steps = 0
        self.realized_episodes = 0
        self.censored_episodes = 0

    def _member_seed(self, m, prior):
        return (self.ensemble_seed * 1000003 + m * 7919
                + (104729 if prior else 0)) % (2 ** 31)

    # ---- diagnostics passthroughs (means over members) ----------------
    @property
    def t(self):
        return float(np.mean([m.t for m in self.members]))

    @property
    def coverage(self):
        return float(np.mean([m.coverage for m in self.members]))

    @property
    def bootstrap_coverage(self):
        return float(np.mean([m.bootstrap_coverage for m in self.members]))

    def trainable_param_count(self):
        return sum(p.numel() for m in self.members
                   for p in m.q_net.f.parameters())

    def frozen_param_count(self):
        return sum(p.numel() for m in self.members
                   for p in m.q_net.p.parameters())

    def param_count(self):
        return self.trainable_param_count()

    # ---- epsilon: EXACT reuse of the base warmup rule ------------------
    def _epsilon(self, force_epsilon=None):
        return ControllerAgent._epsilon(self, force_epsilon)

    # ---- credal aggregation --------------------------------------------
    def credal_q(self, tokens_np, c=None):
        """Member candidate intervals stacked + credal hull + disagreement.

        Returns a dict:
          L, U          [M, n_cand]  per-member candidate intervals
          l, u          [n_cand]     credal hull  [min_m l_m, max_m u_m]
          disagreement_mid      mean over candidates of std_m(midpoints)
          disagreement_hurwicz  mean over candidates of std_m(hurwicz@c)
          mean_member_width / mean_hull_width
        """
        Ls, Us = [], []
        for m in self.members:
            cl, cu = m.candidate_q(tokens_np)   # base method, RPF q_net
            Ls.append(cl)
            Us.append(cu)
        L = np.stack(Ls)
        U = np.stack(Us)
        l = L.min(axis=0)
        u = U.max(axis=0)
        use_c = self.c_train if c is None else c
        mid = 0.5 * (L + U)
        hur = L + use_c * (U - L)
        return {
            "L": L, "U": U, "l": l, "u": u,
            "disagreement_mid": float(mid.std(axis=0).mean()),
            "disagreement_hurwicz": float(hur.std(axis=0).mean()),
            "mean_member_width": float((U - L).mean()),
            "mean_hull_width": float((u - l).mean()),
        }

    # ---- action selection: Hurwicz on the CREDAL interval ---------------
    def select_candidate(self, tokens_np, c=None, adaptive=False,
                         reissue_mask=None):
        """Same semantics as ControllerAgent.select_candidate (tie-to-NOOP,
        re-issue masking, adaptive-c sigmoid on the mean width) but scored
        on the credal hull. Returns (idx, mean_hull_width, c_used, cq)."""
        use_c = c if c is not None else self.c_train
        cq = self.credal_q(tokens_np, c=use_c)
        l, u = cq["l"], cq["u"]
        mean_width = float((u - l).mean()) if l.size else 0.0
        if adaptive:
            w_mid = getattr(self, "adaptive_w_mid", 4.0)
            use_c = float(bid.IntervalDQNAgent.adaptive_c(
                torch.tensor(mean_width), w_mid=w_mid))
        scores = l + use_c * (u - l)
        if reissue_mask is not None:
            scores = np.where(reissue_mask, -np.inf, scores)
        best = scores.max()
        idx = 0 if scores[0] == best else int(scores.argmax())
        return idx, mean_width, float(use_c), cq

    def calibrate_w_mid(self, token_arrays):
        """Adaptive-c midpoint from the median mean CREDAL width over a
        probe sample (ensemble analogue of the base calibrate_w_mid)."""
        widths = []
        for t in token_arrays:
            if t.shape[0] == 0:
                continue
            cq = self.credal_q(t)
            widths.append(float((cq["u"] - cq["l"]).mean()))
        self.adaptive_w_mid = float(np.median(widths)) if widths else 4.0
        return self.adaptive_w_mid

    def generate_action(self, env, obs_dict, info_dict, c=None,
                        force_epsilon=None, adaptive=False,
                        last_issued=None):
        """Mirror of ControllerAgent.generate_action (same contract, same
        epsilon/masking semantics — the ~20 lines of glue are duplicated
        here because the base method hard-binds its own select_candidate)
        with credal selection and disagreement telemetry in aux."""
        cs_list = sorted(obs_dict.keys())
        tokens = build_tokens(env, obs_dict, info_dict, cs_list)
        rmask = None
        if self.mask_reissue and last_issued:
            rmask = build_reissue_mask(cs_list, last_issued, self.n_instr)
        idx, mean_width, c_used, cq = self.select_candidate(
            tokens, c=c, adaptive=adaptive, reissue_mask=rmask)
        eps = self._epsilon(force_epsilon)
        if random.random() < eps:
            n_cand = 1 + self.n_instr * len(cs_list)
            if rmask is not None:
                idx = random.choice(
                    [k for k in range(n_cand) if not rmask[k]])
            else:
                idx = random.randrange(n_cand)
        actions = {cs: 0 for cs in cs_list}
        if idx > 0:
            i, j = divmod(idx - 1, self.n_instr)
            actions[cs_list[i]] = j + 1
        aux = {"cand_idx": idx, "tokens": tokens,
               "callsigns": tuple(cs_list), "mean_width": mean_width,
               "c_used": c_used,
               "disagreement_mid": cq["disagreement_mid"],
               "disagreement_hurwicz": cq["disagreement_hurwicz"],
               "mean_member_width": cq["mean_member_width"]}
        return actions, aux

    # ---- training orchestration (design decision 6) ---------------------
    def train_step(self):
        """One base-agent gradient step PER MEMBER, each sampling its own
        minibatch from the SHARED buffer. Per-member optimizers, target
        nets and stratified-t trackers are untouched base machinery."""
        losses = [m.train_step() for m in self.members]
        return float(np.mean(losses))

    def record_realized_episode(self, states, a_idxs, strata, rewards):
        """Delegate to every member: realized coverage is measured against
        each member's OWN intervals and drives ITS stratified t."""
        for m in self.members:
            m.record_realized_episode(states, a_idxs, strata, rewards)
        self.realized_episodes += 1


# ===========================================================================
# CHECKPOINTS — all members + priors + metadata in ONE file
# ===========================================================================

def save_ensemble(ens, path, episode, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "ensemble_controller": True,
        "n_members": ens.n_members,
        "beta": ens.beta,
        "ensemble_seed": ens.ensemble_seed,
        "token_dim": ens.token_dim,
        "n_instr": ens.n_instr,
        "gamma": ens.gamma,
        "c_train": ens.c_train,
        "mask_reissue": ens.mask_reissue,
        "episode": episode,
        "total_env_steps": ens.total_env_steps,
        "adaptive_w_mid": getattr(ens, "adaptive_w_mid", None),
        "coverage": ens.coverage,
        "members": [{
            "q_f": m.q_net.f.state_dict(),
            "target_f": m.target_net.f.state_dict(),
            "prior": m.q_net.p.state_dict(),
            "strat_t_values": [tr.t for tr in m.trackers],
            "step_count": m.step_count,
        } for m in ens.members],
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_ensemble(path, device="cpu", **kwargs):
    """Exact restore: trainable nets, target nets, PRIORS, tracker t's,
    step counts, beta/M/metadata. Returns (ensemble, ckpt)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt.get("ensemble_controller"), \
        f"{path} is not an ensemble controller checkpoint"
    ens = EnsembleControllerAgent(
        token_dim=ckpt["token_dim"],
        n_members=ckpt["n_members"],
        beta=ckpt["beta"],
        ensemble_seed=ckpt.get("ensemble_seed", 0),
        device=device,
        n_instr=ckpt.get("n_instr", N_INSTR),
        c_train=ckpt.get("c_train", 0.5),
        gamma=ckpt.get("gamma", GAMMA),
        mask_reissue=ckpt.get("mask_reissue", True),
        **kwargs)
    for m, md in zip(ens.members, ckpt["members"]):
        m.q_net.f.load_state_dict(md["q_f"])
        m.q_net.p.load_state_dict(md["prior"])
        m.target_net.f.load_state_dict(md["target_f"])
        m.target_net.p.load_state_dict(md["prior"])   # shared instance;
        # explicit for clarity — q_net.p IS target_net.p
        for tr, tv in zip(m.trackers, md.get("strat_t_values", [])):
            tr.t = tv
        m.step_count = md.get("step_count", 0)
        m.total_env_steps = 10 ** 9
        m.q_net.eval()
        m.target_net.eval()
    if ckpt.get("adaptive_w_mid") is not None:
        ens.adaptive_w_mid = ckpt["adaptive_w_mid"]
    ens.total_env_steps = 10 ** 9   # past warmup: no epsilon at eval
    return ens, ckpt


# ===========================================================================
# TRAINING HARNESS — thin: bcd.run_episode drives the ensemble directly
# ===========================================================================

def run_training(args):
    """Structural mirror of bcd.run_training with the ensemble agent;
    windowing/CBP/objective/eval logic is bcd's, reused via run_episode
    duck-typing. NOT exercised by --selftest (no training runs allowed
    in this session); the selftests cover generate_action against the
    live env and train_step against fake replay data."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 100)
    print(f"BLUEBIRD CONTROLLER ENSEMBLE-RPF INTERVAL DQN — TRAINING "
          f"(seed={args.seed}, M={args.members}, beta={args.beta}, "
          f"c_train={args.c_train}, duration={args.duration}s, "
          f"episodes={args.episodes}, gamma={args.gamma}, "
          f"nstep={args.nstep}, cbp={'ON' if args.cbp else 'OFF'})")
    print("=" * 100)

    env = make_controller_env(scenario_duration=args.duration,
                              k_nearest=args.k)
    obs, info = env.reset(seed=args.seed)
    obs_dim = int(next(iter(obs.values())).shape[0])
    token_dim = obs_dim + KIN_FEATS

    ens = EnsembleControllerAgent(
        token_dim, n_members=args.members, beta=args.beta,
        ensemble_seed=args.seed, device=args.device,
        lr=args.lr, gamma=args.gamma, c_train=args.c_train,
        batch_size=args.batch, buffer_size=args.buffer,
        warmup_steps=args.warmup_steps,
        mask_reissue=args.mask_reissue)
    print(f"trainable params: {ens.trainable_param_count()} "
          f"({args.members} x {ens.trainable_param_count() // args.members}); "
          f"frozen priors: {ens.frozen_param_count()}")

    os.makedirs(ENSEMBLE_CKPT_DIR, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(ENSEMBLE_CKPT_DIR,
                            f"train_seed{args.seed}_{run_tag}.jsonl")
    log_file = open(log_path, "w")
    print(f"Logging to {log_path}")

    for ep in range(args.episodes):
        t0 = time.time()
        stats = bcd.run_episode(env, ens, seed=args.seed + ep, train=True,
                                nstep=args.nstep, cbp=args.cbp,
                                cbp_lag=args.cbp_lag, objective_v2=True)
        wall = time.time() - t0
        record = {"episode": ep + 1, "seed": args.seed + ep, **stats,
                  "buffer_size": len(ens.buffer),
                  "coverage": round(ens.coverage, 4),
                  "t": round(ens.t, 4),
                  "wall_time_s": round(wall, 2)}
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        vio = (f"VIOLATION[{stats['violation_kind']}]"
               f"@{stats['time_to_violation']}s" if stats["violated"]
               else f"clean({stats['time_to_violation']}s)")
        print(f"  Ep {ep+1:4d} | {vio:>38} | G:{stats['ep_return']:8.2f} "
              f"| del:{stats['deliveries']} cmd:{stats['commands']:3d} "
              f"| Whull:{stats['mean_width']:.3f} rCvg:{ens.coverage:.2f} "
              f"| buf:{len(ens.buffer):6d} | {wall:.1f}s")
        if (ep + 1) % args.ckpt_every == 0:
            path = os.path.join(ENSEMBLE_CKPT_DIR,
                                f"train_seed{args.seed}_{run_tag}"
                                f"_ep{ep+1}.pt")
            save_ensemble(ens, path, ep + 1)
            print(f"    Saved: {path}")
    final = os.path.join(ENSEMBLE_CKPT_DIR,
                         f"train_seed{args.seed}_{run_tag}_final.pt")
    save_ensemble(ens, final, args.episodes)
    print(f"Final ensemble checkpoint: {final}")
    env.close()
    log_file.close()
    return ens


# ===========================================================================
# SELF-TESTS
# ===========================================================================

_D_TOK = 60   # k=3 relative obs (54) + 6 kinematics


def _fake_state(rng, n, d=_D_TOK):
    toks = rng.standard_normal((n, d)).astype(np.float32)
    return (toks, tuple(f"FAKE{i:02d}" for i in range(n)))


def _fill_fake_replay(ens, rng, n_items=64):
    """Synthetic transitions in the exact replay layout
    (state, a_idx, R_n, next_state, ns_mask, disc, stratum), mixing
    bootstrap and terminal items and variable aircraft counts."""
    for k in range(n_items):
        n = int(rng.integers(2, 6))
        s = _fake_state(rng, n)
        a_idx = int(rng.integers(0, 1 + N_INSTR * n))
        r = float(rng.normal(0.0, 5.0))
        terminal = (k % 5 == 0)
        if terminal:
            ns = (np.zeros((0, _D_TOK), dtype=np.float32), ())
            ns_mask = np.zeros(1, dtype=bool)
            disc = 0.0
            r -= 50.0
        else:
            n2 = int(rng.integers(2, 6))
            ns = _fake_state(rng, n2)
            ns_mask = np.zeros(1 + N_INSTR * n2, dtype=bool)
            disc = GAMMA ** NSTEP
        ens.buffer.push(s, a_idx, r, ns, ns_mask, disc,
                        int(rng.integers(0, 3)))


def _selftest_member_independence():
    """1. Pairwise parameter distance > 0 for all trainable nets AND all
    priors (independent seeds actually took effect)."""
    print("[selftest 1] member independence")
    ens = EnsembleControllerAgent(_D_TOK, n_members=4, beta=BETA_PRIOR,
                                  ensemble_seed=42)
    min_f, min_p = float("inf"), float("inf")
    for a in range(ens.n_members):
        for b in range(a + 1, ens.n_members):
            df = _param_l1_dist(ens.members[a].q_net.f,
                                ens.members[b].q_net.f)
            dp = _param_l1_dist(ens.members[a].q_net.p,
                                ens.members[b].q_net.p)
            assert df > 0.0, f"FAIL: trainable nets {a},{b} identical"
            assert dp > 0.0, f"FAIL: priors {a},{b} identical"
            min_f, min_p = min(min_f, df), min(min_p, dp)
        # a member's prior must also differ from its own trainable net
        d_self = _param_l1_dist(ens.members[a].q_net.f,
                                ens.members[a].q_net.p)
        assert d_self > 0.0, f"FAIL: member {a} prior == trainable"
    print(f"  OK  all pairs differ (min L1: trainable {min_f:.1f}, "
          f"priors {min_p:.1f})")
    return ens


def _selftest_priors_frozen():
    """2. No prior param requires grad; after real optimization steps on
    fake replay data the priors are BITWISE unchanged while every
    member's trainable net moved."""
    print("[selftest 2] priors frozen under optimization")
    rng = np.random.default_rng(7)
    ens = EnsembleControllerAgent(_D_TOK, n_members=3, beta=BETA_PRIOR,
                                  ensemble_seed=7, batch_size=16)
    for m in ens.members:
        for p in m.q_net.p.parameters():
            assert not p.requires_grad, "FAIL: prior param requires_grad"
    _fill_fake_replay(ens, rng, n_items=64)
    prior_before = [[p.detach().clone() for p in m.q_net.p.parameters()]
                    for m in ens.members]
    train_before = [[p.detach().clone() for p in m.q_net.f.parameters()]
                    for m in ens.members]
    losses = [ens.train_step() for _ in range(3)]
    assert all(np.isfinite(losses)), f"FAIL: non-finite losses {losses}"
    for i, m in enumerate(ens.members):
        for p0, p1 in zip(prior_before[i], m.q_net.p.parameters()):
            assert torch.equal(p0, p1), \
                f"FAIL: member {i} prior param changed under optimization"
        moved = any(not torch.equal(p0, p1) for p0, p1
                    in zip(train_before[i], m.q_net.f.parameters()))
        assert moved, f"FAIL: member {i} trainable net did not move"
    print(f"  OK  3 ensemble train steps (losses {[f'{l:.3f}' for l in losses]}): "
          f"priors bitwise unchanged, all trainable nets moved, "
          f"no prior grads")
    return ens


def _selftest_rpf_plumbing():
    """3. Member output actually shifts with beta: beta=0 reproduces the
    bare trainable net exactly; two members with IDENTICAL trainable
    weights agree at beta=0 and diverge at beta>0 (their priors differ)."""
    print("[selftest 3] RPF plumbing (beta wiring)")
    torch.manual_seed(303)
    ens = EnsembleControllerAgent(_D_TOK, n_members=2, beta=0.0,
                                  ensemble_seed=303)
    m0, m1 = ens.members
    m1.q_net.f.load_state_dict(m0.q_net.f.state_dict())
    tok = torch.randn(2, 5, _D_TOK)
    mask = torch.ones(2, 5, dtype=torch.bool)

    def fw(net):
        with torch.no_grad():
            return net(tok, mask)

    # beta=0 == bare trainable net (hook path fidelity). Tolerance 1e-6:
    # softplus dispatches a different kernel for the base net's strided
    # head view than for our contiguous raw sum -> 1-ulp differences on
    # u only (l is exactly equal, asserted below).
    bare = ControllerQNet(_D_TOK)
    bare.load_state_dict(m0.q_net.f.state_dict())
    for got, want, name in zip(fw(m0.q_net), fw(bare),
                               ("ac_l", "ac_u", "noop_l", "noop_u")):
        err = (got - want).abs().max().item()
        tol = 0.0 if name in ("ac_l", "noop_l") else 1e-6
        assert err <= tol, \
            f"FAIL: beta=0 {name} differs from bare net by {err:.2e}"
    # identical trainable + beta=0 -> members agree despite different priors
    for a, b, name in zip(fw(m0.q_net), fw(m1.q_net),
                          ("ac_l", "ac_u", "noop_l", "noop_u")):
        assert (a - b).abs().max().item() == 0.0, \
            f"FAIL: beta=0 members with same trainable disagree on {name}"
    # beta>0 shifts a member's own output...
    out_b0 = fw(m0.q_net)
    m0.q_net.beta = BETA_PRIOR
    m1.q_net.beta = BETA_PRIOR
    out_b3 = fw(m0.q_net)
    shift = max((x - y).abs().max().item() for x, y in zip(out_b3, out_b0))
    assert shift > 0.0, "FAIL: beta=3 output identical to beta=0"
    # ...and makes identical-trainable members disagree (priors differ)
    dis = max((x - y).abs().max().item()
              for x, y in zip(fw(m0.q_net), fw(m1.q_net)))
    assert dis > 0.0, \
        "FAIL: beta=3 members with different priors still agree"
    print(f"  OK  beta=0 == bare net (bitwise); beta=0 same-f members "
          f"agree; beta={BETA_PRIOR} shifts output (max {shift:.3f}) and "
          f"separates members (max {dis:.3f})")


def _selftest_credal_aggregation(ens):
    """4. Aggregate interval contains every member interval; per-member
    u >= l survives the prior addition (incl. large beta); disagreement
    is 0 when members are forced identical and > 0 with independent
    seeds."""
    print("[selftest 4] credal aggregation + interval-semantics guardrail")
    rng = np.random.default_rng(44)
    # guardrail: u >= l for every member on random inputs at several betas
    tok = torch.tensor(rng.standard_normal((3, 6, _D_TOK)),
                       dtype=torch.float32)
    mask = torch.ones(3, 6, dtype=torch.bool)
    for beta_probe in (0.0, BETA_PRIOR, 30.0):
        for i, m in enumerate(ens.members):
            old = m.q_net.beta
            m.q_net.beta = beta_probe
            with torch.no_grad():
                ac_l, ac_u, nl, nu = m.q_net(tok, mask)
            m.q_net.beta = old
            assert bool((ac_u >= ac_l).all()) and bool((nu >= nl).all()), \
                (f"FAIL: member {i} u < l at beta={beta_probe} — prior "
                 f"addition broke softplus ordering")
    # containment of every member interval in the credal hull
    toks_np = rng.standard_normal((7, _D_TOK)).astype(np.float32)
    cq = ens.credal_q(toks_np)
    assert (cq["l"][None, :] <= cq["L"] + 1e-12).all() and \
           (cq["u"][None, :] >= cq["U"] - 1e-12).all(), \
        "FAIL: credal hull does not contain every member interval"
    assert (cq["U"] >= cq["L"]).all(), "FAIL: some member u < l"
    # forced-identical members -> disagreement exactly 0
    ens_same = copy.deepcopy(ens)
    f_sd = ens_same.members[0].q_net.f.state_dict()
    p_sd = ens_same.members[0].q_net.p.state_dict()
    for m in ens_same.members:
        m.q_net.f.load_state_dict(f_sd)
        m.q_net.p.load_state_dict(p_sd)
        m.q_net.beta = ens_same.members[0].q_net.beta
    cq_same = ens_same.credal_q(toks_np)
    assert cq_same["disagreement_mid"] == 0.0 and \
           cq_same["disagreement_hurwicz"] == 0.0, \
        (f"FAIL: identical members but disagreement "
         f"{cq_same['disagreement_mid']!r}")
    assert cq["disagreement_mid"] > 0.0 and \
           cq["disagreement_hurwicz"] > 0.0, \
        "FAIL: independent members but zero disagreement"
    print(f"  OK  hull contains all member intervals; u>=l at beta 0/"
          f"{BETA_PRIOR}/30; disagreement 0.0 forced-identical, "
          f"{cq['disagreement_mid']:.3f} (mid) / "
          f"{cq['disagreement_hurwicz']:.3f} (hurwicz) independent")


def _selftest_ood_disagreement(seed=42):
    """5. THE POINT OF THE MODULE: untrained-ensemble disagreement on real
    tokens from a live env reset vs shuffled vs garbage tokens. No
    pass/fail threshold — the ratio is REPORTED honestly (untrained nets
    may not separate; that is a finding, not a failure)."""
    print("[selftest 5] OOD disagreement, untrained ensemble "
          "(builds a real env; one reset + a few steps only)")
    env = make_controller_env(scenario_duration=600)
    obs, info = env.reset(seed=seed)
    obs_dim = int(next(iter(obs.values())).shape[0])
    token_dim = obs_dim + KIN_FEATS
    ens = EnsembleControllerAgent(token_dim, n_members=N_MEMBERS,
                                  beta=BETA_PRIOR, ensemble_seed=seed)
    # exercise the full duck-typed selection path against the live env;
    # the X-Plus scenario populates gradually (a reset state can hold as
    # few as 2 aircraft — too thin a candidate set to summarize), so step
    # forward cheaply until the sector holds a decent token set. Not a
    # training rollout: untrained greedy actions, token harvesting only.
    token_sets = []
    for _ in range(80):
        if not obs:
            break
        actions, aux = ens.generate_action(env, obs, info, c=0.5,
                                           force_epsilon=0.0)
        assert set(actions) == set(obs), "FAIL: action dict keys != obs"
        token_sets.append(aux["tokens"])
        if aux["tokens"].shape[0] >= 12:
            break
        # NOTE: the decentralized env returns done/trunc as PER-CALLSIGN
        # DICTS (truthy even when all False) — run_episode ignores them
        # and so do we; the step cap bounds the loop.
        obs, _r, _done, _trunc, info = env.step(actions)
    real = max(token_sets, key=lambda t: t.shape[0])
    env.close()
    n, d = real.shape
    assert n >= 2, f"FAIL: only {n} aircraft at reset; cannot shuffle"
    rng = np.random.default_rng(seed)
    shuffled = real.copy()
    for j in range(d):   # permute each feature independently across
        shuffled[:, j] = rng.permutation(shuffled[:, j])   # aircraft
    garbage = (rng.standard_normal((n, d)) * 3.0).astype(np.float32)
    rows = {}
    for name, t in (("real", real), ("shuffled", shuffled),
                    ("garbage_3x", garbage)):
        cq = ens.credal_q(t)
        rows[name] = cq
        print(f"    {name:<10} disagree_mid {cq['disagreement_mid']:8.3f}  "
              f"disagree_hur {cq['disagreement_hurwicz']:8.3f}  "
              f"hull_w {cq['mean_hull_width']:8.3f}  "
              f"member_w {cq['mean_member_width']:8.3f}")
    r_shuf = rows["shuffled"]["disagreement_mid"] / \
        max(1e-9, rows["real"]["disagreement_mid"])
    r_garb = rows["garbage_3x"]["disagreement_mid"] / \
        max(1e-9, rows["real"]["disagreement_mid"])
    print(f"  OK  (reported, no threshold) disagreement ratios vs real: "
          f"shuffled {r_shuf:.2f}x, garbage {r_garb:.2f}x — UNTRAINED "
          f"nets; separation is only expected after training pulls "
          f"members together on-distribution ({n} aircraft, "
          f"token dim {d})")
    return token_dim


def _selftest_checkpoint_roundtrip():
    """6. save_ensemble -> load_ensemble -> identical outputs (q AND
    target nets, every member) on a fixed batch; tracker t's restored."""
    print("[selftest 6] checkpoint round-trip")
    rng = np.random.default_rng(66)
    ens = EnsembleControllerAgent(_D_TOK, n_members=2, beta=BETA_PRIOR,
                                  ensemble_seed=66, batch_size=16)
    _fill_fake_replay(ens, rng, n_items=48)
    for _ in range(2):      # desync f from target and move trackers
        ens.train_step()
    ens.calibrate_w_mid([rng.standard_normal((4, _D_TOK))
                         .astype(np.float32)])
    path = os.path.join(ENSEMBLE_CKPT_DIR, "selftest_roundtrip.pt")
    save_ensemble(ens, path, episode=0)
    ens2, ckpt = load_ensemble(path)
    # like-for-like comparison: load_ensemble sets eval mode (base
    # load_agent convention), and nn.MultiheadAttention dispatches a
    # fused fast path in eval that differs from the train-mode path by
    # ~1 ulp — so put the ORIGINAL members in eval too.
    for m in ens.members:
        m.q_net.eval()
        m.target_net.eval()
    tok = torch.tensor(rng.standard_normal((2, 5, _D_TOK)),
                       dtype=torch.float32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    for i, (ma, mb) in enumerate(zip(ens.members, ens2.members)):
        with torch.no_grad():
            for net_name in ("q_net", "target_net"):
                oa = getattr(ma, net_name)(tok, mask)
                ob = getattr(mb, net_name)(tok, mask)
                for a, b, hd in zip(oa, ob,
                                    ("ac_l", "ac_u", "noop_l", "noop_u")):
                    assert torch.equal(a, b), \
                        f"FAIL: member {i} {net_name} {hd} differs " \
                        f"after round-trip"
        for tra, trb in zip(ma.trackers, mb.trackers):
            assert tra.t == trb.t, f"FAIL: member {i} tracker t mismatch"
        assert ma.step_count == mb.step_count, \
            f"FAIL: member {i} step_count mismatch"
    assert getattr(ens2, "adaptive_w_mid", None) == ens.adaptive_w_mid, \
        "FAIL: adaptive_w_mid not restored"
    assert ckpt["beta"] == ens.beta and ckpt["n_members"] == ens.n_members
    sz = os.path.getsize(path)
    os.remove(path)
    print(f"  OK  identical q/target outputs for all members, tracker "
          f"t's + w_mid restored ({sz / 1e6:.1f} MB, single file)")


def _selftest_param_budget():
    """7. Parameter/compute budget report."""
    print("[selftest 7] parameter budget")
    base = sum(p.numel() for p in ControllerQNet(_D_TOK).parameters())
    ens = EnsembleControllerAgent(_D_TOK, n_members=N_MEMBERS,
                                  beta=BETA_PRIOR, ensemble_seed=1)
    tr = ens.trainable_param_count()
    fz = ens.frozen_param_count()
    assert tr == N_MEMBERS * base, \
        f"FAIL: trainable {tr} != {N_MEMBERS} x {base}"
    assert fz == N_MEMBERS * base, \
        f"FAIL: frozen {fz} != {N_MEMBERS} x {base}"
    # target nets add another M x base trainable-copy params (not
    # optimized directly, synced from f) — report for completeness.
    tgt = sum(p.numel() for m in ens.members
              for p in m.target_net.f.parameters())
    print(f"  OK  base net {base} params; M={N_MEMBERS}: "
          f"trainable {tr} (= M x {base}), frozen priors {fz}, "
          f"target copies {tgt}")
    print(f"      compute: each member forward = 2 net passes (f + "
          f"prior) -> network compute ~{2 * N_MEMBERS}x single agent; "
          f"env stepping (CBP deepcopies dominate wall time) is shared, "
          f"so expected training wall-clock multiplier is well below "
          f"{2 * N_MEMBERS}x — roughly 1.5-3x on the M4 Pro profile "
          f"where env time >> net time.")


def run_selftest():
    print("=" * 100)
    print("CONTROLLER ENSEMBLE-RPF SELF-TESTS")
    print("=" * 100)
    t0 = time.time()
    ens = _selftest_member_independence()
    _selftest_priors_frozen()
    _selftest_rpf_plumbing()
    _selftest_credal_aggregation(ens)
    _selftest_ood_disagreement()
    _selftest_checkpoint_roundtrip()
    _selftest_param_budget()
    print(f"SELFTEST PASSED ({time.time() - t0:.0f}s)")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble-RPF interval DQN over the controller frame "
                    "(M members, frozen random priors, credal-hull "
                    "Hurwicz selection; run-13 groundwork)")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--members", type=int, default=N_MEMBERS)
    parser.add_argument("--beta", type=float, default=BETA_PRIOR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--c_train", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=bcd.LR)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer", type=int, default=BUFFER_SIZE)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--nstep", type=int, default=NSTEP)
    parser.add_argument("--cbp", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--cbp_lag", type=int, default=CBP_LAG)
    parser.add_argument("--mask_reissue",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt_every", type=int, default=25)
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    elif args.train:
        run_training(args)
    else:
        parser.print_help()
