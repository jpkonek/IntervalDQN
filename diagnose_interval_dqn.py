"""
Component-Level Diagnostic Suite for the BluebirdATC Interval DQN
=================================================================

Verifies each starred component of the credal interval DQN loop
(Helsinki talk, slide "Credal Interval DQN"):

  (2) score intervals  — Q-net emits [Q_lower, Q_upper] per action
  (3) Hurwicz pick     — a* = argmax_a Q_l + c * (Q_u - Q_l)
  (5) update+self-tune — interval Bellman target [T_l, T_u],
                         interval loss L(.; t), coverage tracker -> t ~ 0.85

Checks (each prints evidence + PASS / WARN / FAIL / INFO verdict):

  1. INTERVAL VALIDITY        upper >= lower everywhere; width varies by state
  2. WIDTH-RISK CORRELATION   width vs nearest-neighbour proximity (report-only)
  3. HURWICZ RELEVANCE        policy disagreement across c in {0,.25,.5,.75,1}
  4. LOSS MECHANICS           unit tests of interval_loss_sampled + Bellman
                              target construction (instrumented train_step)
  5. COVERAGE CONTROL LOOP    coverage / t / width / loss trajectories from the
                              run-2 JSONL (PNG dashboard saved to diagnostics/)
  6. EMPIRICAL COVERAGE       realized discounted return-to-go vs predicted
                              [Q_l, Q_u] on fresh eval episodes
  7. TARGET-NET SANITY        Double-DQN target net: stale between hard
                              updates, tracks online net, syncs at update

All logic (network, agent, env, loss) is imported from
bluebird_interval_dqn — nothing is re-implemented except where a check
must instrument internals (noted inline).

IMPORTANT operational constraints honoured here:
  - a training run may be active on this machine: we set a low thread
    count, create only short-lived envs, and NEVER write to
    checkpoints/bluebird/train* paths (outputs go to
    checkpoints/bluebird/diagnostics/).
  - run 3 (route_parallel, 4 actions) is overwriting run-2 checkpoint
    files in place; run-2 (3-action) checkpoints were snapshotted to
    diagnostics/run2_snapshots/ and are loaded from there, with an
    n_actions==3 validity check on every load.

Usage:
    nice -n 15 .venv/bin/python diagnose_interval_dqn.py
    nice -n 15 .venv/bin/python diagnose_interval_dqn.py --heavy   # + every
        250th run-2 checkpoint for checks 1/3/6 (long; do not run while a
        training run needs the machine)
"""

import argparse
import copy
import glob
import json
import math
import os
import random
import re
import sys

import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

import bluebird_interval_dqn as bid  # noqa: E402  (single source of truth)

torch.set_num_threads(2)  # be gentle: a training run may be active
sys.stdout.reconfigure(line_buffering=True)

CKPT_DIR = os.path.join(REPO_DIR, "checkpoints", "bluebird")
DIAG_DIR = os.path.join(CKPT_DIR, "diagnostics")
SNAP_DIR = os.path.join(DIAG_DIR, "run2_snapshots")
RUN2_JSONL = os.path.join(CKPT_DIR, "train_seed42_20260704_124534.jsonl")

# Run-2 geometry (3-action, k=2, extra_minimal encoder, 1 forward fix):
#   [0] centreline distance      in [-3, 3]   (NM / 50, clipped)
#   [1] next-fix relative angle  in [-pi, pi]
#   [2] nbr-1 relative heading   in [-pi, pi]
#   [3] nbr-1 distance           in [0, 3]    (NM / 50, clipped; 0 = padding)
#   [4] nbr-2 relative heading   in [-pi, pi]
#   [5] nbr-2 distance           in [0, 3]
# (source: BluebirdATC bluebird_gymnasium/state_repr/extraminimal.py,
#  ExtraMinimalRepresentation; neighbours sorted nearest-first)
FEATURE_LOW = np.array([-3.0, -np.pi, -np.pi, 0.0, -np.pi, 0.0], np.float32)
FEATURE_HIGH = np.array([3.0, np.pi, np.pi, 3.0, np.pi, 3.0], np.float32)
NBR1_DIST_IDX = 3
NM_PER_UNIT = 50.0  # feature units -> nautical miles

RESULTS = []  # (check_no, name, verdict, one_line_evidence)


def record(no, name, verdict, evidence):
    RESULTS.append((no, name, verdict, evidence))
    print(f"\n  --> CHECK {no} VERDICT: {verdict}  ({evidence})")


def banner(no, title):
    print("\n" + "=" * 100)
    print(f"CHECK {no}: {title}")
    print("=" * 100)


def load_run2_agent(path, device="cpu"):
    """load_agent + guard that this really is a run-2 (3-action) net."""
    agent, ckpt = bid.load_agent(path, device=device)
    if ckpt["n_actions"] != 3 or ckpt.get("route_parallel", False):
        raise RuntimeError(
            f"{path} is not a run-2 checkpoint "
            f"(n_actions={ckpt['n_actions']}, "
            f"route_parallel={ckpt.get('route_parallel', False)}) — "
            f"probably overwritten by the active route_parallel run; "
            f"use diagnostics/run2_snapshots/.")
    return agent, ckpt


def forward_intervals(agent, states):
    """[N, state_dim] -> (lower, upper) numpy arrays [N, n_actions]."""
    with torch.no_grad():
        s = torch.from_numpy(np.asarray(states, np.float32)).to(agent.device)
        lower, upper = agent.q_net(s)
    return lower.cpu().numpy(), upper.cpu().numpy()


# ============================================================================
# STATE COLLECTION (shared by checks 1, 2, 3)
# ============================================================================

def collect_states(agent, duration=600, seed=20777, n_random=1000):
    """Real states from ONE short env episode driven by the loaded policy
    (c = c_train, eps = 0), padded with uniform-random states in the
    documented feature box. Violations do not end the episode here — we
    keep stepping to harvest more states (all remain valid env states)."""
    print(f"Collecting real states: one episode, duration {duration}s, "
          f"seed {seed}, c={agent.c_train} ...")
    env = bid.make_env(scenario_duration=duration, k_nearest=2,
                       route_parallel=False)
    obs, _ = env.reset(seed=seed)
    maxstep = int(getattr(env, "maxstep", duration // bid.SEC_PER_STEP))
    real = []
    for _ in range(maxstep):
        for v in obs.values():
            real.append(np.asarray(v, np.float32))
        actions, _ = agent.generate_action(obs, force_epsilon=0.0)
        obs, _, _, _, _ = env.step(actions)
    env.close()
    real = np.stack(real) if real else np.zeros((0, agent.state_dim))

    rng = np.random.default_rng(seed)
    rand = rng.uniform(FEATURE_LOW, FEATURE_HIGH,
                       size=(n_random, agent.state_dim)).astype(np.float32)
    print(f"Collected {len(real)} real states (+{n_random} random box states)")
    return real, rand


# ============================================================================
# CHECK 1 — INTERVAL VALIDITY (slide step 2)
# ============================================================================

def check_1(agents, real, rand, record_verdict=True, tag=""):
    banner(1, f"INTERVAL VALIDITY {tag}— upper >= lower; widths carry "
              "state information")
    any_inversion = False
    all_constant = []
    for label, agent in agents.items():
        for sname, states in (("real", real), ("random", rand)):
            lo, up = forward_intervals(agent, states)
            n_inv = int((up < lo).sum())
            any_inversion |= n_inv > 0
            w = up - lo
            print(f"\n  [{label}] {sname} states (n={len(states)}): "
                  f"inversions (upper<lower): {n_inv}")
            for a in range(w.shape[1]):
                wa = w[:, a]
                cv = wa.std() / max(1e-12, abs(wa.mean()))
                print(f"    action {a}: width mean {wa.mean():8.4f} | "
                      f"p10 {np.percentile(wa, 10):8.4f} | "
                      f"p90 {np.percentile(wa, 90):8.4f} | "
                      f"std/mean {cv:.4f}")
            if sname == "real":
                cvs = [w[:, a].std() / max(1e-12, abs(w[:, a].mean()))
                       for a in range(w.shape[1])]
                all_constant.append((label, max(cvs)))
    if any_inversion:
        verdict, ev = "FAIL", "interval inversion (upper < lower) detected"
    else:
        flat = [(l, c) for l, c in all_constant if c < 0.05]
        if flat:
            verdict = "WARN"
            ev = ("no inversions, but widths ~constant across states for " +
                  ", ".join(f"{l} (max std/mean {c:.3f})" for l, c in flat))
        else:
            verdict = "PASS"
            ev = ("no inversions in any checkpoint; widths vary with state "
                  f"(max std/mean per ckpt: " +
                  ", ".join(f"{l}={c:.2f}" for l, c in all_constant) + ")")
    if record_verdict:
        record(1, "Interval validity", verdict, ev)
    return verdict


# ============================================================================
# CHECK 2 — WIDTH vs NEIGHBOUR PROXIMITY (slide step 2)
# ============================================================================

def check_2(agent, real, record_verdict=True):
    from scipy import stats as sps
    banner(2, "WIDTH-RISK CORRELATION — interval width vs nearest-neighbour "
              "distance (report-only)")
    print(f"  Nearest-neighbour distance = feature [{NBR1_DIST_IDX}] "
          f"(extra_minimal: nbr-1 distance, NM/50, neighbours sorted "
          f"nearest-first; 0.0 = zero-padding for absent neighbour)")

    lo, up = forward_intervals(agent, real)
    w = up - lo
    mean_w = w.mean(axis=1)
    chosen = (lo + agent.c_train * (up - lo)).argmax(axis=1)
    chosen_w = w[np.arange(len(w)), chosen]

    d = real[:, NBR1_DIST_IDX]
    mask = d > 1e-6  # exclude zero-padded (no-neighbour) states
    print(f"  states with a real nearest neighbour: {mask.sum()}/{len(d)} "
          f"(padding excluded from correlations)")
    if mask.sum() < 30:
        record(2, "Width-risk correlation", "INFO",
               "too few neighboured states to correlate")
        return

    for name, wv in (("mean width (all actions)", mean_w),
                     ("width of chosen action (c=0.5)", chosen_w)):
        pr, pp = sps.pearsonr(d[mask], wv[mask])
        sr, sp = sps.spearmanr(d[mask], wv[mask])
        print(f"  {name:33s}: Pearson r={pr:+.3f} (p={pp:.1e}) | "
              f"Spearman rho={sr:+.3f} (p={sp:.1e})")

    # binned view (feature units -> NM)
    print("\n  width by nearest-neighbour distance bin:")
    edges = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.01]
    for a, b in zip(edges[:-1], edges[1:]):
        m = mask & (d >= a) & (d < b)
        if m.sum() > 0:
            print(f"    [{a * NM_PER_UNIT:5.0f}, {b * NM_PER_UNIT:5.0f}) NM "
                  f"(n={m.sum():4d}): mean width {mean_w[m].mean():8.4f}")

    # perturbation analysis: sweep nbr-1 distance, everything else fixed
    # (isolates the causal effect of proximity, which raw correlations
    # confound with everything else that varies along a trajectory)
    print("\n  perturbation sweep (200 real states, nbr-1 distance forced):")
    base = real[mask][:200].copy()
    sweep = []
    for val in (0.04, 0.10, 0.20, 0.50, 1.00, 2.00, 3.00):
        pert = base.copy()
        pert[:, NBR1_DIST_IDX] = val
        lo_p, up_p = forward_intervals(agent, pert)
        sweep.append((val, float((up_p - lo_p).mean(axis=1).mean())))
        print(f"    nbr dist = {val:4.2f} ({val * NM_PER_UNIT:5.1f} NM"
              f"{', < 5 NM sep. threshold' if val * NM_PER_UNIT < 5 else '':28s}): "
              f"mean width {sweep[-1][1]:8.4f}")

    sr_all, _ = sps.spearmanr(d[mask], mean_w[mask])
    w_close, w_far = sweep[0][1], sweep[-1][1]
    causal = ("wider when a neighbour is forced close" if w_close > 1.05 * w_far
              else "narrower when a neighbour is forced close"
              if w_far > 1.05 * w_close else "insensitive to forced proximity")
    if record_verdict:
        record(2, "Width-risk correlation", "INFO",
               f"raw Spearman(dist, width) = {sr_all:+.3f}; perturbation: "
               f"width {w_close:.2f} @2NM vs {w_far:.2f} @150NM -> {causal}")


# ============================================================================
# CHECK 3 — HURWICZ BEHAVIORAL RELEVANCE (slide step 3)
# ============================================================================

def check_3(agent, real, rand, record_verdict=True, tag=""):
    banner(3, f"HURWICZ BEHAVIORAL RELEVANCE {tag}— policy disagreement "
              "across c")
    cs = [0.0, 0.25, 0.5, 0.75, 1.0]
    lo, up = forward_intervals(agent, real)
    acts = {c: (lo + c * (up - lo)).argmax(axis=1) for c in cs}

    print(f"\n  pairwise policy disagreement (% of {len(real)} real states):")
    print("        " + "".join(f"  c={c:<5}" for c in cs))
    for ci in cs:
        row = f"  c={ci:<4}"
        for cj in cs:
            row += f"  {100.0 * (acts[ci] != acts[cj]).mean():6.2f}%"
        print(row)

    d01 = 100.0 * (acts[0.0] != acts[1.0]).mean()
    lo_r, up_r = forward_intervals(agent, rand)
    d01_rand = 100.0 * ((lo_r.argmax(1)) != ((up_r).argmax(1))).mean()
    print(f"\n  disagreement c=0 vs c=1: {d01:.2f}% on real states, "
          f"{d01_rand:.2f}% on random box states")
    print("  action shares per c (real states): " + " | ".join(
        f"c={c}: " + "/".join(f"{(acts[c] == a).mean():.2f}"
                              for a in range(agent.n_actions)) for c in cs))

    if d01 == 0.0:
        verdict, ev = "FAIL", "argmax identical at c=0 and c=1 — intervals behaviorally inert"
    elif d01 > 2.0:
        verdict, ev = "PASS", f"disagreement(c=0 vs c=1) = {d01:.2f}% > 2% — intervals shape behavior"
    else:
        verdict, ev = "WARN", f"disagreement(c=0 vs c=1) = {d01:.2f}% (>0 but <= 2%)"
    if record_verdict:
        record(3, "Hurwicz behavioral relevance", verdict, ev)
    return verdict


# ============================================================================
# CHECK 4 — LOSS MECHANICS (slide step 5)
# ============================================================================

def _fresh_bare_agent(t=0.5, width_reg=0.0):
    """Minimal agent to unit-test interval_loss_sampled in isolation
    (width_reg=0 so the width-regularizer cannot contaminate gradients;
    the tracker starts empty so no width penalty triggers either)."""
    a = bid.IntervalDQNAgent(state_dim=6, n_actions=3, device="cpu",
                             width_reg=width_reg, warmup_steps=0)
    a.coverage_tracker.t = t
    return a


def check_4():
    banner(4, "LOSS MECHANICS — unit tests of interval_loss_sampled + "
              "Bellman target construction")
    sub = {}

    # -- (a) target strictly inside -> gradient step SHRINKS the width -------
    agent = _fresh_bare_agent(t=0.5)
    l = torch.tensor([-1.0], requires_grad=True)
    u = torch.tensor([2.0], requires_grad=True)
    tl, tu = torch.tensor([-0.5]), torch.tensor([0.5])  # inside [-1, 2]
    loss = agent.interval_loss_sampled(l, u, tl, tu)
    loss.backward()
    g_l, g_u = l.grad.item(), u.grad.item()
    w_grad = g_u - g_l  # dW/dstep < 0 iff this > 0 under gradient descent
    new_w = (u.item() - 0.1 * g_u) - (l.item() - 0.1 * g_l)
    print(f"\n  (a) targets inside [-0.5,0.5] c [-1,2]: dL/dl={g_l:+.4f}, "
          f"dL/du={g_u:+.4f}; width 3.00 -> {new_w:.4f} after lr=0.1 step")
    sub["a: inside target shrinks width"] = w_grad > 0 and new_w < 3.0

    # -- (b) target outside -> nearer bound moves TOWARD the target ----------
    agent = _fresh_bare_agent(t=0.5)
    l = torch.tensor([-1.0], requires_grad=True)
    u = torch.tensor([1.0], requires_grad=True)
    tgt = torch.tensor([2.0])  # above upper -> nearer bound is u
    agent.interval_loss_sampled(l, u, tgt, tgt).backward()
    up_toward = u.grad.item() < 0  # descent raises u toward 2.0
    print(f"  (b) target 2.0 above [-1,1]: dL/du={u.grad.item():+.4f} "
          f"(<0 -> u rises toward target: {up_toward})")
    agent = _fresh_bare_agent(t=0.5)
    l2 = torch.tensor([-1.0], requires_grad=True)
    u2 = torch.tensor([1.0], requires_grad=True)
    tgt2 = torch.tensor([-2.0])  # below lower -> nearer bound is l
    agent.interval_loss_sampled(l2, u2, tgt2, tgt2).backward()
    low_toward = l2.grad.item() > 0  # descent lowers l toward -2.0
    print(f"      target -2.0 below [-1,1]: dL/dl={l2.grad.item():+.4f} "
          f"(>0 -> l falls toward target: {low_toward})")
    sub["b: outside target pulls nearer bound"] = up_toward and low_toward

    # -- (c) increasing t raises the coverage term's relative weight ---------
    print("  (c) excluded target y=1.5 outside [0,1]; coverage term = "
          "t*min_dist^2, width term = (1-t)*max_dist^2:")
    l0, u0, y = 0.0, 1.0, 1.5
    min_sq, max_sq = (y - u0) ** 2, (y - l0) ** 2
    covs, shares, totals = [], [], []
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        agent = _fresh_bare_agent(t=t)
        with torch.no_grad():
            total = agent.interval_loss_sampled(
                torch.tensor([l0]), torch.tensor([u0]),
                torch.tensor([y]), torch.tensor([y])).item()
        cov_term = t * min_sq
        covs.append(cov_term); shares.append(cov_term / total)
        totals.append(total)
        print(f"      t={t:.1f}: coverage term {cov_term:.4f} "
              f"({100 * cov_term / total:5.1f}% of loss {total:.4f})")
    mono_cov = all(b > a for a, b in zip(covs, covs[1:]))
    mono_share = all(b > a for a, b in zip(shares, shares[1:]))
    sub["c: coverage weight grows with t"] = mono_cov and mono_share
    if totals[-1] < totals[0]:
        print("      note: the ABSOLUTE loss for an excluded target falls "
              "with t (loss = max^2 - t*(max^2 - min^2));")
        print("      what grows with t is the miss term and its share — as "
              "t->1 the width-shrink force (1-t)*max^2 vanishes,")
        print("      leaving only the pull to cover the target. Relative "
              "weighting behaves as designed.")

    # -- (d) sampled Bellman targets lie within [r + g*T_l, r + g*T_u] -------
    # Instrumented: replay the exact batch train_step will draw (the RNG is
    # the global `random` module and buffer.sample is its only consumer in
    # train_step), compute the expected Bellman bounds from the CURRENT
    # nets, then hook interval_loss_sampled to capture what train_step
    # actually passed in.
    agent = bid.IntervalDQNAgent(state_dim=6, n_actions=3, device="cpu",
                                 warmup_steps=0)
    rng = np.random.default_rng(0)
    for _ in range(400):
        s = rng.uniform(FEATURE_LOW, FEATURE_HIGH).astype(np.float32)
        ns = rng.uniform(FEATURE_LOW, FEATURE_HIGH).astype(np.float32)
        # n-step replay semantics (run 7+): the last column is the bootstrap
        # discount `disc` (gamma^m for an m-step bootstrapped window, 0.0 at
        # terminals), NOT a done flag
        disc = 0.0 if rng.random() < 0.1 else float(
            agent.gamma ** int(rng.integers(1, 7)))
        agent.buffer.push(s, int(rng.integers(0, 3)),
                          float(rng.normal()), ns, disc)
    random.seed(4242)
    states, actions, rewards, next_states, discs = agent.buffer.sample(
        agent.batch_size)
    with torch.no_grad():
        nl, nu = agent.target_net(torch.from_numpy(
            next_states.astype(np.float32)))
        ol, ou = agent.q_net(torch.from_numpy(next_states.astype(np.float32)))
        na = (ol + agent.c_train * (ou - ol)).argmax(dim=1)
        d = torch.from_numpy(discs)
        tl_exp = (torch.from_numpy(rewards) +
                  d * nl.gather(1, na.unsqueeze(1)).squeeze(1))
        tu_exp = (torch.from_numpy(rewards) +
                  d * nu.gather(1, na.unsqueeze(1)).squeeze(1))
    captured = {}
    orig = agent.interval_loss_sampled

    def hook(lower, upper, target_lower, target_upper):
        captured["tl"] = target_lower.detach().clone()
        captured["tu"] = target_upper.detach().clone()
        return orig(lower, upper, target_lower, target_upper)

    agent.interval_loss_sampled = hook
    random.seed(4242)          # replay the identical batch inside train_step
    agent.train_step()
    agent.interval_loss_sampled = orig

    match_l = torch.allclose(captured["tl"], tl_exp, atol=1e-5)
    match_u = torch.allclose(captured["tu"], tu_exp, atol=1e-5)
    ordered = bool((captured["tu"] >= captured["tl"]).all())
    alphas = torch.linspace(0, 1, bid.IntervalDQNAgent.N_TARGET_SAMPLES)
    samples = (captured["tl"].unsqueeze(0) + alphas.unsqueeze(1) *
               (captured["tu"] - captured["tl"]).unsqueeze(0))
    within = bool(((samples >= captured["tl"].unsqueeze(0) - 1e-6) &
                   (samples <= captured["tu"].unsqueeze(0) + 1e-6)).all())
    print(f"  (d) instrumented train_step (batch {agent.batch_size}): "
          f"T_l == R_n + disc*L_next: {match_l}; "
          f"T_u == R_n + disc*U_next: {match_u}")
    print(f"      T_u >= T_l for all: {ordered}; all {len(alphas)} sampled "
          f"targets within [T_l, T_u]: {within}")
    sub["d: Bellman targets + samples in bounds"] = (match_l and match_u
                                                     and ordered and within)

    print()
    for name, ok in sub.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    n_fail = sum(not ok for ok in sub.values())
    record(4, "Loss mechanics", "PASS" if n_fail == 0 else "FAIL",
           f"{len(sub) - n_fail}/{len(sub)} sub-checks passed")


# ============================================================================
# CHECK 5 — COVERAGE CONTROL LOOP (slide step 5 dashboard)
# ============================================================================

def check_5(jsonl_path):
    banner(5, "COVERAGE CONTROL LOOP — coverage / t / width / loss from "
              "run-2 training log")
    print(f"  log: {jsonl_path}")
    recs = [json.loads(ln) for ln in open(jsonl_path) if ln.strip()]
    ep = np.array([r["episode"] for r in recs])
    cov = np.array([r["coverage"] for r in recs])
    tt = np.array([r["t"] for r in recs])
    wid = np.array([r["mean_width"] for r in recs])
    loss = np.array([r["mean_loss"] for r in recs])
    print(f"  {len(recs)} episodes")

    def roll(x, k=51):
        return np.convolve(x, np.ones(k) / k, mode="valid")

    # text dashboard
    print("\n  ep-range      coverage       t         mean_width   mean_loss")
    for i in range(0, len(ep), max(1, len(ep) // 12)):
        j = min(len(ep), i + max(1, len(ep) // 12))
        print(f"  {ep[i]:5d}-{ep[j - 1]:5d}   {cov[i:j].mean():8.3f}   "
              f"{tt[i:j].mean():8.3f}   {wid[i:j].mean():10.3f}   "
              f"{loss[i:j].mean():9.3f}")

    steady = slice(max(0, len(ep) - 500), len(ep))
    cov_ss = cov[steady].mean()
    t_final, t_max = tt[-1], tt.max()
    rw = roll(wid)
    i_min = int(np.argmin(rw)) + 25  # centre of the rolling window
    w_min, w_end = rw.min(), rw[-1]
    print(f"\n  steady-state coverage (last 500 eps): {cov_ss:.3f} "
          f"(target 0.85 +/- 0.03)")
    print(f"  t: final {t_final:.4f}, max {t_max:.4f} (clamp 0.95); "
          f"t still rising at end: {tt[-1] - tt[-200] > 0.001}")
    print(f"  width: narrows to a rolling-mean minimum of {w_min:.3f} at "
          f"~episode {ep[min(i_min, len(ep) - 1)]}, then inflates to "
          f"{w_end:.3f} by the end ({w_end / max(1e-9, w_min):.1f}x the "
          f"minimum) — width stops narrowing / starts inflating there")

    # PNG dashboard
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        r_ep = ep[25:len(rw) + 25]
        for ax, (y, ry, name, extra) in zip(axes.flat, [
                (cov, roll(cov), "coverage", ("target band", 0.82, 0.88)),
                (tt, roll(tt), "t (loss parameter)", ("clamp", 0.95, None)),
                (wid, roll(wid), "mean interval width", None),
                (loss, roll(loss), "mean loss", None)]):
            ax.plot(ep, y, alpha=0.25, lw=0.6)
            ax.plot(r_ep, ry, lw=1.6)
            if name == "coverage":
                ax.axhspan(0.82, 0.88, alpha=0.15, color="green")
                ax.axhline(0.85, ls="--", lw=0.8, color="green")
            if name.startswith("t "):
                ax.axhline(0.95, ls="--", lw=0.8, color="red")
            if name.startswith("mean interval"):
                ax.axvline(ep[min(i_min, len(ep) - 1)], ls=":", lw=1,
                           color="red")
            ax.set_title(name)
            ax.set_xlabel("episode")
        fig.suptitle("Interval DQN control loop — run 2 (seed 42, 3 actions)")
        fig.tight_layout()
        png = os.path.join(DIAG_DIR, "run2_control_loop.png")
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print(f"  dashboard PNG saved: {png}")
    except Exception as e:  # plotting must never sink the suite
        print(f"  (matplotlib dashboard skipped: {e})")

    problems = []
    if t_max >= 0.945:
        problems.append(f"t saturated at clamp ({t_max:.3f} vs 0.95)")
    if not (0.82 <= cov_ss <= 0.88):
        problems.append(f"steady-state coverage {cov_ss:.3f} outside "
                        f"0.85 +/- 0.03")
    if w_end / max(1e-9, w_min) > 1.5:
        problems.append(f"width inflation: {w_end:.2f} vs minimum "
                        f"{w_min:.2f} at ~ep {ep[min(i_min, len(ep) - 1)]}")
    if problems:
        record(5, "Coverage control loop", "WARN", "; ".join(problems))
    else:
        record(5, "Coverage control loop", "PASS",
               f"coverage {cov_ss:.3f} in band, t {t_final:.3f} below clamp, "
               f"width stable")


# ============================================================================
# CHECK 6 — EMPIRICAL COVERAGE of realized returns (deepest check)
# ============================================================================

def check_6(agent, ckpt, n_episodes=3, base_seed=20042, duration=300, c=0.0,
            violation_penalty=10.0, record_verdict=True, tag=""):
    banner(6, f"EMPIRICAL COVERAGE {tag}— realized discounted return-to-go "
              f"vs predicted [Q_l, Q_u]")
    gamma = ckpt.get("gamma", bid.GAMMA)
    # reward regime from checkpoint metadata (run 4+: outcome anchoring);
    # old checkpoints fall back to the run-2/3 regime they trained under
    exit_bonus = ckpt.get("exit_bonus", 0.0)
    centreline_coeff = ckpt.get("centreline_coeff", 1.0)
    route_parallel = ckpt.get("route_parallel", False)
    violation_penalty = ckpt.get("violation_penalty", violation_penalty)
    progress_coeff = ckpt.get("progress_coeff", 0.0)
    print(f"  gamma = {gamma} ({'from checkpoint' if 'gamma' in ckpt else 'module default; checkpoint does not store gamma'})")
    print(f"  reward regime: exit_bonus={exit_bonus}, "
          f"centreline_coeff={centreline_coeff}, "
          f"violation_penalty={violation_penalty}, "
          f"route_parallel={route_parallel}")
    print(f"  {n_episodes} eval episodes, duration {duration}s, c={c}, "
          f"seeds {base_seed}..{base_seed + n_episodes - 1}")
    print("  episode semantics mirror training: ends at first violation; "
          "involved aircraft get -"
          f"{violation_penalty} and terminal; BEFORE_ENTRY steps skipped")

    env = bid.make_env(scenario_duration=duration, k_nearest=ckpt.get("k", 2),
                       route_parallel=route_parallel,
                       centreline_coeff=centreline_coeff,
                       encoder_cls=ckpt.get("encoder_cls", "extra_minimal"))
    trajs = []  # per (episode, callsign): dict(steps=[(q_l,q_u,r)], terminal)
    for epi in range(n_episodes):
        obs, info = env.reset(seed=base_seed + epi)
        maxstep = int(getattr(env, "maxstep", duration // bid.SEC_PER_STEP))
        streams = {}
        for _ in range(maxstep):
            if not obs:
                break
            callsigns = list(obs.keys())
            # mirrors generate_action, but also captures the chosen action's
            # interval (generate_action returns only actions + mean width)
            with torch.no_grad():
                s = torch.from_numpy(np.stack(
                    [obs[cs] for cs in callsigns]).astype(np.float32))
                lo, up = agent.q_net(s.to(agent.device))
                a_star = (lo + c * (up - lo)).argmax(dim=1).cpu().numpy()
                lo, up = lo.cpu().numpy(), up.cpu().numpy()
            actions = {cs: int(a_star[i]) for i, cs in enumerate(callsigns)}
            phi_prev = ({cs: bid.exit_potential(env, cs) for cs in callsigns}
                        if progress_coeff else {})
            next_obs, rew, done, trunc, info = env.step(actions)
            violated, kind, involved = bid.detect_violation(info)
            for i, cs in enumerate(callsigns):
                d = info.get(cs)
                if isinstance(d, dict) and d.get("pos_status") == "BEFORE_ENTRY":
                    continue
                r = float(rew.get(cs, 0.0))
                terminal = (cs not in next_obs or
                            (bool(done.get(cs, False)) and
                             not bool(trunc.get(cs, False))))
                if violated and cs in involved:
                    r -= violation_penalty
                    terminal = True
                elif terminal and not (isinstance(d, dict)
                                       and d.get("pos_status") == "OUT_SECTOR"):
                    # mirror training's outcome anchoring (0.0 for old ckpts)
                    r += exit_bonus
                if progress_coeff and not terminal:
                    phi = phi_prev.get(cs)
                    phi_next = bid.exit_potential(env, cs)
                    if phi is not None and phi_next is not None:
                        r += progress_coeff * (gamma * phi_next - phi)
                st = streams.setdefault(cs, {"steps": [], "terminal": False})
                if not st["terminal"]:
                    st["steps"].append((float(lo[i, a_star[i]]),
                                        float(up[i, a_star[i]]), r))
                    st["terminal"] = terminal
            obs = next_obs
            if violated:
                print(f"    ep seed {base_seed + epi}: violation [{kind}] "
                      f"involving {involved} -> episode ends (as in training)")
                break
        # censored streams: bootstrap-complete the truncated tail with the
        # net's own interval at the aircraft's FINAL state (turns the check
        # into a multi-step Bellman self-consistency test for those steps)
        for cs, st in streams.items():
            st["tail"] = (0.0, 0.0)
            if not st["terminal"] and cs in obs:
                with torch.no_grad():
                    s = torch.from_numpy(np.asarray(
                        obs[cs], np.float32)).unsqueeze(0).to(agent.device)
                    lo_f, up_f = agent.q_net(s)
                    a_f = (lo_f + c * (up_f - lo_f)).argmax(dim=1)
                st["tail"] = (float(lo_f[0, a_f]), float(up_f[0, a_f]))
        trajs.extend(streams.values())

    env.close()

    hits_all, hits_term, hits_cens = [], [], []
    hits_boot, widths, gaps, rtg_all, ql_all, qu_all = [], [], [], [], [], []
    for st in trajs:
        steps = st["steps"]
        T = len(steps)
        G = 0.0
        rtg = [0.0] * T
        for k in range(T - 1, -1, -1):
            G = steps[k][2] + gamma * G
            rtg[k] = G
        tail_l, tail_u = st["tail"]
        for k, (ql, qu, _r) in enumerate(steps):
            hit = ql <= rtg[k] <= qu
            hits_all.append(hit)
            (hits_term if st["terminal"] else hits_cens).append(hit)
            # completed return interval: G_k + gamma^{T-k} * [tail_l, tail_u]
            g_pow = gamma ** (T - k)
            cl, cu = rtg[k] + g_pow * tail_l, rtg[k] + g_pow * tail_u
            hits_boot.append(not (cu < ql or cl > qu))  # interval overlap
            widths.append(qu - ql)
            rtg_all.append(rtg[k]); ql_all.append(ql); qu_all.append(qu)
            gaps.append(0.0 if hit else min(abs(rtg[k] - ql),
                                            abs(rtg[k] - qu)))
    n = len(hits_all)
    cov_all = float(np.mean(hits_all)) if n else float("nan")
    print(f"\n  aircraft streams: {len(trajs)} "
          f"({sum(s['terminal'] for s in trajs)} terminated, rest censored "
          f"at episode end -> their tails are truncated, biasing G downward)")
    print(f"  per-step records: {n}; mean predicted width "
          f"{np.mean(widths):.3f}")
    print(f"  scale: mean realized return-to-go {np.mean(rtg_all):8.3f}  vs  "
          f"mean predicted [Q_l, Q_u] = [{np.mean(ql_all):.3f}, "
          f"{np.mean(qu_all):.3f}]")
    print(f"  REALIZED coverage of [Q_l, Q_u] over discounted "
          f"return-to-go: {cov_all:.3f}")
    if hits_term:
        print(f"    terminated streams only: {np.mean(hits_term):.3f} "
              f"(n={len(hits_term)})")
    if hits_cens:
        print(f"    censored streams only:   {np.mean(hits_cens):.3f} "
              f"(n={len(hits_cens)}; gamma^H truncation with gamma="
              f"{gamma} and <= {duration // bid.SEC_PER_STEP} steps is "
              f"severe: gamma^50 = {gamma ** 50:.2f})")
    print(f"  bootstrap-completed coverage (censored tail filled with the "
          f"net's own interval at the final state,\n  i.e. multi-step "
          f"Bellman self-consistency): {np.mean(hits_boot):.3f}")
    print(f"  mean miss distance when outside: "
          f"{np.mean([g for g in gaps if g > 0]) if any(g > 0 for g in gaps) else 0.0:.3f}")
    print(f"  tracker's own coverage (sampled Bellman targets, from "
          f"checkpoint): {ckpt.get('coverage', float('nan')):.3f} vs "
          f"target 0.85")
    print("  note: the tracker measures coverage of BOOTSTRAPPED sampled "
          "targets; realized-return coverage is a\n  strictly harder, "
          "model-independent criterion — divergence between the two is a "
          "research finding, not a bug.")

    if not n:
        verdict, ev = "WARN", "no per-step records collected"
    elif cov_all < 0.5:
        verdict, ev = "WARN", (f"realized coverage {cov_all:.3f} < 0.5 vs "
                               f"tracker {ckpt.get('coverage', float('nan')):.2f}; "
                               f"mean G {np.mean(rtg_all):.1f} vs predicted "
                               f"[{np.mean(ql_all):.1f}, {np.mean(qu_all):.1f}]; "
                               f"bootstrap-completed {np.mean(hits_boot):.2f}")
    elif cov_all > 0.99:
        verdict, ev = "WARN", (f"realized coverage {cov_all:.3f} > 0.99 — "
                               f"intervals likely vacuous")
    else:
        verdict, ev = "PASS", (f"realized coverage {cov_all:.3f} vs target "
                               f"0.85 (tracker: "
                               f"{ckpt.get('coverage', float('nan')):.2f})")
    if record_verdict:
        record(6, "Empirical coverage", verdict, ev)
    return cov_all


# ============================================================================
# CHECK 7 — TARGET-NET / DOUBLE-DQN SANITY (slide step 5)
# ============================================================================

def _flat(net):
    return torch.cat([p.detach().flatten() for p in net.parameters()])


def check_7():
    banner(7, "TARGET-NET / DOUBLE-DQN SANITY — staleness, tracking, hard "
              "sync")
    print("  note: checkpoints store only q_net (load_agent copies it into "
          "target_net), so on-disk staleness is\n  unobservable; this check "
          "exercises the live train_step machinery on a fresh agent.")
    agent = bid.IntervalDQNAgent(state_dim=6, n_actions=3, device="cpu",
                                 warmup_steps=0)
    rng = np.random.default_rng(1)
    for _ in range(500):
        agent.buffer.push(
            rng.uniform(FEATURE_LOW, FEATURE_HIGH).astype(np.float32),
            int(rng.integers(0, 3)), float(rng.normal()),
            rng.uniform(FEATURE_LOW, FEATURE_HIGH).astype(np.float32),
            float(rng.random() < 0.1))

    q0, t0 = _flat(agent.q_net), _flat(agent.target_net)
    sub = {}
    sub["init: target == online"] = bool(torch.equal(q0, t0))

    agent.train_step()  # step 1
    q1, t1 = _flat(agent.q_net), _flat(agent.target_net)
    online_moved = not torch.equal(q0, q1)
    target_frozen = torch.equal(t0, t1)
    print(f"\n  after 1 train_step: online moved: {online_moved} "
          f"(|dq|={float((q1 - q0).norm()):.2e}), "
          f"target frozen: {target_frozen}")
    sub["one train_step moves online, not target"] = (online_moved
                                                      and target_frozen)

    while agent.step_count < bid.TARGET_UPDATE_FREQ - 1:  # up to step 99
        agent.train_step()
    q99, t99 = _flat(agent.q_net), _flat(agent.target_net)
    stale = float((q99 - t99).norm())
    cos = float(torch.nn.functional.cosine_similarity(
        q99.unsqueeze(0), t99.unsqueeze(0)).item())
    print(f"  after {agent.step_count} steps (pre-sync): staleness "
          f"|q - target| = {stale:.4f} (> 0), cosine similarity = {cos:.4f}")
    sub["pre-sync: staleness > 0"] = stale > 0
    sub["pre-sync: target tracks online (cos > 0.95)"] = cos > 0.95

    agent.train_step()  # step 100 -> hard update
    q100, t100 = _flat(agent.q_net), _flat(agent.target_net)
    synced = torch.equal(q100, t100)
    print(f"  after step {agent.step_count} (TARGET_UPDATE_FREQ="
          f"{bid.TARGET_UPDATE_FREQ}): hard sync -> target == online: "
          f"{synced}")
    sub[f"hard sync at step {bid.TARGET_UPDATE_FREQ}"] = bool(synced)

    print()
    for name, ok in sub.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    n_fail = sum(not ok for ok in sub.values())
    record(7, "Target-net / Double-DQN sanity",
           "PASS" if n_fail == 0 else "FAIL",
           f"{len(sub) - n_fail}/{len(sub)} sub-checks passed")


# ============================================================================
# HEAVY MODE — component evolution across run-2 checkpoints
# ============================================================================

def run_heavy(real, rand, device):
    print("\n" + "#" * 100)
    print("HEAVY MODE: checks 1/3/6 across every 250th run-2 checkpoint "
          "(from diagnostics/run2_snapshots/)")
    print("#" * 100)
    snaps = sorted(glob.glob(os.path.join(SNAP_DIR, "train_seed42_ep*.pt")),
                   key=lambda p: int(re.search(r"ep(\d+)", p).group(1)))
    snaps = [p for p in snaps
             if int(re.search(r"ep(\d+)", p).group(1)) % 250 == 0]
    rows = []
    for path in snaps:
        epn = int(re.search(r"ep(\d+)", path).group(1))
        try:
            agent, ckpt = load_run2_agent(path, device)
        except RuntimeError as e:
            print(f"  skipping ep{epn}: {e}")
            continue
        tag = f"[ep{epn}] "
        v1 = check_1({f"ep{epn}": agent}, real, rand,
                     record_verdict=False, tag=tag)
        v3 = check_3(agent, real, rand, record_verdict=False, tag=tag)
        cov = check_6(agent, ckpt, record_verdict=False, tag=tag)
        lo, up = forward_intervals(agent, real)
        rows.append((epn, v1, v3, cov, float((up - lo).mean())))
    print("\n  EVOLUTION SUMMARY (run-2 checkpoints):")
    print("    episode  validity  hurwicz  realized_cov  mean_width")
    for epn, v1, v3, cov, w in rows:
        print(f"    {epn:7d}  {v1:8s}  {v3:7s}  {cov:12.3f}  {w:10.3f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Component-level diagnostics for the BluebirdATC "
                    "interval DQN")
    ap.add_argument("--heavy", action="store_true",
                    help="also run checks 1/3/6 on every 250th run-2 "
                         "checkpoint (long; avoid while training is active)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--jsonl", default=RUN2_JSONL,
                    help="run-2 training JSONL for check 5")
    args = ap.parse_args()

    os.makedirs(DIAG_DIR, exist_ok=True)

    print("=" * 100)
    print("INTERVAL DQN COMPONENT DIAGNOSTICS — BluebirdATC run 2 "
          "(seed 42, 3 actions, k=2)")
    print("=" * 100)

    def snap_or_live(name):
        """Prefer the run-2 snapshot (the live file may have been
        overwritten by the active route_parallel run)."""
        snap = os.path.join(SNAP_DIR, name)
        return snap if os.path.exists(snap) else os.path.join(CKPT_DIR, name)

    ckpt_paths = {
        "best (ep2250)": snap_or_live("best_seed42_ep2250.pt"),
        "final (ep3000)": snap_or_live("train_seed42_final.pt"),
        # earliest SURVIVING run-2 checkpoint: the active route_parallel run
        # reuses train_seed42_ep*.pt names and had already overwritten
        # ep25..ep875 when snapshots were taken
        "early (ep900, earliest surviving)":
            snap_or_live("train_seed42_ep900.pt"),
    }
    agents, ckpts = {}, {}
    for label, path in ckpt_paths.items():
        agents[label], ckpts[label] = load_run2_agent(path, args.device)
        c = ckpts[label]
        print(f"loaded {label}: {path}")
        print(f"    episode {c.get('episode')}, state_dim {c['state_dim']}, "
              f"{c['n_actions']} actions, t={c.get('t', 0.5):.4f}, "
              f"tracker coverage={c.get('coverage', float('nan')):.4f}")

    best_label = "best (ep2250)"
    real, rand = collect_states(agents[best_label])

    check_1(agents, real, rand)
    check_2(agents[best_label], real)
    check_3(agents[best_label], real, rand)
    check_4()
    check_5(args.jsonl)
    check_6(agents[best_label], ckpts[best_label])
    check_7()

    if args.heavy:
        run_heavy(real, rand, args.device)

    # ---- summary table ----
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  {'#':>2}  {'check':<32} {'verdict':<8} evidence")
    print("  " + "-" * 96)
    for no, name, verdict, ev in RESULTS:
        print(f"  {no:>2}  {name:<32} {verdict:<8} {ev}")
    n_fail = sum(1 for r in RESULTS if r[2] == "FAIL")
    n_warn = sum(1 for r in RESULTS if r[2] == "WARN")
    print("  " + "-" * 96)
    print(f"  {len(RESULTS)} checks: "
          f"{sum(1 for r in RESULTS if r[2] == 'PASS')} PASS, "
          f"{n_warn} WARN, {n_fail} FAIL, "
          f"{sum(1 for r in RESULTS if r[2] == 'INFO')} INFO")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
