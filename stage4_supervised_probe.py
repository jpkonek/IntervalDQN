"""
Stage 4 — supervised ordering probe (CREDIT_DIAGNOSTICS.md, stage 4)
====================================================================

D2 localized the credit failure to TARGET GEOMETRY: the consequence
horizon is 24-48 steps (median |gap| 0.10 at H<=12 vs ~10 at H>=24)
while the credit window is nstep=12, and the fee anti-orders actions
inside the window.  Stage 4 asks the remaining question: is the
ARCHITECTURE innocent — given FAITHFUL targets (determinized H=48
rollout returns for EVERY candidate), can ControllerQNet represent
candidate ordering?  JK's prior: it can.

DESIGN (fixed before any run)
-----------------------------
* DATASET: ~120 states from the 12c ep2000 checkpoint's own greedy
  policy at standard density (dc.get_env(1200)) — ~60 conflict states
  (min vertically-proximate pair sep < 12 nm AND NOOP-LoS within H=48;
  the D2 gate at the stage-4 horizon) + ~60 general on-policy states
  (collect_probe_states, min_aircraft=2), INTERLEAVED
  conflict/general so a budget cut preserves the mix.  At each state,
  EVERY valid candidate (1 + 5*N_aircraft).  GT = determinized H=48
  discounted CBP-signal return, ONE cd.atc_prefix_rollout per
  candidate (D2 measured repeat-rollout noise ~1e-15, so one rollout
  suffices), frozen greedy continuation with re-issue masking seeded
  with the rolled-out first candidate (C2 caveats i-iii inherited
  verbatim).  Dataset saved to
  checkpoints/bluebird_controller/diagnostics/stage4_dataset_<ts>.npz
  BEFORE any training.
* PACE RULE: at the halfway mark of GT generation, if the projected
  total GT wall time exceeds 2 h, cut to the first ~80 interleaved
  states and say so.  Safety rail (added after the smoke run showed
  dense on-policy states reach 14 aircraft = 71 candidates, ~3x the
  ~21 assumed in the budget): if the GT phase itself exceeds 3.5 h
  wall, stop, save what exists, and flag it — the dataset stays
  usable because states are interleaved.
* PROBE: the SAME ControllerQNet (token_dim=60, n_instr=5,
  width_scalars=True — exactly the ckpt architecture), trained
  SUPERVISED: MSE of the Hurwicz@c=0.5 score against GT for ALL valid
  candidates of each state simultaneously.  At c=0.5 the Hurwicz
  score l + 0.5*(u-l) IS the interval midpoint, and c_train=0.5 in
  the ep2000 ckpt, so the probe readout is exactly the readout the RL
  policy acts on — midpoint-vs-Hurwicz is not a choice here, they
  coincide.  Two variants, identical regime (Adam lr 1e-3, batch 32
  states, 3000 steps, grad-clip 5): (A) fresh init, (B) initialized
  from the ep2000 q_net weights (does pretraining help or hurt?).
* SPLIT: 80/20 by STATE (stratified conflict/general, fixed seed
  20260714) — test measures ordering on unseen states, never unseen
  candidates of a seen state.
* METRICS (test split, final-step model — no test-based model
  selection): per-state Spearman(pred, GT) over that state's valid
  candidates (mean + median across states with nonzero GT spread;
  zero-spread states counted and excluded), and top-1 agreement
  (GT[argmax pred] >= GT.max() - 1e-9, so GT ties count as hits;
  chance ~ 1/21 ~ 5%).  BASELINE: the ep2000 RL net's own
  Hurwicz@c_train scores on the same test states (the C2-style null)
  = variant B evaluated at step 0.

PRE-REGISTERED VERDICT RULES (written before any training)
----------------------------------------------------------
* ARCHITECTURE-INNOCENT: some variant reaches mean test Spearman
  >= 0.5 AND top-1 agreement >= 40% (same variant, final step).
* ARCHITECTURE-GUILTY: both variants' mean test Spearman < 0.2
  DESPITE train-split fit (max train Spearman >= 0.5).
* CAPACITY (reported separately, not "guilty"): both test rho < 0.2
  AND both train rho < 0.5 — the net UNDERFITS even the train split.
* Anything else: AMBIGUOUS (reported with both numbers).

Operational constraints: a 12c training run is LIVE — torch pinned to
2 threads, nothing about the training run touched, run under
caffeinate.  All env-side machinery is reused from credit_diagnostics
/ diagnose_controller; nothing env-facing is reimplemented.

Usage:
    caffeinate -i .venv/bin/python stage4_supervised_probe.py
    .venv/bin/python stage4_supervised_probe.py --smoke
    .venv/bin/python stage4_supervised_probe.py --dataset <path.npz>
"""

import argparse
import json
import os
import time

import numpy as np
import torch

torch.set_num_threads(2)   # live 12c training run on this machine

import credit_diagnostics as cd
import diagnose_controller as dc
import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import (
    GAMMA, build_tokens, sector_snapshot, ControllerQNet,
    candidate_intervals,
)
from bluebird_interval_dqn import detect_violation, SEC_PER_STEP

DIAG_DIR = cd.DIAG_DIR
H_GT = 48
C_PROBE = 0.5
PACE_CAP_S = 2 * 3600      # projected-GT-wall cap at the halfway mark
CUT_TO = 80                # states kept if the pace rule fires
WALL_CAP_S = 3.5 * 3600    # hard GT-phase wall cap (safety rail)
SPLIT_SEED = 20260714
TRAIN_STEPS = 3000
BATCH_STATES = 32
LR = 1e-3
GRAD_CLIP = 5.0
EVAL_EVERY = 250
TIE_EPS = 1e-9

VERDICT_RULES = {
    "innocent": "some variant: mean test rho >= 0.5 AND top-1 >= 0.40",
    "guilty": "both test rho < 0.2 despite max train rho >= 0.5",
    "capacity": "both test rho < 0.2 AND both train rho < 0.5 (underfit)",
    "ambiguous": "anything else",
}


# ===========================================================================
# Dataset generation (all env machinery = cd/dc reuse)
# ===========================================================================

def collect_conflict_states(agent, env, n_states, hmax, budget,
                            seed0=71042, max_seeds=48):
    """cd.collect_conflict_states_atc with a parametrized seed range
    (D2's fixed range tops out well below 60 states).  Gate: on-policy
    (frozen greedy) state, min vertically-proximate pair sep <
    cd.D2_ATC_SEP_NM, NOOP-LoS within hmax on a raw deepcopy, >= 8
    steps between accepted states."""
    found = []
    for seed in range(seed0, seed0 + max_seeds):
        if len(found) >= n_states:
            break
        obs, info = env.reset(seed=seed)
        bcd.reset_delivery_clock(env)
        li = {}
        maxstep = int(getattr(env, "maxstep",
                              env.config.scenario_duration // SEC_PER_STEP))
        snap = sector_snapshot(env, obs.keys())
        last_acc = -100
        for step in range(maxstep):
            sep = cd.min_pair_sep_nm(snap)
            if sep < cd.D2_ATC_SEP_NM and step - last_acc >= 8 and \
                    len(found) < n_states:
                import copy
                env_c = copy.deepcopy(env)
                o = copy.deepcopy(obs)
                los_at = None
                for h in range(1, hmax + 1):
                    o, _, _, _, inf = env_c.step({cs: 0 for cs in o})
                    budget.add(1)
                    v, _, _ = detect_violation(inf)
                    if v:
                        los_at = h
                        break
                if los_at is not None:
                    found.append({"env": copy.deepcopy(env),
                                  "obs": copy.deepcopy(obs),
                                  "kind": "conflict", "seed": seed,
                                  "step": step, "min_sep": sep,
                                  "noop_los_at": los_at})
                    last_acc = step
                    print(f"    conflict state {len(found):2d}: "
                          f"seed={seed} step={step:3d} sep={sep:.1f}nm "
                          f"NOOP-LoS at +{los_at}")
            actions, aux = agent.generate_action(env, obs, info,
                                                 c=agent.c_train,
                                                 force_epsilon=0.0,
                                                 last_issued=li)
            if aux["cand_idx"] != 0 and agent.mask_reissue:
                i, j = divmod(aux["cand_idx"] - 1, agent.n_instr)
                li[aux["callsigns"][i]] = j
            obs, _, _, _, info = env.step(actions)
            budget.add(1)
            snap = sector_snapshot(env, obs.keys())
            v, _, _ = detect_violation(info)
            if v:
                break
    return found


def collect_general_states(agent, env, n_states, seeds=(91042, 91043,
                                                        91044, 91045)):
    """General on-policy states via dc.collect_probe_states (greedy
    policy episodes, min 2 aircraft, evenly spaced snapshots)."""
    per = int(np.ceil(n_states / len(seeds)))
    out = []
    for seed in seeds:
        if len(out) >= n_states:
            break
        raw = dc.collect_probe_states(
            env, seed, dc._agent_act_fn(agent, agent.c_train), per,
            min_aircraft=2)
        for env_s, obs_s, stratum, step in raw:
            snap = sector_snapshot(env_s, obs_s.keys())
            out.append({"env": env_s, "obs": obs_s, "kind": "general",
                        "seed": seed, "step": step,
                        "min_sep": cd.min_pair_sep_nm(snap),
                        "noop_los_at": -1})
        print(f"    general states: seed={seed} -> +{len(raw)} "
              f"(total {len(out)})")
    return out[:n_states]


def interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def save_dataset(path, recs, meta):
    S = len(recs)
    n_ac = np.array([r["tokens"].shape[0] for r in recs], dtype=np.int64)
    Nmax = int(n_ac.max())
    D = recs[0]["tokens"].shape[1]
    n_instr = meta["n_instr"]
    Cmax = 1 + n_instr * Nmax
    tok = np.zeros((S, Nmax, D), dtype=np.float32)
    gt = np.zeros((S, Cmax), dtype=np.float32)
    gtv = np.zeros((S, Cmax), dtype=bool)
    for i, r in enumerate(recs):
        n = n_ac[i]
        tok[i, :n] = r["tokens"]
        nc = 1 + n_instr * n
        gt[i, :nc] = r["gt"]
        gtv[i, :nc] = r["gt_violated"]
    np.savez_compressed(
        path, tok=tok, n_ac=n_ac, gt=gt, gt_violated=gtv,
        is_conflict=np.array([r["kind"] == "conflict" for r in recs]),
        seed=np.array([r["seed"] for r in recs], dtype=np.int64),
        step=np.array([r["step"] for r in recs], dtype=np.int64),
        min_sep=np.array([r["min_sep"] for r in recs], dtype=np.float32),
        noop_los_at=np.array([r["noop_los_at"] for r in recs],
                             dtype=np.int64),
        clone_steps=np.array([r["clone_steps"] for r in recs],
                             dtype=np.int64),
        meta_json=np.array(json.dumps(meta)))
    return path


def build_dataset(args):
    print("=" * 78)
    print(f"STAGE-4 DATASET — {args.n_conflict} conflict + "
          f"{args.n_general} general states, EVERY candidate, "
          f"GT = one determinized H={args.h_gt} CBP-signal rollout per "
          f"candidate (D2 noise ~1e-15)")
    print("=" * 78)
    dc.validate_cbp_mirror()
    agent, ckpt, path = cd.atc_agent(args)
    assert agent.n_instr == 5 and agent.c_train == C_PROBE
    env = dc.get_env(1200)
    budget = cd.StepBudget(10 ** 9)   # pace rule governs, not the cap

    t_scan = time.time()
    print("  [scan] conflict states ...")
    conf = collect_conflict_states(agent, env, args.n_conflict,
                                   args.h_gt, budget)
    if len(conf) < args.n_conflict:
        print(f"  [FLAG] only {len(conf)}/{args.n_conflict} conflict "
              f"states found in the seed range — proceeding")
    print("  [scan] general states ...")
    gen = collect_general_states(agent, env, args.n_general)
    states = interleave(conf, gen)
    scan_wall = time.time() - t_scan
    print(f"  [scan] {len(conf)} conflict + {len(gen)} general = "
          f"{len(states)} states in {scan_wall:.0f}s "
          f"({budget.n} priced scan steps)")

    total = len(states)
    target = total
    recs = []
    raw_clone_steps = 0
    priced0 = budget.n
    t_gt = time.time()
    cut_fired = False
    wall_capped = False
    for s_i, st in enumerate(states):
        if s_i >= target:
            print(f"  [pace] stopping at {s_i} states (cut applied)")
            break
        if time.time() - t_gt > WALL_CAP_S:
            wall_capped = True
            print(f"  [FLAG] hard wall cap {WALL_CAP_S / 3600:.1f}h hit "
                  f"at state {s_i} — stopping; interleaving keeps the "
                  f"conflict/general mix")
            break
        env_s, obs_s = st["env"], st["obs"]
        cs_list = sorted(obs_s.keys())
        snap_s = sector_snapshot(env_s, obs_s.keys())
        toks = build_tokens(env_s, obs_s, None, cs_list)
        n_cand = 1 + agent.n_instr * len(cs_list)
        gts = np.zeros(n_cand, dtype=np.float64)
        viol = np.zeros(n_cand, dtype=bool)
        steps_here = 0
        for k in range(n_cand):
            acts, issued, li0 = cd.cand_to_actions(k, cs_list, obs_s,
                                                   agent.n_instr)
            g, stp, vio, _ = cd.atc_prefix_rollout(
                env_s, obs_s, snap_s, acts, issued, (args.h_gt,),
                cd.make_continue(agent, agent.c_train, li0), budget)
            gts[k] = g[args.h_gt]
            viol[k] = vio
            steps_here += stp
        raw_clone_steps += steps_here
        recs.append({"tokens": toks, "gt": gts.astype(np.float32),
                     "gt_violated": viol, "kind": st["kind"],
                     "seed": st["seed"], "step": st["step"],
                     "min_sep": st["min_sep"],
                     "noop_los_at": st["noop_los_at"],
                     "clone_steps": steps_here})
        el = time.time() - t_gt
        print(f"  state {s_i:3d}/{target} [{st['kind']:8s}] "
              f"{len(cs_list)} ac, {n_cand:2d} cands, {steps_here:4d} "
              f"clone steps | GT spread "
              f"{gts.max() - gts.min():7.2f} | {el:6.0f}s elapsed")
        # informational projection at the quarter mark (no action)
        if s_i + 1 == total // 4:
            proj = el / (s_i + 1) * total
            print(f"  [pace] quarter mark: projected GT wall "
                  f"{proj / 3600:.2f}h (informational only)")
        # pre-registered pace rule: single check at the halfway mark
        if not cut_fired and s_i + 1 == total // 2:
            proj = el / (s_i + 1) * total
            if proj > PACE_CAP_S:
                target = min(total, CUT_TO)
                cut_fired = True
                print(f"  [pace] halfway: projected GT wall "
                      f"{proj / 3600:.2f}h > 2h -> cutting to first "
                      f"{target} interleaved states (mix preserved)")
            else:
                print(f"  [pace] halfway: projected GT wall "
                      f"{proj / 3600:.2f}h — full {total} states")
        if (s_i + 1) % 10 == 0:
            save_dataset(args.partial_path, recs, dataset_meta(
                args, path, ckpt, scan_wall, raw_clone_steps,
                budget.n - priced0, time.time() - t_gt, cut_fired))
    gt_wall = time.time() - t_gt
    meta = dataset_meta(args, path, ckpt, scan_wall, raw_clone_steps,
                        budget.n - priced0, gt_wall, cut_fired,
                        wall_capped=wall_capped)
    save_dataset(args.dataset_path, recs, meta)
    if os.path.exists(args.partial_path):
        os.remove(args.partial_path)
    print(f"\n  DATASET saved: {args.dataset_path}")
    print(f"  {len(recs)} states, {raw_clone_steps} raw clone steps "
          f"({budget.n - priced0} priced), GT wall {gt_wall:.0f}s "
          f"({gt_wall / 3600:.2f}h), cut_fired={cut_fired}")
    return args.dataset_path


def dataset_meta(args, ckpt_path, ckpt, scan_wall, raw_steps, priced,
                 gt_wall, cut_fired, wall_capped=False):
    return {"doc": "CREDIT_DIAGNOSTICS.md stage 4",
            "wall_capped": bool(wall_capped),
            "ckpt": os.path.basename(ckpt_path),
            "episode": int(ckpt.get("episode", -1)),
            "h_gt": args.h_gt, "gamma": GAMMA, "c_train": C_PROBE,
            "n_instr": 5, "token_dim": 60,
            "gt_signal": "CBP training signal (dc.sector_step_cbp), "
                         "C2 caveats i-iii inherited",
            "scan_wall_s": round(scan_wall),
            "raw_clone_steps": int(raw_steps),
            "priced_steps": int(priced),
            "gt_wall_s": round(gt_wall), "cut_fired": bool(cut_fired),
            "timestamp": time.strftime("%F %T")}


# ===========================================================================
# Probe training + metrics
# ===========================================================================

def load_dataset(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    d = {k: z[k] for k in z.files if k != "meta_json"}
    S, Nmax, _D = d["tok"].shape
    mask = np.zeros((S, Nmax), dtype=bool)
    for i in range(S):
        mask[i, :d["n_ac"][i]] = True
    d["mask"] = mask
    d["n_cand"] = 1 + meta["n_instr"] * d["n_ac"]
    return d, meta


def stratified_split(is_conflict, frac_test=0.2, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for grp in (True, False):
        idx = np.where(is_conflict == grp)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(frac_test * len(idx)))) if len(idx) \
            else 0
        test.extend(idx[:n_test].tolist())
        train.extend(idx[n_test:].tolist())
    return sorted(train), sorted(test)


def forward_scores(net, tok_t, mask_t):
    """Hurwicz@c=0.5 (== midpoint) candidate scores [B, Cmax]."""
    ac_l, ac_u, nl, nu = net(tok_t, mask_t)
    cl, cu, valid = candidate_intervals(ac_l, ac_u, nl, nu, mask_t)
    return cl + C_PROBE * (cu - cl), valid


def eval_split(net, data, idxs, device="cpu"):
    """Per-state Spearman + top-1 on the given state indices.  States
    with zero GT spread are excluded from rho (counted) but kept for
    top-1 (any argmax is a GT-argmax there)."""
    net.eval()
    tok_t = torch.from_numpy(data["tok"][idxs]).to(device)
    mask_t = torch.from_numpy(data["mask"][idxs]).to(device)
    with torch.no_grad():
        scores, _ = forward_scores(net, tok_t, mask_t)
    scores = scores.cpu().numpy()
    rhos, top1, skipped = [], [], 0
    for row, i in enumerate(idxs):
        nc = int(data["n_cand"][i])
        gt = data["gt"][i, :nc]
        pr = scores[row, :nc]
        rho = dc.spearman(gt, pr)
        if rho is None:
            skipped += 1
        else:
            rhos.append(rho)
        top1.append(bool(gt[int(np.argmax(pr))] >= gt.max() - TIE_EPS))
    return {"rho_mean": float(np.mean(rhos)) if rhos else float("nan"),
            "rho_median": float(np.median(rhos)) if rhos else
            float("nan"),
            "top1": float(np.mean(top1)),
            "n_rho": len(rhos), "n_skipped_zero_spread": skipped,
            "rhos": rhos}


def train_probe(name, init_sd, data, train_idx, test_idx, meta,
                steps=TRAIN_STEPS, bs=BATCH_STATES, lr=LR, seed=1):
    torch.manual_seed(seed)
    net = ControllerQNet(meta["token_dim"], n_instr=meta["n_instr"],
                         width_scalars=True)
    if init_sd is not None:
        net.load_state_dict(init_sd)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tok_t = torch.from_numpy(data["tok"])
    mask_t = torch.from_numpy(data["mask"])
    gt_t = torch.from_numpy(data["gt"])
    cand_valid = torch.zeros(gt_t.shape, dtype=torch.bool)
    for i in range(gt_t.shape[0]):
        cand_valid[i, :int(data["n_cand"][i])] = True
    rng = np.random.default_rng(seed)
    tr = np.array(train_idx)
    hist = []
    print(f"\n  [{name}] training: {steps} steps, batch {bs} states, "
          f"lr {lr}, {len(train_idx)} train / {len(test_idx)} test "
          f"states")
    t0 = time.time()
    for step in range(1, steps + 1):
        net.train()
        b = tr[rng.integers(len(tr), size=min(bs, len(tr)))]
        scores, _ = forward_scores(net, tok_t[b], mask_t[b])
        v = cand_valid[b]
        Cb = scores.shape[1]
        err = (scores - gt_t[b][:, :Cb]) ** 2
        loss = (err * v[:, :Cb]).sum() / v[:, :Cb].sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        opt.step()
        if step % EVAL_EVERY == 0 or step == steps:
            etr = eval_split(net, data, train_idx)
            ete = eval_split(net, data, test_idx)
            hist.append({"step": step, "loss": float(loss.item()),
                         "train_rho": etr["rho_mean"],
                         "test_rho": ete["rho_mean"],
                         "test_top1": ete["top1"]})
            print(f"    step {step:5d} loss {loss.item():10.3f} | "
                  f"train rho {etr['rho_mean']:+.3f} | test rho "
                  f"{ete['rho_mean']:+.3f} top1 {ete['top1']:.2f}")
    final_train = eval_split(net, data, train_idx)
    final_test = eval_split(net, data, test_idx)
    print(f"  [{name}] FINAL ({time.time() - t0:.0f}s): train rho "
          f"{final_train['rho_mean']:+.3f} (med "
          f"{final_train['rho_median']:+.3f}, top1 "
          f"{final_train['top1']:.2f}) | test rho "
          f"{final_test['rho_mean']:+.3f} (med "
          f"{final_test['rho_median']:+.3f}), top1 "
          f"{final_test['top1']:.2f}, {final_test['n_skipped_zero_spread']}"
          f" zero-spread test states skipped for rho")
    return {"name": name, "history": hist, "train": final_train,
            "test": final_test, "wall_s": round(time.time() - t0)}


def verdict(variants):
    """Pre-registered rules (see module header)."""
    innocent = any(v["test"]["rho_mean"] >= 0.5 and
                   v["test"]["top1"] >= 0.40 for v in variants)
    max_test = max(v["test"]["rho_mean"] for v in variants)
    max_train = max(v["train"]["rho_mean"] for v in variants)
    if innocent:
        return "ARCHITECTURE-INNOCENT"
    if max_test < 0.2:
        if max_train >= 0.5:
            return "ARCHITECTURE-GUILTY"
        return "CAPACITY (underfits train split — separate verdict)"
    return "AMBIGUOUS"


def run_training(args, dataset_path):
    data, meta = load_dataset(dataset_path)
    S = data["tok"].shape[0]
    train_idx, test_idx = stratified_split(data["is_conflict"])
    spread = np.array([np.ptp(data["gt"][i, :int(data["n_cand"][i])])
                       for i in range(S)])
    print("\n" + "=" * 78)
    print(f"STAGE-4 PROBE TRAINING — dataset {os.path.basename(dataset_path)}")
    print(f"  {S} states ({int(data['is_conflict'].sum())} conflict / "
          f"{int((~data['is_conflict']).sum())} general), candidates "
          f"{int(data['n_cand'].min())}-{int(data['n_cand'].max())} "
          f"(mean {data['n_cand'].mean():.1f})")
    print(f"  GT spread per state: median {np.median(spread):.2f}, "
          f"mean {spread.mean():.2f}, min {spread.min():.2f}, max "
          f"{spread.max():.2f}; GT range [{data['gt'].min():.2f}, "
          f"{data['gt'].max():.2f}]")
    print(f"  split: {len(train_idx)} train / {len(test_idx)} test "
          f"(stratified, seed {SPLIT_SEED})")
    print("  PRE-REGISTERED:", json.dumps(VERDICT_RULES))
    print("=" * 78)

    # RL-net baseline (C2-style null) = pretrained weights at step 0
    agent, ckpt, ckpt_path = cd.atc_agent(args)
    base_net = ControllerQNet(meta["token_dim"], n_instr=meta["n_instr"],
                              width_scalars=True)
    base_net.load_state_dict(agent.q_net.state_dict())
    base_test = eval_split(base_net, data, test_idx)
    base_train = eval_split(base_net, data, train_idx)
    print(f"\n  [RL-baseline ep2000, Hurwicz@c={C_PROBE}] test rho "
          f"{base_test['rho_mean']:+.3f} (med {base_test['rho_median']:+.3f})"
          f", top1 {base_test['top1']:.2f} | train-side rho "
          f"{base_train['rho_mean']:+.3f}")

    steps = args.train_steps
    vA = train_probe("A fresh-init", None, data, train_idx, test_idx,
                     meta, steps=steps, seed=1)
    vB = train_probe("B ep2000-init", agent.q_net.state_dict(), data,
                     train_idx, test_idx, meta, steps=steps, seed=2)
    vd = verdict([vA, vB])

    print("\n" + "=" * 78)
    print("STAGE-4 SUMMARY (pre-registered rules, final-step models)")
    print("=" * 78)
    print(f"  {'variant':<22}{'train rho':>10}{'test rho':>10}"
          f"{'test med':>10}{'top-1':>8}")
    for v in (vA, vB):
        print(f"  {v['name']:<22}{v['train']['rho_mean']:>10.3f}"
              f"{v['test']['rho_mean']:>10.3f}"
              f"{v['test']['rho_median']:>10.3f}"
              f"{v['test']['top1']:>8.2f}")
    print(f"  {'RL baseline (null)':<22}{base_train['rho_mean']:>10.3f}"
          f"{base_test['rho_mean']:>10.3f}"
          f"{base_test['rho_median']:>10.3f}{base_test['top1']:>8.2f}")
    print(f"  chance top-1 ~ {1.0 / data['n_cand'].mean():.3f}")
    print(f"\n  VERDICT: {vd}")

    out = {"timestamp": time.strftime("%F %T"),
           "doc": "CREDIT_DIAGNOSTICS.md stage 4",
           "dataset": dataset_path, "dataset_meta": meta,
           "n_states": int(S),
           "n_conflict": int(data["is_conflict"].sum()),
           "cand_mean": float(data["n_cand"].mean()),
           "gt_spread_median": float(np.median(spread)),
           "gt_spread_mean": float(spread.mean()),
           "split": {"train": train_idx, "test": test_idx,
                     "seed": SPLIT_SEED},
           "prereg_rules": VERDICT_RULES,
           "baseline": {"train": base_train, "test": base_test},
           "variants": [vA, vB], "verdict": vd,
           "train_steps": steps}
    os.makedirs(DIAG_DIR, exist_ok=True)
    out_path = os.path.join(
        DIAG_DIR, f"stage4_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(dc.to_jsonable(out), f, indent=1)
    print(f"\nJSON written: {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Stage-4 supervised "
                                             "ordering probe")
    ap.add_argument("--dataset", default=None,
                    help="existing stage4 npz: skip generation")
    ap.add_argument("--atc_ckpt", default=None,
                    help="default: 12c tag from run12c_console.log, "
                         "_ep2000.pt")
    ap.add_argument("--n_conflict", type=int, default=60)
    ap.add_argument("--n_general", type=int, default=60)
    ap.add_argument("--h_gt", type=int, default=H_GT)
    ap.add_argument("--train_steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_conflict, args.n_general = 2, 2
        args.h_gt = 12
        args.train_steps = 200
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = "smoke_" if args.smoke else ""
    os.makedirs(DIAG_DIR, exist_ok=True)
    args.dataset_path = os.path.join(
        DIAG_DIR, f"stage4_dataset_{tag}{ts}.npz")
    args.partial_path = os.path.join(
        DIAG_DIR, f"stage4_dataset_{tag}{ts}_partial.npz")

    torch.manual_seed(0)
    dataset_path = args.dataset or build_dataset(args)
    run_training(args, dataset_path)


if __name__ == "__main__":
    main()
