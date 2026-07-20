"""
ROUND-3 D1 ordering-instrument rebuild (2026-07-20)
===================================================

Replaces the round-1/2 rho instrument that the 20-July audit voided:
T5 measured a 30-init untrained null band of roughly [-0.6, +0.85] that
swallows the 0.4 bar (the 20 "states" were timestamps of ONE trajectory,
effective n ~ 1), and T6 measured a +0.18 basis ceiling (NOOP-basis GT
vs frozen-policy training basis). This module is the D1 rebuild per
ROUND3_DESIGN.md D1 + ROUND3_REVIEW.md "D1 (GT/null-band lifecycle)":

  * probe sets drawn from N >= 8 INDEPENDENT scenario seeds/geometries
    per test family (states_per_seed x n_seeds; autocorrelation within a
    seed is quarantined by reporting the per-seed statistic);
  * every instrument ships its 30-random-init untrained NULL BAND
    (mean/sd/2.5-97.5 pct) beside every reading — NOOP basis only (the
    null-basis rule: a shared-basis frozen null is invalid);
  * DUAL-BASIS GT lifecycle: the NOOP-basis GT table is built ONCE per
    probe set and rescored per checkpoint at full cadence (net forward
    passes only); frozen-policy-basis tables are built ONLY at the
    pre-registered checkpoint set {init, quarter, half, final} and are
    read via the T6-style basis-agreement number beside them;
  * rho reported at c in {0, c_train, 1}; rho_mid DROPPED (algebraic
    duplicate of the Hurwicz score at c = 0.5);
  * probe-set FINGERPRINT (scenario seeds, probe steps, strata, env
    config, GT-relevant pricing echo) stored with every table and band;
    readers HARD-FAIL on mismatch (SystemExit, never a warning).

PRE-REGISTERED SIZES AND BUDGET (fixed before any run; unit measured in
audit_measurements/t6_run.log — ~240 s wall per 20-state dual-basis
table, 440 branches, deepcopies included, i.e. ~0.55 s/branch and
~120 s per 20-state single-basis table):

  states_per_seed = 3, n_seeds = 8  ->  24 states / family
  families: "conflict" (M1/M3 geometry) and "holdfire" (M2 geometry)

  phase                          branches      predicted wall
  NOOP-basis GT, one family      24 x 11=264   ~145 s (once per set)
  null bands, one family         forward only  ~1-2 min (once per set)
  per-checkpoint NOOP rescore    forward only  ~1 s
  frozen-basis table, 1 ckpt     264           ~145 s  (x4 ckpts/arm:
                                                ~10 min/arm/family)

Measured walls are written into every table JSON and printed by
--budget; ROUND3_INSTRUMENTS_STATUS.md carries the measured table.

MANDATORY RECOMPUTE TRIGGERS (staleness governance): probe regeneration,
any pricing change (vertical_ramp / delta_conflict / reward constants),
any ENV-LEVEL termination change. A1 (all-delivered terminal) QUALIFIES:
when it lands, bcd must define/bump ENV_TERMINATION_VERSION — the
fingerprint includes getattr(bcd, "ENV_TERMINATION_VERSION", "pre-A1"),
so the A1 build bumping that constant hard-fails every pre-A1 table
exactly as the review requires. A2 does NOT trigger under the
no-segmentation default (it never touches the scripted NOOP probe
trajectory).

Usage (all runnable NOW — nothing here needs a trained net):
    .venv/bin/python round3_instruments.py --scan            # seed banks
    .venv/bin/python round3_instruments.py --build           # NOOP GT
    .venv/bin/python round3_instruments.py --nulls           # null bands
    .venv/bin/python round3_instruments.py --frozen --ckpt P # dual basis
    .venv/bin/python round3_instruments.py --read --ckpt P   # full read
    .venv/bin/python round3_instruments.py --budget
"""

import argparse
import copy
import glob
import hashlib
import json
import os
import time

import numpy as np
import torch

import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import (
    GAMMA, N_INSTR, ControllerAgent, sector_snapshot, build_tokens,
)
from bluebird_interval_dqn import SEC_PER_STEP
from diagnose_controller import (
    collect_probe_states, rollout_return, spearman, noop_action,
    make_custom_density_env, to_jsonable,
)
from micro_battery_common import (
    bar, bcd_config_echo, noop_trace_multi, pair_series,
)
from micro_battery_m3 import lateral_keep

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "checkpoints", "round3_instruments")

# ---- pre-registered constants (fixed 20 July 2026, before any run) ----
STATES_PER_SEED = 3
N_SEEDS = 8
NULL_INITS = 30
NULL_TORCH_SEED0 = 7000          # seeds 7000..7029 (T5 discipline)
FROZEN_CADENCE = ("init", "quarter", "half", "final")
HOT_SPREAD = 5.0                 # gt_spread threshold for the hot subset
H_CAP = 80                       # GT horizon cap (T6 unit)
# scenario-family acceptance (identical to the round-1 finders)
DFL_LO, DFL_HI = 10.0, 20.0      # holdfire: level-separated band
WOULD_BE_CPA_NM = 5.0            # holdfire: laterally proximate
ENV_CFG = {"duration": 600, "initial_spawn_rate": 0.0,
           "max_spawn_rate": 0.0, "num_starter_aircraft": 2}

DEFAULT_SCAN = [10043, 20042] + list(range(100, 400))


def make_env():
    return make_custom_density_env(
        duration=ENV_CFG["duration"],
        initial_spawn_rate=ENV_CFG["initial_spawn_rate"],
        max_spawn_rate=ENV_CFG["max_spawn_rate"],
        num_starter_aircraft=ENV_CFG["num_starter_aircraft"])


# ===========================================================================
# Fingerprint + hard-fail staleness governance
# ===========================================================================

def gt_relevant_echo():
    """The slice of the D2 config echo the GT tables actually depend on
    (pricing + reward constants + env termination semantics). Run-side
    knobs (agent seed, episode counts) deliberately excluded — they can
    never invalidate a GT rollout."""
    return {
        "vertical_ramp": bcd.VERTICAL_RAMP,
        "delta_conflict": bcd.DELTA_CONFLICT,
        "gamma": GAMMA,
        "cmd_cost": bcd.CMD_COST,
        "delivery_bonus": bcd.DELIVERY_BONUS,
        "delivery_floor": bcd.DELIVERY_FLOOR,
        "violation_penalty": bcd.VIOLATION_PENALTY,
        "objective_v2": True,
        "gt_basis_rule": "NOOP continuation, CBP-priced, horizon capped "
                         "at episode end (build_probe_set_capped rule)",
        # A1 recompute trigger: the A1 build MUST bump this constant in
        # bcd; every pre-A1 table then hard-fails (fingerprint rule).
        "env_termination_version": getattr(bcd, "ENV_TERMINATION_VERSION",
                                           "pre-A1"),
    }


def fingerprint_of(family, entries, probe):
    fp = {
        "instrument": "round3_instruments D1 rebuild",
        "family": family,
        "scenario_seeds": [e["seed"] for e in entries],
        "states_per_seed": STATES_PER_SEED,
        "h_per_seed": {str(e["seed"]): e["h"] for e in entries},
        "probe_steps": {str(e["seed"]):
                        [p["step"] for p in probe
                         if p["seed"] == e["seed"]] for e in entries},
        "strata": {str(e["seed"]):
                   [p["stratum"] for p in probe
                    if p["seed"] == e["seed"]] for e in entries},
        "env_config": ENV_CFG,
        "sec_per_step": SEC_PER_STEP,
        "n_instr": N_INSTR,
        "config_echo_gt": gt_relevant_echo(),   # D2 echo folded in
    }
    return fp


def fp_hash(fp):
    return hashlib.sha256(
        json.dumps(to_jsonable(fp), sort_keys=True).encode()).hexdigest()


def check_fingerprint(stored_fp, context):
    """HARD-FAIL staleness check: the stored table's GT-relevant echo
    must match the LIVE module state, else every reading through it is
    a category error (T6's lesson, made mechanical)."""
    cur = gt_relevant_echo()
    old = stored_fp.get("config_echo_gt", {})
    diffs = {k: (old.get(k, "<missing>"), v) for k, v in cur.items()
             if to_jsonable(old.get(k, "<missing>")) != to_jsonable(v)}
    if diffs:
        raise SystemExit(
            f"[STALENESS HARD-FAIL] {context}: stored fingerprint does "
            f"not match live pricing/env semantics: "
            + "; ".join(f"{k}: stored={a!r} live={b!r}"
                        for k, (a, b) in diffs.items())
            + " — rebuild the probe set/GT tables (mandatory recompute "
              "trigger fired); older readings are a separate series.")


def save_table(name, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    payload = dict(payload)
    payload["fingerprint_hash"] = fp_hash(payload["fingerprint"])
    payload["built"] = time.strftime("%F %T")
    path = os.path.join(OUT_DIR, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_jsonable(payload), f)
    os.replace(tmp, path)
    print(f"  table written: {path}")
    return path


def load_table(name, context):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        raise SystemExit(f"[{context}] missing table {path} — run the "
                         f"build step first (see module docstring)")
    with open(path) as f:
        payload = json.load(f)
    if fp_hash(payload["fingerprint"]) != payload["fingerprint_hash"]:
        raise SystemExit(f"[STALENESS HARD-FAIL] {context}: {path} "
                         f"fingerprint hash mismatch (file edited or "
                         f"corrupt) — rebuild")
    check_fingerprint(payload["fingerprint"], f"{context}: {path}")
    return payload


# ===========================================================================
# Scenario banks: N >= 8 independent seeds/geometries per family
# ===========================================================================

def qualify_conflict(env, seed):
    """M1/M3 geometry: 2-starter NOOP -> loss_of_separation."""
    rows, violated, kind, vstep, inv = noop_trace_multi(env, seed)
    if violated and kind == "loss_of_separation":
        return {"seed": seed, "vstep": vstep,
                "h": min(H_CAP, max(15, vstep + 2))}
    return None


def qualify_holdfire(env, seed):
    """M2 geometry: NOOP-clean, |dFL| in [10, 20) throughout, would-be
    lateral CPA < 5 nm (micro_battery_m2.find_scenario criterion)."""
    rows, violated, kind, vstep, inv = noop_trace_multi(env, seed)
    if violated:
        return None
    paired = [r for r in rows if r["pairs"]]
    if not paired:
        return None
    pair = sorted(paired[0]["pairs"])[0]
    ser = pair_series(rows, pair)
    dmin_step, dmin, dfl_at = min(ser, key=lambda x: x[1])
    dfl_min = min(x[2] for x in ser)
    if DFL_LO <= dfl_at < DFL_HI and dfl_min >= DFL_LO \
            and dmin < WOULD_BE_CPA_NM:
        return {"seed": seed, "cpa_step": dmin_step,
                "cpa_d_nm": round(dmin, 2), "h": H_CAP}
    return None


QUALIFIERS = {"conflict": qualify_conflict, "holdfire": qualify_holdfire}


def scan_family(env, family, seeds, need):
    entries = []
    t0 = time.time()
    for seed in seeds:
        got = QUALIFIERS[family](env, seed)
        if got:
            entries.append(got)
            print(f"  [{family}] seed {seed} QUALIFIES "
                  f"({len(entries)}/{need}) {got}")
        if len(entries) >= need:
            break
    print(f"  [{family}] scan: {len(entries)} seeds in "
          f"{time.time() - t0:.0f}s")
    if len(entries) < need:
        print(f"  [FLAG] only {len(entries)}/{need} independent "
              f"{family} seeds found — N >= {need} is pre-registered; "
              f"widen the scan before reading bars through this set")
    return entries


# ===========================================================================
# Probe harvest + NOOP-basis GT (built once per probe set)
# ===========================================================================

def harvest_states(env, entry):
    """Frozen states along the seed's NOOP trajectory (deterministic
    env; identical call regenerates identical states — the frozen-basis
    rebuild relies on this and asserts steps+strata)."""
    return collect_probe_states(
        env, entry["seed"], lambda e, o, i: (noop_action(o), False),
        STATES_PER_SEED, min_aircraft=2)


def build_noop_gt(env, family, entries):
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    probe = []
    t0 = time.time()
    n_branches = 0
    for entry in entries:
        states = harvest_states(env, entry)
        for env_s, obs_s, stratum, step in states:
            cs_list = sorted(obs_s.keys())
            snap_s = sector_snapshot(env_s, obs_s.keys())
            toks = build_tokens(env_s, obs_s, None, cs_list)
            n_cand = 1 + N_INSTR * len(cs_list)
            h_s = max(1, min(entry["h"], maxstep - step))
            gts = []
            for k in range(n_cand):
                acts = noop_action(obs_s)
                issued = k != 0
                if issued:
                    i, j = divmod(k - 1, N_INSTR)
                    acts[cs_list[i]] = j + 1
                G, _st, _vio, _dl = rollout_return(
                    env_s, copy.deepcopy(obs_s), snap_s, acts, issued,
                    h_s, continue_policy=None, use_cbp=True)
                gts.append(float(G))
                n_branches += 1
            probe.append({"seed": entry["seed"], "step": step,
                          "stratum": stratum, "h": h_s,
                          "cs_list": cs_list, "n_cand": n_cand,
                          "tokens": toks.tolist(), "gt": gts,
                          "gt_spread": float(max(gts) - min(gts))})
        print(f"  [{family}] seed {entry['seed']}: {len(states)} states "
              f"harvested ({time.time() - t0:.0f}s)")
    hot = sum(1 for p in probe if p["gt_spread"] > HOT_SPREAD)
    wall = time.time() - t0
    print(f"  [{family}] NOOP-basis GT: {len(probe)} states / "
          f"{len(entries)} seeds, {n_branches} branches, {hot} hot "
          f"(spread > {HOT_SPREAD}), {wall:.0f}s wall")
    return probe, {"n_branches": n_branches, "wall_s": round(wall),
                   "n_hot": hot}


def probe_tokens(p):
    return np.asarray(p["tokens"], dtype=np.float32)


def probe_gt(p, key="gt"):
    return np.asarray(p[key], dtype=float)


# ===========================================================================
# The instrument read (candidate_q ONLY — C-instruments lock applies)
# ===========================================================================

def ordering_read(agent, probe, cs, cand_keep=None, gt_key="gt"):
    """All D1 scalar readings for one net on one probe table.

    C-INSTRUMENTS LOCK: reads candidate_q directly; must never route
    through select_candidate/generate_action (tolerance + count bonus
    live there). Locked by test_c_instruments_lock.py.

    Per Hurwicz c in cs (pre-registered: {0, c_train, 1}; rho_mid
    DROPPED): pooled per-state rho, hot-subset rho, per-seed mean rho
    (the independent-n statistic, n = number of seeds) with its sd,
    NOOP-argmax fraction, NOOP margin (best non-NOOP minus NOOP; mean +
    p95 — the C2 tolerance derivation's noise-band input), and the
    NOOP-level median |Hurwicz(NOOP) - GT(NOOP)|."""
    out = {}
    for c in cs:
        rhos, rhos_hot, by_seed = [], [], {}
        noop_hits, margins, level_devs = [], [], []
        for p in probe:
            n = p["n_cand"]
            cl, cu = agent.candidate_q(probe_tokens(p))
            scores = (cl + c * (cu - cl))[:n]
            gt = probe_gt(p, gt_key)
            if cand_keep is not None:
                keep = np.array([cand_keep(k) for k in range(n)])
                scores_k, gt_k = scores[keep], gt[keep]
            else:
                scores_k, gt_k = scores, gt
            rho = spearman(gt_k, scores_k)
            if rho is not None:
                rhos.append(rho)
                by_seed.setdefault(p["seed"], []).append(rho)
                if float(gt_k.max() - gt_k.min()) > HOT_SPREAD:
                    rhos_hot.append(rho)
            # cand_keep (if any) must keep candidate 0 (NOOP) — asserted
            # so subset margins stay NOOP-relative
            assert cand_keep is None or cand_keep(0)
            best = scores_k.max()
            noop_hits.append(scores_k[0] == best)
            if len(scores_k) > 1:
                margins.append(float(scores_k[1:].max() - scores_k[0]))
            level_devs.append(abs(float(scores_k[0]) - float(gt[0])))
        seed_means = [float(np.mean(v)) for v in by_seed.values()]
        out[f"c{c}"] = {
            "rho": float(np.mean(rhos)) if rhos else float("nan"),
            "rho_hot": float(np.mean(rhos_hot)) if rhos_hot
            else float("nan"),
            "rho_perseed_mean": float(np.mean(seed_means))
            if seed_means else float("nan"),
            "rho_perseed_sd": float(np.std(seed_means, ddof=1))
            if len(seed_means) > 1 else float("nan"),
            "n_states": len(rhos), "n_hot": len(rhos_hot),
            "n_seeds": len(seed_means),
            "noop_argmax_frac": float(np.mean(noop_hits)),
            "noop_margin_mean": float(np.mean(margins))
            if margins else float("nan"),
            "noop_margin_p95": float(np.percentile(margins, 95))
            if margins else float("nan"),
            "noop_level_med_dev": float(np.median(level_devs))
            if level_devs else float("nan"),
        }
    return out


NULL_SCALARS = ("rho", "rho_hot", "rho_perseed_mean", "noop_argmax_frac",
                "noop_margin_mean", "noop_margin_p95",
                "noop_level_med_dev")


def build_null_bands(probe, cs, cand_keep=None):
    """30 fresh random-init nets scored by the SAME read on the SAME
    probe table — NOOP basis ONLY (null-basis rule: frozen-basis
    readings get the basis-agreement number, never a null of their
    own). Returns {c: {scalar: band}}."""
    token_dim = probe_tokens(probe[0]).shape[1]
    vals = {f"c{c}": {s: [] for s in NULL_SCALARS} for c in cs}
    t0 = time.time()
    for i in range(NULL_INITS):
        torch.manual_seed(NULL_TORCH_SEED0 + i)
        agent = ControllerAgent(token_dim=token_dim, n_instr=N_INSTR,
                                device="cpu")
        agent.q_net.eval()
        r = ordering_read(agent, probe, cs, cand_keep=cand_keep)
        for ck, d in r.items():
            for s in NULL_SCALARS:
                vals[ck][s].append(d[s])
    wall = time.time() - t0

    def band(v):
        v = np.asarray([x for x in v if x == x], float)
        if len(v) == 0:
            return None
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "p2.5": float(np.percentile(v, 2.5)),
                "p97.5": float(np.percentile(v, 97.5)),
                "min": float(v.min()), "max": float(v.max()),
                "n": int(len(v))}

    bands = {ck: {s: band(v) for s, v in d.items()}
             for ck, d in vals.items()}
    print(f"  null bands: {NULL_INITS} inits (torch seeds "
          f"{NULL_TORCH_SEED0}..{NULL_TORCH_SEED0 + NULL_INITS - 1}, "
          f"token_dim {token_dim}) in {wall:.0f}s")
    return bands, round(wall)


# ===========================================================================
# Frozen-policy-basis GT (pre-registered checkpoints only) + agreement
# ===========================================================================

def frozen_checkpoint_episodes(total_episodes):
    """The pre-registered frozen-basis cadence {init, quarter, half,
    final} in episode numbers for a run of total_episodes."""
    return {"init": 0, "quarter": total_episodes // 4,
            "half": total_episodes // 2, "final": total_episodes}


def _frozen_policy(agent, li):
    """Greedy mask-aware continuation (T6 pattern; cf_rng so no live
    RNG stream is advanced)."""
    def pol(env, obs):
        acts, aux = agent.generate_action(env, obs, None,
                                          force_epsilon=0.0,
                                          last_issued=li,
                                          rng=agent.cf_rng)
        idx = aux["cand_idx"]
        if agent.mask_reissue and idx > 0:
            csl = sorted(obs)
            i, j = divmod(idx - 1, agent.n_instr)
            li[csl[i]] = j
        return acts, idx != 0
    return pol


def build_frozen_gt(env, agent, table, ckpt_tag):
    """Frozen-policy-basis GT for every state in a NOOP-basis table,
    regenerating the frozen states from the fingerprint recipe
    (deterministic env) with a steps+strata hard-fail assert. Also
    computes the per-state basis-agreement Spearman(gt_noop, gt_frozen)
    — the number frozen-basis readings are interpreted through."""
    fp = table["fingerprint"]
    probe = table["probe"]
    maxstep = int(getattr(env, "maxstep",
                          env.config.scenario_duration // SEC_PER_STEP))
    t0 = time.time()
    frozen_gt, agreement = [], []
    n_branches = 0
    for seed_str, steps in fp["probe_steps"].items():
        seed = int(seed_str)
        entry = {"seed": seed, "h": fp["h_per_seed"][seed_str]}
        states = harvest_states(env, entry)
        got_steps = [s[3] for s in states]
        got_strata = [s[2] for s in states]
        if got_steps != steps or got_strata != fp["strata"][seed_str]:
            raise SystemExit(
                f"[STALENESS HARD-FAIL] frozen-basis rebuild: seed "
                f"{seed} regenerated steps/strata {got_steps}/"
                f"{got_strata} != fingerprint {steps}/"
                f"{fp['strata'][seed_str]} — env or probe drift; "
                f"rebuild the probe set")
        for env_s, obs_s, stratum, step in states:
            rec = next(p for p in probe
                       if p["seed"] == seed and p["step"] == step)
            cs_list = sorted(obs_s.keys())
            assert cs_list == rec["cs_list"], (seed, step)
            snap_s = sector_snapshot(env_s, obs_s.keys())
            h_s = rec["h"]
            gtf = []
            for k in range(rec["n_cand"]):
                acts = noop_action(obs_s)
                issued = k != 0
                li = {}
                if issued:
                    i, j = divmod(k - 1, N_INSTR)
                    acts[cs_list[i]] = j + 1
                    if agent.mask_reissue:
                        li[cs_list[i]] = j
                G, _st, _vio, _dl = rollout_return(
                    env_s, copy.deepcopy(obs_s), snap_s, dict(acts),
                    issued, h_s,
                    continue_policy=_frozen_policy(agent, li),
                    use_cbp=True)
                gtf.append(float(G))
                n_branches += 1
            frozen_gt.append({"seed": seed, "step": step,
                              "gt_frozen": gtf})
            agr = spearman(rec["gt"], gtf)
            agreement.append(agr)
    wall = time.time() - t0
    agr_vals = [a for a in agreement if a is not None]
    out = {
        "fingerprint": fp, "ckpt_tag": ckpt_tag,
        "frozen_gt": frozen_gt,
        "basis_agreement_per_state": agreement,
        "basis_agreement_mean": float(np.mean(agr_vals))
        if agr_vals else float("nan"),
        "basis_agreement_sd": float(np.std(agr_vals, ddof=1))
        if len(agr_vals) > 1 else float("nan"),
        "n_branches": n_branches, "wall_s": round(wall),
    }
    print(f"  frozen-basis GT [{ckpt_tag}]: {len(frozen_gt)} states, "
          f"{n_branches} branches, basis agreement "
          f"{out['basis_agreement_mean']:+.3f} "
          f"(sd {out['basis_agreement_sd']:.3f}), {wall:.0f}s wall")
    return out


def merge_frozen(table, frozen_payload):
    """Attach gt_frozen to the probe records (in memory) so
    ordering_read(..., gt_key='gt_frozen') works."""
    by_key = {(r["seed"], r["step"]): r["gt_frozen"]
              for r in frozen_payload["frozen_gt"]}
    for p in table["probe"]:
        p["gt_frozen"] = by_key[(p["seed"], p["step"])]
    return table


# ===========================================================================
# Reporting: every reading printed WITH its null band
# ===========================================================================

def report_read(label, read, nulls, bars=()):
    """Print every scalar beside its null band; then any pre-registered
    bars as (name, c_key, scalar, op, thresh)."""
    print(f"\n  [{label}] readings (NOOP basis; null = 30 untrained "
          f"inits on the same probe table):")
    for ck, d in read.items():
        nb = nulls.get(ck, {}) if nulls else {}
        for s in NULL_SCALARS:
            b = nb.get(s)
            btxt = (f"null mean {b['mean']:+.3f} sd {b['sd']:.3f} band "
                    f"[{b['p2.5']:+.3f}, {b['p97.5']:+.3f}]"
                    if b else "null band MISSING")
            v = d[s]
            vtxt = f"{v:+.3f}" if v == v else "nan"
            inside = (b and v == v and b["p2.5"] <= v <= b["p97.5"])
            print(f"    {ck:>6} {s:<20} {vtxt:>8}   [{btxt}]"
                  f"{'  << inside null band' if inside else ''}")
    results = {}
    for (name, ck, scalar, op, thresh) in bars:
        results[name] = bar(name, read[ck][scalar], op, thresh,
                            null_band=(nulls or {}).get(ck, {}).get(scalar))
    return results


# ===========================================================================
# CLI
# ===========================================================================

def cs_for(c_train):
    return (0.0, float(c_train), 1.0)


FAMILY_KEEP = {"conflict": None, "holdfire": None}
# lateral-subset read (M3's trained support) is reported on the conflict
# family beside the full read
SUBSET_READS = {"conflict": [("lateral", lateral_keep)], "holdfire": []}


def main():
    ap = argparse.ArgumentParser(description="ROUND3 D1 ordering "
                                             "instrument rebuild")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--nulls", action="store_true")
    ap.add_argument("--frozen", action="store_true",
                    help="build the frozen-policy-basis table for --ckpt "
                         "(pre-registered cadence: init/quarter/half/"
                         "final only)")
    ap.add_argument("--read", action="store_true",
                    help="full instrument read for --ckpt with null "
                         "bands (and frozen basis if a table exists)")
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--family", type=str, default="both",
                    choices=["conflict", "holdfire", "both"])
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--ckpt_tag", type=str, default=None,
                    help="tag for the frozen table filename (default: "
                         "checkpoint basename)")
    ap.add_argument("--c_train", type=float, default=0.5)
    ap.add_argument("--scan_seeds", type=int, nargs="+",
                    default=DEFAULT_SCAN)
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--vertical_ramp", type=str, default="gate",
                    choices=["gate", "smooth"])
    ap.add_argument("--delta_conflict", type=float, default=None)
    args = ap.parse_args()

    bcd.set_conflict_pricing(args.vertical_ramp, args.delta_conflict)
    print("=" * 74)
    print("ROUND3 D1 ORDERING INSTRUMENT (multi-seed, dual-basis, "
          "null-banded)")
    print(f"[d1] {bcd.conflict_pricing_str()}; states_per_seed="
          f"{STATES_PER_SEED}, n_seeds={args.n_seeds}, "
          f"c in {cs_for(args.c_train)}")
    print("=" * 74)
    families = (["conflict", "holdfire"] if args.family == "both"
                else [args.family])
    env = make_env()
    cs = cs_for(args.c_train)

    if args.scan:
        for fam in families:
            entries = scan_family(env, fam, args.scan_seeds, args.n_seeds)
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(os.path.join(OUT_DIR, f"bank_{fam}.json"),
                      "w") as f:
                json.dump({"family": fam, "entries": entries,
                           "scan_seeds": [args.scan_seeds[0],
                                          args.scan_seeds[-1]],
                           "built": time.strftime("%F %T")}, f, indent=1)
            print(f"  bank written: bank_{fam}.json")

    if args.build:
        for fam in families:
            with open(os.path.join(OUT_DIR, f"bank_{fam}.json")) as f:
                entries = json.load(f)["entries"]
            probe, stats = build_noop_gt(env, fam, entries)
            fp = fingerprint_of(fam, entries, probe)
            save_table(f"gt_noop_{fam}.json",
                       {"fingerprint": fp, "probe": probe,
                        "build_stats": stats})

    if args.nulls:
        for fam in families:
            table = load_table(f"gt_noop_{fam}.json", "nulls")
            bands, wall = build_null_bands(table["probe"], cs)
            payload = {"fingerprint": table["fingerprint"],
                       "n_inits": NULL_INITS,
                       "torch_seeds": f"{NULL_TORCH_SEED0}.."
                                      f"{NULL_TORCH_SEED0 + NULL_INITS - 1}",
                       "basis": "NOOP only (null-basis rule)",
                       "bands": bands, "wall_s": wall}
            for sub_name, keep in SUBSET_READS[fam]:
                sub_bands, sub_wall = build_null_bands(table["probe"], cs,
                                                       cand_keep=keep)
                payload[f"bands_{sub_name}"] = sub_bands
                payload["wall_s"] += sub_wall
            save_table(f"nulls_{fam}.json", payload)

    if args.frozen:
        assert args.ckpt, "--frozen needs --ckpt"
        agent, ck = bcd.load_agent(args.ckpt)
        tag = args.ckpt_tag or \
            os.path.splitext(os.path.basename(args.ckpt))[0]
        for fam in families:
            table = load_table(f"gt_noop_{fam}.json", "frozen")
            fr = build_frozen_gt(env, agent, table, tag)
            save_table(f"gt_frozen_{fam}_{tag}.json", fr)

    if args.read:
        assert args.ckpt, "--read needs --ckpt"
        agent, ck = bcd.load_agent(args.ckpt)
        tag = args.ckpt_tag or \
            os.path.splitext(os.path.basename(args.ckpt))[0]
        for fam in families:
            table = load_table(f"gt_noop_{fam}.json", "read")
            nulls_pl = load_table(f"nulls_{fam}.json", "read")
            read = ordering_read(agent, table["probe"], cs)
            report_read(f"{fam} / {tag} / full candidate set", read,
                        nulls_pl["bands"])
            for sub_name, keep in SUBSET_READS[fam]:
                sub = ordering_read(agent, table["probe"], cs,
                                    cand_keep=keep)
                report_read(f"{fam} / {tag} / {sub_name} subset", sub,
                            nulls_pl.get(f"bands_{sub_name}"))
            fpath = os.path.join(OUT_DIR, f"gt_frozen_{fam}_{tag}.json")
            if os.path.exists(fpath):
                fr = load_table(f"gt_frozen_{fam}_{tag}.json", "read")
                merge_frozen(table, fr)
                fread = ordering_read(agent, table["probe"], cs,
                                      gt_key="gt_frozen")
                print(f"\n  [{fam} / {tag} / FROZEN basis] interpreted "
                      f"via basis agreement "
                      f"{fr['basis_agreement_mean']:+.3f} (sd "
                      f"{fr['basis_agreement_sd']:.3f}) — NO null band "
                      f"by rule (a shared-basis frozen null is invalid)")
                for ck_key, d in fread.items():
                    print(f"    {ck_key:>6} rho={d['rho']:+.3f} "
                          f"hot={d['rho_hot']:+.3f} "
                          f"perseed={d['rho_perseed_mean']:+.3f}")
            else:
                print(f"\n  [{fam} / {tag}] no frozen-basis table "
                      f"(pre-registered cadence: "
                      f"{'/'.join(FROZEN_CADENCE)} checkpoints only)")

    if args.budget:
        print("\n  BUDGET TABLE (pre-registered unit: ~240 s per "
              "20-state dual-basis table, t6_run.log; measured walls "
              "from the built tables):")
        for fam in ("conflict", "holdfire"):
            for pat, what in ((f"gt_noop_{fam}.json", "NOOP-basis GT"),
                              (f"nulls_{fam}.json", "null bands")):
                path = os.path.join(OUT_DIR, pat)
                if os.path.exists(path):
                    with open(path) as f:
                        d = json.load(f)
                    w = d.get("build_stats", {}).get("wall_s",
                                                     d.get("wall_s"))
                    nb = d.get("build_stats", {}).get("n_branches", "-")
                    print(f"    {fam:<9} {what:<16} wall={w}s "
                          f"branches={nb}")
            for path in sorted(glob.glob(
                    os.path.join(OUT_DIR, f"gt_frozen_{fam}_*.json"))):
                with open(path) as f:
                    d = json.load(f)
                print(f"    {fam:<9} frozen[{d['ckpt_tag']}] "
                      f"wall={d['wall_s']}s branches={d['n_branches']} "
                      f"agreement={d['basis_agreement_mean']:+.3f}")


if __name__ == "__main__":
    main()
