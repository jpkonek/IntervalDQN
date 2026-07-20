"""T8 overnight audit measurement (20 July 2026): M4 round-1 forensics.

Evidence table backing the FULL-MISS re-score of M4 round-1 (600 ep,
summary m4_summary_seed134_20260719_230841.json, jsonl
m4_seed134_20260719_210020.jsonl — started 21:00, wall 7699 s -> 23:08).

DEVIATION (flagged): the task asked for summary/jsonl+ckpt, but M4
round-1 saved NO .pt checkpoints — run_training_arm only checkpoints
when rho_fn is given and micro_battery_m4.train_arm passes rho_fn=None.
Nothing net-side can be recomputed; everything below is extracted from
the run's own recorded outputs.

Extracted:
  - greedy eval episode's full command list (step, candidate index,
    aircraft, decoded instruction), commands-per-aircraft counts,
    ordinal + time of the first through-flight (AIR-02) command;
  - per-probe argmax targets (20 frozen states) and the
    NOOP/acceptable/through split;
  - training-episode first-command target distribution from the jsonl
    (candidate index decoded via the constant 3-starter sorted callsign
    list [AIR-00, AIR-01, AIR-02]; spawn rate 0, so the mapping holds
    while all three are in obs — early steps, where first commands land)
    -> the empirical Bernoulli rate behind the b1 "coin flip" claim.
Writes m4_forensics.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bluebird_controller_dqn import INSTR_NAMES, N_INSTR

OUT = os.path.dirname(os.path.abspath(__file__))
SUMMARY = "checkpoints/micro_battery/m4/m4_summary_seed134_20260719_230841.json"
JSONL = "checkpoints/micro_battery/m4/m4_seed134_20260719_210020.jsonl"
CS3 = ["AIR-00", "AIR-01", "AIR-02"]


def decode(k):
    i, j = divmod(k - 1, N_INSTR)
    return i, INSTR_NAMES[j]


def main():
    s = json.load(open(SUMMARY))
    acceptable = set(s["oracle"]["acceptable"])
    through = set(s["oracle"]["through"])

    # ---- greedy eval command list -------------------------------------
    issues = s["greedy_eval"]["issues"]  # [step, cand_idx, callsign]
    cmd_rows = []
    per_ac = {}
    first_through = None
    for ordinal, (step, k, cs) in enumerate(issues, 1):
        _i, instr = decode(k)
        per_ac[cs] = per_ac.get(cs, 0) + 1
        cmd_rows.append({"ordinal": ordinal, "step": step, "sim_s": step * 6,
                         "cand_idx": k, "aircraft": cs, "instr": instr,
                         "class": ("acceptable" if cs in acceptable
                                   else "through" if cs in through
                                   else "other")})
        if first_through is None and cs in through:
            first_through = {"ordinal": ordinal, "step": step,
                             "sim_s": step * 6, "instr": instr}
    n_through_cmds = sum(1 for r in cmd_rows if r["class"] == "through")

    # ---- probe argmax table -------------------------------------------
    probe_rows = []
    for step, k, target, cat in s["probe_detail"]:
        instr = None if k == 0 else decode(k)[1]
        probe_rows.append({"probe_step": step, "argmax_cand": k,
                           "target": target, "instr": instr, "class": cat})

    # ---- training first-command distribution from jsonl ----------------
    first_targets = {"AIR-00": 0, "AIR-01": 0, "AIR-02": 0, "none": 0}
    first_targets_last200 = dict(first_targets)
    eps = []
    with open(JSONL) as f:
        for line in f:
            rec = json.loads(line)
            eps.append(rec)
    for rec in eps:
        iss = rec.get("issues") or []
        if not iss:
            first_targets["none"] += 1
            tgt = "none"
        else:
            _step, k = iss[0]
            i, _instr = decode(k)
            tgt = CS3[i] if i < len(CS3) else f"idx{i}"
            first_targets[tgt] = first_targets.get(tgt, 0) + 1
        if rec["episode"] > len(eps) - 200:
            first_targets_last200[tgt] = \
                first_targets_last200.get(tgt, 0) + 1
    n_ep = len(eps)
    acc_first = sum(first_targets.get(cs, 0) for cs in acceptable)
    acc_rate = acc_first / max(1, n_ep)
    l200_total = sum(first_targets_last200.values())
    acc_rate_l200 = sum(first_targets_last200.get(cs, 0)
                        for cs in acceptable) / max(1, l200_total)

    res = {
        "summary_file": SUMMARY, "jsonl_file": JSONL,
        "deviation": ("no M4 .pt checkpoint exists (train_arm passes "
                      "rho_fn=None; run_training_arm only saves "
                      "checkpoints on rho evaluations) — evidence is "
                      "extracted from recorded outputs, nothing "
                      "net-side recomputed"),
        "oracle": {"acceptable": sorted(acceptable),
                   "through": sorted(through),
                   "through_gap": s["oracle"]["through_gap"],
                   "noop_G": s["oracle"]["noop_G"],
                   "best": s["oracle"]["best"]},
        "greedy_eval": {"G": s["greedy_eval"]["G"],
                        "commands": s["greedy_eval"]["commands"],
                        "violated": s["greedy_eval"]["violated"],
                        "first_command": cmd_rows[0] if cmd_rows else None,
                        "first_through_command": first_through,
                        "commands_per_aircraft": per_ac,
                        "n_through_commands": n_through_cmds,
                        "command_list": cmd_rows},
        "probe": {"split": s["probe_split"], "rows": probe_rows,
                  "n_through_argmax": s["probe_split"]["through"],
                  "n_states": len(probe_rows)},
        "bars_as_scored": s["bars"],
        "training_first_command": {
            "n_episodes": n_ep,
            "first_target_counts": first_targets,
            "acceptable_first_rate_all": round(acc_rate, 4),
            "first_target_counts_last200": first_targets_last200,
            "acceptable_first_rate_last200": round(acc_rate_l200, 4),
            "note": ("b1 as written is ONE draw from roughly this "
                     "Bernoulli rate — the chance-level context for "
                     "the green first-command bar")}}
    with open(os.path.join(OUT, "m4_forensics.json"), "w") as f:
        json.dump(res, f, indent=1)

    print(f"[t8] greedy: {len(cmd_rows)} commands, per-aircraft {per_ac}")
    print(f"[t8] first command: {cmd_rows[0] if cmd_rows else None}")
    print(f"[t8] first through-flight command: {first_through}")
    print(f"[t8] probe split: {s['probe_split']}")
    print(f"[t8] training first-cmd targets over {n_ep} eps: "
          f"{first_targets} -> acceptable-first rate {acc_rate:.3f} "
          f"(last200: {acc_rate_l200:.3f})")
    print("[t8] wrote m4_forensics.json")


if __name__ == "__main__":
    main()
