"""
M4 give-way (run-13 micro battery; RUN13_FIX_C_DESIGN.md S3, A15)
=================================================================

Scenario: THREE aircraft — two on a crossing course (NOOP ->
loss_of_separation between them) plus one THROUGH-FLIGHT that transits
uninvolved. The learner must aim its first intervention at the crossing,
never at the through-flight.

ACCEPTABLE-SET ORACLE (A15 — replaces the ill-posed scripted-oracle
match): grid over (aircraft, instruction, timing); each cell is ONE
scripted single-clearance episode through bcd.run_episode (CBP-priced,
deterministic env — deepcopy-free exact rollout via run_scripted). An
aircraft is ACCEPTABLE when its best cell's UNDISCOUNTED episode return
is within eps = 1.0 of the global best (undiscounted: A15's "leaves the
crossing to ~-50 LoS" arithmetic is in raw return units; a discounted
basis would shrink late LoS below the gap threshold by construction).
Cells whose command was never issued (target not yet in obs at the
timing step) are excluded — they are NOOP episodes in disguise.

STRICT-SUBSET GATE (pre-registered): the through-flight's best-cell gap
from the global best must be > 5 (a lone through-flight instruction
leaves the crossing to its -50), else RE-SEED — the scenario cannot
discriminate.

PRE-REGISTERED BARS — ROUND3 D3 REDESIGN (20 July 2026; supersedes the
round-1 B1, which the results audit re-scored as a single Bernoulli
trial passed at chance ~0.60-0.67):
  (B1-HARD) ANY through-flight command in the greedy eval episode =
       HARD FAIL (count must be 0) — not just the first command; the
       round-1 policy passed first-command by luck while giving the
       through-flight 10 of 60 commands.
  (B1-FI) FIRST-INTERVENTION test over >= 10 probe episodes on
       INDEPENDENT qualifying scenario seeds: in EVERY episode the
       first issued command targets a crossing member (chance floor
       ~0.667^10 ~ 0.018 vs the old single trial's ~0.65). No command
       in an episode = MISS for that episode (the crossing runs to
       LoS). DOCUMENTED OPERATIONALIZATION: on non-primary seeds
       "acceptable" is CROSSING MEMBERSHIP from the finder (the full
       A15 oracle costs ~10 min/seed; crossing-vs-through is the 2/3
       chance arithmetic the review's floor states); the primary seed
       is additionally scored against its oracle acceptable set.
  (B2) commands/ep <= 2 (greedy eval episode) — UNCHANGED (D3:
       command-discipline bar unchanged);
  (B3) per-state greedy argmax on FROZEN probe states (post-warmup
       epsilon is exactly 0, so episode-level fractions degenerate —
       A15.3): number of probe states whose argmax targets the
       through-flight == 0 (hard); the NOOP / acceptable / through
       split is reported. UNCHANGED (C-instruments locked read).

Usage:
    .venv/bin/python micro_battery_m4.py --gates            # finder+oracle
    .venv/bin/python micro_battery_m4.py --gates --train --episodes 40
"""

import argparse
import copy
import os
import time

import numpy as np

import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import build_tokens, N_INSTR, INSTR_NAMES
from micro_battery_common import (
    bar, base_argparser, announce, write_summary, make_agent,
    noop_trace_multi, run_training_arm, greedy_eval_episode,
    BATTERY_DIR, SEC_PER_STEP,
)
from micro_level_allocation import SingleActionPolicy
from diagnose_controller import (
    make_custom_density_env, NoopPolicy, run_scripted,
    collect_probe_states, noop_action,
)

OUT_DIR = os.path.join(BATTERY_DIR, "m4")

EPS_ACCEPT = 1.0        # A15: within-eps-of-global-best acceptance band
THROUGH_GAP = 5.0       # strict-subset gate on the through-flight


def make_m4_env(duration=600):
    return make_custom_density_env(duration=duration,
                                   initial_spawn_rate=0.0,
                                   max_spawn_rate=0.0,
                                   num_starter_aircraft=3)


def find_candidates(env, seeds):
    """Seeds whose 3-starter NOOP episode ends in loss_of_separation
    between a pair, with the third aircraft (through-flight) present."""
    out = []
    for seed in seeds:
        rows, violated, kind, vstep, inv = noop_trace_multi(env, seed)
        tag = (f"VIOL[{kind}]@{vstep} inv={inv}" if violated else "clean")
        all_cs = sorted({cs for r in rows for cs in r["in_sector"]})
        ok = (violated and kind == "loss_of_separation" and len(inv) == 2
              and len(all_cs) >= 3)
        print(f"  seed {seed}: {tag:48s}"
              f"{' <- M4 CANDIDATE' if ok else ''}")
        if ok:
            through = [cs for cs in all_cs if cs not in inv]
            out.append((seed, rows, vstep, sorted(inv), through))
    return out


def acceptable_set_oracle(env, seed, vstep, all_cs, args):
    """A15 oracle: GT grid over (aircraft, instruction, timing); each
    cell one scripted single-clearance episode (system-priced, cbp).
    Returns (table, best, global_best, noop_G)."""
    timings = list(range(0, vstep + 1, args.timing_stride))
    g_noop = run_scripted(env, NoopPolicy(), seed, cbp=True)
    noop_G = g_noop["ep_return"]
    print(f"  oracle grid: {len(all_cs)} aircraft x {N_INSTR} instr x "
          f"{len(timings)} timings (stride {args.timing_stride}); "
          f"NOOP G={noop_G:.2f}")
    t0 = time.time()
    table = {}          # (cs, j, s) -> G  (issued cells only)
    best = {cs: (-np.inf, None) for cs in all_cs}
    for cs in all_cs:
        for j in range(1, N_INSTR + 1):
            for s in timings:
                st = run_scripted(env, SingleActionPolicy(cs, j, s), seed,
                                  cbp=True)
                if st["commands"] == 0:
                    continue        # never issued (not in obs yet)
                G = st["ep_return"]
                table[(cs, j, s)] = G
                if G > best[cs][0]:
                    best[cs] = (G, (INSTR_NAMES[j - 1], s,
                                    not st["violated"]))
    global_best = max(v[0] for v in best.values())
    for cs in all_cs:
        G, how = best[cs]
        print(f"    {cs}: best G={G:+8.2f} (gap {global_best - G:6.2f}) "
              f"via {how}")
    print(f"  oracle built: {len(table)} issued cells in "
          f"{time.time() - t0:.0f}s")
    return table, best, global_best, noop_G


def run_gates(env, seed, rows, vstep, crossing, through, args):
    all_cs = sorted(set(crossing) | set(through))
    print(f"\n  crossing pair {crossing}, through-flight {through}, NOOP "
          f"LoS at step {vstep} ({(vstep + 1) * SEC_PER_STEP} s)")
    table, best, global_best, noop_G = acceptable_set_oracle(
        env, seed, vstep, all_cs, args)
    acceptable = sorted(cs for cs in all_cs
                        if global_best - best[cs][0] <= EPS_ACCEPT)
    through_gap = min(global_best - best[cs][0] for cs in through)
    print(f"  ACCEPTABLE SET (eps={EPS_ACCEPT}): {acceptable}")
    gate = bar("M4-gate through-flight best gap", float(through_gap), ">",
               THROUGH_GAP, note="strict subset; else re-seed")
    subset_ok = all(cs not in acceptable for cs in through) and \
        len(acceptable) >= 1 and set(acceptable) <= set(crossing)
    print(f"  strict-subset check: acceptable ⊆ crossing and through "
          f"excluded -> {'PASS' if subset_ok else 'FAIL'}")
    return {"crossing": crossing, "through": through, "vstep": vstep,
            "acceptable": acceptable, "through_gap": float(through_gap),
            "global_best": float(global_best), "noop_G": float(noop_G),
            "best": {cs: {"G": float(best[cs][0]), "how": best[cs][1]}
                     for cs in all_cs},
            "gate_pass": bool(gate and subset_ok)}


def collect_m4_probe(env, seed, n_states):
    """Frozen probe states along the NOOP trajectory with all three
    aircraft in the sector (tokens + callsign list; no per-candidate GT —
    M4's bars are set-membership, not rank)."""
    states = collect_probe_states(
        env, seed, lambda e, o, i: (noop_action(o), False),
        n_states, min_aircraft=3)
    probe = []
    for env_s, obs_s, stratum, step in states:
        cs_list = sorted(obs_s.keys())
        toks = build_tokens(env_s, obs_s, None, cs_list)
        probe.append({"step": step, "tokens": toks, "cs_list": cs_list,
                      "n_cand": 1 + N_INSTR * len(cs_list),
                      "stratum": stratum})
    print(f"  probe set: {len(probe)} frozen states (steps "
          f"{[p['step'] for p in probe]})")
    return probe


def probe_target_split(agent, probe, c, acceptable, through):
    """Per-state greedy argmax classification: NOOP / acceptable /
    through-flight / other-aircraft."""
    split = {"noop": 0, "acceptable": 0, "through": 0, "other": 0}
    detail = []
    for p in probe:
        cl, cu = agent.candidate_q(p["tokens"])
        scores = (cl + c * (cu - cl))[:p["n_cand"]]
        best = scores.max()
        k = 0 if scores[0] == best else int(scores.argmax())
        if k == 0:
            cat, target = "noop", None
        else:
            target = p["cs_list"][(k - 1) // agent.n_instr]
            cat = ("acceptable" if target in acceptable
                   else "through" if target in through else "other")
        split[cat] += 1
        detail.append((p["step"], k, target, cat))
    return split, detail


def collect_fi_seeds(env, primary_seed, args):
    """ROUND3 D3: qualifying probe-episode seeds for the
    first-intervention test — the primary seed plus independently
    qualifying 3-starter LoS+through seeds from the scan list, until
    >= args.fi_episodes. Each entry: (seed, crossing, through)."""
    out = []
    seen = set()
    pool = [s for s in [primary_seed] + list(args.fi_scan_seeds)
            if not (s in seen or seen.add(s))]   # dedupe, order-stable
    for s in pool:
        rows, violated, kind, vstep, inv = noop_trace_multi(env, s)
        all_cs = sorted({cs for r in rows for cs in r["in_sector"]})
        if (violated and kind == "loss_of_separation" and len(inv) == 2
                and len(all_cs) >= 3):
            out.append((s, sorted(inv),
                        [cs for cs in all_cs if cs not in inv]))
        if len(out) >= args.fi_episodes:
            break
    if len(out) < args.fi_episodes:
        print(f"  [FLAG] only {len(out)}/{args.fi_episodes} qualifying "
              f"first-intervention seeds — widen --fi_scan_seeds "
              f"(pre-registered minimum is 10)")
    return out


def first_intervention_test(env, agent, fi_seeds, args):
    """Greedy eval on every qualifying seed; per episode, does the FIRST
    issued command target a crossing member? Also counts through-flight
    commands anywhere in each episode (feeds the composite report)."""
    rows = []
    for s, crossing, through in fi_seeds:
        g = greedy_eval_episode(env, agent, s, cbp=args.cbp)
        first = g["issue_list"][0] if g["issue_list"] else None
        n_through = sum(1 for (_st, _k, tgt) in g["issue_list"]
                        if tgt in through)
        hit = first is not None and first[2] in crossing
        rows.append({"seed": s, "first": first, "hit": bool(hit),
                     "n_through_cmds": n_through,
                     "commands": g["commands"], "G": g["ep_return"],
                     "violated": g["violated"]})
        print(f"    fi seed {s}: first="
              f"{first if first else 'NONE'} -> "
              f"{'HIT' if hit else 'MISS'} "
              f"(through cmds {n_through}, total {g['commands']})")
    return rows


def train_arm(env, seed, gates_out, args):
    agent = make_agent(env, seed, args)
    probe = collect_m4_probe(env, seed, args.probe_states)
    out = run_training_arm(env, seed, agent, args, label="m4",
                           rho_fn=None, out_dir=OUT_DIR)

    acceptable, through = gates_out["acceptable"], gates_out["through"]
    print("\n" + "=" * 74)
    print("M4 PRE-REGISTERED VERDICT BLOCK (give-way, ROUND3 D3 bars)")
    print(f"  [{bcd.conflict_pricing_str()}] episodes={args.episodes} "
          f"(SMOKE run if < 300)")
    g_eval = greedy_eval_episode(env, agent, seed, cbp=args.cbp)
    issues = g_eval["issue_list"]
    first = issues[0] if issues else None
    print(f"  greedy eval (primary seed): G={g_eval['ep_return']:.2f} "
          f"cmd={g_eval['commands']} "
          f"{'clean' if not g_eval['violated'] else 'VIOLATION[' + str(g_eval['violation_kind']) + ']'}"
          f"; first command: "
          f"{first if first else 'NONE (crossing left to run)'}")
    # ---- B1-HARD (D3): ANY through-flight command in the greedy
    # episode is the hard fail — not just the first command
    n_through_cmds = sum(1 for (_s, _k, tgt) in issues if tgt in through)
    b1_hard = bar("M4-1H through-flight commands in greedy eval",
                  n_through_cmds, "==", 0,
                  note="ROUND3 D3 hard-fail: ANY through-flight command "
                       "= FAIL, escape hatch INVALID")
    # ---- B1-FI (D3): first-intervention over >= 10 probe episodes ----
    print(f"  first-intervention test ({args.fi_episodes} probe "
          f"episodes, independent seeds; chance floor "
          f"~{0.667 ** args.fi_episodes:.3f}):")
    fi_seeds = collect_fi_seeds(env, seed, args)
    fi_rows = first_intervention_test(env, agent, fi_seeds, args)
    fi_hits = sum(r["hit"] for r in fi_rows)
    b1_fi = bar("M4-1F first interventions hitting crossing member",
                fi_hits, ">=", len(fi_rows) if fi_rows else 1,
                note=f"{fi_hits}/{len(fi_rows)} episodes; crossing "
                     f"membership basis (finder-level; oracle "
                     f"acceptable-set on the primary seed only — "
                     f"documented operationalization)")
    # primary-seed oracle check (the stricter acceptable-set basis)
    if first is not None:
        print(f"    primary-seed oracle basis: first target {first[2]} "
              f"{'IN' if first[2] in acceptable else 'NOT IN'} "
              f"acceptable set {acceptable}")
    b2 = bar("M4-2 commands/ep (greedy eval)", g_eval["commands"], "<=", 2,
             note="command-discipline bar UNCHANGED (D3)")
    split, detail = probe_target_split(agent, probe, args.c_train,
                                       acceptable, through)
    b3 = bar("M4-3 probe states with through-flight argmax",
             split["through"], "==", 0,
             note=f"split {split} over {len(probe)} states")
    print(f"  per-state argmax detail: {detail}")
    hard_miss = n_through_cmds > 0
    print(f"  M4 composite: "
          f"{'MEET' if (b1_hard and b1_fi and b2 and b3) else 'MISS'} "
          f"(bars {int(b1_hard)}{int(b1_fi)}{int(b2)}{int(b3)}"
          + ("; HARD MISS — through-flight instructed" if hard_miss
             else "") + ")")
    print("=" * 74)

    payload = {"test": "M4", "scenario_seed": seed,
               "episodes": args.episodes, "nstep": args.nstep,
               "cf_replay": args.cf_replay, "oracle": gates_out,
               "greedy_eval": {"G": g_eval["ep_return"],
                               "commands": g_eval["commands"],
                               "violated": g_eval["violated"],
                               "issues": issues,
                               "n_through_cmds": n_through_cmds},
               "first_intervention": {"episodes": fi_rows,
                                      "hits": fi_hits,
                                      "n": len(fi_rows),
                                      "chance_floor":
                                          0.667 ** max(1, len(fi_rows))},
               "probe_split": split, "probe_detail": detail,
               "bars": {"b1_hard_no_through": b1_hard,
                        "b1_first_intervention": b1_fi,
                        "b2_cmd_count": b2,
                        "b3_probe_through": b3, "hard_miss": hard_miss},
               "wall_seconds": out["wall_seconds"]}
    write_summary(OUT_DIR, f"m4_summary_seed{seed}", payload)
    return payload


def main():
    ap = argparse.ArgumentParser(description="M4 give-way micro test")
    base_argparser(ap)
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--scan_seeds", type=int, nargs="+",
                    default=[126, 134, 138, 116, 120, 144, 152, 168, 108],
                    help="shortlist from the 19 Jul 3-starter scan "
                         "(NOOP LoS seeds); the finder re-verifies each")
    ap.add_argument("--timing_stride", type=int, default=4)
    ap.add_argument("--fi_episodes", type=int, default=10,
                    help="ROUND3 D3: probe episodes for the first-"
                         "intervention test (pre-registered >= 10)")
    ap.add_argument("--fi_scan_seeds", type=int, nargs="+",
                    default=[126, 134, 138, 116, 120, 144, 152, 168,
                             108] + list(range(100, 260)),
                    help="seed pool for first-intervention probe "
                         "episodes (finder re-verifies each)")
    args = ap.parse_args()
    announce("M4 GIVE-WAY (3 starters: crossing pair + through-flight, "
             "spawn 0.0)", args)
    env = make_m4_env(duration=args.duration)

    seeds = [args.scenario_seed] if args.scenario_seed is not None \
        else args.scan_seeds
    print("scanning 3-starter seeds for LoS pair + through-flight ...")
    cands = find_candidates(env, seeds)
    assert cands, "no 3-aircraft LoS candidate in the scan range"

    gates_out, seed = None, None
    for cand_seed, rows, vstep, crossing, through in cands:
        print(f"\nrunning acceptable-set oracle on seed {cand_seed} ...")
        g = run_gates(env, cand_seed, rows, vstep, crossing, through, args)
        if g["gate_pass"]:
            gates_out, seed = g, cand_seed
            break
        print(f"  seed {cand_seed} fails the strict-subset gate — "
              f"re-seeding (A15)")
    assert gates_out is not None, \
        "no candidate seed passed the strict-subset gate — report, do " \
        "not relax"
    print(f"\nM4 scenario seed: {seed} (crossing {gates_out['crossing']}, "
          f"through {gates_out['through']}, acceptable "
          f"{gates_out['acceptable']})")

    if args.train:
        train_arm(env, seed, gates_out, args)
    else:
        write_summary(OUT_DIR, f"m4_gates_seed{seed}",
                      {"test": "M4", "scenario_seed": seed,
                       "oracle": gates_out})


if __name__ == "__main__":
    main()
