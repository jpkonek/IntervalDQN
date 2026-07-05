#!/usr/bin/env python
"""
Sequential, fault-tolerant stress-test driver for the BluebirdATC
interval DQN + interval MCTS.

Stages (each wrapped in try/except; a failure records the traceback in the
report and the suite CONTINUES; results are flushed after every stage):

  0  WAIT              until no "--episodes 6000" training process remains
  1  RUN-6 CURVE       250-ep bin table from the newest COMPLETE training
                       JSONL (+ top-3 50-ep fine bins by mean TTV)
  2  CKPT SELECTION    eval candidate checkpoints at c=0, copy winner to
                       checkpoints/bluebird/best_run6.pt
  3  C-SWEEP           winner at c in {0.0, 0.2, 0.5, 1.0}
  4  NOOP BASELINE     all-NOOP episodes, same env, duration 600
  5  ACCEPTANCE        diagnose_interval_dqn.check_6 realized coverage
  6  DQN OOD STRESS    duration 1200 (denser traffic), c in {0.0, 1.0} + NOOP
  7  MCTS STRESS       subprocess model-free vs hybrid over seeds
  8  MCTS AUTOPSY      subprocess diagnose_interval_mcts.py --heavy
  9  DQN COMPONENTS    check_1 (interval validity) + check_3 (Hurwicz
                       disagreement) on the winner, in-process
 10  FINAL REPORT      executive summary / verdicts

Outputs:  checkpoints/bluebird/stress/results.json
          checkpoints/bluebird/stress/STRESS_REPORT.md
          checkpoints/bluebird/stress/*.log (subprocess stdout captures)

Operational constraints honoured:
  - torch.set_num_threads(2); intended to run under `nice -n 15`
  - NEVER writes to checkpoints/bluebird/train* paths
  - stage 0 waits for active `--episodes 6000` training runs (skipped
    with --quick)

Usage:
  nice -n 15 .venv/bin/python run_stress_suite.py            # full suite
  nice -n 15 .venv/bin/python run_stress_suite.py --quick    # dry run
  ... --skip-mcts            # skip stages 7 and 8
  ... --only 3 4             # run only the listed stages
"""

import argparse
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

PY = os.path.join(REPO, ".venv", "bin", "python")
CKPT_DIR = os.path.join(REPO, "checkpoints", "bluebird")
STRESS_DIR = os.path.join(CKPT_DIR, "stress")
BEST_PATH = os.path.join(CKPT_DIR, "best_run6.pt")
RESULTS_PATH = os.path.join(STRESS_DIR, "results.json")
REPORT_PATH = os.path.join(STRESS_DIR, "STRESS_REPORT.md")
MCTS_SCRIPT = os.path.join(REPO, "bluebird_interval_mcts.py")
DIAG_DQN_SCRIPT = os.path.join(REPO, "diagnose_interval_dqn.py")
DIAG_MCTS_SCRIPT = os.path.join(REPO, "diagnose_interval_mcts.py")

# Recorded reference values for the calibration trajectory (stage 5 / 10).
# Provenance: diagnose_interval_dqn.py check_6 (realized discounted
# return-to-go coverage of [Q_l, Q_u]) as recorded in the session log.
REF_REALIZED_COVERAGE = [
    {"run": "run 2", "value": 0.000,
     "provenance": "check_6 on run-2 snapshot best_seed42_ep2250 "
                   "(tag 20260704_124534; income-dominated reward regime, "
                   "gamma 0.99 — Q ~ +40 vs realized ~ -6)"},
    {"run": "run 4", "value": 0.125,
     "provenance": "check_6 on run-4 winner (tag 20260704_153244; "
                   "outcome-anchored regime, gamma 0.97)"},
]

STAGE_TITLES = {
    0: "Wait for active training run",
    1: "Run-6 training curve",
    2: "Checkpoint selection",
    3: "C-sweep on winner",
    4: "NOOP baseline (600 s)",
    5: "Acceptance: realized-coverage calibration (check_6)",
    6: "DQN OOD stress (1200 s)",
    7: "MCTS stress (model-free vs hybrid)",
    8: "MCTS heavy autopsy (seed 10043)",
    9: "DQN component diagnostics (checks 1 & 3)",
    10: "Final report",
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def to_jsonable(x):
    """Recursively coerce numpy scalars/arrays etc. into JSON-safe types."""
    import numpy as np
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and (x != x):  # NaN
        return None
    return x


def fmt(x, spec=".3f", none="-"):
    if x is None or (isinstance(x, float) and x != x):
        return none
    try:
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def discover_runtag():
    """Newest COMPLETE training run: <tag>_final.pt AND its JSONL exist."""
    tags = []
    for p in glob.glob(os.path.join(CKPT_DIR, "train_seed42_*_final.pt")):
        m = re.match(r"train_seed42_(\d{8}_\d{6})_final\.pt",
                     os.path.basename(p))
        if m and os.path.exists(os.path.join(
                CKPT_DIR, f"train_seed42_{m.group(1)}.jsonl")):
            tags.append(m.group(1))
    if not tags:
        raise RuntimeError(
            "no complete training run found "
            "(need train_seed42_<TAG>_final.pt + matching .jsonl)")
    return max(tags)  # %Y%m%d_%H%M%S sorts lexicographically


def run_tag_paths(tag):
    jsonl = os.path.join(CKPT_DIR, f"train_seed42_{tag}.jsonl")
    final = os.path.join(CKPT_DIR, f"train_seed42_{tag}_final.pt")
    eps = {}
    for p in glob.glob(os.path.join(CKPT_DIR, f"train_seed42_{tag}_ep*.pt")):
        m = re.search(r"_ep(\d+)\.pt$", p)
        if m:
            eps[int(m.group(1))] = p
    return jsonl, final, eps


class Driver:
    def __init__(self, args):
        self.args = args
        self.quick = args.quick
        os.makedirs(STRESS_DIR, exist_ok=True)
        self.prev = {}
        if os.path.exists(RESULTS_PATH):
            try:
                with open(RESULTS_PATH) as f:
                    self.prev = json.load(f).get("stages", {})
            except (OSError, json.JSONDecodeError):
                self.prev = {}
        self.results = {"created": now(), "quick": self.quick,
                        "argv": sys.argv[1:], "stages": {}}
        self.sections = {}   # stage no -> markdown text
        if args.only:
            # --only reruns update the previous results/report in place
            # instead of discarding the other stages
            self.results["stages"] = dict(self.prev)
            for no_s, e in self.prev.items():
                if e.get("report_md") is not None:
                    self.sections[int(no_s)] = e["report_md"]
        self._envs = {}      # (duration, k, rp, cc) -> env
        self._bid = None
        self._runtag = None

    # ---------------- shared helpers ----------------

    @property
    def bid(self):
        if self._bid is None:
            import torch
            torch.set_num_threads(2)
            import bluebird_interval_dqn as bid
            self._bid = bid
        return self._bid

    @property
    def runtag(self):
        if self._runtag is None:
            self._runtag = discover_runtag()
        return self._runtag

    def env_for(self, ckpt, duration):
        key = (duration, ckpt.get("k", 2), ckpt.get("route_parallel", False),
               ckpt.get("centreline_coeff", 1.0),
               ckpt.get("encoder_cls", "extra_minimal"))
        if key not in self._envs:
            print(f"  creating env (duration={key[0]}s, k={key[1]}, "
                  f"route_parallel={key[2]}, centreline_coeff={key[3]}, "
                  f"encoder={key[4]})")
            self._envs[key] = self.bid.make_env(
                scenario_duration=key[0], k_nearest=key[1],
                route_parallel=key[2], centreline_coeff=key[3],
                encoder_cls=key[4])
        return self._envs[key]

    def close_envs(self):
        for env in self._envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._envs.clear()

    def stage_data(self, no):
        """Data from a stage run this session, else from a prior results.json
        (supports --only re-runs). None if unavailable."""
        e = self.results["stages"].get(str(no)) or self.prev.get(str(no))
        if e and e.get("status") == "ok":
            return e.get("data")
        return None

    def winner_path(self):
        d = self.stage_data(2)
        if d and d.get("winner_path") and os.path.exists(d["winner_path"]):
            return d["winner_path"]
        if os.path.exists(BEST_PATH):
            return BEST_PATH
        raise RuntimeError("no winner checkpoint available "
                           "(stage 2 did not run and best_run6.pt missing)")

    def load_winner(self):
        path = self.winner_path()
        agent, ckpt = self.bid.load_agent(path, device="cpu")
        print(f"  winner: {path} (ep {ckpt.get('episode', '?')}, "
              f"{ckpt['n_actions']} actions, t={ckpt.get('t', 0.5):.3f})")
        return path, agent, ckpt

    # ---------------- flushing ----------------

    def flush(self):
        tmp = RESULTS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(to_jsonable(self.results), f, indent=2)
        os.replace(tmp, RESULTS_PATH)

        lines = ["# BluebirdATC Interval DQN + MCTS — Stress-Test Report",
                 "",
                 f"- generated by `run_stress_suite.py` "
                 f"({'QUICK dry run' if self.quick else 'full suite'})",
                 f"- started {self.results['created']}, "
                 f"last flush {now()}",
                 f"- python: `{PY}`",
                 ""]
        for no in sorted(self.sections):
            lines.append(f"## Stage {no}: {STAGE_TITLES.get(no, '')}")
            lines.append("")
            lines.append(self.sections[no])
            lines.append("")
        tmp = REPORT_PATH + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(lines))
        os.replace(tmp, REPORT_PATH)

    def run_stage(self, no, fn, skip_reason=None):
        title = STAGE_TITLES[no]
        print("\n" + "=" * 88)
        print(f"STAGE {no}: {title}  [{now()}]")
        print("=" * 88, flush=True)
        entry = {"stage": no, "title": title, "started": now()}
        t0 = time.time()
        if skip_reason:
            if (self.args.only and no not in self.args.only
                    and str(no) in self.results["stages"]):
                # --only rerun: leave the carried-over prior result alone
                print(f"  SKIPPED: {skip_reason} (prior result kept)")
                return
            print(f"  SKIPPED: {skip_reason}")
            entry.update(status="skipped", reason=skip_reason)
            self.sections[no] = f"_Skipped: {skip_reason}._"
        else:
            try:
                data, md = fn()
                entry.update(status="ok", data=to_jsonable(data))
                self.sections[no] = md
            except Exception:
                tb = traceback.format_exc()
                print(tb, file=sys.stderr, flush=True)
                entry.update(status="failed", traceback=tb)
                self.sections[no] = ("**STAGE FAILED** — suite continued.\n\n"
                                     f"```\n{tb}```")
        entry["elapsed_s"] = round(time.time() - t0, 1)
        entry["report_md"] = self.sections.get(no)
        self.results["stages"][str(no)] = entry
        self.flush()
        print(f"  stage {no} {entry['status']} "
              f"({entry['elapsed_s']:.1f} s)", flush=True)

    # ---------------- stage 0 ----------------

    def stage0(self):
        pat = "episodes 6000"
        waited = 0
        while True:
            r = subprocess.run(["pgrep", "-f", pat],
                               capture_output=True, text=True)
            pids = [p for p in r.stdout.split() if p.strip()]
            if not pids:
                break
            print(f"  [{now()}] {len(pids)} process(es) matching "
                  f"'{pat}' still running (pids {', '.join(pids)}); "
                  f"sleeping 60 s ...", flush=True)
            time.sleep(60)
            waited += 60
        md = (f"No process matching `{pat}` at {now()}"
              + (f" (waited {waited // 60} min)." if waited else
                 " (no wait needed)."))
        print("  " + md)
        return {"waited_s": waited}, md

    # ---------------- stage 1 ----------------

    def stage1(self):
        import numpy as np
        tag = self.runtag
        jsonl, final, eps = run_tag_paths(tag)
        rows = []
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # torn tail line
        if not rows:
            raise RuntimeError(f"no rows parsed from {jsonl}")
        n_ep = max(r.get("episode", 0) for r in rows)
        has_boot = any("bootstrap_coverage" in r for r in rows)

        def summarize(sub):
            def m(key):
                vals = [r[key] for r in sub
                        if key in r and r[key] is not None]
                return float(np.mean(vals)) if vals else None
            return {
                "n": len(sub),
                "mean_ttv": m("time_to_violation"),
                "clean": sum(1 for r in sub if not r.get("violated", True)),
                "mean_width": m("mean_width"),
                "coverage": m("coverage"),
                "bootstrap_coverage": m("bootstrap_coverage"),
                "t": m("t"),
                "mean_loss": m("mean_loss"),
            }

        def bins(size):
            out = {}
            for r in rows:
                b = (r.get("episode", 1) - 1) // size
                out.setdefault(b, []).append(r)
            return [(b * size + 1, (b + 1) * size, summarize(sub))
                    for b, sub in sorted(out.items())]

        coarse = bins(250)
        fine = [x for x in bins(50) if x[2]["n"] >= 25]  # drop torn tail bin
        fine.sort(key=lambda x: (-(x[2]["mean_ttv"] or 0.0), -x[2]["clean"]))
        top_fine = [{"start": lo, "end": hi, "center": (lo + hi) // 2,
                     "mean_ttv": s["mean_ttv"], "clean": s["clean"],
                     "n": s["n"]}
                    for lo, hi, s in fine[:3]]

        hdr = ("| episodes | n | mean TTV (s) | clean | mean width | "
               "coverage | bootstrap cov | t | loss |\n"
               "|---|---|---|---|---|---|---|---|---|")
        tbl = [hdr]
        for lo, hi, s in coarse:
            tbl.append(
                f"| {lo}-{hi} | {s['n']} | {fmt(s['mean_ttv'], '.1f')} | "
                f"{s['clean']}/{s['n']} | {fmt(s['mean_width'])} | "
                f"{fmt(s['coverage'])} | {fmt(s['bootstrap_coverage'])} | "
                f"{fmt(s['t'])} | {fmt(s['mean_loss'])} |")
        md = [f"Newest complete run tag: `{tag}` "
              f"({n_ep} episodes; JSONL `{os.path.basename(jsonl)}`; "
              f"{len(eps)} periodic checkpoints + final)."]
        if not has_boot:
            md.append("\n_Note: this JSONL has no `bootstrap_coverage` key "
                      "(pre-run-6 format); column shown as `-`. Its "
                      "`coverage` key is the tracker/realized coverage as "
                      "logged by that run._")
        md.append("\n### 250-episode bins\n\n" + "\n".join(tbl))
        md.append("\n### Top-3 50-episode fine bins (by mean TTV, then "
                  "clean count)\n")
        for i, b in enumerate(top_fine, 1):
            md.append(f"{i}. episodes {b['start']}-{b['end']}: mean TTV "
                      f"{fmt(b['mean_ttv'], '.1f')} s, clean "
                      f"{b['clean']}/{b['n']}")
        data = {"tag": tag, "jsonl": jsonl, "n_episodes": n_ep,
                "has_bootstrap_coverage": has_boot,
                "bins_250": [{"start": lo, "end": hi, **s}
                             for lo, hi, s in coarse],
                "top_fine_bins": top_fine}
        return data, "\n".join(md)

    # ---------------- stage 2 ----------------

    def stage2(self):
        bid = self.bid
        s1 = self.stage_data(1)
        tag = (s1 or {}).get("tag") or self.runtag
        _, final, eps = run_tag_paths(tag)
        avail = sorted(eps)

        cands = []  # (label, path)
        if s1 and avail:
            for b in s1["top_fine_bins"]:
                nearest = min(avail, key=lambda e: abs(e - b["center"]))
                cands.append((f"ep{nearest} (bin {b['start']}-{b['end']})",
                              eps[nearest]))
        if os.path.exists(final):
            cands.append(("final", final))
        # dedupe by path, preserve order
        seen, uniq = set(), []
        for label, path in cands:
            if path not in seen:
                seen.add(path)
                uniq.append((label, path))
        if not uniq:
            raise RuntimeError(f"no candidate checkpoints for tag {tag}")

        n_eps = 2 if self.quick else 10
        results = []
        for label, path in uniq:
            agent, ckpt = bid.load_agent(path, device="cpu")
            env = self.env_for(ckpt, duration=600)
            print(f"  evaluating {label}: {os.path.basename(path)} "
                  f"(c=0.0, {n_eps} eps, base_seed 10042)")
            r = bid.evaluate(env, agent, c=0.0, n_episodes=n_eps,
                             base_seed=10042)
            r.update(label=label, path=path,
                     episode=ckpt.get("episode"))
            results.append(r)

        winner = max(results, key=lambda r: (r["mean_time_to_violation"],
                                             -r["violation_rate"]))
        shutil.copy2(winner["path"], BEST_PATH)
        print(f"  WINNER: {winner['label']} -> {BEST_PATH}")

        tbl = ["| candidate | mean TTV (s) | std | violated | kinds | "
               "mean width |",
               "|---|---|---|---|---|---|"]
        for r in results:
            star = " **<- winner**" if r is winner else ""
            tbl.append(
                f"| {r['label']}{star} | "
                f"{fmt(r['mean_time_to_violation'], '.1f')} | "
                f"{fmt(r['std_time_to_violation'], '.1f')} | "
                f"{int(r['violation_rate'] * n_eps)}/{n_eps} | "
                f"{r['violation_kinds'] or '-'} | "
                f"{fmt(r['mean_width'])} |")
        md = (f"Candidates from run `{tag}`, evaluated in-process at c=0.0, "
              f"{n_eps} episodes each, base_seed 10042 (same seeds for all "
              f"candidates), duration 600 s.\n\n" + "\n".join(tbl) +
              f"\n\nWinner `{winner['label']}` copied to "
              f"`{os.path.relpath(BEST_PATH, REPO)}`.")
        return {"tag": tag, "n_episodes": n_eps, "candidates": results,
                "winner_label": winner["label"],
                "winner_src": winner["path"], "winner_path": BEST_PATH}, md

    # ---------------- stage 3 ----------------

    def stage3(self):
        bid = self.bid
        path, agent, ckpt = self.load_winner()
        env = self.env_for(ckpt, duration=600)
        cs = [0.0] if self.quick else [0.0, 0.2, 0.5, 1.0]
        n_eps = 2 if self.quick else 15
        results = []
        for c in cs:
            print(f"  c-sweep: c={c}, {n_eps} eps, base_seed 10042")
            r = bid.evaluate(env, agent, c=c, n_episodes=n_eps,
                             base_seed=10042)
            results.append(r)
        tbl = ["| c | mean TTV (s) | std | violated | kinds | mean width |",
               "|---|---|---|---|---|---|"]
        for r in results:
            tbl.append(f"| {r['c']} | "
                       f"{fmt(r['mean_time_to_violation'], '.1f')} | "
                       f"{fmt(r['std_time_to_violation'], '.1f')} | "
                       f"{int(r['violation_rate'] * n_eps)}/{n_eps} | "
                       f"{r['violation_kinds'] or '-'} | "
                       f"{fmt(r['mean_width'])} |")
        md = (f"Winner `{os.path.basename(path)}`, {n_eps} episodes per c, "
              f"base_seed 10042, duration 600 s.\n\n" + "\n".join(tbl))
        return {"winner": path, "n_episodes": n_eps, "sweep": results}, md

    # ---------------- stage 4 (and stage-6 reuse) ----------------

    def _noop_eval(self, env, n_episodes, base_seed):
        """All-NOOP rollouts mirroring evaluate(): episode ends at the first
        violation (detect_violation) or the time limit; TTV censored at the
        scenario duration. NOOP is action index 0 (see bluebird_interval_mcts
        NOOP constant; the DQN env shares the action head layout)."""
        import numpy as np
        bid = self.bid
        ttvs, kinds = [], {}
        n_violated = 0
        for i in range(n_episodes):
            obs, info = env.reset(seed=base_seed + i)
            maxstep = int(getattr(env, "maxstep",
                                  env.config.scenario_duration
                                  // bid.SEC_PER_STEP))
            violated, kind = False, None
            step_i = -1
            for step_i in range(maxstep):
                actions = {cs: 0 for cs in obs}  # NOOP
                obs, rew, done, trunc, info = env.step(actions)
                violated, kind, _involved = bid.detect_violation(info)
                if violated:
                    break
            ttv = ((step_i + 1) * bid.SEC_PER_STEP if violated
                   else maxstep * bid.SEC_PER_STEP)
            ttvs.append(ttv)
            if violated:
                n_violated += 1
                kinds[kind] = kinds.get(kind, 0) + 1
            print(f"    NOOP seed {base_seed + i}: TTV {ttv} s"
                  f"{f' [{kind}]' if violated else ' (clean)'}")
        return {"mean_time_to_violation": float(np.mean(ttvs)),
                "std_time_to_violation": float(np.std(ttvs)),
                "violated": n_violated, "n_episodes": n_episodes,
                "violation_kinds": kinds, "ttvs": ttvs}

    def stage4(self):
        _, _, ckpt = self.load_winner()  # env config only
        env = self.env_for(ckpt, duration=600)
        n_eps = 2 if self.quick else 15
        print(f"  NOOP baseline: {n_eps} eps, seeds 10042..{10042 + n_eps - 1}")
        r = self._noop_eval(env, n_eps, 10042)
        md = (f"All-NOOP baseline, same env as stages 2-3 (duration 600 s), "
              f"seeds 10042..{10042 + n_eps - 1}.\n\n"
              f"- mean TTV: **{fmt(r['mean_time_to_violation'], '.1f')} s** "
              f"± {fmt(r['std_time_to_violation'], '.1f')}\n"
              f"- violated: {r['violated']}/{n_eps} "
              f"({r['violation_kinds'] or 'none'})")
        return r, md

    # ---------------- stage 5 ----------------

    def stage5(self):
        import diagnose_interval_dqn as diag
        path, agent, ckpt = self.load_winner()
        n_eps = 2 if self.quick else 5
        buf = io.StringIO()
        with redirect_stdout(buf):
            cov = diag.check_6(agent, ckpt, n_episodes=n_eps,
                               base_seed=20042, duration=300, c=0.0,
                               record_verdict=False, tag="(stress) ")
        out = buf.getvalue()
        print(out)
        out_path = os.path.join(STRESS_DIR, "check6_acceptance.log")
        with open(out_path, "w") as f:
            f.write(out)

        def grab(pattern, groups=1):
            m = re.search(pattern, out)
            if not m:
                return None if groups == 1 else (None,) * groups
            if groups == 1:
                return float(m.group(1))
            return tuple(float(m.group(i + 1)) for i in range(groups))

        boot = grab(r"Bellman self-consistency\):\s*([0-9.]+|nan)")
        g_mean, ql, qu = grab(
            r"scale: mean realized return-to-go\s+([-0-9.]+)\s+vs\s+"
            r"mean predicted \[Q_l, Q_u\] = \[([-0-9.]+),\s*([-0-9.]+)\]", 3)

        traj = REF_REALIZED_COVERAGE + [
            {"run": "this run", "value": cov,
             "provenance": f"check_6 on {os.path.basename(path)} "
                           f"({n_eps} eps, duration 300, seeds 20042+)"}]
        md = [f"check_6 on the winner, {n_eps} episodes, duration 300 s, "
              f"c=0.0, seeds 20042+ (full output: "
              f"`stress/{os.path.basename(out_path)}`).",
              "",
              f"- realized coverage: **{fmt(cov)}** (target 0.85)",
              f"- bootstrap-completed coverage: {fmt(boot)}",
              f"- scale: mean realized return-to-go {fmt(g_mean)} vs mean "
              f"predicted [{fmt(ql)}, {fmt(qu)}]",
              "",
              "Calibration trajectory (realized coverage):",
              "",
              "| run | realized coverage | provenance |",
              "|---|---|---|"]
        for t in traj:
            md.append(f"| {t['run']} | {fmt(t['value'])} | "
                      f"{t['provenance']} |")
        data = {"winner": path, "n_episodes": n_eps,
                "realized_coverage": cov, "bootstrap_coverage": boot,
                "mean_realized_return": g_mean,
                "mean_predicted_interval": [ql, qu],
                "reference": REF_REALIZED_COVERAGE,
                "output_file": out_path}
        return data, "\n".join(md)

    # ---------------- stage 6 ----------------

    def stage6(self):
        bid = self.bid
        path, agent, ckpt = self.load_winner()
        env = self.env_for(ckpt, duration=1200)
        n_eps = 8
        sweep = []
        for c in (0.0, 1.0):
            print(f"  OOD eval: duration 1200, c={c}, {n_eps} eps, "
                  f"base_seed 30042")
            r = bid.evaluate(env, agent, c=c, n_episodes=n_eps,
                             base_seed=30042)
            sweep.append(r)
        print(f"  OOD NOOP: duration 1200, {n_eps} eps, base_seed 30042")
        noop = self._noop_eval(env, n_eps, 30042)

        tbl = ["| agent | mean TTV (s) | std | violated | kinds |",
               "|---|---|---|---|---|"]
        for r in sweep:
            tbl.append(f"| winner c={r['c']} | "
                       f"{fmt(r['mean_time_to_violation'], '.1f')} | "
                       f"{fmt(r['std_time_to_violation'], '.1f')} | "
                       f"{int(r['violation_rate'] * n_eps)}/{n_eps} | "
                       f"{r['violation_kinds'] or '-'} |")
        tbl.append(f"| NOOP | {fmt(noop['mean_time_to_violation'], '.1f')} | "
                   f"{fmt(noop['std_time_to_violation'], '.1f')} | "
                   f"{noop['violated']}/{n_eps} | "
                   f"{noop['violation_kinds'] or '-'} |")

        edge_1200 = (sweep[0]["mean_time_to_violation"]
                     - sweep[1]["mean_time_to_violation"])
        s3 = self.stage_data(3)
        edge_600 = None
        if s3:
            by_c = {r["c"]: r for r in s3["sweep"]}
            if 0.0 in by_c and 1.0 in by_c:
                edge_600 = (by_c[0.0]["mean_time_to_violation"]
                            - by_c[1.0]["mean_time_to_violation"])
        md = [f"Duration 1200 s (spawn ramp -> denser traffic than the "
              f"600 s training regime), {n_eps} episodes, base_seed 30042.",
              "", "\n".join(tbl), "",
              f"Pessimism edge (TTV[c=0] - TTV[c=1]) at 1200 s: "
              f"**{fmt(edge_1200, '.1f')} s**"
              + (f"; at 600 s (stage 3): {fmt(edge_600, '.1f')} s -> "
                 f"{'GROWS' if edge_1200 > edge_600 else 'does NOT grow'} "
                 f"under stress." if edge_600 is not None else
                 " (600 s edge unavailable: stage 3 lacks c=1.0).")]
        return {"winner": path, "n_episodes": n_eps, "sweep": sweep,
                "noop_1200": noop, "edge_1200": edge_1200,
                "edge_600": edge_600}, "\n".join(md)

    # ---------------- stage 7 ----------------

    def _parse_mcts_stdout(self, out):
        d = {}
        m = re.search(r"time-to-first-violation : ([0-9.]+) s \(step (\d+)\)",
                      out)
        if m:
            d["violated"] = True
            d["ttv_s"] = float(m.group(1))
        else:
            m = re.search(r"clean episode\s*\(censored at ([0-9.]+) s\)", out)
            if m:
                d["violated"] = False
                d["ttv_s"] = float(m.group(1))
        m = re.search(r"first violation\s+:\s*(.+)", out)
        if m:
            d["first_violation_desc"] = m.group(1).strip()
        m = re.search(r"episode return\s+:\s*([-0-9.]+)", out)
        if m:
            d["episode_return"] = float(m.group(1))
        m = re.search(r"decision latency\s+:\s*mean ([0-9.]+) ms, "
                      r"max ([0-9.]+) ms", out)
        if m:
            d["mean_latency_ms"] = float(m.group(1))
            d["max_latency_ms"] = float(m.group(2))
        m = re.search(r"realized H-step coverage:\s*([0-9.]+)\s*"
                      r"\((\d+) scored, (\d+) censored\)", out)
        if m:
            d["realized_h_coverage"] = float(m.group(1))
            d["scored"] = int(m.group(2))
            d["censored"] = int(m.group(3))
        m = re.search(r"Episode wall time: ([0-9.]+) s", out)
        if m:
            d["wall_s"] = float(m.group(1))
        m = re.search(r"MCTS searches run\s+:\s*(\d+)", out)
        if m:
            d["n_searches"] = int(m.group(1))
        m = re.search(r"mean root interval width:\s*([0-9.]+)", out)
        if m:
            d["mean_root_width"] = float(m.group(1))
        return d

    def stage7(self):
        winner = self.winner_path()
        seeds = [10042] if self.quick else [10042, 10043, 10044]
        # quick: tiny budget + wide-open alert gates (like the script's own
        # --smoke mode) so the search machinery is actually exercised —
        # with production gates the well-separated early traffic at 120 s
        # triggers zero searches and the plumbing goes untested
        extra = (["--sims", "8", "--horizon", "8", "--duration", "120",
                  "--alert-radius", "1e9", "--alert-fl", "1e9",
                  "--max-planned", "2"]
                 if self.quick else [])
        timeout = 1200 if self.quick else 3600
        runs = []
        for seed in seeds:
            for variant, args in (
                    ("modelfree", []),
                    ("hybrid", ["--leaf-value", winner])):
                cmd = [PY, MCTS_SCRIPT, "--run", "--seed", str(seed)] \
                    + extra + args
                log_path = os.path.join(
                    STRESS_DIR, f"mcts_{variant}_seed{seed}.log")
                rec = {"seed": seed, "variant": variant,
                       "cmd": " ".join(cmd), "log": log_path}
                print(f"  [{now()}] MCTS {variant} seed {seed} "
                      f"(timeout {timeout} s) ...", flush=True)
                t0 = time.time()
                try:
                    p = subprocess.run(cmd, cwd=REPO, capture_output=True,
                                       text=True, timeout=timeout)
                    out = p.stdout + ("\n[stderr]\n" + p.stderr
                                      if p.stderr.strip() else "")
                    rec["returncode"] = p.returncode
                    rec.update(self._parse_mcts_stdout(p.stdout))
                    rec["status"] = ("ok" if p.returncode == 0
                                     and "ttv_s" in rec else "error")
                except subprocess.TimeoutExpired as e:
                    out = ((e.stdout or "")
                           + "\n[TIMEOUT after %d s — killed]" % timeout)
                    if isinstance(out, bytes):
                        out = out.decode(errors="replace")
                    rec["status"] = "timeout"
                except Exception:
                    out = traceback.format_exc()
                    rec["status"] = "error"
                rec["wall_clock_s"] = round(time.time() - t0, 1)
                with open(log_path, "w") as f:
                    f.write(out)
                print(f"    -> {rec['status']}, TTV "
                      f"{rec.get('ttv_s', '?')} s, wall "
                      f"{rec['wall_clock_s']} s", flush=True)
                runs.append(rec)

        tbl = ["| seed | variant | status | TTV (s) | violated | return | "
               "latency mean/max (ms) | searches | H-cov (scored/cens) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in runs:
            lat = (f"{fmt(r.get('mean_latency_ms'), '.1f')} / "
                   f"{fmt(r.get('max_latency_ms'), '.1f')}")
            hcov = (f"{fmt(r.get('realized_h_coverage'))} "
                    f"({r.get('scored', '-')}/{r.get('censored', '-')})"
                    if r.get("realized_h_coverage") is not None else "-")
            tbl.append(f"| {r['seed']} | {r['variant']} | {r['status']} | "
                       f"{fmt(r.get('ttv_s'), '.0f')} | "
                       f"{'yes' if r.get('violated') else 'no' if 'violated' in r else '-'} | "
                       f"{fmt(r.get('episode_return'), '.2f')} | {lat} | "
                       f"{r.get('n_searches', '-')} | "
                       f"{hcov} |")
        cfg = ("sims 8, horizon 8, duration 120 s, alert gates wide open, "
               "max_planned 2 (quick)" if self.quick
               else "default budget (sims 24, horizon 15, duration 600 s)")
        md = (f"Sequential subprocess runs, {cfg}; hybrid uses "
              f"`--leaf-value {os.path.relpath(winner, REPO)}`. Stdout "
              f"captures in `stress/mcts_*_seed*.log`.\n\n" + "\n".join(tbl))
        return {"runs": runs, "seeds": seeds, "quick": self.quick}, md

    # ---------------- stage 8 ----------------

    def stage8(self):
        cmd = [PY, DIAG_MCTS_SCRIPT, "--heavy"]
        log_path = os.path.join(STRESS_DIR, "mcts_heavy_autopsy.log")
        print(f"  [{now()}] running {' '.join(cmd)} (timeout 5400 s)",
              flush=True)
        status = "ok"
        try:
            p = subprocess.run(cmd, cwd=REPO, capture_output=True,
                               text=True, timeout=5400)
            out = p.stdout + ("\n[stderr]\n" + p.stderr
                              if p.stderr.strip() else "")
            if p.returncode != 0:
                status = f"exit {p.returncode}"
        except subprocess.TimeoutExpired as e:
            out = (e.stdout.decode(errors="replace")
                   if isinstance(e.stdout, bytes) else (e.stdout or ""))
            out += "\n[TIMEOUT after 5400 s — killed]"
            status = "timeout"
        with open(log_path, "w") as f:
            f.write(out)

        verdicts = [l.strip() for l in out.splitlines()
                    if re.search(r"verdict|VERDICT|autops", l)
                    and l.strip()][:20]
        tee_logs = sorted(glob.glob(os.path.join(
            CKPT_DIR, "diagnostics", "diagnose_*.log")))
        tee_log = tee_logs[-1] if tee_logs else None
        md = [f"`diagnose_interval_mcts.py --heavy` (seed-10043 autopsy): "
              f"status **{status}**.",
              f"- captured stdout: `stress/{os.path.basename(log_path)}`"
              + (f"; suite's own tee log: `{os.path.relpath(tee_log, REPO)}`"
                 if tee_log else ""),
              "", "Verdict lines:", ""]
        md += [f"> {v}" for v in verdicts] or ["> (no verdict lines found)"]
        return {"status": status, "log": log_path, "tee_log": tee_log,
                "verdict_lines": verdicts}, "\n".join(md)

    # ---------------- stage 9 ----------------

    def stage9(self):
        import numpy as np
        try:
            import diagnose_interval_dqn as diag
            path, agent, ckpt = self.load_winner()
            dur = 120 if self.quick else 600
            env = self.env_for(ckpt, duration=dur)
            # Own state collection (diag.collect_states hardcodes
            # route_parallel=False, which breaks for 4-action winners):
            # one policy-driven episode, violations do NOT end it.
            print(f"  collecting real states: one {dur} s episode, "
                  f"seed 20777, c={agent.c_train}")
            obs, _ = env.reset(seed=20777)
            maxstep = int(getattr(env, "maxstep",
                                  dur // self.bid.SEC_PER_STEP))
            real = []
            for _ in range(maxstep):
                if not obs:
                    break
                real.extend(np.asarray(v, np.float32)
                            for v in obs.values())
                actions, _w = agent.generate_action(obs, force_epsilon=0.0)
                obs, _r, _d, _t, _i = env.step(actions)
            real = (np.stack(real) if real
                    else np.zeros((0, agent.state_dim), np.float32))
            rng = np.random.default_rng(20777)
            n_rand = 200 if self.quick else 1000
            if agent.state_dim == len(diag.FEATURE_LOW):
                rand = rng.uniform(diag.FEATURE_LOW, diag.FEATURE_HIGH,
                                   size=(n_rand, agent.state_dim)
                                   ).astype(np.float32)
            else:  # fall back to the empirical box
                lo_b, hi_b = real.min(0), real.max(0)
                rand = rng.uniform(lo_b, hi_b,
                                   size=(n_rand, agent.state_dim)
                                   ).astype(np.float32)
            print(f"  {len(real)} real states, {n_rand} random box states")

            buf = io.StringIO()
            with redirect_stdout(buf):
                v1 = diag.check_1({"winner": agent}, real, rand,
                                  record_verdict=False, tag="(stress) ")
                v3 = diag.check_3(agent, real, rand,
                                  record_verdict=False, tag="(stress) ")
            out = buf.getvalue()
            print(out)
            out_path = os.path.join(STRESS_DIR, "dqn_component_checks.log")
            with open(out_path, "w") as f:
                f.write(out)
            m = re.search(r"disagreement c=0 vs c=1: ([0-9.]+)% on real",
                          out)
            d01 = float(m.group(1)) if m else None
            md = (f"In-process `check_1` / `check_3` on the winner "
                  f"(`{os.path.basename(path)}`), with driver-side state "
                  f"collection honouring the checkpoint's route_parallel "
                  f"action space (full output: "
                  f"`stress/{os.path.basename(out_path)}`).\n\n"
                  f"- check 1 (interval validity): **{v1}**\n"
                  f"- check 3 (Hurwicz behavioral relevance): **{v3}**"
                  + (f" — c=0 vs c=1 disagreement {d01}% on real states"
                     if d01 is not None else ""))
            return {"mode": "in-process", "winner": path,
                    "check_1": v1, "check_3": v3,
                    "disagreement_c0_c1_pct": d01,
                    "n_real": int(len(real)), "n_rand": n_rand,
                    "output_file": out_path}, md
        except Exception:
            tb = traceback.format_exc()
            print("  in-process checks failed; falling back to the "
                  "subprocess light suite\n" + tb, flush=True)
            s1 = self.stage_data(1) or {}
            jsonl = s1.get("jsonl") or os.path.join(
                CKPT_DIR, f"train_seed42_{self.runtag}.jsonl")
            cmd = [PY, DIAG_DQN_SCRIPT, "--jsonl", jsonl]
            log_path = os.path.join(STRESS_DIR, "dqn_light_suite.log")
            p = subprocess.run(cmd, cwd=REPO, capture_output=True,
                               text=True, timeout=2400)
            with open(log_path, "w") as f:
                f.write(p.stdout + "\n[stderr]\n" + p.stderr)
            summary = [l for l in p.stdout.splitlines()
                       if re.search(r"VERDICT|checks:", l)][:20]
            md = ("In-process checks failed (traceback in results.json); "
                  "fell back to the subprocess light suite "
                  f"(`stress/{os.path.basename(log_path)}`).\n\n"
                  "**Limitation**: the light suite's checks 1/3 run against "
                  "run-2 snapshots (its hardcoded checkpoints), NOT the "
                  "stress winner; only check 5 used this run's JSONL.\n\n"
                  + "\n".join(f"> {s}" for s in summary))
            return {"mode": "subprocess-fallback",
                    "inprocess_traceback": tb, "jsonl": jsonl,
                    "returncode": p.returncode, "log": log_path,
                    "summary_lines": summary}, md

    # ---------------- stage 10 ----------------

    def stage10(self):
        s3, s4, s5 = (self.stage_data(n) for n in (3, 4, 5))
        s6, s7, s8, s9 = (self.stage_data(n) for n in (6, 7, 8, 9))

        # --- executive summary table ---
        rows = []  # (agent, duration, n, mean ttv, std, violated)
        if s3:
            n = s3["n_episodes"]
            for r in s3["sweep"]:
                rows.append((f"DQN winner c={r['c']}", 600, n,
                             r["mean_time_to_violation"],
                             r["std_time_to_violation"],
                             f"{int(r['violation_rate'] * n)}/{n}"))
        if s4:
            rows.append(("NOOP baseline", 600, s4["n_episodes"],
                         s4["mean_time_to_violation"],
                         s4["std_time_to_violation"],
                         f"{s4['violated']}/{s4['n_episodes']}"))
        if s6:
            n = s6["n_episodes"]
            for r in s6["sweep"]:
                rows.append((f"DQN winner c={r['c']} (OOD)", 1200, n,
                             r["mean_time_to_violation"],
                             r["std_time_to_violation"],
                             f"{int(r['violation_rate'] * n)}/{n}"))
            noop = s6["noop_1200"]
            rows.append(("NOOP baseline (OOD)", 1200, n,
                         noop["mean_time_to_violation"],
                         noop["std_time_to_violation"],
                         f"{noop['violated']}/{n}"))
        if s7:
            import numpy as np
            for variant in ("modelfree", "hybrid"):
                sub = [r for r in s7["runs"] if r["variant"] == variant
                       and r.get("ttv_s") is not None]
                if sub:
                    ttvs = [r["ttv_s"] for r in sub]
                    nv = sum(1 for r in sub if r.get("violated"))
                    dur = 120 if s7.get("quick") else 600
                    rows.append((f"Interval MCTS ({variant})", dur,
                                 len(sub), float(np.mean(ttvs)),
                                 float(np.std(ttvs)), f"{nv}/{len(sub)}"))
        tbl = ["| agent | duration (s) | eps | mean TTV (s) | std | "
               "violated |", "|---|---|---|---|---|---|"]
        for a, d, n, m, sd, v in rows:
            tbl.append(f"| {a} | {d} | {n} | {fmt(m, '.1f')} | "
                       f"{fmt(sd, '.1f')} | {v} |")
        if not rows:
            tbl.append("| (no TTV results available) | | | | | |")

        # --- calibration trajectory ---
        cal = [f"{r['run']}: {fmt(r['value'])}"
               for r in REF_REALIZED_COVERAGE]
        if s5 and s5.get("realized_coverage") is not None:
            cal.append(f"this run: {fmt(s5['realized_coverage'])} "
                       f"(bootstrap-completed "
                       f"{fmt(s5.get('bootstrap_coverage'))})")
            cal_verdict = ("IMPROVING towards the 0.85 target"
                           if s5["realized_coverage"] > 0.125
                           else "NOT improving over run 4 (0.125)")
        else:
            cal.append("this run: unavailable (stage 5 did not complete)")
            cal_verdict = "inconclusive"

        # --- c-sweep monotonicity ---
        if s3 and len(s3["sweep"]) >= 2:
            ttvs = [(r["c"], r["mean_time_to_violation"])
                    for r in s3["sweep"]]
            diffs = [b[1] - a[1] for a, b in zip(ttvs, ttvs[1:])]
            if all(d <= 0 for d in diffs):
                mono = ("MONOTONE: TTV non-increasing in c — pessimism "
                        "(low c) is uniformly safer")
            elif all(d >= 0 for d in diffs):
                mono = ("INVERTED: TTV non-decreasing in c — optimism "
                        "safer here (unexpected)")
            else:
                mono = "NON-MONOTONE: " + ", ".join(
                    f"c={c}: {fmt(t, '.0f')}s" for c, t in ttvs)
        else:
            mono = ("inconclusive (single c value in quick mode)"
                    if self.quick else "inconclusive (stage 3 incomplete)")

        # --- OOD verdict ---
        if s6:
            e12, e6 = s6.get("edge_1200"), s6.get("edge_600")
            if e6 is not None:
                ood = (f"pessimism edge {fmt(e6, '.1f')} s @600 -> "
                       f"{fmt(e12, '.1f')} s @1200: the safety edge "
                       f"{'GROWS' if e12 > e6 else 'does NOT grow'} "
                       f"under stress")
            else:
                ood = (f"edge @1200 = {fmt(e12, '.1f')} s "
                       f"(no 600 s c=1.0 reference)")
        else:
            ood = "skipped/unavailable"

        # --- MCTS comparison ---
        if s7:
            def mean_ttv(variant):
                import numpy as np
                sub = [r["ttv_s"] for r in s7["runs"]
                       if r["variant"] == variant
                       and r.get("ttv_s") is not None]
                return float(np.mean(sub)) if sub else None
            mf, hy = mean_ttv("modelfree"), mean_ttv("hybrid")
            n_to = sum(1 for r in s7["runs"] if r["status"] == "timeout")
            if mf is not None and hy is not None:
                mcts = (f"model-free mean TTV {fmt(mf, '.1f')} s vs hybrid "
                        f"{fmt(hy, '.1f')} s -> "
                        f"{'hybrid' if hy > mf else 'model-free' if mf > hy else 'tie'}"
                        f"{' ahead' if hy != mf else ''}"
                        + (f" ({n_to} timeout(s))" if n_to else ""))
            else:
                mcts = f"incomplete ({n_to} timeout(s)/errors)"
        else:
            mcts = "skipped/unavailable"

        autopsy = ("; ".join(s8["verdict_lines"][:3])
                   if s8 and s8.get("verdict_lines")
                   else "skipped/unavailable")

        # --- open issues ---
        issues = []
        for no in sorted(self.results["stages"], key=int):
            e = self.results["stages"][no]
            if e["status"] == "failed":
                first = e["traceback"].strip().splitlines()[-1]
                issues.append(f"stage {no} FAILED: {first}")
            elif e["status"] == "skipped":
                issues.append(f"stage {no} skipped ({e.get('reason', '')})")
        if s5 and (s5.get("realized_coverage") or 0) < 0.5:
            issues.append(f"realized coverage "
                          f"{fmt(s5['realized_coverage'])} still far below "
                          f"the 0.85 target (acceptance criterion unmet)")
        if s7:
            for r in s7["runs"]:
                if r["status"] != "ok":
                    issues.append(f"MCTS {r['variant']} seed {r['seed']}: "
                                  f"{r['status']}")
        if s9 and s9.get("mode") == "subprocess-fallback":
            issues.append("stage 9 checks 1/3 ran against run-2 snapshots "
                          "(light-suite fallback), not the winner")
        if not issues:
            issues.append("none")

        md = ["### Executive summary — time to first violation", "",
              "\n".join(tbl), "",
              "### Verdicts", "",
              f"- **Calibration trajectory (realized coverage)**: "
              f"{' -> '.join(cal)} — {cal_verdict}",
              f"- **C-sweep monotonicity**: {mono}",
              f"- **OOD stress**: {ood}",
              f"- **MCTS model-free vs hybrid**: {mcts}",
              f"- **MCTS autopsy**: {autopsy}",
              "",
              "### Open issues", ""]
        md += [f"- {i}" for i in issues]
        md += ["", f"_Report finalized {now()}._"]
        data = {"summary_rows": rows, "calibration": cal,
                "calibration_verdict": cal_verdict, "monotonicity": mono,
                "ood_verdict": ood, "mcts_verdict": mcts,
                "autopsy_verdict": autopsy, "open_issues": issues}
        return data, "\n".join(md)

    # ---------------- orchestration ----------------

    def run(self):
        a = self.args
        only = set(a.only) if a.only else None

        def reason(no):
            if only is not None and no not in only:
                return "not in --only selection"
            if self.quick and no == 0:
                return "--quick skips the stage-0 wait"
            if self.quick and no in (6, 8):
                return "--quick skips this stage"
            if a.skip_mcts and no in (7, 8):
                return "--skip-mcts"
            return None

        stage_fns = {0: self.stage0, 1: self.stage1, 2: self.stage2,
                     3: self.stage3, 4: self.stage4, 5: self.stage5,
                     6: self.stage6, 7: self.stage7, 8: self.stage8,
                     9: self.stage9, 10: self.stage10}
        try:
            for no in range(11):
                self.run_stage(no, stage_fns[no], skip_reason=reason(no))
        finally:
            self.close_envs()
        n_fail = sum(1 for e in self.results["stages"].values()
                     if e["status"] == "failed")
        print(f"\nDONE: {len(self.results['stages'])} stages, "
              f"{n_fail} failed. Report: {REPORT_PATH}")
        return 0


def main():
    ap = argparse.ArgumentParser(
        description="Stress-test driver for BluebirdATC interval DQN + MCTS")
    ap.add_argument("--quick", action="store_true",
                    help="end-to-end dry run: 2 eval eps, 1 c value, 1 MCTS "
                         "seed at 120 s (sims 8, horizon 8); skips stages "
                         "0, 6, 8")
    ap.add_argument("--skip-mcts", action="store_true",
                    help="skip stages 7 and 8")
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="run only these stage numbers")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    return Driver(args).run()


if __name__ == "__main__":
    sys.exit(main())
