"""Evaluate every credal-LunarLander (c, seed) checkpoint at its training c.

Worst-wind MC, mirrors `Credal DQN/scripts/eval_per_c.py`. Each model is
evaluated at c=its training c (deployment matches training).
"""

from __future__ import annotations
import argparse
import json
import statistics
import subprocess
import time
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-values", type=str, default="0.0,0.2,0.5,0.8,1.0")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--results-root",
                    default="/Users/jk13942/Documents/GitHub/IP-ML/Credal LunarLander/results/per_c")
    ap.add_argument("--n-rollouts-per-wind", type=int, default=30)
    ap.add_argument("--n-wind", type=int, default=11)
    ap.add_argument("--wind-lo", type=float, default=0.0)
    ap.add_argument("--wind-hi", type=float, default=15.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    c_values = [float(c) for c in args.c_values.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    root = pathlib.Path(args.results_root)
    out_path = pathlib.Path(args.out) if args.out else root / "c_curve.json"

    eval_script = pathlib.Path(__file__).parent / "eval_credal_lunarlander.py"
    python = "/Users/jk13942/Documents/GitHub/IP-ML/.venv/bin/python"

    per_c: dict = {}
    for c in c_values:
        per_seed = []
        for seed in seeds:
            run_dir = root / f"c{c}_seed{seed}"
            ckpt = run_dir / "ckpt_final.pt"
            if not ckpt.exists():
                print(f"  c={c} seed={seed}: NO CHECKPOINT", flush=True)
                continue
            out_json = run_dir / "eval_at_train_c.json"
            t0 = time.time()
            cmd = [python, str(eval_script),
                   "--ckpt", str(ckpt),
                   "--seed", str(seed),
                   "--c-eval", str(c),
                   "--n-rollouts-per-wind", str(args.n_rollouts_per_wind),
                   "--n-wind", str(args.n_wind),
                   "--wind-lo", str(args.wind_lo),
                   "--wind-hi", str(args.wind_hi),
                   "--out", str(out_json)]
            env = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PATH": "/usr/bin:/bin"}
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
            elapsed = time.time() - t0
            if r.returncode != 0:
                print(f"  c={c} seed={seed}: FAILED  {r.stderr[-300:]}", flush=True)
                continue
            d = json.loads(out_json.read_text())
            per_seed.append({
                "seed": seed,
                "robust_value": d["robust_value_mc"],
                "worst_wind": d["worst_wind"],
                "wall_s": elapsed,
            })
            print(f"  c={c} seed={seed}: robust_R={d['robust_value_mc']:.2f} "
                  f"@ wind={d['worst_wind']:.2f}  ({elapsed:.1f}s)", flush=True)
        if per_seed:
            vals = [p["robust_value"] for p in per_seed]
            per_c[str(c)] = {
                "c": c,
                "n_seeds": len(per_seed),
                "robust_values": vals,
                "median": statistics.median(vals),
                "mean": statistics.mean(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "per_seed": per_seed,
            }

    print("\n=== c-curve (LunarLander credal, deployment c = training c) ===",
          flush=True)
    print(f"{'c':>6} {'n':>3} {'median':>10} {'mean':>10} {'min':>10} {'max':>10}",
          flush=True)
    cs_sorted = sorted(per_c.keys(), key=lambda x: float(x))
    for c_str in cs_sorted:
        d = per_c[c_str]
        print(f"{d['c']:>6} {d['n_seeds']:>3} {d['median']:>10.2f} "
              f"{d['mean']:>10.2f} {d['min']:>10.2f} {d['max']:>10.2f}",
              flush=True)
    if len(per_c) >= 2:
        meds = [per_c[c]['median'] for c in cs_sorted]
        # NOTE: LunarLander reward is positive-good (vs collision cost which is positive-bad).
        # Spread c=high vs c=low: more positive = better. We report raw spread.
        spread = meds[-1] - meds[0]
        rel = spread / abs(meds[0]) if meds[0] != 0 else float("inf")
        mono_inc = all(meds[i] <= meds[i + 1] + 1e-3 for i in range(len(meds) - 1))
        mono_dec = all(meds[i] >= meds[i + 1] - 1e-3 for i in range(len(meds) - 1))
        print(f"\n  spread (c=1 vs c=0): {spread:+.2f} ({rel*100:+.1f}% of c=0)",
              flush=True)
        print(f"  monotone increasing in c: {mono_inc}  decreasing: {mono_dec}",
              flush=True)

    out = {"c_values": c_values, "seeds": seeds, "per_c": per_c}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
