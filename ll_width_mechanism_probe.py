"""
LL width-mechanism probe — does LunarLander's OOD interval widening track
input MAGNITUDE (the bluebird pilot-net accident) or genuine UNFAMILIARITY?

Adjudicates the MORNING_BRIEF §4 claim JK challenged (8 July 2026). Same
design as a4_diagnosis_pilotmech on the bluebird pilot nets, plus the
condition LL uniquely enables: states actually visited under the headline
wind-OOD conditions, placed against a norm->width curve fit on scaled
FAMILIAR states.

PRE-REGISTERED (before first run):
  H-accident (mechanism transfers to LL):
    - calm x2/x5/x10/x100 (same familiar states, scaled) reproduce multi-x
      widening, tracking hidden norm;
    - shuffled (per-dim batch permutation: joint structure destroyed,
      marginals/norms preserved) stays ~1x the curve (residual in [0.5, 2]);
    - wind10/wind20 visited states land ON the curve (residual ~ 1): the
      headline widening is priced by state magnitude, not unfamiliarity.
  H-genuine (LL learned real epistemics):
    - shuffled and/or wind residuals >= 2-3: width responds to
      unfamiliarity at matched magnitude.
A miss either way is a finding; numbers reported regardless.

Widths use the same measure as the original ood_eval (mean over actions).
"""
import argparse
import json
import math
import pathlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import gymnasium as gym

from lunarlander_interval_dqn import IntervalQNetwork

PER_C = pathlib.Path("checkpoints/per_c")
SCALES = [1.0, 2.0, 5.0, 10.0, 100.0]
WINDS = [10.0, 20.0]


def make_env(wind_power=0.0):
    return gym.make("LunarLander-v3",
                    enable_wind=(wind_power > 0.0),
                    wind_power=wind_power,
                    gravity=-10.0)


def load_net(ckpt_path):
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = IntervalQNetwork()
    net.load_state_dict(d["q_net_state_dict"])
    net.eval()
    return net, d["args"]


def hurwicz_action(net, state, c):
    with torch.no_grad():
        lo, up = net(torch.FloatTensor(state).unsqueeze(0))
        score = lo + c * (up - lo)
        return int(score.argmax(dim=1).item())


def collect_states(net, c, wind_power, episodes, seed0, eps=0.02,
                   max_steps=500, cap=3000):
    env = make_env(wind_power)
    rng = np.random.default_rng(seed0)
    states = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed0 + ep)
        for _ in range(max_steps):
            states.append(np.asarray(s, dtype=np.float32))
            a = (env.action_space.sample() if rng.random() < eps
                 else hurwicz_action(net, s, c))
            s, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    env.close()
    S = np.stack(states)
    if len(S) > cap:
        S = S[rng.choice(len(S), cap, replace=False)]
    return S


def measure(net, states):
    """Mean width (mean over actions — the ood_eval measure) and mean
    hidden norm at the last shared layer."""
    captured = {}

    def hook(_m, _i, out):
        captured["h"] = out.detach()

    h = net.shared.register_forward_hook(hook)
    with torch.no_grad():
        lo, up = net(torch.from_numpy(states))
    h.remove()
    width = (up - lo).mean(dim=1)
    hnorm = captured["h"].norm(dim=1)
    return float(width.mean()), float(hnorm.mean())


def probe_ckpt(ckpt_path, seed0):
    net, args = load_net(ckpt_path)
    c = float(args["c_train"])

    calm = collect_states(net, c, 0.0, episodes=8, seed0=seed0)
    conds = {}

    # magnitude family: same familiar states, scaled
    for k in SCALES:
        name = "calm" if k == 1.0 else f"calm_x{k:g}"
        conds[name] = measure(net, calm * np.float32(k))

    # unfamiliarity at matched norm: per-dim batch permutation
    rng = np.random.default_rng(seed0)
    shuf = calm.copy()
    for d in range(shuf.shape[1]):
        rng.shuffle(shuf[:, d])
    conds["shuffled"] = measure(net, shuf)

    # box garbage: uniform in the observed per-dim range
    box = rng.uniform(calm.min(0), calm.max(0),
                      size=calm.shape).astype(np.float32)
    conds["box_garbage"] = measure(net, box)

    # gaussian at 5x the calm per-dim std (large + unfamiliar)
    conds["gauss_sd5"] = measure(
        net, (rng.standard_normal(calm.shape) * calm.std(0) * 5.0
              ).astype(np.float32))

    # the headline conditions: states actually visited under wind
    for w in WINDS:
        wind_states = collect_states(net, c, w, episodes=8,
                                     seed0=seed0 + 1000 * int(w))
        conds[f"wind{w:g}"] = measure(net, wind_states)

    # norm->width curve from the magnitude family (log-log least squares)
    pts = [conds["calm" if k == 1.0 else f"calm_x{k:g}"] for k in SCALES]
    X = np.array([[math.log(n), 1.0] for _, n in pts])
    y = np.array([math.log(w) for w, _ in pts])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    def predicted(hnorm):
        return math.exp(coef[0] * math.log(hnorm) + coef[1])

    out = {"ckpt": str(ckpt_path), "c_train": c, "slope": float(coef[0]),
           "conds": {}}
    w_calm = conds["calm"][0]
    for name, (w, n) in conds.items():
        out["conds"][name] = {
            "width": w, "hidden_norm": n,
            "ratio_vs_calm": w / w_calm,
            "residual_vs_norm_curve": w / predicted(n),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs", default="0.0,0.5,1.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=str(PER_C / "ll_mechanism_probe.json"))
    a = ap.parse_args()

    results = []
    for cv in a.cs.split(","):
        for sd in a.seeds.split(","):
            ck = PER_C / f"c{cv}_seed{sd}" / "ckpt_final.pt"
            if not ck.exists():
                print(f"skip (missing): {ck}")
                continue
            r = probe_ckpt(ck, seed0=90000 + 17 * int(sd))
            results.append(r)
            print(f"\n=== c={cv} seed={sd}  (norm-curve slope "
                  f"{r['slope']:.2f}) ===")
            print(f"{'condition':<12}{'width':>9}{'h-norm':>9}"
                  f"{'x calm':>8}{'resid':>7}")
            for name, v in r["conds"].items():
                print(f"{name:<12}{v['width']:>9.2f}"
                      f"{v['hidden_norm']:>9.1f}"
                      f"{v['ratio_vs_calm']:>8.2f}"
                      f"{v['residual_vs_norm_curve']:>7.2f}")

    # cross-net summary of the two adjudicating quantities
    def agg(name, key):
        vals = [r["conds"][name][key] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    print("\n===== SUMMARY over", len(results), "nets =====")
    print(f"{'condition':<12}{'x calm (mean+/-sd)':>22}"
          f"{'residual vs norm curve':>26}")
    for name in results[0]["conds"]:
        rm, rs = agg(name, "ratio_vs_calm")
        em, es = agg(name, "residual_vs_norm_curve")
        print(f"{name:<12}{rm:>13.2f} +/- {rs:<6.2f}"
              f"{em:>15.2f} +/- {es:<6.2f}")

    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
