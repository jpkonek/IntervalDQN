"""
Width-mechanism probes P1-P5 (see WIDTH_MECHANISM_PROBES.md for the
pre-registered hypotheses H1-H4 and predictions).

Isolates the mechanism behind the ~2x above-norm-curve width signal on
structure-destroyed inputs found by ll_width_mechanism_probe.py.

  P1  emergence over training (untrained + snapshot ladder)
  P2  raw-head decomposition: mean shift vs Jensen dispersion
  P3  activation-pattern novelty vs width (within shuffled)
  P4  additive-marginal model: is the signal just feature recombination?
  P5  dose-response (alpha-interpolation, k-dim shuffle) + per-state AUROC

Usage:
  .venv/bin/python ll_width_mechanism_probe2.py                # P1-P5, per_c c0.5
  .venv/bin/python ll_width_mechanism_probe2.py --ckpts a.pt b.pt --label wr0.0
                                                 # P2-P5 residual on given ckpts (P6 sweep)
"""
import argparse
import json
import math
import pathlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

from lunarlander_interval_dqn import IntervalQNetwork
import ll_width_mechanism_probe as p1

PER_C = pathlib.Path("checkpoints/per_c")
SCALES = [1.0, 2.0, 5.0, 10.0, 100.0]


# --------------------------------------------------------------------------
# shared instrumentation
# --------------------------------------------------------------------------

def full_forward(net, states):
    """Returns (width per state [mean over actions], hidden norm,
    raw delta pre-softplus [mean over actions], relu on/off patterns)."""
    caps = {}

    def mk(name):
        def hook(_m, _i, out):
            caps[name] = out.detach()
        return hook

    hs = [net.shared[1].register_forward_hook(mk("h1")),
          net.shared[3].register_forward_hook(mk("h2"))]
    with torch.no_grad():
        x = torch.from_numpy(states)
        lo, up = net(x)
        raw = net.head(caps["h2"]).view(len(states), net.n_actions, 2)[:, :, 1]
    for h in hs:
        h.remove()
    width = (up - lo).mean(dim=1).numpy()
    hnorm = caps["h2"].norm(dim=1).numpy()
    pattern = torch.cat([caps["h1"] > 0, caps["h2"] > 0], dim=1).numpy()
    return width, hnorm, raw.mean(dim=1).numpy(), pattern


def norm_curve(net, calm):
    """log-log width~hnorm fit on the calm x-scales family; returns
    predict(hnorm) callable."""
    xs, ys = [], []
    for k in SCALES:
        w, n, _, _ = full_forward(net, calm * np.float32(k))
        xs.append(math.log(float(n.mean())))
        ys.append(math.log(float(w.mean())))
    A = np.stack([np.array(xs), np.ones(len(xs))], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    return lambda hn: math.exp(coef[0] * math.log(hn) + coef[1])


def make_shuffled(calm, rng):
    s = calm.copy()
    for d in range(s.shape[1]):
        rng.shuffle(s[:, d])
    return s


def residual(net, calm, states):
    pred = norm_curve(net, calm)
    w, n, _, _ = full_forward(net, states)
    return float(w.mean()) / pred(float(n.mean()))


def auroc(w_real, w_ood):
    """Rank-based AUC: P(width_ood > width_real)."""
    allw = np.concatenate([w_real, w_ood])
    ranks = allw.argsort().argsort().astype(np.float64)
    r_ood = ranks[len(w_real):]
    n0, n1 = len(w_real), len(w_ood)
    return float((r_ood.sum() - n1 * (n1 - 1) / 2) / (n0 * n1))


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------

def probe_p1(calm_by_seed, rng):
    """Emergence: untrained nets + c0.5 snapshot ladder."""
    rows = []
    for seed in [0, 1, 2]:
        calm = calm_by_seed[seed]
        shuf = make_shuffled(calm, rng)
        torch.manual_seed(seed)
        net = IntervalQNetwork().eval()
        rows.append({"seed": seed, "stage": "untrained", "tracker_t": None,
                     "resid_shuffled": residual(net, calm, shuf)})
        for ep in [200, 400, 600, 800]:
            ck = PER_C / f"c0.5_seed{seed}" / f"ckpt_ep{ep:05d}.pt"
            d = torch.load(ck, map_location="cpu", weights_only=False)
            net = IntervalQNetwork()
            net.load_state_dict(d["q_net_state_dict"])
            net.eval()
            rows.append({"seed": seed, "stage": f"ep{ep}",
                         "tracker_t": d.get("tracker_t"),
                         "resid_shuffled": residual(net, calm, shuf)})
    return rows


def probe_p2_p5(net, calm, rng, label=""):
    """P2 raw decomposition, P3 pattern novelty, P4 additive model,
    P5 dose-response + AUROC — one trained net."""
    out = {"label": label}
    shuf = make_shuffled(calm, rng)
    box = rng.uniform(calm.min(0), calm.max(0),
                      size=calm.shape).astype(np.float32)

    w_c, n_c, raw_c, pat_c = full_forward(net, calm)
    w_s, n_s, raw_s, pat_s = full_forward(net, shuf)
    w_b, n_b, raw_b, _ = full_forward(net, box)

    # ---- P2: mean shift vs Jensen dispersion (norms are matched, so direct)
    def sp(x):
        return math.log1p(math.exp(min(x, 30.0))) if x < 30 else x

    def p2_block(w_o, raw_o):
        d_width = float(w_o.mean() - w_c.mean())
        d_mean_driven = sp(float(raw_o.mean())) - sp(float(raw_c.mean()))
        return {"raw_mean_calm": float(raw_c.mean()),
                "raw_mean_ood": float(raw_o.mean()),
                "raw_std_calm": float(raw_c.std()),
                "raw_std_ood": float(raw_o.std()),
                "width_gap": d_width,
                "mean_driven_share": d_mean_driven / d_width if d_width else None}

    out["p2_shuffled"] = p2_block(w_s, raw_s)
    out["p2_box"] = p2_block(w_b, raw_b)

    # ---- P3: pattern novelty vs width within shuffled
    ref = pat_c[:1500].astype(np.float32) * 2 - 1
    qry = pat_s[:1500].astype(np.float32) * 2 - 1
    ham = (ref.shape[1] - qry @ ref.T) / 2.0        # hamming to each calm pattern
    novelty = ham.min(axis=1)
    ws = w_s[:1500]
    rr = np.argsort(np.argsort(novelty)).astype(float)
    rw = np.argsort(np.argsort(ws)).astype(float)
    rho = float(np.corrcoef(rr, rw)[0, 1])
    # baseline: same statistic within calm (self-novelty, exclude self-match)
    ham_c = (ref.shape[1] - ref @ ref.T) / 2.0
    np.fill_diagonal(ham_c, np.inf)
    nov_c = ham_c.min(axis=1)
    rho_c = float(np.corrcoef(np.argsort(np.argsort(nov_c)).astype(float),
                              np.argsort(np.argsort(w_c[:1500])).astype(float))[0, 1])
    out["p3"] = {"novelty_mean_shuffled": float(novelty.mean()),
                 "novelty_mean_calm_self": float(nov_c.mean()),
                 "spearman_novelty_width_shuffled": rho,
                 "spearman_novelty_width_calm": rho_c}

    # ---- P4: additive-marginal model fit on calm, applied to shuffled
    nbin = 16
    edges = [np.quantile(calm[:, d], np.linspace(0, 1, nbin + 1))
             for d in range(calm.shape[1])]
    gmean = float(w_c.mean())

    def bin_idx(S, d):
        return np.clip(np.searchsorted(edges[d][1:-1], S[:, d]), 0, nbin - 1)

    fd = []
    for d in range(calm.shape[1]):
        idx = bin_idx(calm, d)
        f = np.zeros(nbin)
        for b in range(nbin):
            m = idx == b
            f[b] = w_c[m].mean() - gmean if m.any() else 0.0
        fd.append(f)

    def additive_pred(S):
        p = np.full(len(S), gmean)
        for d in range(S.shape[1]):
            p += fd[d][bin_idx(S, d)]
        return p

    pred_c = additive_pred(calm)
    ss_res = float(((w_c - pred_c) ** 2).sum())
    ss_tot = float(((w_c - gmean) ** 2).sum())
    pred_s = additive_pred(shuf)
    out["p4"] = {"r2_on_calm": 1 - ss_res / ss_tot,
                 "additive_pred_shuffled_mean": float(pred_s.mean()),
                 "actual_shuffled_mean": float(w_s.mean()),
                 "actual_over_additive": float(w_s.mean() / pred_s.mean())}

    # ---- P5: dose-response + AUROC
    alphas = {}
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mix = ((1 - a) * calm + a * shuf).astype(np.float32)
        w_m, n_m, _, _ = full_forward(net, mix)
        alphas[f"{a:g}"] = {"width": float(w_m.mean()),
                            "hnorm": float(n_m.mean())}
    kdims = {}
    for k in [1, 2, 4, 8]:
        dims = rng.choice(calm.shape[1], size=k, replace=False)
        s = calm.copy()
        for d in dims:
            rng.shuffle(s[:, d])
        w_k, _, _, _ = full_forward(net, s)
        kdims[str(k)] = float(w_k.mean())
    out["p5"] = {"alpha": alphas, "kdim_width": kdims,
                 "kdim_calm_width": float(w_c.mean()),
                 "auroc_shuffled": auroc(w_c, w_s),
                 "auroc_box": auroc(w_c, w_b)}
    return out


# --------------------------------------------------------------------------

def load_final(ck):
    d = torch.load(ck, map_location="cpu", weights_only=False)
    net = IntervalQNetwork()
    net.load_state_dict(d["q_net_state_dict"])
    net.eval()
    return net, d["args"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=None,
                    help="explicit ckpt paths: run P2-P5 + residual only "
                         "(P6 sweep mode)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(PER_C / "ll_mechanism_probe2.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(4242)

    # calm state sets, one per seed, collected with the trained final policy
    calm_by_seed = {}
    for seed in [0, 1, 2]:
        net, args = load_final(PER_C / f"c0.5_seed{seed}" / "ckpt_final.pt")
        calm_by_seed[seed] = p1.collect_states(
            net, float(args["c_train"]), 0.0, episodes=8,
            seed0=90000 + 17 * seed)

    results = {}

    if a.ckpts:  # sweep mode (P6): residual + P2-P5 on the given nets
        rows = []
        for ck in a.ckpts:
            net, args_ck = load_final(pathlib.Path(ck))
            # calm manifold from the probed net's OWN policy
            calm = p1.collect_states(net, float(args_ck["c_train"]), 0.0,
                                     episodes=8, seed0=91000)
            shuf = make_shuffled(calm, rng)
            r = probe_p2_p5(net, calm, rng, label=ck)
            r["resid_shuffled"] = residual(net, calm, shuf)
            rows.append(r)
            print(f"{ck}: resid_shuffled={r['resid_shuffled']:.2f} "
                  f"mean_share={r['p2_shuffled']['mean_driven_share']:.2f}")
        results = {"sweep_" + a.label: rows}
        out_path = a.out.replace(".json", f"_{a.label}.json")
    else:
        print("=== P1: emergence over training ===")
        results["p1"] = probe_p1(calm_by_seed, rng)
        for r in results["p1"]:
            t = f"t={r['tracker_t']:.3f}" if r["tracker_t"] else "        "
            print(f"  seed{r['seed']} {r['stage']:<10} {t}  "
                  f"resid_shuffled={r['resid_shuffled']:.2f}")

        print("\n=== P2-P5 on final nets (c0.5, seeds 0-2) ===")
        results["p2_p5"] = []
        for seed in [0, 1, 2]:
            net, _ = load_final(PER_C / f"c0.5_seed{seed}" / "ckpt_final.pt")
            r = probe_p2_p5(net, calm_by_seed[seed], rng,
                            label=f"c0.5_seed{seed}")
            results["p2_p5"].append(r)
            p2, p3, p4, p5 = r["p2_shuffled"], r["p3"], r["p4"], r["p5"]
            print(f"\n  --- seed {seed} ---")
            print(f"  P2 raw mean {p2['raw_mean_calm']:.2f}->"
                  f"{p2['raw_mean_ood']:.2f}, std {p2['raw_std_calm']:.2f}->"
                  f"{p2['raw_std_ood']:.2f}, mean-driven share "
                  f"{p2['mean_driven_share']:.2f}")
            print(f"  P3 novelty(shuf) {p3['novelty_mean_shuffled']:.1f} bits "
                  f"vs calm-self {p3['novelty_mean_calm_self']:.1f}; "
                  f"spearman(novelty,width) shuf={p3['spearman_novelty_width_shuffled']:.2f} "
                  f"calm={p3['spearman_novelty_width_calm']:.2f}")
            print(f"  P4 additive R2(calm)={p4['r2_on_calm']:.2f}; "
                  f"actual/additive on shuffled={p4['actual_over_additive']:.2f}")
            aw = {k: v['width'] for k, v in p5['alpha'].items()}
            print(f"  P5 alpha widths {aw}")
            print(f"     k-dim widths {p5['kdim_width']} "
                  f"(calm {p5['kdim_calm_width']:.2f})")
            print(f"     AUROC shuffled={p5['auroc_shuffled']:.2f} "
                  f"box={p5['auroc_box']:.2f}")
        out_path = a.out

    json.dump(results, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
