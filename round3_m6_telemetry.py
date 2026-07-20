"""
M6 recovery-instrument SCAFFOLD — data side only (ROUND3, 20 July 2026)
=======================================================================

Parsing + reporting for the per-arc recovery telemetry that ROUND3
mandates (ROUND3_REVIEW "D (new instrument M6 + D2 telemetry)" and
ROUND3_REREVIEW "M6 flag-coverage instrument" + "Boundary telemetry +
laundering alarm + coverage feed"). The ARMING MACHINERY LIVES IN
bluebird_controller_dqn.py AND DOES NOT EXIST YET — this module reads
the JSONL fields bcd will emit and DEGRADES GRACEFULLY (per-field
"absent" reporting) until that build lands. Nothing here trains,
simulates or edits bcd.

FIELD CONTRACT (proposed names for the bcd build — the re-review
specifies the CONTENT of every field but not literal key names; this
table is therefore the binding name registry, flagged in
ROUND3_INSTRUMENTS_STATUS.md; if the bcd build ships different names,
update FIELD_MAP here, nowhere else):

per-episode JSONL record keys
  risk_episodes: list of per-arc dicts —
      onset_step        int    band-entry step
      pair              [a, b] the named risk pair
      causal_would_enter_noop
                        bool   the per-pair cbp_noop_snapshot CAUSAL
                               TEST at onset ("would this pair have
                               entered the band under NOOP within H?")
                               — GROUND TRUTH for flag coverage;
                               False => endogenous (command-caused)
      armed             bool   arming verdict (causal test primary)
      disqualifying_command   {step, cand_idx, target} | None
      resolved          bool   K=10 out-of-band + clearance-state
      resolution_step   int | None
      time_in_band_steps int
      w_rec_paid        float  (own budget column, never delivery)
      flag_events: list —     (sign channel: SUPPLEMENTARY telemetry)
          step, pair, per_pair_diff, aggregate, issuance_d_nm,
          issuance_dfl, inside_ramp (bool: d < 15 nm AND |dFL| < 20),
          armed_outcome
  boundaries: list of per-boundary dicts (global K=10 collapse) —
      step, armed, live_clearance, m_per_window (list of int),
      q_target_pre_floor, q_target_post_floor,
      band_reentry_within_nstep (bool)
  armed_win_count   int      per episode (config-echo companion)
  w_rec_paid_total  float    per episode
  tolerance_invoked int      C2 tolerance counter (per episode)
  bonus_override    int      count-bonus overrode the tolerance verdict
  in_band_tolerance_flips
                    int      stratum-guard breach counter (hard bar 0)

PRE-REGISTERED (flag coverage, re-review "M6 flag-coverage
instrument"): prediction = near-zero sign-flag coverage for
OUTSIDE-ramp endogenous onsets, partial coverage inside; FAIL criterion
= material W_rec mass on unflagged outside-ramp endogenous onsets
(draft threshold: > 10% of total W_rec mass — DRAFT, review must
ratify; the fraction is always printed either way).

LAUNDERING ALARM (re-review "Boundary telemetry"): exposure metric =
joint rate of floor-active AND band-reentry-within-nstep; alarm = that
joint rate rising, or floor-active fraction on live-clearance
boundaries failing to fall — both time series are emitted for the
battery reader; this module prints the whole-run rates.

Usage:
    .venv/bin/python round3_m6_telemetry.py --jsonl PATH [PATH ...]
"""

import argparse
import json

import numpy as np

# single indirection point if bcd ships different key names
FIELD_MAP = {
    "risk_episodes": "risk_episodes",
    "boundaries": "boundaries",
    "armed_win_count": "armed_win_count",
    "w_rec_paid_total": "w_rec_paid_total",
    "tolerance_invoked": "tolerance_invoked",
    "bonus_override": "bonus_override",
    "in_band_tolerance_flips": "in_band_tolerance_flips",
}

RAMP_D_NM = 15.0      # pair_conflict_f lateral support edge
RAMP_DFL = 20.0       # CONFLICT_FL vertical gate
MATERIAL_WREC_FRAC = 0.10   # DRAFT fail threshold (flagged for review)


class FieldTracker:
    """Counts per-field presence so 'absent' is reported per field, not
    as one all-or-nothing failure."""

    def __init__(self):
        self.present = {}
        self.n_records = 0

    def get(self, rec, name, default=None):
        key = FIELD_MAP[name]
        hit = key in rec
        self.present[name] = self.present.get(name, 0) + int(hit)
        return rec.get(key, default)

    def coverage_line(self, name):
        n = self.present.get(name, 0)
        if n == 0:
            return (f"ABSENT (bcd not yet emitting "
                    f"'{FIELD_MAP[name]}' — arming machinery pending)")
        return f"present in {n}/{self.n_records} records"


def _flagged(arc):
    """Sign-channel verdict for one arc: any flag event with a NEGATIVE
    per-pair conflict differential on this pair (the issuance-time sign
    test saying 'this command endangered the pair')."""
    return any(f.get("per_pair_diff", 0.0) < 0.0
               for f in arc.get("flag_events", []))


def _issuance_inside_ramp(arc):
    """inside_ramp from the flag events if stamped, else from geometry."""
    evs = arc.get("flag_events", [])
    for f in evs:
        if "inside_ramp" in f:
            return bool(f["inside_ramp"])
        if "issuance_d_nm" in f:
            return (f["issuance_d_nm"] < RAMP_D_NM
                    and abs(f.get("issuance_dfl", 0.0)) < RAMP_DFL)
    return None    # no issuance geometry logged


def analyze(paths):
    ft = FieldTracker()
    arcs, boundaries = [], []
    per_ep = {"armed_win_count": [], "w_rec_paid_total": [],
              "tolerance_invoked": [], "bonus_override": [],
              "in_band_tolerance_flips": []}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "episode" not in rec:
                    continue
                ft.n_records += 1
                arcs.extend(ft.get(rec, "risk_episodes", []) or [])
                boundaries.extend(ft.get(rec, "boundaries", []) or [])
                for k in per_ep:
                    v = ft.get(rec, k)
                    if v is not None:
                        per_ep[k].append(v)

    print("=" * 74)
    print("M6 RECOVERY TELEMETRY REPORT (scaffold — data side)")
    print(f"  {ft.n_records} episode records from {len(paths)} file(s)")
    print("=" * 74)

    # ---- field presence (graceful degradation) -----------------------
    print("\n  field presence:")
    for name in FIELD_MAP:
        print(f"    {name:<26} {ft.coverage_line(name)}")

    out = {"n_records": ft.n_records,
           "field_presence": dict(ft.present)}

    # ---- risk-episode / arc aggregates -------------------------------
    if arcs:
        n = len(arcs)
        resolved = [a for a in arcs if a.get("resolved")]
        armed = [a for a in arcs if a.get("armed")]
        endo = [a for a in arcs
                if a.get("causal_would_enter_noop") is False]
        tib = [a["time_in_band_steps"] for a in arcs
               if a.get("time_in_band_steps") is not None]
        print(f"\n  risk episodes: {n} | resolved {len(resolved)} "
              f"(rate {len(resolved) / n:.3f}) | armed {len(armed)} "
              f"(armed fraction {len(armed) / n:.3f} — discriminates "
              f"'A+B insufficient' from 'A2 disarmed') | endogenous "
              f"(causal test) {len(endo)}")
        if tib:
            print(f"  time-in-band: mean {np.mean(tib):.1f} steps, "
                  f"p95 {np.percentile(tib, 95):.0f}")
        out["arcs"] = {"n": n, "resolved": len(resolved),
                       "armed": len(armed), "endogenous": len(endo),
                       "mean_time_in_band": (float(np.mean(tib))
                                             if tib else None)}

        # ---- flag coverage vs causal-test ground truth ---------------
        cov = {"inside": [0, 0], "outside": [0, 0], "unknown": [0, 0]}
        wrec_unflagged_outside = 0.0
        wrec_total = sum(a.get("w_rec_paid", 0.0) or 0.0 for a in arcs)
        for a in endo:
            where = _issuance_inside_ramp(a)
            key = ("unknown" if where is None
                   else "inside" if where else "outside")
            cov[key][1] += 1
            if _flagged(a):
                cov[key][0] += 1
            elif key == "outside":
                wrec_unflagged_outside += a.get("w_rec_paid", 0.0) or 0.0
        print("\n  FLAG COVERAGE (sign channel vs causal-test GT; "
              "pre-registered: ~0 outside ramp, partial inside):")
        for k, (hit, tot) in cov.items():
            frac = hit / tot if tot else float("nan")
            print(f"    {k:<8} ramp issuance: {hit}/{tot} flagged"
                  + (f" ({frac:.2f})" if tot else ""))
        frac_mass = (wrec_unflagged_outside / wrec_total
                     if wrec_total > 0 else 0.0)
        print(f"    W_rec mass on UNFLAGGED outside-ramp endogenous "
              f"onsets: {wrec_unflagged_outside:.2f} of "
              f"{wrec_total:.2f} total ({frac_mass:.1%}) -> "
              f"{'FAIL (mechanism-purpose failure)' if frac_mass > MATERIAL_WREC_FRAC else 'ok'}"
              f"  [draft materiality threshold "
              f"{MATERIAL_WREC_FRAC:.0%} — review must ratify]")
        out["flag_coverage"] = {k: {"flagged": h, "n": t}
                                for k, (h, t) in cov.items()}
        out["wrec_unflagged_outside"] = wrec_unflagged_outside
        out["wrec_total"] = wrec_total
    else:
        print("\n  risk episodes: NONE LOGGED — "
              + ft.coverage_line("risk_episodes"))

    # ---- boundary telemetry + laundering alarm -----------------------
    if boundaries:
        nb = len(boundaries)
        floor_active = [b for b in boundaries
                        if b.get("q_target_pre_floor") is not None
                        and b.get("q_target_post_floor") is not None
                        and b["q_target_post_floor"]
                        != b["q_target_pre_floor"]]
        reentry = [b for b in boundaries
                   if b.get("band_reentry_within_nstep")]
        joint = [b for b in floor_active
                 if b.get("band_reentry_within_nstep")]
        livecl = [b for b in boundaries if b.get("live_clearance")]
        fa_live = [b for b in livecl if b in floor_active]
        m_all = [m for b in boundaries
                 for m in (b.get("m_per_window") or [])]
        print(f"\n  boundaries (global K=10 collapses): {nb} "
              f"(fire-rate counter — 'per-pair resolutions nonzero, "
              f"global collapses zero' must be visible here)")
        print(f"    floor-active: {len(floor_active)}/{nb} | "
              f"band re-entry within nstep: {len(reentry)}/{nb}")
        print(f"    LAUNDERING EXPOSURE (floor-active AND re-entry): "
              f"{len(joint)}/{nb} "
              f"({len(joint) / nb:.3f}) — alarm = this rate RISING")
        if livecl:
            print(f"    floor-active on live-clearance boundaries: "
                  f"{len(fa_live)}/{len(livecl)} — alarm = failing to "
                  f"fall as belief repair lands")
        if m_all:
            print(f"    m per window: mean {np.mean(m_all):.1f}, "
                  f"min {min(m_all)}, max {max(m_all)} (effective "
                  f"credit horizon vs nstep)")
        out["boundaries"] = {
            "n": nb, "floor_active": len(floor_active),
            "reentry": len(reentry),
            "laundering_joint_rate": len(joint) / nb,
            "live_clearance": len(livecl),
            "floor_active_live_clearance": len(fa_live)}
    else:
        print("\n  boundaries: NONE LOGGED — "
              + ft.coverage_line("boundaries"))

    # ---- per-episode counters ----------------------------------------
    print("\n  per-episode counters:")
    for k, vals in per_ep.items():
        if vals:
            line = (f"total {sum(vals)}, mean/ep {np.mean(vals):.2f} "
                    f"over {len(vals)} eps")
            if k == "in_band_tolerance_flips":
                line += (" -> HARD BAR (== 0): "
                         + ("MEET" if sum(vals) == 0 else "MISS"))
        else:
            line = ft.coverage_line(k)
        print(f"    {k:<26} {line}")
        out[k] = sum(vals) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser(description="M6 recovery telemetry "
                                             "report (scaffold)")
    ap.add_argument("--jsonl", type=str, nargs="+", required=True,
                    help="battery / bcd per-episode JSONL file(s)")
    args = ap.parse_args()
    analyze(args.jsonl)


if __name__ == "__main__":
    main()
