"""
C-INSTRUMENTS LOCK regression test (ROUND3 re-review, [medium] "C
instruments lock (tolerance x pre-registered bars)")
====================================================================

Asserts that the four locked probe instruments —
    probe_argmax / noop_argmax_fraction   (micro_battery_common)
    rho_eval                              (micro_level_allocation)
    rho_eval_subset                       (micro_battery_common)
    the M2-XL read                        (micro_battery_m2xl.state_read)
plus the round-3 additions (noop_level_hits, round3_instruments.
ordering_read) — are BIT-IDENTICAL under ANY NOOP tie-break tolerance
value. They read candidate_q directly and must never inherit
selection-path modifications; the amendment exists to stop a future
implementer from mirroring the tolerance into probe_argmax (whose old
docstring invited it).

Two lock mechanisms, both must hold:
  (1) tolerance sweep: every plausible tolerance attribute (agent-level
      and bcd module-level, the same probe list the dual-mode reporting
      hook uses) is set to values from 0 to 1e9 — outputs must not move
      by a single bit;
  (2) selection-path poisoning: select_candidate / generate_action are
      replaced with functions that RAISE — the instruments must still
      run, proving they never route through the selection path (where
      the tolerance and the count bonus will live).

Run:  .venv/bin/pytest test_c_instruments_lock.py -q
  or: .venv/bin/python test_c_instruments_lock.py
"""

import numpy as np
import torch

import bluebird_controller_dqn as bcd
from bluebird_controller_dqn import ControllerAgent, N_INSTR
from micro_battery_common import (
    probe_argmax, noop_argmax_fraction, noop_level_hits, rho_eval_subset,
    _AGENT_TOL_ATTRS,
)
from micro_level_allocation import rho_eval
from micro_battery_m2xl import state_read
from micro_battery_m3 import lateral_keep
from round3_instruments import ordering_read

TOKEN_DIM = 60
C_TRAIN = 0.5
TOL_VALUES = (0.0, 0.1, 0.7, 5.0, 1e9)
BCD_TOL_GLOBALS = ("NOOP_TOLERANCE", "NOOP_TIE_TOLERANCE",
                   "TIE_BREAK_TOLERANCE")


def _synthetic_probe(n_states=6, n_ac=2, seed=1234):
    """Synthetic frozen probe entries (random tokens + GT): the lock is
    about the NET-READ path, not about GT quality, so no env rollouts
    are needed and the test stays sub-second."""
    rng = np.random.default_rng(seed)
    probe = []
    n_cand = 1 + N_INSTR * n_ac
    for i in range(n_states):
        gt = rng.normal(0.0, 10.0, size=n_cand)
        probe.append({"seed": 100 + i // 3, "step": i, "stratum": i % 3,
                      "h": 15, "n_cand": n_cand,
                      "cs_list": [f"AIR-0{k}" for k in range(n_ac)],
                      "tokens": rng.normal(0, 1, size=(n_ac, TOKEN_DIM))
                      .astype(np.float32),
                      "gt": gt, "gt_spread": float(gt.max() - gt.min())})
    return probe


def _make_agent():
    torch.manual_seed(4242)
    agent = ControllerAgent(token_dim=TOKEN_DIM, n_instr=N_INSTR,
                            device="cpu")
    agent.q_net.eval()
    return agent


def _read_all(agent, probe):
    """One tuple of every locked instrument's full output."""
    argmaxes = tuple(probe_argmax(agent, p, C_TRAIN) for p in probe)
    naf = noop_argmax_fraction(agent, probe, C_TRAIN)
    nlh_frac, nlh_devs = noop_level_hits(agent, probe, C_TRAIN)
    re = rho_eval(agent, probe, C_TRAIN)
    res = rho_eval_subset(agent, probe, C_TRAIN, lateral_keep)
    m2xl = tuple(state_read(agent, np.asarray(p["tokens"]), p["n_cand"],
                            C_TRAIN) for p in probe)
    ordr = ordering_read(agent, probe, (0.0, C_TRAIN, 1.0))
    return (argmaxes, naf, nlh_frac, tuple(nlh_devs),
            tuple(sorted(re.items())), tuple(sorted(res.items())),
            m2xl, repr(ordr))


def _assert_identical(a, b, context):
    assert a == b, (f"C-INSTRUMENTS LOCK VIOLATED [{context}]: a locked "
                    f"instrument's output moved — it inherited a "
                    f"selection-path modification. Instruments must "
                    f"read candidate_q directly.")


def test_tolerance_sweep_bit_identical():
    agent = _make_agent()
    probe = _synthetic_probe()
    baseline = _read_all(agent, probe)
    saved_bcd = {n: getattr(bcd, n) for n in BCD_TOL_GLOBALS
                 if hasattr(bcd, n)}
    try:
        for val in TOL_VALUES:
            for n in _AGENT_TOL_ATTRS:
                setattr(agent, n, val)
            for n in BCD_TOL_GLOBALS:
                setattr(bcd, n, val)
            _assert_identical(_read_all(agent, probe), baseline,
                              f"tolerance={val}")
    finally:
        for n in _AGENT_TOL_ATTRS:
            if hasattr(agent, n):
                delattr(agent, n)
        for n in BCD_TOL_GLOBALS:
            if n in saved_bcd:
                setattr(bcd, n, saved_bcd[n])
            elif hasattr(bcd, n):
                delattr(bcd, n)


def test_selection_path_never_consulted():
    agent = _make_agent()
    probe = _synthetic_probe()
    baseline = _read_all(agent, probe)

    def _poisoned(*a, **kw):
        raise AssertionError(
            "locked instrument routed through the SELECTION path "
            "(select_candidate/generate_action) — the tolerance and the "
            "count bonus live there; instruments must read candidate_q")

    agent.select_candidate = _poisoned
    agent.generate_action = _poisoned
    _assert_identical(_read_all(agent, probe), baseline,
                      "selection path poisoned")


def test_probe_argmax_is_candidate_q_argmax():
    """probe_argmax == direct Hurwicz argmax over candidate_q with the
    exact-tie-to-NOOP rule — pinned so a 'helpful' rewrite through
    select_candidate changes this test's expectation visibly."""
    agent = _make_agent()
    probe = _synthetic_probe()
    for p in probe:
        cl, cu = agent.candidate_q(np.asarray(p["tokens"]))
        scores = (cl + C_TRAIN * (cu - cl))[:p["n_cand"]]
        expect = 0 if scores[0] == scores.max() else int(scores.argmax())
        assert probe_argmax(agent, p, C_TRAIN) == expect


if __name__ == "__main__":
    test_tolerance_sweep_bit_identical()
    print("PASS tolerance sweep: all locked instruments bit-identical "
          f"across tolerance values {TOL_VALUES}")
    test_selection_path_never_consulted()
    print("PASS selection-path poisoning: no locked instrument consults "
          "select_candidate/generate_action")
    test_probe_argmax_is_candidate_q_argmax()
    print("PASS probe_argmax == direct candidate_q Hurwicz argmax "
          "(exact-tie-to-NOOP)")
