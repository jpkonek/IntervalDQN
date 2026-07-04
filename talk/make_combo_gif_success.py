"""Pick a *successful* combo-OOD trial for the credal-DQN GIF.

Scans 5 credal model seeds × 9 rollout seeds = 45 trials at wind=25,
gravity=-11.99 with the same adaptive-c policy as the original. Picks
the lowest-reward trial among those that solve (reward >= 200) — a
"typical successful" landing rather than an extreme outlier. Falls back
to the highest reward if no trial solves.

Writes PC_credal_combo.gif. Leaves PC_std_combo.gif (scalar baseline,
single seed) untouched.
"""

from __future__ import annotations
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_progression_comparison import load_credal, rollout, save_gif, OUT_DIR


def main():
    rs_pool = [1, 3, 5, 7, 11, 17, 23, 29, 31]
    cred_seeds = [0, 1, 2, 3, 4]
    wind, gravity = 25.0, -11.99

    rolls = []
    for ms in cred_seeds:
        agent = load_credal("0.5", ms)
        for rs in rs_pool:
            frames, total = rollout(agent, wind, gravity, rs, adaptive=True)
            rolls.append((ms, rs, total, frames))
            print(f"  ms={ms} rs={rs:2d}  R={total:7.1f}")

    successful = [r for r in rolls if r[2] >= 200]
    print(f"\n{len(successful)} of {len(rolls)} trials solved (R >= 200)")

    if successful:
        chosen = min(successful, key=lambda r: r[2])
        label = "lowest-reward solved (typical successful)"
    else:
        chosen = max(rolls, key=lambda r: r[2])
        label = "highest reward (no trial solved)"

    print(f"\n[{label}] ms={chosen[0]} rs={chosen[1]} R={chosen[2]:.1f}")
    save_gif(chosen[3], OUT_DIR / "PC_credal_combo.gif")


if __name__ == "__main__":
    main()
