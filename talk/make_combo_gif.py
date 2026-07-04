"""Generate the combo-OOD comparison GIFs.

Wind=25 + gravity=-11.99 — the hardest condition in the eval grid for both
controllers. Same methodology as `make_progression_comparison.py`:
  • scalar DQN: single existing seed (seed42), median across rollout seeds
  • credal interval DQN: median across model_seeds × rollout_seeds, adaptive c

Output: gifs/PC_std_combo.gif and gifs/PC_credal_combo.gif.
"""

from __future__ import annotations
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_progression_comparison import (
    find_typical_standard, find_typical_credal, save_gif, OUT_DIR
)


def main():
    rs_pool = [1, 3, 5, 7, 11, 17, 23, 29, 31]
    cred_seeds = [0, 1, 2, 3, 4]

    wind, gravity = 25.0, -11.99

    print(f"=== Combo OOD: wind={wind}, gravity={gravity} ===")

    print("  -- Standard DQN --")
    frames_std, R_std = find_typical_standard(wind, gravity, rs_pool, label="std combo")
    save_gif(frames_std, OUT_DIR / "PC_std_combo.gif")

    print("  -- Credal interval DQN (adaptive c, c_train=0.5) --")
    frames_cred, R_cred = find_typical_credal("0.5", wind, gravity,
                                                cred_seeds, rs_pool,
                                                label="credal combo")
    save_gif(frames_cred, OUT_DIR / "PC_credal_combo.gif")

    print(f"\nDone. Rewards: std={R_std:.1f}  credal={R_cred:.1f}")
    for f in sorted(OUT_DIR.glob("PC_*combo*.gif")):
        print(f"  {f.name}: {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
