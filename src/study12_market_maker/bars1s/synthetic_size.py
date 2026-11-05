#!/usr/bin/env python3
"""Synthetic baseline for the 1-second bar files: file size and build time per symbol-day for three activity levels.
Quote/trade counts follow the probe (complete_nbbo 249k-522k session rows, ctm 60k-131k). Writes the .npz files to the
directory given as argv[1] (outside the repository) and prints sizes and seconds only."""
import os
import sys
import time

import numpy as np

import bars

LEVELS = {"thin": (50_000, 5_000), "typical": (360_000, 86_000), "heavy": (520_000, 131_000)}


def day(nq, nt, seed):
    r = np.random.default_rng(seed)
    qt = np.sort(r.integers(bars.T0 * bars.NS, bars.T1 * bars.NS, nq))
    mid = 100 + np.cumsum(r.choice([-0.01, 0.0, 0.0, 0.01], nq))
    half = r.choice([0.005, 0.01, 0.015], nq, p=[0.6, 0.3, 0.1])
    tt = np.sort(r.integers(bars.T0 * bars.NS, bars.T1 * bars.NS, nt))
    j = np.clip(np.searchsorted(qt, tt) - 1, 0, nq - 1)
    side = r.choice([-1.0, 0.0, 1.0], nt, p=[0.45, 0.1, 0.45])
    return bars.build_bars(qt, mid - half, mid + half, r.integers(1, 3000, nq), r.integers(1, 3000, nq), np.ones(nq, bool),
                           tt, mid[j] + side * half[j], r.integers(1, 2000, nt), np.ones(nt, bool))


def main(out):
    os.makedirs(out, exist_ok=True)
    for name, (nq, nt) in LEVELS.items():
        t = time.time()
        b, _ = day(nq, nt, seed=12)
        sec = time.time() - t
        p = os.path.join(out, f"synthetic_{name}.npz")
        with open(p, "wb") as f:
            bars.save(f, b)
        print(f"{name}: quotes {nq}, trades {nt} -> build {sec:.2f}s, file {os.path.getsize(p):,} bytes")


if __name__ == "__main__":
    main(sys.argv[1])
