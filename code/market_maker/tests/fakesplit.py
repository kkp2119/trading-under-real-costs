"""Synthetic split folders (MANIFEST + npz with the 21 int32 columns) under a temporary base, for loader tests."""
import hashlib
import os

import numpy as np
import pandas as pd

from synth import COLS, random_day

MAN = ["sym", "date", "file", "rows", "bytes", "sha256", "n_quote_rows", "n_trade_rows", "last_trade_sec"]


def write_split(base, split, asset, days, lot=1, seed=0, mutate=None, vol_scale=1):
    """days: [(sym, YYYYMMDD, bars or None)]; bars None -> random_day. Sizes divided by `lot` (lot units).
    mutate(sym, day, arrays) may corrupt the arrays before saving; vol_scale multiplies every volume column (the
    identity holds). Returns the folder."""
    root = os.path.join(base, f"bars_1s_{split}{'_stocks' if asset == 'stocks' else ''}")
    rows = []
    for i, (sym, day, b) in enumerate(days):
        b = random_day(seed + i) if b is None else b
        if vol_scale != 1:
            b = {c: (v * vol_scale if c.startswith("vol") else v) for c, v in b.items()}
        arr = {c: b[c].astype(np.int32) for c in COLS}
        arr["bsz"] = (b["bsz"] // lot).astype(np.int32)
        arr["asz"] = (b["asz"] // lot).astype(np.int32)
        if mutate:
            arr = mutate(sym, day, arr)
        rel = os.path.join(day[:4], f"{sym}_{day}.npz")
        os.makedirs(os.path.join(root, day[:4]), exist_ok=True)
        with open(os.path.join(root, rel), "wb") as f:
            np.savez_compressed(f, **arr)
        h = hashlib.sha256(open(os.path.join(root, rel), "rb").read()).hexdigest()
        nq = int((b["n_upd"] > 0).sum() + (b["bid"] > 0).any())
        rows.append({"sym": sym, "date": day, "file": rel, "rows": 23400, "bytes": 0, "sha256": h,
                     "n_quote_rows": nq, "n_trade_rows": int(b["ntr"].sum()), "last_trade_sec": -1})
    pd.DataFrame(rows, columns=MAN).to_csv(os.path.join(root, "MANIFEST.tsv"), sep="\t", index=False)
    return root
