"""Bar loader with hash checks and the holdout lock (SPEC 2.1, 17; M11). The holdout and reserve folders are not listed
or read unless results/HOLDOUT_UNLOCKED exists (checked before any path in them is touched). Every file is checked
against its MANIFEST sha256, the 21 int32 columns of 23,400 rows, the split's date range and the volume identity."""
import hashlib
import os

import numpy as np
import pandas as pd

from config import COLS, DATA, LOT_FACTOR, NSEC, RESULTS, SPLITS, UNLOCK


class Refused(Exception):
    pass


def folder(split, asset, base=None):
    return os.path.join(base or DATA, f"bars_1s_{split}{'_stocks' if asset == 'stocks' else ''}")


def check_unlock(split, results_dir=None):
    if split in ("holdout", "reserve") and not os.path.exists(os.path.join(results_dir or RESULTS, UNLOCK)):
        raise Refused(f"{split} is locked: no {UNLOCK}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class Split:
    """One split folder (dev/holdout/reserve, stocks/etf). meta: {(sym, day): MANIFEST row as dict of str}."""

    def __init__(self, split, asset, base=None, results_dir=None, lot_factor=LOT_FACTOR):
        check_unlock(split, results_dir)                               # before touching the folder
        self.split, self.asset, self.lot_factor = split, asset, lot_factor
        self.root = folder(split, asset, base)
        lo, hi = SPLITS[split]
        m = pd.read_csv(os.path.join(self.root, "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False)
        bad = m[(m.date < lo) | (m.date > hi)]
        if len(bad):
            raise Refused(f"{len(bad)} MANIFEST rows outside {split} {lo}..{hi}")
        self.meta = {(r["sym"], r["date"]): r for r in m.to_dict("records")}
        p = os.path.join(self.root, "skipped_absent.tsv")
        a = pd.read_csv(p, sep="\t", dtype=str) if os.path.exists(p) else pd.DataFrame({"sym": [], "date": []})
        self.absent = set(zip(a.sym, a.date))

    def available(self):
        return set(self.meta)

    def zero_days(self):
        return {k for k, r in self.meta.items() if int(r["n_quote_rows"]) == 0 and int(r["n_trade_rows"]) == 0}

    def load(self, sym, day):
        """21 columns as int64 (bsz, asz x lot_factor). Refuses on any check failure."""
        r = self.meta.get((sym, day))
        if r is None:
            raise Refused(f"{sym} {day}: not in MANIFEST")
        path = os.path.join(self.root, r["file"])
        if sha256(path) != r["sha256"]:
            raise Refused(f"{sym} {day}: sha256 differs from MANIFEST")
        with np.load(path, allow_pickle=False) as z:
            if sorted(z.files) != sorted(COLS):
                raise Refused(f"{sym} {day}: columns {sorted(z.files)}")
            b = {}
            for c in COLS:
                a = z[c]
                if a.dtype != np.int32 or a.shape != (NSEC,):
                    raise Refused(f"{sym} {day}: {c} {a.dtype} {a.shape}")
                b[c] = a.astype(np.int64)
        if not np.array_equal(b["vol"], b["vol_bid"] + b["vol_ask"] + b["vol_in"] + b["vol_unc"]):
            raise Refused(f"{sym} {day}: vol identity fails")
        b["bsz"] = b["bsz"] * self.lot_factor
        b["asz"] = b["asz"] * self.lot_factor
        return b

    def verify_all(self):
        """Hash every MANIFEST file (holdout pre-check, SPEC 17 step 4). Returns the list of mismatching keys."""
        return [k for k, r in sorted(self.meta.items()) if sha256(os.path.join(self.root, r["file"])) != r["sha256"]]
