#!/usr/bin/env python3
"""Mutation check for M1-M7 (Kyle 2026-10-27): copy the engine and tests to $R12_TEST_DIR/<id>/, inject one deliberate
defect, run the test file of that M-item and require it to FAIL. Prints a table. Not collected by pytest.
  R12_TEST_DIR=... python research12/tests/run_mutations.py"""
import os
import shutil
import subprocess
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MUTANTS = [  # (M item, test file, module, original text, defect, description)
    ("M1", "test_m01_lookahead.py", "features.py", "s, d = bsz + asz, bsz - asz",
     "s, d = np.roll(bsz, -1) + asz, np.roll(bsz, -1) - asz", "OBI reads next bar's bid size (lookahead 1 bar)"),
    ("M1", "test_m01_lookahead.py", "simulator.py", "        pb = bid[u]\n", "        pb = bid[min(u + 1, df)]\n",
     "buy price from the next bar (lookahead 1 bar)"),
    ("M2", "test_m02_fills.py", "fillmodel.py", "if bid_max[u] > 0 and bid_max[u] < price:",
     "if bid_max[u] > 0 and bid_max[u] <= price:", "Q case 2 inequality <= instead of <"),
    ("M2", "test_m02_fills.py", "fillmodel.py", "if ask_min[u] > 0 and ask_min[u] <= price:",
     "if ask_min[u] > 0 and ask_min[u] < price:", "cross needs ask strictly below the bid"),
    ("M2", "test_m02_fills.py", "fillmodel.py", "    credit = vol_bid[u] - vol_bid_odd[u]",
     "    credit = vol_bid[u]", "odd lots move the queue"),
    ("M3", "test_m03_inventory.py", "simulator.py", "        b_size = min(Q, LQ - q)", "        b_size = min(Q, LQ - q + 1)",
     "inventory limit off by one share"),
    ("M3", "test_m03_inventory.py", "simulator.py", "u - hit >= stale_s", "u - hit > stale_s",
     "stale liquidation one decision late at the boundary"),
    ("M4", "test_m04_market.py", "simulator.py", "        if u == df:\n            b_rem = 0",
     "        if u == df - 1:\n            b_rem = 0", "flatten one bar early"),
    ("M4", "test_m04_market.py", "fillmodel.py", "if bid_min[u] > 0 and bid_min[u] < bid[dec]:",
     "if bid_min[u] > 0 and bid_min[u] > bid[dec]:", "market sell takes the better price"),
    ("M5", "test_m05_accounting.py", "accounting.py", 'out["inv2"] = int(np.sum(-s * qty * ref))',
     'out["inv2"] = int(np.sum(-s * qty * 2 * p))', "inventory P&L marked at the fill price"),
    ("M5", "test_m05_accounting.py", "config.py", 'COST = {"R": (30, 0), "R+": (30, -20), "C": (100, -50)}',
     'COST = {"R": (30, 0), "R+": (30, 20), "C": (100, -50)}', "R+ rebate charged as a fee"),
    ("M6", "test_m06_markout.py", "accounting.py", "        t = u[li] + k\n        good = t < close_bar",
     "        t = u[li] + k\n        good = t <= close_bar", "markout past the close kept"),
    ("M6", "test_m06_markout.py", "accounting.py",
     'out[f"mk{k}_sum2"] = int(np.sum(s[j] * qty[j] * (mid2[u[j] + k] - 2 * p[j])))',
     'out[f"mk{k}_sum2"] = int(np.sum(s[j] * qty[j] * (mid2[u[j] + k - 1] - 2 * p[j])))', "markout one bar early"),
    ("M7", "test_m07_event.py", "simulator.py", "if mode == 1 and buy_ok and b_size > 0 and b_rem > 0 and b_px == pb:",
     "if mode == 1 and buy_ok and b_size > 0 and b_rem > 0:", "event keeps a buy after the best moved"),
    ("M7", "test_m07_event.py", "simulator.py", "                b_px, b_rem, b_a4, b_dec, b_hit, b_slot = pb, b_size, theta4 * bsz[u], u, 0, no",
     "                b_px, b_rem, b_a4, b_dec, b_hit, b_slot = pb, b_size, theta4 * bsz[u] + 4, u, 0, no",
     "new buy queue one share too long"),
]


def copy_tree(dst):
    os.makedirs(os.path.join(dst, "tests"))
    os.makedirs(os.path.join(dst, "bars1s"))
    for f in os.listdir(ROOT):
        if f.endswith(".py"):
            shutil.copy(os.path.join(ROOT, f), dst)
    for f in ("calendar.csv",):
        if os.path.exists(os.path.join(ROOT, f)):
            shutil.copy(os.path.join(ROOT, f), dst)
    shutil.copy(os.path.join(ROOT, "bars1s", "bars.py"), os.path.join(dst, "bars1s"))
    for f in os.listdir(HERE):
        if f.endswith(".py"):
            shutil.copy(os.path.join(HERE, f), os.path.join(dst, "tests"))


def main():
    base = os.environ.get("R12_TEST_DIR")
    if not base:
        raise SystemExit("set R12_TEST_DIR")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    print(f"{'item':4s}  {'defect':52s} {'test':26s} result")
    rows = []
    for item, test, mod, old, new, desc in MUTANTS:
        d = os.path.join(base, "mut_" + uuid.uuid4().hex[:8])
        copy_tree(d)
        p = os.path.join(d, mod)
        src = open(p).read()
        assert src.count(old) == 1, (mod, old)
        open(p, "w").write(src.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-x", os.path.join("tests", test)],
                           cwd=d, env=env, capture_output=True, text=True)
        caught = r.returncode != 0
        rows.append((item, desc, test, caught))
        print(f"{item:4s}  {desc:52s} {test:26s} {'caught (test failed)' if caught else 'NOT CAUGHT'}")
    print(f"caught {sum(r[3] for r in rows)}/{len(rows)}")
    return rows


if __name__ == "__main__":
    main()
