"""M15 universe selection (SPEC 2.2): the screen's criteria boundaries through universe_rolling.screen with a fake DB
(writes go to a temporary folder), the seed 12 + Y order (2018: the saved draw order), the first-day rule (FCB -> AKAM)
and selection with a fake DB. Only ticker / flag columns of the saved universe files are read."""
import importlib
import os
import re
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import wrds  # noqa: F401
except ImportError:                                                  # the module only needs the name at import time
    sys.modules["wrds"] = types.ModuleType("wrds")
ur = importlib.import_module("universe_rolling")

X = 2030
DATES = [f"{X}{m:02d}15" for m in range(1, 13)]
BASE = dict(price=50.0, dv=2e7, trades=5000, spread=5.0, days=12)
CASES = {"P10": dict(price=10.0), "P9": dict(price=9.99), "P150": dict(price=150.0), "P151": dict(price=150.01),
         "S3": dict(spread=3.0), "S2": dict(spread=2.99), "S12": dict(spread=12.0), "S13": dict(spread=12.01),
         "D5": dict(dv=5e6), "D4": dict(dv=4.99e6), "DH": dict(dv=1e8), "DHP": dict(dv=1.0001e8),
         "T3K": dict(trades=3000), "T2K": dict(trades=2999), "N10": dict(days=10), "N9": dict(days=9), "OK": {},
         "OKB": {}, "OKC": {}}
PASS = {"P10", "P150", "S3", "S12", "D5", "DH", "T3K", "N10", "OK", "OKB", "OKC"}


class FakeDB:
    def __init__(self, first_day_missing=()):
        self.missing = set(first_day_missing)

    def raw_sql(self, sql):
        m = re.search(r"table_name ~ '\^(\w+)_(\d{4})", sql)
        if m:
            kind, y = m.group(1), m.group(2)
            days = DATES if y == str(X) else [f"{y}0102", f"{y}0103"]
            return pd.DataFrame({"table_name": [f"{kind}_{d}" for d in days]})
        if sql.startswith("select symbol_15"):
            return pd.DataFrame({"symbol_15": list(CASES) + ["AAPL"]})
        d = re.search(r"_(\d{8})\b", sql).group(1)
        if "count(*) as n from" in sql:
            syms = re.findall(r"'([A-Z0-9]+)'", sql.split("sym_root in")[-1])
            return pd.DataFrame({"sym": [s for s in syms if s not in self.missing], "n": 1})
        rows = [(s, {**BASE, **CASES[s]}) for s in CASES if DATES.index(d) < {**BASE, **CASES[s]}["days"]]
        if "n_trades" in sql:
            return pd.DataFrame({"sym": [s for s, _ in rows], "n_trades": [v["trades"] for _, v in rows],
                                 "dollar_volume": [v["dv"] for _, v in rows], "mean_price": [v["price"] for _, v in rows]})
        return pd.DataFrame({"sym": [s for s, _ in rows], "n_quotes": 100, "spread_ticks": [v["spread"] for _, v in rows]})


def test_screen_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(ur, "HERE", str(tmp_path))
    passing = ur.screen(FakeDB(), X, lambda m: None)
    assert set(passing) == PASS
    s = pd.read_csv(tmp_path / f"universe_screen_{X}.tsv", sep="\t")
    assert "AAPL" not in set(s.sym)                                 # exclusion list


def test_draw_order_and_selection_with_first_day_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(ur, "HERE", str(tmp_path))
    order = ur.draw_order(sorted(PASS), 12 + X + 1)
    assert order == ur.draw_order(sorted(PASS), 12 + X + 1) and sorted(order) == sorted(PASS)
    syms = ur.select(FakeDB(first_day_missing={order[1]}), X, X + 1, order, 12 + X + 1, lambda m: None)
    assert len(syms) == 10 and syms == [s for s in order if s != order[1]][:10]


def test_saved_2018_order_reproduces_and_fcb_is_replaced_by_akam():
    scr = pd.read_csv(os.path.join(ROOT, "universe_screen_2018.tsv"), sep="\t", usecols=["sym", "pass_all"])
    passing = sorted(scr.sym[scr.pass_all])
    saved = pd.read_csv(os.path.join(ROOT, "universe_draw_order_2018.tsv"), sep="\t",
                        usecols=["order_rank", "sym", "has_trades", "has_quotes", "selected"])
    assert ur.draw_order(passing, 12) == saved.sort_values("order_rank").sym.tolist()
    first10 = saved[saved.has_trades & saved.has_quotes].sort_values("order_rank").sym.tolist()[:10]
    u19 = pd.read_csv(os.path.join(ROOT, "universe_2019.tsv"), sep="\t").sym.tolist()
    assert u19 == first10 and "FCB" not in u19 and u19[-1] == "AKAM"
    assert not bool(saved.set_index("sym").loc["FCB", "selected"])
