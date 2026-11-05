#!/usr/bin/env python3
"""Research 15: tick size pilot test-group symbols (Kyle decision 1, 2026-10-29) from the FINRA files in data_meta/.
No network access; the files were downloaded once (data_meta/SOURCES.tsv gives URLs and sha256).
  Tick_Pilot_Test_Group_Assignments.txt  consolidated exchange list with the planned groups (END_FILE 20161028)
  TSPilotChanges.txt                     daily change list 2016-09-08 .. 2019-04-01 (rollouts 'R' 2016-10-03 .. 10-31,
                                         rollbacks 'R' 2018-10-01, deletions, symbol changes)
A symbol is a test-group symbol when, in either file, it carries group G1, G2 or G3 (new or old group), or it is the
Old_Ticker_Symbol of a change row whose group is G1, G2 or G3. Exchange test symbols (ATEST/NTEST/Z*ZZT) are dropped.
Matching to the TAQ universe is by ticker only. Writes data_meta/tsp_test_group_symbols.tsv and prints counts."""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "data_meta")
OUT = os.path.join(META, "tsp_test_group_symbols.tsv")
G = {"G1", "G2", "G3"}
TEST_SYMS = r"^(ATEST|NTEST|Z[A-Z]ZZT)"


def read(name):
    t = pd.read_csv(os.path.join(META, name), sep="|", dtype=str, keep_default_na=False)
    return t[t.iloc[:, 0] != "END_FILE"]


def test_group_symbols():
    a = read("Tick_Pilot_Test_Group_Assignments.txt")
    c = read("TSPilotChanges.txt")
    rows = [(r.Ticker_Symbol, r.Tick_Size_Pilot_Program_Group, r.Effective_Date, "assignment")
            for r in a.itertuples() if r.Tick_Size_Pilot_Program_Group in G]
    for r in c.itertuples():
        for grp in {r.Tick_Size_Pilot_Program_Group, r.Old_Tick_Size_Pilot_Program_Group} & G:
            rows.append((r.Ticker_Symbol, grp, r.Effective_Date, f"changes:{r.Reason_for_Change}"))
            if r.Old_Ticker_Symbol:
                rows.append((r.Old_Ticker_Symbol, grp, r.Effective_Date, f"changes:old_ticker:{r.Reason_for_Change}"))
    t = pd.DataFrame(rows, columns=["sym", "group", "effective_date", "source"])
    t = t[~t.sym.str.match(TEST_SYMS)]
    return (t.groupby("sym").agg(groups=("group", lambda x: ",".join(sorted(set(x)))),
                                 first_effective=("effective_date", "min"), last_effective=("effective_date", "max"),
                                 sources=("source", lambda x: ",".join(sorted(set(x)))))
            .reset_index().sort_values("sym"))


def main():
    t = test_group_symbols()
    t.to_csv(OUT, sep="\t", index=False)
    print(f"test-group symbols: {len(t)} -> {OUT}; by group: "
          + ", ".join(f"{g} {int(t.groups.str.contains(g).sum())}" for g in sorted(G))
          + f"; more than one group: {int(t.groups.str.contains(',').sum())}")


if __name__ == "__main__":
    main()
