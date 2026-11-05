#!/usr/bin/env python3
"""Research 15 gate for the CRSP common-stock rule (Kyle 2026-10-29): on taqm_2016.mastm_20161230, the first year-end
master with a filled sec_type, and on taqm_2017.mastm_20171229, compare TAQ sec_type 'A' with universe15.COMMON_CRSP
(CUSIP[:8] = crsp.stocknames ncusip, the shrcd set of universe15, namedt <= master date <= nameenddt). Population:
no-suffix symbols minus universe15.EXCLUDE. Kyle's rule: agreement over all symbols below 95% -> stop and report
(15:41 ET: judged over all symbols; the first run at 01:09 ET used min(all, Jaccard) with shrcd 10/11 and stopped).
The union agreement (Jaccard) is reported only. Counts only (logs/crsp_agree15.log); no per-symbol CRSP field is written."""
import os
import sys

import wrds

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import universe15 as U  # noqa: E402

MASTERS = ((2016, "20161230"), (2017, "20171229"))
GATE = 0.95


def main():
    db = wrds.Connection()
    lines, ok = [f"rule: {U.COMMON_CRSP.split('s.shrcd')[1].split(chr(10))[0].strip()}"], True
    try:
        for x, m in MASTERS:
            allsym = set(db.raw_sql(U.ALL_NOSUFFIX.format(x=x, m=m)).symbol_15) - U.EXCLUDE
            taq = set(db.raw_sql(U.COMMON.format(x=x, m=m)).symbol_15) & allsym
            crsp = set(db.raw_sql(U.COMMON_CRSP.format(x=x, m=m, d=f"{m[:4]}-{m[4:6]}-{m[6:]}")).symbol_15) & allsym
            both, union = taq & crsp, taq | crsp
            overall = 1 - len(taq ^ crsp) / len(allsym)
            ok &= overall >= GATE
            lines += [f"mastm_{m}: no-suffix symbols minus EXCLUDE {len(allsym)}; TAQ sec_type A {len(taq)}; CRSP {len(crsp)}; "
                      f"both {len(both)}; TAQ only {len(taq - crsp)}; CRSP only {len(crsp - taq)}",
                      f"   agreement over all symbols {overall:.4f}; over the union (Jaccard, reported only) {len(both) / len(union):.4f}"]
    finally:
        db.close()
    lines.append(f"gate {GATE} on the agreement over all symbols: {'PASS' if ok else 'STOP (report to Kyle)'}")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "crsp_agree15.log"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
