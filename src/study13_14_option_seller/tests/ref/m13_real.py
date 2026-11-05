"""M13 on real development data (Kyle permission 2026-10-28; memo 4b v2 10; A13-3 change 7, E13-15). Five
non-overlapping 3-month windows inside the development statistics window (1996-03-06..2012-12-31), start months drawn
with seed 13, empty ladder at each window start, settings O0..O10, scenarios base/stress/opt.
Printed and written: per window and setting the passing and failing day counts and the largest relative error; on the
first failure only the date, the setting, the differing scenarios, the relative error and the per-event-type count
difference of that day, then stop. No P&L level, sum or statistic is printed or written. Development files only."""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(os.path.dirname(HERE)), HERE]
import config13 as C  # noqa: E402
import ladder  # noqa: E402
import market  # noqa: E402
import refsim13  # noqa: E402
from compare_ref import choose_windows  # noqa: E402

STATS_START = 19960306
SCEN = ("base", "stress", "opt")


def main():
    paths = []
    cal_src = market.RealSource()
    days = [int(d) for d in cal_src.calendar("dev").days if STATS_START <= d <= C.DEV_END]
    paths += cal_src.paths()
    wins = choose_windows(days)
    ids = [s.id for s in C.SETTINGS]
    out = [f"windows (seed 13): {wins}", "window\tsetting\tpass_days\tfail_days\tmax_rel_err"]
    print(out[0])
    for a, b in wins:
        src_e, src_r = market.RealSource(), market.RealSource()
        mkt = market.Market(src_e, "SPX", ("dev",))
        eng = ladder.run(mkt, C.SETTINGS, a, b)
        trace = []
        states = refsim13._run(src_r, ids, a, b, trace=trace)
        ref = {st.sid: refsim13.pd.DataFrame(st.rows, columns=["date", "P_base", "P_stress", "P_opt"]) for st in states}
        trace = [t for t in trace if t[0] != "day"]                    # events only; day rows carry values
        paths += src_e.paths() + src_r.paths()
        for sid in ids:
            e, r = eng[sid].ledger.frame(), ref[sid]
            if list(e.date) != list(r.date):
                print(f"STOP: dates differ in {a}-{b} {sid}")
                return 1
            fail, worst = 0, 0.0
            for i in range(len(e)):
                bad = []
                for sc in SCEN:
                    x, y = float(e[f"P_{sc}"].iloc[i]), float(r[f"P_{sc}"].iloc[i])
                    rel = abs(x - y) / max(abs(x), abs(y), 1e-12)
                    if abs(x - y) > 1e-6:
                        worst = max(worst, rel)
                    if abs(x - y) > 1e-6 * max(abs(x), abs(y)) + 1e-6:
                        bad.append((sc, rel))
                if bad:
                    d = int(e.date.iloc[i])
                    ev_e = collections.Counter(x["event"] for x in eng[sid].ledger.legs if x["date"] == d)
                    ev_r = collections.Counter({"open": "open", "close": "normal", "fallback_exit": "kappa"}.get(t[0], t[0])
                                               for t in trace if t[1] == sid and t[2] == d)
                    print(f"STOP: first failure {a}-{b} {sid} date {d} scenarios {[s for s, _ in bad]} "
                          f"rel_err {max(x for _, x in bad):.3e} events engine {dict(ev_e)} reference {dict(ev_r)}")
                    return 1
            line = f"{a}-{b}\t{sid}\t{len(e) - fail}\t{fail}\t{worst:.3e}"
            out.append(line)
        print(f"window {a}-{b}: {len(ids)} settings x 3 scenarios compared, failures 0")
    hold = sum("data_holdout" in p for p in paths)
    out.append(f"read paths {len(set(paths))}, holdout paths {hold}")
    with open(os.path.join(HERE, "M13_real_result.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(HERE, "M13_real_read_paths.txt"), "w") as f:
        f.write("\n".join(sorted(set(paths))) + "\n")
    print("\n".join(out[1:]))
    return 0 if hold == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
