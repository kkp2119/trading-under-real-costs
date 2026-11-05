"""M10 verdict boundaries (SPEC 11; E1-45 (b): the MDE comes from the observation days)."""
import judge
import stats

MDE = stats.mde(1441)


def v(x, a, b):
    return judge.verdict(x, a, b, "B0", 69, MDE, 1441 / 252)


def test_boundaries():
    assert v(0.6, 0.0, 1.2)[0] == "c"                                # a > 0 is strict
    code, s = v(0.5, 0.01, 1.0)
    assert code == "a" and "확정적이지 않다" in s and f"{MDE:.2f}" in s
    assert v(0.49, 0.01, 1.0)[0] == "c"                              # x >= 0.5 needed
    assert v(-0.5, -1.0, 0.5)[0] == "c"                              # b < 0.5 is strict
    assert v(-0.5, -1.0, 0.4999)[0] == "b"
    assert "확정적이지 않다" not in v(MDE, 0.3, 2.0)[1]                # x = MDE: no caveat
    assert "확정적이지 않다" in v(MDE - 1e-9, 0.3, 2.0)[1]


def test_every_sentence_ends_with_the_contamination_clause():
    for x, a, b in ((2.0, 1.0, 3.0), (-1.0, -2.0, 0.1), (0.3, -0.5, 1.1)):
        s = v(x, a, b)[1]
        assert s.endswith(judge.suffix(69)) and "탐색적 증거" in s and "설정 69개 중 1개" in s


def test_side_line_ci_low_above_sr_star():
    lines = judge.side_lines(0.51, 2.0, 1.0, {"S": 0.1}, -0.001, 0.2, -0.3)
    assert lines[0].endswith("예") and judge.side_lines(0.5, 2, 1, {}, 0, 0, 0)[0].endswith("아니오")
