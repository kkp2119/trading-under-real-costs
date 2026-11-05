"""SPEC 11 verdict sentences and the G1 gate (SPEC 9; E1-37, E1-38, E1-45 (b): the MDE in the sentence is computed from
the holdout's observation days)."""
from config import G1_QUOTABLE, G1_SHARPE, SR_STAR
from stats import classify

SUBJECT = "개발 구간에서 선택한 시장조성 설정은 2021~2026년 홀드아웃에서 비용 후"
NOUN = "개발 구간에서 선택한 시장조성 설정의 2021~2026년 홀드아웃 비용 후 수익"


def suffix(n_trials):
    return (f"(대기열 모델이 앞 수량의 취소를 무시한 보수적 체결 가정이고, 지정가 리베이트가 없는 실제 비용 가정이며, 개발 구간에서 "
            f"설정 {n_trials}개 중 1개를 골랐다. 같은 시장·기간의 일간·월간 수익률 결과를 이미 본 구간, 탐색적 증거)")


def verdict(x, a, b, setting, n_trials, mde, years):
    """(code, sentence) for point x and 95% CI [a, b] (Q, R)."""
    code = classify(x, a, b)
    ci_txt = f"순 샤프 {x:.2f}, 95% CI [{a:.2f}, {b:.2f}]"
    if code == "a":
        s = (f"{SUBJECT}에도 경제적으로 의미 있는 수준으로 남는다({ci_txt}). 선택한 설정은 {setting}이다.")
        if x < mde:
            s += f" 다만 이 표본({years:.2f}년)으로 검출 가능한 최소 샤프({mde:.2f})보다 작아 확정적이지 않다."
    elif code == "b":
        s = f"{NOUN}은 경제적으로 의미 있는 수준(샤프 {SR_STAR})에 못 미친다({ci_txt})."
    else:
        s = f"이 표본으로는 {NOUN}이 경제적으로 의미 있는 수준인지 확정할 수 없다({ci_txt}, 검출 가능한 최소 샤프 {mde:.2f})."
    return code, f"{s} {suffix(n_trials)}"


def side_lines(a, dev_sr, hold_sr, model_sr, breakeven, b0_sr, naive_sr):
    """SPEC 11 separate report lines (do not change the verdict)."""
    return [f"CI 하한 > SR*: {'예' if a > SR_STAR else '아니오'}",
            f"개발 샤프 대비 홀드아웃 샤프(선택 낙관 정도): 개발 {dev_sr:.2f}, 홀드아웃 {hold_sr:.2f}, 차이 {dev_sr - hold_sr:.2f}",
            "체결 모델·시나리오 순 샤프: " + ", ".join(f"{k} {v:.2f}" for k, v in model_sr.items()),
            f"손익분기 리베이트: {breakeven:.4f} 달러/주 (0 이하면 필요 없음)",
            f"B0 홀드아웃 순 샤프 {b0_sr:.2f}, A_naive {naive_sr:.2f} (기술통계)"]


def gate_g1(dev_sharpe, nested_ok, quotable_ratio, flat_ok, symbols_with_fills, symbols):
    """SPEC 9 G1 (with E1-37: 2 = per-order fill nesting on a common order stream, E1-38: 3 = data-condition ratio)."""
    g = {"1_sharpe": bool(dev_sharpe == dev_sharpe and dev_sharpe >= G1_SHARPE),
         "2_fill_nesting": bool(nested_ok),
         "3_quotable": bool(quotable_ratio >= G1_QUOTABLE),
         "3_flat": bool(flat_ok),
         "3_every_symbol_filled": bool(set(symbols) <= set(symbols_with_fills))}
    g["pass"] = all(g.values())
    return g
