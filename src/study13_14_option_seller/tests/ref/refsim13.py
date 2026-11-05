# -*- coding: utf-8 -*-
"""M13 독립 참조 구현: 연구 13 SPX 옵션 매도 사다리의 일 손익.

근거 문서(이것만 읽고 작성, 엔진 코드는 보지 않음):
  research13/SPEC.md (3.1~3.9, 4, 5.1, 7)
  research13/SPEC_AMENDMENTS.md (A13-1, A13-2, A13-3의 E13-01~E13-18)
  research13/AUDIT_G0.md 6절 확정 스위치 표
      - delta: OptionMetrics delta (결측이면 블랙-숄즈, 그것도 없으면 전일 delta)
      - T 기준: T1 = (정산일 - 날짜)/365 (정산일은 3.5절, 달력 끝 규칙 포함)
      - 비표준 expiry_indicator 코드: ['m', 'w']
  research13/design/ENGINE_DESIGN_13A_DATA.md 1~3절, ENGINE_DESIGN_13B.md 10절
  research13/tests/ref/BRIEF.md

해석을 고른 곳은 같은 폴더의 REF_NOTES.md에 번호(R-xx)로 적었고, 코드 주석에 같은 번호를 단다.

공개 함수:
    simulate(source, setting_ids, start, end)
        -> {setting_id: DataFrame(columns=["date", "P_base", "P_stress", "P_opt"])}
"""

import datetime as _dt
import math

import numpy as np
import pandas as pd
from scipy.special import ndtr

# ---------------------------------------------------------------------------
# 상수 (SPEC 3.2, 3.4, 3.5, 3.7, 3.8, 3.9, 4, 7절, AUDIT_G0 6절)
# ---------------------------------------------------------------------------

N0 = 1_000_000.0               # 3.2 기준 노셔널
UND = "SPX"
SPLIT = "dev"                  # BRIEF 4절: 이 비교는 dev 달력·zerocd만 쓴다
MULT = 100.0                   # 계약 승수

TAU = {"S": 0.25, "G": 0.50}   # 3.7 유효 호가 V의 상대 스프레드 상한
TAU_FWD = 0.25                 # 3.7 / A13-1: forward용 체인은 구조와 무관하게 0.25
G_TARGET = 0.16                # 3.3 G 목표 delta
N_CAND = 3                     # 3.7 후보 최대 3개
DTE_RANGE = {21: (35, 65, 45), 10: (19, 49, 30)}   # 3.4 (하한, 상한, 목표), 달력일
DTE_EARLY_EXIT = 5             # 3.5 DTE <= 5
MAX_DEFER = 3                  # 3.5 연기 최대 3영업일
MIN_DTE_DEFER = 2              # 3.5 연기 가능일의 DTE >= 2
NONSTD_EXPIRY_CODES = frozenset(["m", "w"])          # AUDIT_G0 6절
IV30_OPEN_RANGE = (0.01, 3.0)  # E13-02 (개발 데이터에서는 멈춤 조건)
RV_N = 21                      # 3.8 RV21: 로그수익률 21개 (종가 22개)
ANN = 252.0

SCENARIOS = ("base", "stress", "opt")
SPREAD_MULT = {"base": 1.0, "stress": 1.5, "opt": 0.0}     # 4절: bid/offer, mid -+ 1.5 반스프레드, mid
FEE = {"base": 1.00, "stress": 2.00, "opt": 0.0}           # 4절: 계약당 편도
HEDGE_BP = {"base": 1.0e-4, "stress": 3.0e-4, "opt": 0.25e-4}
KAPPA = {"base": 1.25, "stress": 1.50, "opt": 1.00}        # 3.5, 4절 대체 청산 배수

# 7절 설정 격자: (구조, H, 게이트, 사이징)
SETTINGS = {
    "O0": ("S", 21, "G0", "K"),
    "O1": ("S", 10, "G0", "K"),
    "O2": ("G", 21, "G0", "K"),
    "O3": ("S", 21, "G1", "K"),
    "O4": ("S", 21, "G2", "K"),
    "O5": ("S", 21, "G0", "V"),
    "O6": ("S", 21, "G1", "V"),
    "O7": ("S", 21, "G2", "V"),
    "O8": ("G", 21, "G0", "V"),
    "O9": ("G", 21, "G1", "V"),
    "O10": ("S", 10, "G1", "V"),
}

_ONE_DAY = _dt.timedelta(days=1)


# ---------------------------------------------------------------------------
# 날짜 도구
# ---------------------------------------------------------------------------

def _to_date(d):
    d = int(d)
    return _dt.date(d // 10000, (d // 100) % 100, d % 100)


def _to_int(x):
    return x.year * 10000 + x.month * 100 + x.day


def _ndays(a, b):
    """b - a (달력일)."""
    return (_to_date(b) - _to_date(a)).days


def _third_friday(year, month):
    first = _dt.date(year, month, 1)
    return first + _dt.timedelta(days=(4 - first.weekday()) % 7 + 14)


# ---------------------------------------------------------------------------
# 영업일 달력 (3.4 영업일 달력, 달력 끝 이후의 날짜, 3.5 정산일·마지막 거래일)
# ---------------------------------------------------------------------------

class _Calendar:
    def __init__(self, days):
        arr = np.unique(np.asarray(days, dtype=np.int64))
        if arr.size == 0:
            raise ValueError("빈 영업일 달력")
        self.days = arr
        self.pos = {int(d): i for i, d in enumerate(arr)}
        self.first = int(arr[0])
        self.last = int(arr[-1])
        self._settle = {}
        self._ltd = {}
        self._cat = {}

    def is_bday(self, d):
        """달력 안이면 달력 소속 여부, 달력 끝 뒤이면 주말이 아닌 날(3.4 달력 끝 규칙)."""
        d = int(d)
        if d > self.last:
            return _to_date(d).weekday() < 5
        return d in self.pos

    def prev_bday(self, d):
        """d 직전 영업일(d 미포함). 달력 시작 전으로 가면 None."""
        x = _to_date(d)
        while True:
            x -= _ONE_DAY
            y = _to_int(x)
            if y < self.first:
                return None
            if self.is_bday(y):
                return y

    def next_bday(self, i):
        """달력 인덱스 i 다음 영업일. 달력 끝이면 달력 끝 규칙(다음 평일) (R-19)."""
        if i + 1 < len(self.days):
            return int(self.days[i + 1])
        x = _to_date(self.days[i])
        while True:
            x += _ONE_DAY
            if x.weekday() < 5:
                return _to_int(x)

    def settlement(self, exdate):
        """3.5: exdate가 영업일이면 exdate, 아니면 직전 영업일."""
        e = int(exdate)
        if e not in self._settle:
            self._settle[e] = e if self.is_bday(e) else self.prev_bday(e)
        return self._settle[e]

    def last_trading(self, exdate):
        """3.5: 정산일 직전 영업일."""
        e = int(exdate)
        if e not in self._ltd:
            s = self.settlement(e)
            self._ltd[e] = None if s is None else self.prev_bday(s)
        return self._ltd[e]

    def std_category(self, exdate):
        """3.4 (i): 1 = 셋째 금요일, 2 = 그다음 토요일, 3 = 셋째 금요일 휴장 시 직전 영업일, 0 = 해당 없음."""
        e = int(exdate)
        if e in self._cat:
            return self._cat[e]
        d = _to_date(e)
        tf = _third_friday(d.year, d.month)
        tfi = _to_int(tf)
        cat = 0
        if e == tfi:
            cat = 1
        elif e == _to_int(tf + _ONE_DAY):
            cat = 2
        elif not self.is_bday(tfi):
            pb = self.prev_bday(tfi)
            if pb is not None and e == pb:
                cat = 3
        self._cat[e] = cat
        return cat


# ---------------------------------------------------------------------------
# 금리 (3.7, A13-1 변경 4)
# ---------------------------------------------------------------------------

class _Rates:
    """그날 이하에서 가장 최근의, 영업일 달력에 있는 zerocd 날짜의 곡선. days 축 선형 보간, 범위 밖 평탄 외삽.
    rate는 % 단위 연속복리(U11) -> 소수로 바꾼다."""

    def __init__(self, z, cal):
        date = np.asarray(z["date"], dtype=np.int64)
        days = np.asarray(z["days"], dtype=np.float64)
        rate = np.asarray(z["rate"], dtype=np.float64)
        ok = np.isfinite(days) & np.isfinite(rate)          # R-21
        date, days, rate = date[ok], days[ok], rate[ok]
        order = np.lexsort((days, date))                    # 안정 정렬: 날짜, days
        date, days, rate = date[order], days[order], rate[order]
        self.curves = {}
        for d in np.unique(date):
            d = int(d)
            if d not in cal.pos:                            # 달력에 없는 zerocd 날짜는 쓰지 않는다
                continue
            m = date == d
            xs, ys = days[m], rate[m]
            keep = np.ones(xs.size, dtype=bool)             # 같은 days 중복은 첫 행만 (R-21)
            keep[1:] = xs[1:] != xs[:-1]
            self.curves[d] = (xs[keep], ys[keep])
        self.dates = np.array(sorted(self.curves), dtype=np.int64)
        self._cache = {}

    def curve_date(self, day):
        k = int(np.searchsorted(self.dates, int(day), side="right")) - 1
        if k < 0:
            raise RuntimeError("금리: %d 이전 곡선이 없음 (A13-1 변경 4, 실행 멈춤)" % day)
        return int(self.dates[k])

    def rate(self, day, tdays):
        key = (int(day), float(tdays))
        if key not in self._cache:
            xs, ys = self.curves[self.curve_date(day)]
            self._cache[key] = float(np.interp(float(tdays), xs, ys)) / 100.0
        return self._cache[key]


# ---------------------------------------------------------------------------
# 블랙-숄즈 (forward 입력, 연속복리 할인)
# ---------------------------------------------------------------------------

def _d1(F, K, T, sigma):
    return (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))


def bs_price(F, K, T, r, sigma, cp):
    """forward F 기준 유럽형 가격: e^{-rT}[F N(d1) - K N(d2)] (콜), e^{-rT}[K N(-d2) - F N(-d1)] (풋)."""
    d1 = _d1(F, K, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    df = math.exp(-r * T)
    if cp > 0:
        return df * (F * float(ndtr(d1)) - K * float(ndtr(d2)))
    return df * (K * float(ndtr(-d2)) - F * float(ndtr(-d1)))


def bs_spot_delta(F, K, T, r, sigma, cp, S):
    """지수(현물) delta = dV/dS, F = S e^{(r-q)T} 이므로 e^{-qT} = e^{-rT} F/S (R-12).
    콜 e^{-rT}(F/S)N(d1), 풋 -e^{-rT}(F/S)N(-d1)."""
    d1 = _d1(F, K, T, sigma)
    scale = math.exp(-r * T) * F / S
    if cp > 0:
        return scale * float(ndtr(d1))
    return -scale * float(ndtr(-d1))


def _valid_quote(bid, offer, tau):
    """3.7 V: bid > 0, offer > bid, (offer - bid)/mid <= tau. NaN은 무효."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mid = (bid + offer) / 2.0
        return (bid > 0) & (offer > bid) & ((offer - bid) / mid <= tau)


def _iv_valid(iv):
    """3.7 IV 유효: 존재하고 0보다 큼."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(iv) & (iv > 0)


# ---------------------------------------------------------------------------
# 원자료 읽기
# ---------------------------------------------------------------------------

def _code_str(c):
    if isinstance(c, bytes):
        c = c.decode("utf-8", "replace")
    return str(c).strip()


class _OptionDays:
    """opprcd를 시세 날짜별 배열 묶음으로 나눈다. 같은 날짜 안의 원래 행 순서를 지킨다."""

    COLS = ("exdate", "cp_flag", "strike_price", "best_bid", "best_offer",
            "impl_volatility", "delta", "open_interest", "optionid", "am_settlement")

    def __init__(self):
        self.by_day = {}

    def add_year(self, raw):
        date = np.asarray(raw["date"], dtype=np.int64)
        if date.size == 0:
            return
        order = np.argsort(date, kind="stable")
        arrs = {
            "exdate": np.asarray(raw["exdate"], dtype=np.int64),
            "cp_flag": np.asarray(raw["cp_flag"], dtype=np.int64),
            "strike_price": np.asarray(raw["strike_price"], dtype=np.float64),
            "best_bid": np.asarray(raw["best_bid"], dtype=np.float64),
            "best_offer": np.asarray(raw["best_offer"], dtype=np.float64),
            "impl_volatility": np.asarray(raw["impl_volatility"], dtype=np.float64),
            "delta": np.asarray(raw["delta"], dtype=np.float64),
            "open_interest": np.asarray(raw["open_interest"], dtype=np.float64),
            "optionid": np.asarray(raw["optionid"], dtype=np.int64),
            "am_settlement": np.asarray(raw["am_settlement"], dtype=np.int64),
        }
        # 3.4 (iii): 비표준 코드(['m', 'w'])만 제외, 결측(-1)과 그 밖의 값은 허용 (R-04)
        codes = [_code_str(c) for c in np.asarray(raw["expiry_indicator__codes"]).tolist()]
        code_bad = np.array([c in NONSTD_EXPIRY_CODES for c in codes], dtype=bool)
        ei = np.asarray(raw["expiry_indicator"], dtype=np.int64)
        ei_bad = np.zeros(ei.size, dtype=bool)
        has = ei >= 0
        if code_bad.size:
            ei_bad[has] = code_bad[ei[has]]
        arrs["ei_bad"] = ei_bad

        sdate = date[order]
        uniq, starts = np.unique(sdate, return_index=True)
        ends = np.append(starts[1:], sdate.size)
        for u, s, e in zip(uniq, starts, ends):
            idx = order[s:e]
            self.by_day[int(u)] = {k: v[idx] for k, v in arrs.items()}


def _safe_read(fn, *args):
    """신호용 이전 해 파일이 없을 수 있다(데이터 첫해). 그때는 빈 결과."""
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001  (합성 source가 어떤 예외를 낼지 모름)
        return None


# ---------------------------------------------------------------------------
# 하루의 옵션 체인 (3.4 표준 월물·병합·중복 정리, 3.7 forward·진입)
# ---------------------------------------------------------------------------

class _DayChain:
    def __init__(self, day, data, cal, rates, S):
        self.day = int(day)
        self.cal = cal
        self.rates = rates
        self.S = S
        if data is None:
            data = {k: np.zeros(0, dtype=np.float64) for k in _OptionDays.COLS}
            data["ei_bad"] = np.zeros(0, dtype=bool)
            for k in ("exdate", "cp_flag", "optionid", "am_settlement"):
                data[k] = np.zeros(0, dtype=np.int64)
        # 8d (Kyle 결정 2026-10-28, REF_NOTES 5절): (optionid, date)가 같은 행이 둘 이상이면 파일에 저장된 순서의
        # 첫 행만 남긴다. 나머지 행은 그날 어디에도 쓰지 않는다(열린 다리 행 찾기, 개시 후보 체인, forward 체인).
        # _OptionDays가 같은 날짜 안의 저장 순서를 지킨다(안정 정렬).
        oid_all = np.asarray(data["optionid"])
        if oid_all.size:
            _, first = np.unique(oid_all, return_index=True)
            if first.size != oid_all.size:
                keep = np.sort(first)
                data = {k: v[keep] for k, v in data.items()}
        self.exd = data["exdate"]
        self.cp = data["cp_flag"]
        self.sp = data["strike_price"]
        self.bid = data["best_bid"]
        self.offer = data["best_offer"]
        self.iv = data["impl_volatility"]
        self.delta = data["delta"]
        self.oi = data["open_interest"]
        self.oid = data["optionid"]
        self.am = data["am_settlement"]
        self.ei_bad = data["ei_bad"]
        self.mid = (self.bid + self.offer) / 2.0
        self.ivok = _iv_valid(self.iv)
        self.v = {t: _valid_quote(self.bid, self.offer, t) for t in (0.25, 0.50)}
        # 열린 다리는 optionid로 추적. (optionid, date) 중복이면 첫 행 (R-20)
        self.row_of = {}
        for i, o in enumerate(self.oid.tolist()):
            if o not in self.row_of:
                self.row_of[o] = i
        self.std_idx, self.std_exdates = self._standard_rows()
        self._dedup_cache = {}
        self._fwd_chain_cache = {}
        self._fwd_cache = {}
        self._tr_cache = {}

    # --- 3.4 표준 월물 ---------------------------------------------------------
    def _standard_rows(self):
        n = self.exd.size
        if n == 0:
            return np.zeros(0, dtype=np.int64), []
        cat_of = {int(e): self.cal.std_category(int(e)) for e in np.unique(self.exd)}
        cat = np.array([cat_of[int(e)] for e in self.exd.tolist()], dtype=np.int64)
        # (i) 달력 조건, (ii) am_settlement = 1, (iii) 비표준 코드 아님 (행 단위)
        ok = (cat > 0) & (self.am == 1) & (~self.ei_bad)
        # 한 달에 둘 이상이면 1 > 2 > 3 우선. (ii)(iii)을 통과한 행이 있는 exdate끼리 비교 (R-02)
        best = {}
        for e in np.unique(self.exd[ok]).tolist():
            ym = e // 100
            c = cat_of[e]
            if ym not in best or c < best[ym][0]:
                best[ym] = (c, e)
        keep = sorted(v[1] for v in best.values())
        ok &= np.isin(self.exd, np.array(keep, dtype=np.int64))
        return np.nonzero(ok)[0], keep

    # --- 시간·금리 -------------------------------------------------------------
    def settle_of(self, exdate):
        """3.5 정산일(달력 끝 규칙 포함)."""
        settle = self.cal.settlement(int(exdate))
        if settle is None:
            raise RuntimeError("정산일을 정할 수 없음: exdate %d" % int(exdate))
        return settle

    def T_r_settle(self, settle):
        """T1 = (정산일 - 날짜)/365 (AUDIT 6절), r = 그 days에서 보간한 zerocd 금리 (R-11).
        열린 다리는 개시 때 계산한 정산일을 넘긴다(8e)."""
        s = int(settle)
        if s not in self._tr_cache:
            nd = _ndays(self.day, s)
            if nd <= 0:
                raise RuntimeError("불변식 위반: T1 <= 0 (날짜 %d, 정산일 %d)" % (self.day, s))
            self._tr_cache[s] = (nd / 365.0, self.rates.rate(self.day, nd))
        return self._tr_cache[s]

    # --- 3.4 중복 정리 -----------------------------------------------------------
    def dedup(self, exdate, tau):
        """신규 개시 후보 체인: 그날 표준 월물(3.4 (i)~(iii)과 1>2>3 우선순위) 행 중 그 exdate를 tau로 정리한다."""
        key = (int(exdate), tau)
        if key not in self._dedup_cache:
            idx = self.std_idx[self.exd[self.std_idx] == int(exdate)]
            self._dedup_cache[key] = self._dedup_rows(idx, tau)
        return self._dedup_cache[key]

    def fwd_chain(self, exdate):
        """forward 체인 (5번 규칙, Kyle 결정 2026-10-28, REF_NOTES 4절): 그날 그 exdate에서
        (ii) am_settlement = 1과 (iii) 비표준 코드 아님을 만족하는 모든 행을 tau = 0.25로 정리한다.
        (i) 달력 조건과 1>2>3 우선순위는 쓰지 않는다. 표준 exdate에서는 dedup(exdate, 0.25)와 같은 행 집합이다."""
        e = int(exdate)
        if e not in self._fwd_chain_cache:
            idx = np.nonzero((self.exd == e) & (self.am == 1) & (~self.ei_bad))[0]
            self._fwd_chain_cache[e] = self._dedup_rows(idx, TAU_FWD)
        return self._fwd_chain_cache[e]

    def _dedup_rows(self, idx, tau):
        """(행사가, cp_flag) 중복을 1) V(tau)와 IV 유효 모두 만족, 2) open_interest 큰 행(NaN은 가장 작게, R-05),
        3) optionid 작은 행 순으로 하나만 남긴다."""
        if idx.size == 0:
            return idx
        good = self.v[tau][idx] & self.ivok[idx]
        oi = self.oi[idx]
        neg_oi = np.where(np.isfinite(oi), -oi, np.inf)
        order = np.lexsort((self.oid[idx], neg_oi, ~good, self.cp[idx], self.sp[idx]))
        sidx = idx[order]
        out = []
        last = None
        for i in sidx.tolist():
            k = (self.sp[i], self.cp[i])
            if k != last:
                out.append(i)
                last = k
        return np.array(out, dtype=np.int64)

    # --- 3.7 forward (A13-1 변경 1) --------------------------------------------
    def forward(self, exdate, settle=None):
        """(1) fwd_chain(5번 규칙)에서 콜·풋 모두 V(0.25)인 행사가 중 |C_mid-P_mid| 최소(동점 낮은 행사가) K*:
        F = K* + e^{rT}(C_mid - P_mid). (2) 없으면 S e^{rT}. forward_price는 쓰지 않는다.
        개시(표준 exdate)와 열린 다리(평가, 대체 평가, 대체 청산, 블랙-숄즈 delta 대체) 모두 이 함수를 쓴다.
        exdate는 체인의 만기, settle은 T와 r의 기준 정산일이다. 생략하면 그 exdate의 정산일이다.
        열린 다리는 exdate = 그날 행의 exdate(행이 없으면 개시 때 exdate), settle = 개시 때 정산일을 넘긴다(8e)."""
        e = int(exdate)
        s = self.settle_of(e) if settle is None else int(settle)
        key = (e, s)
        if key in self._fwd_cache:
            return self._fwd_cache[key]
        T, r = self.T_r_settle(s)
        ch = self.fwd_chain(e)
        calls, puts = {}, {}
        for i in ch.tolist():
            if not self.v[TAU_FWD][i]:
                continue
            (calls if self.cp[i] > 0 else puts)[self.sp[i]] = self.mid[i]
        best = None
        for s in sorted(set(calls) & set(puts)):          # 오름차순이라 엄격 < 로 동점 시 낮은 행사가 유지
            diff = calls[s] - puts[s]
            if best is None or abs(diff) < best[0]:
                best = (abs(diff), s, diff)
        if best is not None:
            F = best[1] / 1000.0 + math.exp(r * T) * best[2]
            src = "parity"
        else:
            F = self.S * math.exp(r * T)
            src = "spot"
        self._fwd_cache[key] = (F, src)
        return self._fwd_cache[key]

    # --- 블랙-숄즈 보조 ----------------------------------------------------------
    def model_price(self, exdate, settle, K, cp, sigma):
        """exdate: forward 체인의 만기, settle: T와 r의 기준 정산일 (8e)."""
        T, r = self.T_r_settle(settle)
        F, _ = self.forward(exdate, settle)
        return bs_price(F, K, T, r, sigma, cp)

    def model_delta(self, exdate, settle, K, cp, sigma):
        """exdate: forward 체인의 만기, settle: T와 r의 기준 정산일 (8e)."""
        T, r = self.T_r_settle(settle)
        F, _ = self.forward(exdate, settle)
        return bs_spot_delta(F, K, T, r, sigma, cp, self.S)

    # --- 3.4 만기 선택 ----------------------------------------------------------
    def pick_expiry(self, H):
        lo, hi, tgt = DTE_RANGE[H]
        best = None
        for e in self.std_exdates:
            dte = _ndays(self.day, e)
            if lo <= dte <= hi:
                key = (abs(dte - tgt), e)                   # 동점이면 이른 exdate (R-07)
                if best is None or key < best[0]:
                    best = (key, e)
        return None if best is None else best[1]

    def _eligible(self, i, tau):
        return bool(self.v[tau][i] and self.ivok[i])

    # --- 3.7 진입 S ------------------------------------------------------------
    def entry_S(self, exdate):
        tau = TAU["S"]
        ch = self.dedup(exdate, tau)
        if ch.size == 0:
            return None
        F, _ = self.forward(exdate)
        if not np.isfinite(F):
            return None
        sps = np.unique(self.sp[ch])                        # 콜 또는 풋 행이 있는 행사가 (R-08)
        dist = np.abs(sps / 1000.0 - F)
        order = np.lexsort((sps, dist))                     # 거리, 동점이면 낮은 행사가
        for s in sps[order][:N_CAND].tolist():
            c = [i for i in ch.tolist() if self.sp[i] == s and self.cp[i] > 0]
            p = [i for i in ch.tolist() if self.sp[i] == s and self.cp[i] < 0]
            if c and p and self._eligible(c[0], tau) and self._eligible(p[0], tau):
                return (c[0], p[0])
        return None

    # --- 3.7 진입 G ------------------------------------------------------------
    def entry_G(self, exdate):
        tau = TAU["G"]
        ch = self.dedup(exdate, tau)
        if ch.size == 0:
            return None
        calls, puts = [], []
        for i in ch.tolist():
            d = self.delta[i]
            if not np.isfinite(d):                          # OM delta 결측 -> 블랙-숄즈, IV 무효면 후보 제외
                if not self.ivok[i]:
                    continue
                d = self.model_delta(exdate, self.settle_of(exdate), self.sp[i] / 1000.0, self.cp[i], self.iv[i])
                if not np.isfinite(d):
                    continue
            if self.cp[i] > 0:
                calls.append((abs(d - G_TARGET), -self.sp[i], i))   # 동점이면 높은 행사가
            else:
                puts.append((abs(d + G_TARGET), self.sp[i], i))     # 동점이면 낮은 행사가
        calls.sort()
        puts.sort()
        c = next((t[2] for t in calls[:N_CAND] if self._eligible(t[2], tau)), None)
        p = next((t[2] for t in puts[:N_CAND] if self._eligible(t[2], tau)), None)
        if c is None or p is None:
            return None
        return (c, p)

    def select_entry(self, struct, H):
        e = self.pick_expiry(H)
        if e is None:
            return None
        return self.entry_S(e) if struct == "S" else self.entry_G(e)


# ---------------------------------------------------------------------------
# 다리와 설정 상태
# ---------------------------------------------------------------------------

class _Leg:
    __slots__ = ("optionid", "cp", "K", "exdate", "settle", "ltd", "q", "tau", "H",
                 "open_idx", "mark", "last_iv", "last_delta", "sched_idx")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _State:
    def __init__(self, sid):
        if sid not in SETTINGS:
            raise KeyError("알 수 없는 설정: %r" % (sid,))
        self.sid = sid
        self.struct, self.H, self.gate, self.sizing = SETTINGS[sid]
        self.legs = []
        self.Hd_prev = 0.0
        self.rows = []


# ---------------------------------------------------------------------------
# 시장 신호 (3.8, 3.9, 3.1: 신호일 = 직전 영업일)
# ---------------------------------------------------------------------------

class _Market:
    def __init__(self, source, cal, i_first, i_last, end_year):
        self.cal = cal
        self.i_lo = max(0, i_first - (RV_N + 1) - 8)       # RV21 창 + 앞쪽 결측 대비 여유
        y_lo_sec = int(cal.days[self.i_lo]) // 10000
        y_first = int(cal.days[i_first]) // 10000
        # --- 종가 (secprd), 결함일은 직전 종가 (3.4) ---
        close_of = {}
        for y in range(y_lo_sec, end_year + 1):
            raw = source.secprd(UND, y) if y >= y_first else _safe_read(source.secprd, UND, y)
            if raw is None:
                continue
            ds = np.asarray(raw["date"], dtype=np.int64).tolist()
            cs = np.asarray(raw["close"], dtype=np.float64).tolist()
            for d, c in zip(ds, cs):
                if d not in close_of:                       # 같은 날짜 중복이면 첫 행 (R-20)
                    close_of[d] = c
        self.S = {}
        last = float("nan")
        for i in range(self.i_lo, i_last + 1):
            c = close_of.get(int(cal.days[i]), float("nan"))
            if np.isfinite(c):
                last = float(c)
            self.S[i] = last
        # --- vsurfd: IV30, IV91 (days, delta=50, cp_flag=1 콜) ---
        self.iv30, self.iv91 = {}, {}
        y_lo_vs = int(cal.days[max(0, i_first - 1)]) // 10000
        for y in range(y_lo_vs, end_year + 1):
            raw = source.vsurfd(UND, y) if y >= y_first else _safe_read(source.vsurfd, UND, y)
            if raw is None:
                continue
            ds = np.asarray(raw["date"], dtype=np.int64)
            dy = np.asarray(raw["days"], dtype=np.float64)
            de = np.asarray(raw["delta"], dtype=np.float64)
            cp = np.asarray(raw["cp_flag"], dtype=np.float64)
            iv = np.asarray(raw["impl_volatility"], dtype=np.float64)
            m = (de == 50) & (cp == 1)
            for d, n, v in zip(ds[m].tolist(), dy[m].tolist(), iv[m].tolist()):
                tgt = self.iv30 if n == 30 else (self.iv91 if n == 91 else None)
                if tgt is not None and d not in tgt:        # 중복이면 첫 행 (R-20)
                    tgt[d] = v

    def close(self, i):
        return self.S.get(i, float("nan"))

    def iv30_at(self, t):
        if t < 0:
            return float("nan")
        v = self.iv30.get(int(self.cal.days[t]), float("nan"))
        if np.isfinite(v) and not (IV30_OPEN_RANGE[0] < v < IV30_OPEN_RANGE[1]):
            raise ValueError("E13-02: IV30 범위 밖 (%d, %r)" % (int(self.cal.days[t]), v))
        return v

    def iv91_at(self, t):
        if t < 0:
            return float("nan")
        return self.iv91.get(int(self.cal.days[t]), float("nan"))

    def rv21_at(self, t):
        """직전 21 영업일(t-20~t)의 로그수익률 표본 표준편차(ddof=1) x sqrt(252). 종가 22개(t-21~t) 필요."""
        if t - RV_N < self.i_lo:
            return float("nan")
        c = np.array([self.S[k] for k in range(t - RV_N, t + 1)], dtype=np.float64)
        if not np.all(np.isfinite(c)) or np.any(c <= 0):
            return float("nan")
        r = np.diff(np.log(c))
        return float(np.std(r, ddof=1) * math.sqrt(ANN))


def _gate_and_size(st, mkt, t):
    """신호일 t(=거래일의 직전 영업일) 종가 정보로 (개시 여부, m)."""
    need_iv30 = st.gate in ("G1", "G2") or st.sizing == "V"
    iv30 = mkt.iv30_at(t) if need_iv30 else float("nan")
    if st.gate == "G1":
        rv = mkt.rv21_at(t)
        if not (np.isfinite(iv30) and np.isfinite(rv)):
            return False, None
        if not (iv30 - rv > 0):
            return False, None
    elif st.gate == "G2":
        iv91 = mkt.iv91_at(t)
        if not (np.isfinite(iv30) and np.isfinite(iv91)):
            return False, None
        if not (iv30 <= iv91):
            return False, None
    if st.sizing == "V":
        if not np.isfinite(iv30):
            return False, None
        return True, min(max(0.16 / iv30, 0.25), 1.5)
    return True, 1.0


# ---------------------------------------------------------------------------
# 시뮬레이션
# ---------------------------------------------------------------------------

def _can_defer(cal, leg, i):
    """3.5 / E13-03: 다음 영업일로 미룰 수 있는가. 연기 3영업일 이하, 미룬 날 DTE >= 2이고 마지막 거래일 이하."""
    if i - leg.sched_idx >= MAX_DEFER:
        return False
    nxt = cal.next_bday(i)
    if leg.ltd is None:
        return False
    return _ndays(nxt, leg.exdate) >= MIN_DTE_DEFER and nxt <= leg.ltd


def _run(source, setting_ids, start, end, trace=None):
    start, end = int(start), int(end)
    sids = list(setting_ids)
    cal = _Calendar(source.calendar(SPLIT).days)
    window = [i for i, d in enumerate(cal.days.tolist()) if start <= d <= end]
    states = [_State(s) for s in sids]
    if not window:
        return states
    i_first, i_last = window[0], window[-1]
    # BRIEF 1절: end가 속한 해보다 뒤의 해는 읽지 않는다. 처리하는 마지막 영업일의 해까지만 읽는다 (R-28).
    end_year = min(end // 10000, int(cal.days[i_last]) // 10000)

    rates = _Rates(source.zerocd(SPLIT), cal)
    mkt = _Market(source, cal, i_first, i_last, end_year)
    opt = _OptionDays()
    for y in range(int(cal.days[i_first]) // 10000, end_year + 1):
        opt.add_year(source.opprcd(UND, y))
    rates.curve_date(int(cal.days[i_first]))                # 시작 때 이전 곡선 존재 확인 (A13-1 변경 4)

    for i in window:
        day = int(cal.days[i])
        S = mkt.close(i)
        S_prev = mkt.close(i - 1)
        chain = _DayChain(day, opt.by_day.get(day), cal, rates, S)
        entry_cache = {}
        for st in states:
            P = {sc: 0.0 for sc in SCENARIOS}
            cash = {sc: 0.0 for sc in SCENARIOS}             # trace용 현금흐름 (M1 자체 점검)
            n_traded = 0.0

            # 1) 전일 종가 뒤 열려 있던 다리: 평가 변화, 청산·연기 판정 (3.5, 3.7, 5.1)
            keep = []
            for leg in st.legs:
                r = chain.row_of.get(leg.optionid)          # 행이 없으면 V 아님 (E13-04)
                isV = r is not None and bool(chain.v[leg.tau][r])
                if isV:
                    mark = float(chain.mid[r])
                else:                                       # 대체 평가: 직전 유효일 IV, 당일 forward·금리·T1
                    # 8e: forward 체인은 그날 행의 exdate(행이 없으면 개시 때 exdate), T·r은 개시 때 정산일
                    ch_ex = leg.exdate if r is None else int(chain.exd[r])
                    mark = chain.model_price(ch_ex, leg.settle, leg.K, leg.cp, leg.last_iv)
                dval = -leg.q * MULT * (mark - leg.mark)
                for sc in SCENARIOS:
                    P[sc] += dval
                leg.mark = mark
                if isV and chain.ivok[r]:
                    leg.last_iv = float(chain.iv[r])
                if leg.sched_idx is None and (i - leg.open_idx >= leg.H
                                              or _ndays(day, leg.exdate) <= DTE_EARLY_EXIT):
                    leg.sched_idx = i
                closed = False
                if leg.sched_idx is not None:
                    if isV:                                  # 정상 청산: 매수 best_offer (C-base)
                        half = float(chain.offer[r]) - mark
                        for sc in SCENARIOS:
                            P[sc] -= leg.q * MULT * half * SPREAD_MULT[sc]
                            cash[sc] -= leg.q * MULT * (mark + half * SPREAD_MULT[sc])
                        closed = True
                    elif not _can_defer(cal, leg, i):        # 연기 상한: kappa x 모형 mid
                        for sc in SCENARIOS:
                            P[sc] -= leg.q * MULT * (KAPPA[sc] - 1.0) * mark
                            cash[sc] -= leg.q * MULT * KAPPA[sc] * mark
                        closed = True
                        if trace is not None:
                            trace.append(("fallback_exit", st.sid, day, leg.optionid))
                if closed:
                    n_traded += leg.q
                    if trace is not None:
                        trace.append(("close", st.sid, day, leg.optionid, i - leg.open_idx))
                else:
                    keep.append(leg)
            st.legs = keep

            # 2) 신규 트랜치 (3.1 신호일 = 직전 영업일, 3.2~3.4, 3.7~3.9)
            ok, m = _gate_and_size(st, mkt, i - 1)
            if ok:
                key = (st.struct, st.H)
                if key not in entry_cache:
                    entry_cache[key] = chain.select_entry(st.struct, st.H)
                sel = entry_cache[key]
                if sel is not None and np.isfinite(S) and S > 0:
                    q = N0 * (1.0 / st.H) * m / (S * MULT)
                    for r in sel:
                        e = int(chain.exd[r])
                        mid = float(chain.mid[r])
                        half = mid - float(chain.bid[r])
                        for sc in SCENARIOS:
                            P[sc] -= q * MULT * half * SPREAD_MULT[sc]
                            cash[sc] += q * MULT * (mid - half * SPREAD_MULT[sc])
                        n_traded += q
                        st.legs.append(_Leg(
                            optionid=int(chain.oid[r]), cp=int(chain.cp[r]),
                            K=float(chain.sp[r]) / 1000.0, exdate=e,
                            settle=cal.settlement(e), ltd=cal.last_trading(e),
                            q=q, tau=TAU[st.struct], H=st.H, open_idx=i,
                            mark=mid, last_iv=float(chain.iv[r]),
                            last_delta=float("nan"), sched_idx=None))
                        if trace is not None:
                            trace.append(("open", st.sid, day, int(chain.oid[r]), e,
                                          float(chain.sp[r]), int(chain.cp[r]), q))

            # 3) 수수료 (q x 2 기준 = 다리별 q)
            for sc in SCENARIOS:
                P[sc] -= FEE[sc] * n_traded
                cash[sc] -= FEE[sc] * n_traded

            # 4) 델타 헤지 (3.6): 거래 후 열린 다리, OM delta -> BS(당일 IV) -> 전일 delta
            Hd = 0.0
            for leg in st.legs:
                r = chain.row_of.get(leg.optionid)
                if r is not None and np.isfinite(chain.delta[r]):
                    dl = float(chain.delta[r])
                elif r is not None and chain.ivok[r]:
                    dl = chain.model_delta(int(chain.exd[r]), leg.settle, leg.K, leg.cp,
                                           float(chain.iv[r]))      # 8e: 그날 행의 exdate 체인, 개시 때 정산일
                else:
                    dl = leg.last_delta
                    if trace is not None:
                        trace.append(("delta_prev", st.sid, day, leg.optionid))
                leg.last_delta = dl
                Hd += leg.q * MULT * dl
            hedge_pnl = st.Hd_prev * (S - S_prev) if st.Hd_prev != 0.0 else 0.0
            turn = abs(Hd - st.Hd_prev)
            for sc in SCENARIOS:
                P[sc] += hedge_pnl - HEDGE_BP[sc] * turn * S
                cash[sc] += hedge_pnl - HEDGE_BP[sc] * turn * S
            st.Hd_prev = Hd

            st.rows.append((day, P["base"], P["stress"], P["opt"]))
            if trace is not None:
                open_mark = sum(-lg.q * MULT * lg.mark for lg in st.legs)
                trace.append(("day", st.sid, day, dict(P), dict(cash), open_mark,
                              len(st.legs), Hd))
    return states


def simulate(source, setting_ids, start, end):
    """source: BRIEF 4절 인터페이스. setting_ids: ["O0", ...] (SPEC 7절).
    start, end: YYYYMMDD 정수. start일에 빈 사다리로 시작해 end일까지 매 영업일을 처리한다.
    데이터 끝 강제 청산은 하지 않는다(창 비교 모드).
    반환: {setting_id: pandas.DataFrame(columns=["date", "P_base", "P_stress", "P_opt"])}"""
    states = _run(source, setting_ids, start, end)
    out = {}
    for st in states:
        out[st.sid] = pd.DataFrame(st.rows, columns=["date", "P_base", "P_stress", "P_opt"])
        out[st.sid]["date"] = out[st.sid]["date"].astype(np.int64)
    return out
