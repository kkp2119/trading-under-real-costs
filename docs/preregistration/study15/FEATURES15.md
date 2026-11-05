(Korean)
# 연구 15 입력값(X)과 정답(y) 정의

작성 2026-10-29 ET · 코드: research15/xy15.py · 시험: research15/tests/test_xy15.py · 근거: research15_PLAN_draft4.md (sha256 `64884835…2f10`), 2단계 지시(Kyle, 2026-10-29)

## 0. 표기
- 막대 u는 [09:30:00 + u초, +1초) 구간이다. "막대 u의 끝"은 09:30:00 + (u + 1)초다.
- 가격은 1e-4달러 정수다. 1센트 = 1틱 = 100이다. mid2 = bid + ask(mid의 2배)다.
- 막대 열은 1단계 bars.py의 21개다. bid/ask/bsz/asz는 막대 끝의 NBBO 상태다.
- **유효 호가(quote_ok):** bid > 0, ask > 0, ask ≥ bid, bsz > 0, asz > 0. 락은 포함한다(bars.py 유효 정의와 같음).
- **유효 mid(mid_ok):** bid > 0, ask > 0, ask ≥ bid.
- **창 합계 S_w(x)[u]:** 막대 u−w+1 .. u의 합이다. u−w+1 < 0이면 NaN이다(창이 모자람).

## 1. 결정 시점
- 결정은 u = 4, 9, …, F−5에서 한다(5초마다). 첫 결정은 09:30:05, 정상일 마지막 결정은 15:58:55다(하루 4,667개).
- 정리 막대 F = 마감 막대 − 61이다. 정상일은 F = 23,339(15:59:00 끝), 반일장(13:00)은 F = 12,539다.
- 결정 막대가 quote_ok가 아니면 no_quote = 1이다. 이때도 X는 계산하지만 정답은 모두 NaN이다.
- **저장 표본:**
  - 2015–2019년은 결정 6개 중 1개(30초 간격)만 저장한다. 시작 위치 k = numpy default_rng(15 + int(YYYYMMDD)).integers(0, 6)이고, 그날 모든 종목에 같은 k를 쓴다.
  - 2020년은 모든 결정을 저장한다.

## 2. 입력값 X(u) — 막대 u 이하만 사용 (31개)

| 이름 | 정의 | 단위 | NaN 조건 |
|---|---|---|---|
| spread_tk | (ask − bid) / 100 | 틱 | quote_ok 아님 |
| spread_bp | (ask − bid) / mid × 1e4 = 2e4 (ask − bid) / mid2 | bp | quote_ok 아님 |
| bsz, asz | 표시 잔량 | 주 | quote_ok 아님 |
| imb | (bsz − asz) / (bsz + asz) | — | quote_ok 아님 |
| mid_usd | mid2 / 2e4 | 달러 | mid_ok 아님 |
| dmid_w (w = 1, 5, 30, 60, 300) | (mid2[u] − mid2[u−w]) / 200 | 틱 | u−w < 0, 또는 u나 u−w가 mid_ok 아님 |
| ofi_w (w = 5, 30, 60, 300) | (S_w(vol_ask) − S_w(vol_bid)) / S_w(vol) | — | 창 부족, 또는 S_w(vol) = 0 |
| vol_w (w = 5, 30, 60, 300) | S_w(vol) | 주 | 창 부족 |
| ntr_w (w = 5, 30, 60, 300) | S_w(ntr) | 건 | 창 부족 |
| sd_w (w = 30, 300) | 아래 설명 | 틱 | u < w, 또는 유효 변화가 2개 미만 |
| hl_60 | 막대 u−59..u 중 mid_ok인 막대의 (max mid2 − min mid2) / 200 | 틱 | u < 59, 또는 창 안에 mid_ok 막대가 없음 |
| nupd_w (w = 5, 60) | S_w(n_upd) | 건 | 창 부족 |
| t_open | u + 1 (09:30:00부터 지난 초) | 초 | — |
| t_close | 마감 막대 − (u + 1) (마감까지 남은 초) | 초 | — |
| dmid_day | (mid2[u] − mid2[f0]) / 200, f0 = 그날 첫 mid_ok 막대 | 틱 | u가 mid_ok 아님, 또는 u < f0 |

- **sd_w:** 막대 v = u−w+1 .. u의 1초 mid 변화 (mid2[v] − mid2[v−1]) / 200 중, v와 v−1이 모두 mid_ok인 것의 모표준편차(ddof 0)다.
- 계획에 있던 "현재 재고"는 넣지 않았다. 정답이 주문 하나 단위라 재고와 무관하기 때문이다. 재고는 3단계 정책에서 쓴다(Kyle 2단계 지시).

## 3. 정답 y(u, 방향) — quote_ok인 결정만
**가상 주문**
- 막대 u의 끝에 100주 지정가를 낸다. 매수는 bid[u], 매도는 ask[u]다.
- 주문은 막대 u+1 .. u+5에서 유효하고, 다음 결정에서 취소된다(연구 12 B0 reset5).
- **체결 규칙:** 연구 12 Q 모델(θ = 1)이다. research12/fillmodel.py의 fill_buy / fill_sell(kind 2)을 그대로 따르고, 막대마다 순서대로 적용한다.
  - 대기열 a4 = 4 × 표시 잔량(bsz[u] 또는 asz[u])이고, 단위는 1/4주다.
  - **매수**
    1. ask_min ≤ 가격이면 남은 수량이 모두 체결된다.
    2. 아니고 0 < bid_max < 가격이면 min(남은 수량, credit)이 체결되고 a4 = 0이 된다.
    3. 아니고 bid_min = bid_max = 가격이면 대기열 소진 규칙을 쓴다. 4·credit > a4이면 min(남은 수량, (4·credit − a4)//4)가 체결되고 a4 = 0이 된다. 아니면 a4 −= 4·credit이다.
    4. 그 밖에는 0이다.
  - credit = max(vol_bid − vol_bid_odd, 0)이다.
  - **매도:** 매수와 대칭이다. bid ↔ ask, min ↔ max, 부등호를 뒤집는다. credit = max(vol_ask − vol_ask_odd, 0)이다.
- 연구 12 simulator.replay와 같은지는 시험으로 확인했다(2019-01 전체, 주문 1,960,136개, 막대별 누적 체결 수량까지 일치).

**정답 열** (접두 b_ = 매수, s_ = 매도, 방향 d = +1 / −1)

| 이름 | 정의 | 단위 |
|---|---|---|
| fill | 수명 안에 1주라도 체결되면 1, 아니면 0 | — |
| frac | 체결된 주식 수 / 100 (부분 체결 확인용) | — |
| tf | t_f − u, t_f = 첫 체결 막대(1..5). 체결이 없으면 NaN | 초 |
| mk_h (h = 5, 30, 60) | d × (mid(v) − 체결가) = d × (mid2[v] − 2·가격) / 200, v = min(t_f + h, F). 체결이 없으면 NaN | 센트/주 |
| val_h | fill × mk_h. 체결이 없으면 0 | 센트/주 |
| tk_h | 매수: (bid[w] − ask[u]) / 100, 매도: (bid[u] − ask[w]) / 100, w = min(u + h, F) | 센트/주 |
| tr_tk_h | u + h > F이면 1 (매수·매도 공통) | — |
| tr_mkb_h, tr_mks_h | t_f + h > F이면 1. 체결이 없으면 NaN | — |

- 수수료와 리베이트는 넣지 않았다. 비용은 3단계 시뮬레이션에서 더한다.
- 부분 체결이어도 fill = 1이고, mk_h는 첫 체결 막대 기준이다. 체결 비율은 frac에 따로 둔다. 2019-01에서 부분 체결은 체결 주문의 0.8%(409 / 52,272)다.

## 4. 저장 형식
- 위치: $DATA_DIR/research15_data/xy/<Y>/<SYM>_<YYYYMMDD>.npz. 배열 xy는 float32 (행 수 × 66)이고, 배열 columns는 열 이름이다.
- 열 순서: u, no_quote, X 31개, 정답 33개(매수 12, 매도 12, 잘림 표시 9).
- xy/MANIFEST.tsv의 열:
  - 파일 정보: sym, date, file, rows, bytes, sha256, written_utc
  - 결정과 표본: decisions, no_quote, close_bar, offset(표본 시작 위치, 2020은 −1)
  - 원본 막대: source_sha256(원본 막대의 sha256)
  - 거래량: vol, vol_bid, vol_ask, vol_in, vol_unc(장중 합계)
