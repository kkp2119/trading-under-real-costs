(Korean)
# A3 (초안): 막대 다운로드의 NULL 행 처리 — 2026-10-27

**상태: Kyle 지시(2026-10-27)로 적용했다. 필터와 막대 정의는 NULL 처리 외에는 바뀌지 않는다.**

## 계기
- 2026-10-27 13:28 dev 다운로드가 143개 종목·일을 받은 뒤 `ValueError: cannot convert to 'int64'-dtype NumPy array with missing values`로 멈췄다.
- **원인**: `taqm_2019.complete_nbbo_20190123`의 **XLE 09:30 이전 행** 1개에서 best_bid, best_bidsizeshares, best_ask, best_asksizeshares가 NULL이다(열마다 1행).
  - XLE의 09:30 이전 행은 216개라, 다운로더가 읽는 마지막 500행에 그 행이 들어왔다.
  - wrds의 `raw_sql`은 정수 열을 nullable 형식으로 돌려주므로, 수량 열의 정수 변환에서 멈췄다.
- **같은 날 다른 종목**(개수만 확인, 값은 보지 않음):
  - TLT: 09:30 이전 행 1개에서 4개 열이 NULL
  - IEF: 09:30 이전 행 1개에서 best_ask와 best_asksizeshares가 NULL
  - 7종목 모두 정규장 호가 행, time_m_nano, 체결의 price·size·tr_corr에는 NULL이 0이다.
  - 체결의 tr_scond NULL은 흔하다(예: XLE 81,610행 중 40,519행). 기존 코드도 빈칸으로 처리했다.

## 규칙 (코드: `bars1s/download.py` `day_bars`, `bars1s/bars.py` `trade_exclusions`)

| 행 | NULL 열 | 처리 |
|---|---|---|
| 호가 (complete_nbbo, 09:30 이전·정규장 모두) | best_bid, best_ask, best_bidsizeshares, best_asksizeshares 중 하나라도 | 그 값을 0으로 바꾼다. 기존 규칙(가격·수량 > 0)에 따라 **무효 행**이 되어 상태와 극값에 들어가지 않는다. 정규장 행이면 n_upd에는 세고, n_xl에는 세지 않는다(가격이 0이므로) |
| 호가 | qu_cond | 빈칸 → 'R'이 아니므로 무효 (기존과 같음) |
| 호가 | qu_source, natbbo_ind | 빈칸 → "NBBO 없음" 코드 쌍이 아니므로 통과 (기존과 같음) |
| 체결 (ctm) | price 또는 size | **제외**. `excl_by_code`에 **'NULL'**로 세고 `n_trade_excluded`에도 넣는다 |
| 체결 | tr_scond | 빈칸으로 본다 → 허용 문자 집합에 들어가므로 통과 (기존과 같음) |
| 체결 | tr_corr | 'NULL'로 둔다 → '00'이 아니므로 **제외**, `excl_by_code`에 **'corr=NULL'**로 센다. 이전 코드는 빈칸을 `zfill(2)`해 '00'으로 만들었으므로(유효로 잘못 처리) 고쳤다 |
| 호가·체결 | time_m_nano | **규칙 없음**: 그 종목·일은 실패로 errors.tsv에 적고 보고한다 |

- **`excl_by_code`의 우선순위**: price·size NULL('NULL') → tr_corr('corr=…') → tr_scond('scond=…'). 정규장 체결만 센다. `n_trade_excluded`는 `excl_by_code` 합계와 같다.
- **MANIFEST 열은 바꾸지 않았다**(받는 도중이라 형식을 유지). 호가·체결의 NULL 행 수는 다운로드 로그의 "null quote rows / null trade rows"에 남는다.

## 이전 결과와의 관계
- **NULL이 없는 입력에서는 21개 막대 열과 MANIFEST 통계가 이전 코드와 같다.** 합성 입력 3개(잠금 호가, 코드 필터 대상, 정정 체결 포함)로 시험했다.
- 이미 받은 143개 파일은 NULL 없이 만들어졌으므로 그대로 유효하다. 이어받기에서 해시가 맞으면 건너뛴다.

## 실패 처리 (같은 날 지시)
- 실패한 종목·일(오류나 경고)은 `bars_1s_dev/errors.tsv`(sym, date, kind, message, utc)에 적고 **재시도하지 않은 채** 다음으로 넘어간다.
  - **오류 종류(kind)** = 예외 이름 + 메시지(숫자를 #으로 바꾼 앞 80자)
- **멈추는 조건**:
  - 실패가 20건을 넘을 때
  - 서로 다른 오류 종류가 3개를 넘을 때
  - 기존 파일의 해시가 MANIFEST와 맞지 않을 때 (덮어쓰지 않음)
  - 여유 디스크가 15 GiB 미만일 때
- 연결은 3개다. 쓰기는 `.part` 파일을 만든 뒤 MANIFEST에 기록하고 이름을 바꾼다.

## 시험
- **NULL 시험** (`tests/test_bars.py`):
  - 09:30 이전 NULL 호가 → 이전 유효 호가가 시작 상태
  - 정규장 NULL 호가 → n_upd만 셈
  - 체결의 price·size·tr_scond·tr_corr NULL
  - time_m_nano NULL → 오류
  - 경고가 나지 않음 (경고를 오류로 바꾼 상태에서 시험)
  - NULL이 없을 때 이전 코드와 같음 (이전 코드는 `tests/ref_v1_*.py`로 고정)
- **실패 처리 시험** (`tests/test_download.py`): errors.tsv 기록과 계속 진행, 경고는 실패로 기록, 실패 21건째에 멈춤, 오류 종류 4개째에 멈춤
- **결과**: 전체 29개 통과. 버그를 일부러 넣은 사본 6개를 모두 시험이 잡았다.
