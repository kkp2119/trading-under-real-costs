(Korean)
# SPEC_AMENDMENTS 초안 A4 — 0행 조회 결과 처리 (2026-10-27, Kyle 승인, 커밋 전)

## 사건
- 주식 개발 다운로드(`download.py --split dev --stocks --years 2019,2020`, 17:52:30 시작)가 17:55:38에 실패 한도를 넘어 스스로 멈췄다.
  - 실패 21건(한도 20), 오류 종류는 1개: `FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated`.
  - 종목별로 WCC 11, CBT 5, MSI 3, CATM 2건.
  - 그때까지 1,146개를 썼고 3,788개는 시작하지 않았다. 재시도는 없었다.
- 원인은 `day_bars`의 `pd.concat([pre, nbbo])`다.
  - raw_sql은 0행 결과를 object 열로 돌려준다.
  - 한쪽만 0행이면 다른 쪽의 nullable dtype과 달라 pandas 2.2.3이 FutureWarning을 내고, 다운로더는 경고를 실패로 다룬다.
  - 합성 재현 결과, 경고는 한쪽만 0행일 때만 난다. 전부 NULL인 열이거나 둘 다 0행이면 나지 않는다.
  - ETF는 개장 전 호가가 없는 날이 없어서 지금까지 드러나지 않았다.
  - 실패한 21일에 어느 쪽이 0행이었는지는 재개 뒤 집계해 아래에 덧붙인다.

## 변경 (download.py 한 줄)
- 전: `q = pd.concat([pre, nbbo], ignore_index=True)`
- 후: `q = pd.concat([f for f in (pre, nbbo) if len(f)] or [nbbo], ignore_index=True)`
  - 0행 프레임은 합치기 전에 뺀다. 둘 다 0행이면 nbbo를 그대로 쓴다.
  - pandas 2.2.3이 경고와 함께 내던 결과(0행 항목은 dtype 결정에서 제외)와 같다.
  - 개장 전 호가가 없으면 첫 정규장 유효 갱신 전까지 상태는 0이다(SPEC 2.1 "첫 갱신 전 0"과 같음).
- 경고를 실패로 다루는 설정은 바꾸지 않았다.
- download.py sha256:
  - 전 `a6f7612f18c1adb0378203f3add9da655eeef2f7c2ef0d54b1946dece313ea60`
  - 후 `b4415b27f790150d5ce4720e15305c9057cc352b8ef964a55d219076748ea644`
- bars.py는 바뀌지 않았다(`9bc34fcad37b33fb3b07b4d9a82cdae485b3a48270b5ccdd7295ca322770d8a6`).

## 출처 기록
- `bars_1s_dev_stocks`의 처음 1,146개 파일(17:52~17:55 작성, 2019년 일부)은 **수정 전 코드(a6f7612f…)**로 만들었다. 이후 파일은 수정 후 코드(b4415b27…)로 만든다.
- 두 코드는 두 프레임이 모두 1행 이상일 때 같은 결과를 낸다. 1,146개 파일은 경고 없이 쓰인 날, 곧 두 프레임이 모두 1행 이상이었던 날이다.
- ETF 개발 5,050개 파일은 모두 수정 전 코드로 만들었고, 그 실행에서는 이 경고가 한 번도 나지 않았다.

## 같은 파일의 다른 지점 점검
- download.py에서 pd.concat는 이 한 곳뿐이다.
- groupby는 ETF 모드 거래일 목록(테이블 이름, 데이터 아님) 한 곳뿐이고 merge는 없다.
- 0행 object 프레임이 지나가는 다른 처리(to_timedelta, astype Float64/Int64, fillna, map)는 합성 시험으로 점검했다:
  - 체결(ctm) 0행인 날
  - 호가는 있는데 유효 체결이 전부 제외된 날
  - 개장 전과 정규장이 모두 0행인 날
  - 세 조회가 모두 0행인 날
  - 이 경우들은 수정 전 코드에서도 경고가 나지 않았으므로 **추가로 막은 곳은 없다**.

## 시험 (tests/test_empty_frames.py, sha256 f1090bb1db53341a930a02a58afbff9e4d0207377542cbcf94996029f6c72fae)
- 모든 시험은 경고를 오류로 다룬 상태에서 실행했다.
- 기준은 수정 전 코드 사본(ref_v1_download)에 0행 object 프레임 대신 그 조회의 dtype을 가진 0행 프레임을 넣은 결과다. 이를 막대 21열(dtype, 형태, 바이트)과 통계까지 비트 단위로 비교한다.
- 시험 목록(합성 시드 1, 2, 3):
  - test_preopen_zero_rows
  - test_session_zero_rows
  - test_preopen_and_session_zero_rows
  - test_trades_zero_rows
  - test_all_trades_excluded
  - test_preopen_zero_rows_and_all_trades_excluded
  - test_all_three_zero_rows
- 결과: 전체 55개 통과(기존 36개 + 새 19개, NULL 없는 등가 시험 3개 포함).
- 변이 점검(scratchpad 사본에서 180행만 수정 전으로 되돌림):
  - 새 시험 9개가 실패했다(개장 전 0행 3개, 정규장 0행 3개, 개장 전 0행 + 전부 제외 3개).
  - 나머지 10개는 수정 전 코드로도 경고 없이 통과했다. 이것이 위 점검 결과의 근거다.

## 2차 정지와 정정 (2026-10-27 18:52)
- 수정 후 코드(b4415b27…)로 재개했다(18:51:52 시작).
  - 18:52:14에 같은 FutureWarning으로 **같은 21개 종목-일**이 다시 실패해 한도에서 멈췄다.
  - 새로 쓴 파일은 2개(AMN·WCC 2019-06-19)다.
  - errors.tsv는 42행(21 × 2회), MANIFEST·npz는 1,148개다.
- **정정**: 위 "사건" 항목의 "전부 NULL인 열이면 나지 않는다"는 틀렸다.
  - 합성 시험은 전부 NULL인 열을 Float64 같은 nullable dtype으로 만들었다.
  - 그러나 raw_sql(`pd.read_sql_query(..., dtype_backend="numpy_nullable")`)은 **값이 전부 NULL인 열을 object로 돌려준다**(합성 확인: 전부 None인 object 열은 convert_dtypes 뒤에도 object).
  - 이런 열이 있는 프레임을 다른 프레임의 Float64/Int/string 열과 합치면 0행이 아니어도 같은 경고가 난다.
  - 실패한 21일은 개장 전(또는 정규장) 행은 있지만 어떤 열이 전부 NULL인 날로 보인다. 어느 열인지는 데이터를 열지 않았으므로 모른다.
- **전수 점검**(현재 코드에 열 하나만 전부 NULL(object)인 합성 프레임을 넣고 경고를 오류로 다룸):
  - 개장 전 7개 열(best_bid, best_bidsizeshares, best_ask, best_asksizeshares, qu_cond, qu_source, natbbo_ind)과 정규장 7개 열은 모두 182행 concat에서 FutureWarning이 났다.
  - 체결 4개 열(price, size, tr_scond, tr_corr)은 경고가 없었다.
  - 두 프레임의 같은 열이 둘 다 전부 NULL인 경우도 경고가 없었다(둘 다 object).
  - concat 밖의 처리(to_timedelta, astype, fillna, map)에서는 경고가 나지 않았다.
- **제안하는 2차 수정(Kyle 승인 전, 적용하지 않음)**: 0행 프레임을 뺀 뒤 두 프레임이 남으면, 한쪽에서 전부 NULL이고 다른 쪽에는 값이 있는 열을 값이 있는 쪽의 dtype으로 바꾼 다음 합친다.
  - 값이 없는 열의 dtype만 바뀌므로 결과는 pandas 2.2.3이 경고와 함께 내던 결과(전부 NA인 항목은 dtype 결정에서 제외)와 같다.
  - 이후 A3 규칙(가격·수량 NULL → 무효 행)이 그대로 적용된다.
- **시험 보강 제안**: raw_sql 형식(object, 전부 None)의 전부 NULL 열을 세 프레임의 각 열에 하나씩 넣은 경우(14 + 4가지)와 두 프레임 모두 전부 NULL인 경우.
  - 기준: 같은 열을 nullable dtype의 전부 NA로 둔 입력의 결과(현재 코드가 경고 없이 내는 결과)와 비트 단위로 같아야 한다.
  - 1차 수정 코드로 되돌리면 실패해야 한다.

## 2차 수정 (Kyle 승인 2026-10-27)
### 형식 재현 (먼저 확인)
- 메모리 SQLite 표를 wrds `raw_sql`과 같은 인자로 읽었다(`pd.read_sql_query(..., coerce_float=True, dtype_backend="numpy_nullable")`). raw_sql 기본 chunksize 500000은 결과가 그보다 작으면 같은 결과다.
  - sqlite3 연결과 SQLAlchemy 연결(wrds와 같은 `engine.connect()`) 둘 다로 읽었다.
- 결과:
  - **전부 NULL인 열은 object가 아니라 `string`(StringDtype, 값은 pd.NA)으로 나왔다.** SQL 열 형식(real, integer, text)과 관계없었다.
  - 0행 결과는 모든 열이 object였다.
  - 둘 다 Float64·Int64 열과 합치면 FutureWarning이 난다.
- pandas는 SQL 열 형식이 아니라 돌려받은 값으로 dtype을 추론한다. 그래서 Postgres(WRDS)에서도 같은 형식일 것으로 본다. WRDS로 직접 확인하지는 않았다.
- **정정**: 위 "2차 정지와 정정"의 "object로 돌려준다"는 정확하지 않다. 전부 NULL 열은 string이고, object는 0행 결과다.
- 시험 입력은 이 SQLite 경로로 만든다(tests/rawsql.py). WRDS 드라이버가 object를 줄 가능성에 대비해, 손으로 만든 object 열(전부 None, 전부 pd.NA)도 보조 시험으로 넣었다.

### 변경
- `download.concat_raw(frames)` 함수 하나로 모으고, `day_bars`의 concat(download.py의 유일한 concat 지점)이 이 함수를 쓴다.
  - 0행 프레임은 뺀다. 모든 프레임이 0행이면 마지막 프레임(nbbo)을 그대로 쓴다.
  - 남은 프레임에서 한쪽이 전부 NULL이고 다른 쪽에 값이 있는 열은 값이 있는 쪽의 dtype으로 바꾼 뒤 합친다.
  - 값이 없는 열의 dtype만 바뀌므로, pandas 2.2.3이 경고와 함께 내던 결과와 같다.
- 경고를 실패로 다루는 설정은 그대로다. bars1s의 다른 파일에는 concat가 없다(tests/ref_v1_download.py는 고정된 비교용 사본이라 바꾸지 않음).
- download.py sha256:
  - 1차 수정 `b4415b27f790150d5ce4720e15305c9057cc352b8ef964a55d219076748ea644`
  - 2차 수정 `b53dd9392a6cca598b8059a2bb10ac93704e5337fa8c7ea629eaa1c44a31d999`
  - 1차 수정 코드로 만든 파일은 AMN·WCC 2019-06-19 두 개다.

### 시험
- 파일:
  - tests/rawsql.py `0f787fd8e7f20071d7f46d3f8110869f8afb242aa9b6fc45a77fd46d760c71a6`
  - tests/test_all_null_columns.py `66cd9f83c70589789e38cf99ab76881d71c6fa6dd11a1259afb6515eb4a3057f`
  - tests/test_empty_frames.py(0행 프레임을 SQLite로 만들도록 바꿈) `26f5b276fa336ff6c8337080c97db8205577a0f71a4a795f70933b63e56fd6e0`
- 새 시험(모두 경고를 오류로 다룸, 기준은 같은 열을 nullable dtype의 전부 NA로 둔 입력의 결과, 막대 21열과 통계를 비트 단위로 비교):
  - test_raw_sql_format_is_reproduced: string 형식, 0행 object, 이 환경에서 경고가 실제로 나는지 확인
  - test_one_all_null_column: 18가지 × 시드 2
  - test_same_column_all_null_in_both_quote_frames: 7열 × 시드 2
  - test_all_null_column_with_the_other_quote_frame_zero_rows: 7열, 개장 전·정규장 양방향
  - test_object_all_null_column_supplementary: 18가지 × (None, pd.NA)
  - test_concat_raw_changes_only_value_free_columns
  - 1차 수정 시험(test_empty_frames, 0행 조합 19개)도 함께 실행
- 결과: 전체 150개 통과(기존 55 + 새 95).
- 변이 점검(scratchpad에서 1차 수정 코드로 복원, sha256이 b4415b27…와 같음을 확인): 새 시험 41개가 실패했다.
  - test_one_all_null_column 12개: 정규장의 가격·수량 4열, 개장 전의 bid·ask 가격 2열(각 시드 2개).
  - 보조 object 시험 28개: 호가 14열 × 2.
  - concat_raw 시험 1개.
  - 문자열 열(qu_cond, qu_source, natbbo_ind)은 SQLite 형식에서 양쪽 모두 string이라 1차 수정 코드로도 경고가 없었다. 체결 열, 양쪽 모두 NULL인 열, 0행 조합도 같았다.
