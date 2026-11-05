(Korean)
# SPEC_AMENDMENTS 초안 A5 — 주식 모드의 테이블 없는 날과 조회 실패 롤백 (2026-10-27, Kyle 승인, 커밋 전)

SPEC 내용은 바꾸지 않는다. SPEC 2.1 "2026-02-24는 complete_nbbo가 없어 막대가 없다(건너뛴 날로 세고 보고)"를 다운로더에 구현한 기록이다.

## 이유
- 주식 모드는 calendar.csv(ctm 테이블이 있는 날)를 그대로 작업 날짜로 썼다. 그래서 complete_nbbo가 없는 2026-02-24도 종목 10개 작업으로 들어간다. ETF 모드는 두 테이블이 모두 있는 날만 썼으므로 해당 없다.
- wrds `raw_sql`은 SQL 오류가 나도 롤백하지 않는다(소스 확인: `except sa.exc.ProgrammingError as e: raise e`). 테이블 없음 오류가 한 번 나면 그 SQLAlchemy 연결은 실패한 트랜잭션 상태로 남는다. 그 뒤 같은 연결의 모든 조회가 실패해 실패 한도(20)에 걸릴 가능성이 컸다.

## 수정 (download.py)
- (a) 주식 모드의 계획 단계에서 `table_gaps(db, days)`가 테이블 이름만으로(information_schema) 그 실행의 달력 날짜 중 complete_nbbo, ctm, mastm 중 하나라도 없는 날을 찾는다.
  - complete_nbbo·ctm 기준은 `trading_days`와 같다(가짜 DB 시험으로 일치 확인).
  - 그런 날은 작업과 마스터 확인에서 빠진다. 그날의 종목-일은 skipped_absent.tsv에 사유 `no <테이블> table`로 적힌다. 기존 사유는 `not in mastm`.
  - `split_days()`를 분리했고, `record_absent()`는 (종목, 날짜[, 사유])를 받는다.
- (b) 모든 조회는 `query(db, sql)`를 거친다. 실패하면 먼저 `db.connection.rollback()`을 하고(롤백 자체가 실패해도 무시), 원래 예외를 그대로 다시 던진다.
  - `wrds_fetcher()`는 `make_fetch(db)`로 나눴다. 가짜 DB 시험에서 이 함수를 쓴다.
- 경고를 실패로 다루는 설정, 실패 한도 20·종류 3, 디스크 하한 15 GiB, MANIFEST·null_counts 형식은 그대로다.

## 시험
- 개발과 시험은 scratchpad의 복사본에서 했다. 실행 중이던 원본은 1단계가 끝날 때까지 건드리지 않았다.
- 새 시험 tests/test_gaps_rollback.py(sha256 `7fdb2d8863a7b8fb91c024acda72cbbb44a1776509ee1320d48f4f6cbe7ecc80`)는 가짜 DB를 쓴다. 가짜 DB에서는 테이블이 없으면 오류가 나고 rollback 전까지 모든 조회가 실패한다.
  1. test_days_without_a_table_are_skipped_with_the_reason
     - 확인 내용: complete_nbbo 없는 날, mastm 없는 날, ctm·mastm 없는 날이 작업에서 빠지는지, 마스터 확인을 하지 않는지, 사유가 기록되는지, 재개 시 중복이 없는지, trading_days와 일치하는지.
  2. test_failed_query_rolls_back_and_the_next_task_succeeds
     - 확인 내용: 연결 1개, 작업 5개 중 테이블 없는 1개만 실패하고 나머지 4개가 쓰이는지, rollback이 1회인지.
  3. test_original_error_is_raised_even_when_rollback_fails
- 결과:
  - 복사본: 153개 통과(기존 150 + 새 3).
  - 교체 후 저장소: 153개 통과.
- 변이 점검(scratchpad 사본):

  | 사본 | 결과 |
  |---|---|
  | 수정 전 원본(`b53dd939…`) | 새 시험 3개 실패 |
  | 롤백만 뺀 코드 | 롤백 시험 2개 실패(연쇄 실패 재현) |
  | 날짜 제외만 끈 코드 | 날짜 제외 시험 1개 실패 |

## 교체
- 1단계(2025년 집계 20:30 완료, 주식 홀드아웃 2021~2025 20:47:56 완료)가 끝난 뒤 2026-10-27 20:48:12에 교체했다.
- download.py sha256:
  - 교체 전 `b53dd9392a6cca598b8059a2bb10ac93704e5337fa8c7ea629eaa1c44a31d999`(A4 2차 수정)
  - 교체 후 `4b7bcf7311cabb32f7492a2eb9e695436fe863aecd37f16dbc8485412aafa007`
  - bars.py는 그대로다(`9bc34fcad37b33fb3b07b4d9a82cdae485b3a48270b5ccdd7295ca322770d8a6`).
- 출처:
  - bars_1s_holdout_stocks의 2021~2025년 12,307개 파일은 교체 전 코드(b53dd939…)로 만들었다. 이 실행에는 테이블 없는 날이 없었다(아래 점검).
  - 이후 실행은 교체 후 코드로 한다.
  - 각 출력 폴더의 CODE_VERSION.tsv에 실행마다 코드 해시와 Python·numpy·pandas·wrds 버전을 1행씩 적는다. wrds 버전은 importlib.metadata로 읽는다. 첫 행(주식 홀드아웃 2021~2025)의 wrds 칸은 `?`로 남았다.

## 테이블 이름 점검 (홀드아웃·봉인 전 구간, 데이터 조회 없음)
- 대상: calendar.csv의 2021-01-04~2026-10-23 거래일 1,459일, information_schema 테이블 이름 조회 1회(연결 1개).
- complete_nbbo, ctm, mastm 중 하나가 없는 날: **2026-02-24(complete_nbbo 없음) 하루**. 그 밖의 날에는 세 테이블이 모두 있다.
- 따라서 같은 규칙을 적용할 다른 날은 없다. 2026-02-24는 2단계(2026년 주식) 실행에서 사유 `no complete_nbbo table`로 skipped_absent에 적힌다.
