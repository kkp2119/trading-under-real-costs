(Korean)

# 연구 19 · 동결 기록 (FREEZE19, 2단계 작성, 커밋 전)

3단계에서 Kyle의 "커밋"으로 태그 `code-freeze-19`를 단다. run_test19.py는 봉인 실행 전에 아래 두 표의 sha256을 하나씩 확인하고, 하나라도 다르면 멈춘다.

## A*와 선택 근거
- **A* = CS40-M:** 월말 BAA10Y(신용 스프레드)가 전월 말보다 0.40%p 넘게 오르면 다음 달 포지션을 비운다. 확인은 월 1회다.
- **근거 (ADDENDUM 4절, 개발 보유 월 1990-01 ~ 2012-12, 276개월, 비용 4bp 차감 후):**
  - 절반 조건(1990–2001과 2002–2012의 수익 합이 모두 양수)은 필터 후보 21개 중 20개가 통과했다. V90-D만 탈락했다(앞 절반 −2.0%).
  - 샤프가 가장 높은 후보는 CS20-M(0.511)이고, C0는 0.351이다. 차이 +0.160은 기준 0.05보다 크다.
  - 0.05 안의 동률은 CS20-M(0.5105)과 CS40-M(0.5103) 둘이다. 그다음 후보 DV40-M(0.459)은 0.05 밖이다.
  - 둘 다 월 단위라서 비운 날 비율이 낮은 쪽을 골랐다: CS40-M 4.0%(11개월), CS20-M 8.0%(22개월).
- **봉인 가설:** H1 샤프(C0) > 0, H2 샤프(CS40-M) − 샤프(C0) > 0. 가설이 2개라 Holm 수준은 0.025, 0.05다.
- **검정력 (둘 다 "검정력 부족"):**
  - H1: MDE 0.685(α 0.05), 0.772(α 0.025). 개발 샤프는 0.351이다. SPEC19 5절에서 미리 표시한 대로다.
  - H2: 개발 차이 +0.160, 158개월로 환산한 표준오차 0.118, MDE 0.293(α 0.05), 0.330(α 0.025).
- **시행 수:** 751 → 773(개발 후보 22개). 봉인 실행 뒤 775(판정 2개).
- **개발 결과는 재실행본이다:** 세션이 한 번 중단돼서, Kyle 지시로 합성 테스트와 dev19.py를 12:14에 처음부터 한 번에 다시 돌렸다.
  - 중단 전 실행(11:20)과 숫자가 모두 같다: 요약 숫자 546개와 월 수익 6,072개의 차이 0, A*도 같다(DEVIATIONS19 2절).
  - 엔진을 쓰지 않는 별도 계산(`crosscheck19.py`)도 22개 후보 × 276개월에서 최대 차이 1.9e-16이다.
- **A1 경로:** h10. 일별 마르크 환율은 연준 H.10 과거 자료 페이지에서 받았다.
- **Kyle 승인 (2026-11-10):** DEVIATIONS19 1절의 해석 전부를 승인했다. 1번(A1은 연준 H.10 과거 자료 파일 사용)과 11번(2012-12를 2012-12-31 환율로 평가)을 포함한다.
- **전략 묶음 (PORTFOLIO19.md, 판정 아님):** 캐리(A*)와 옵션(연구 13·14의 설정 O2)을 개발 구간 변동성의 역수로 묶는다.
  - 고정 비중: 캐리 0.2625, 옵션 0.7375 (개발 1996-03 ~ 2012-12, results/portfolio19_dev.json).
  - 봉인 구간(2013-01 ~ 2025-08) 계산은 `portfolio19.py --sealed`가 하고, results/test19_summary.json과 test19_report.md가 있어야만 돈다.

## 문서 sha256
| 파일 | sha256 |
|---|---|
| `SPEC19.md` | `cff143aebd7a3e6c5e9290ce6969e1e161890ab9eb269b2a37eaafa52e77c9de` |
| `SPEC19_ANSWERS.md` | `d705da5732156314ce0879744b7f8e56ac6d96bb88e00602c8d143ef5fc51030` |
| `SPEC19_ADDENDUM.md` | `43df10c4c82cf1bbf4ad8bc95dd1fa4f5a26de661277123b107d32c99801bfba` |

## 동결 파일 (경로는 research19/ 기준)
| 파일 | sha256 | 역할 |
|---|---|---|
| `engine19.py` | `63cb3cb643345e3f5f5921ddb88575b36301ebdff76ffb2da0662d73bd695213` | 엔진: 달력, 월 1회 구성, 필터(M·W·D), 일별 손익과 비용, 통계, 부트스트랩, A* 선택 |
| `run_test19.py` | `c9783f735b212fdcff3d4f5faae228f3bafb353c47e88cbf399e0e204bf833c8` | 봉인 실행 스크립트 (ASTAR = "CS40-M"), 리허설 모드 포함 |
| `dev19.py` | `d5baa14f6a0a0f7500c75f5a9ec3e1f0f961d46c39635835b0125e8a358ab431` | 개발 실행: 자료 점검, 후보 22개, A* 선택, 검정력 |
| `test_engine19.py` | `c6f5063cbf89faec99bb187578366f48ec2d6c8337a2dba17c427a052e00ef14` | 엔진 합성 테스트 |
| `test_run19.py` | `08761309e46e45f89ce92b6a0f0ea51de17ecf7b5d96567e5c74e692a7340465` | run_test19 합성 테스트 (가짜 자료 폴더) |
| `a1_19.py` | `750f5afab2bb10d94c57ad1c27a5f46abd6ae5633402342ab8d44c229d9c6429` | A1 경로(일별 마르크)와 A2(일본 금리) 확인 |
| `fetch19.py` | `9f8b662fbdd1e4c9b666396ed77f9d2e4382b5e582b40ee863cc64135c627383` | FRED 내려받기와 2013-01-01 분리 |
| `fetch19_h10.py` | `2ab2fe8f5319cd53db70813d0dff1ac9163776174f7b100905a4064d857d0a53` | 연준 H.10 과거 자료(일별 DM/US$) 내려받기 |
| `check19.py` | `0a31e8356eb35cd607d02051c80ea5da22d5b1818cdd4dfecf1627e5b9f3f252` | 1단계 품질 확인 |
| `SEALED_MANIFEST19.tsv` | `e79d60a4340013058b7b5547b14f6a4c368ea760b91a93274c018a2ab5a441b3` | 본 시리즈 22개의 build·sealed 파일 sha256 |
| `SEALED_MANIFEST19_h10.tsv` | `c41fc63c7591aebb882dd58055e9f7ec53394a43456fea4281d0222845ebac05` | H10_DEM 파일 sha256 |
| `SEALED_MANIFEST19_candidates_20261110_104129.tsv` | `8ea0350ebccb29e1909992fecfda7f2a3b0aa388248d7ee3f1d599eb9336ff75` | 후보 시리즈 파일 sha256 (일본 CD 금리 포함) |
| `SEALED_MANIFEST19_candidates_20261110_104201.tsv` | `7b65f89985cd4200a2d179b43a9ccac2ac0ec151bb156295dcf3811f3657ae59` | 후보 시리즈 파일 sha256 (CCUSSP01DEM650N) |
| `DATA_MANIFEST19.tsv` | `0b1bad585ddb5ee9935a1c126fe4d92b5e2486060ef6ce44b4b37e0c983cc59d` | 내려받기 기록: 본 시리즈 |
| `DATA_MANIFEST19_h10.tsv` | `132116f8415fd2c73eb524552ad423b9eb6f17bb3d87d528518de4d04aeb39fb` | 내려받기 기록: 연준 H.10 독일 파일 |
| `DATA_MANIFEST19_candidates_20261110_104129.tsv` | `414c0a21f9e8dce7ea28fe5d0ce76dcbf612b33ca8e6bc0acf56ad99673ebf0c` | 내려받기 기록: 후보 시리즈 |
| `DATA_MANIFEST19_candidates_20261110_104201.tsv` | `17b30c11c01b13768d1f87aaf808cf2fbdaa6322e44486e06303ff3e808e2945` | 내려받기 기록: 후보 시리즈 |
| `results/dev19_summary.json` | `aec68f8253197f24f1193349491183a8ad95283024861bea062fc23b37acda73` | 개발 결과와 A* 선택 (12:14 재실행본, 실행 때의 엔진 sha256 포함) |
| `results/dev19_monthly.csv` | `8e5d1d0fe09e0cedd3f0cdcf681ed2bc2845bdd9afaf737155a32c2f71135ec3` | 개발 276개월 × 후보 22개의 월 순수익 (봉인 실행이 재현을 확인) |
| `results/a1_check19.json` | `08577ee3c37588afc828c2cc3b76cc2c71258bf2443529ac48e27f26344d5b69` | A1 품질 숫자와 경로 h10 |
| `results/engine19_tests.json` | `e28a89cbae6b42b1142ecaa5af1b0dbe40f71512efd8ca450fce7a78595bb27e` | 엔진 합성 테스트 84/84 통과 (12:14 재실행본) |
| `results/run_test19_tests.json` | `0113a0226ebe24865d1b398d146e60594522cbe620633946c1eabbb1506b619d` | run_test19 합성 테스트 21/21 통과 (12:16 재실행본) |
| `PORTFOLIO19.md` | `939e95e26cc4b67684600efe0ac7ba6ff52039043409984ef48eeed1ae11e517` | 전략 묶음 규칙(Kyle 원문 1 ~ 5절), 구현 해석(6절), 고정 비중(7절) |
| `portfolio19.py` | `9dbfb8a21294483ab4bd89cd99e4c3cbce2a5112798841c6734aebd97f3be6c6` | 전략 묶음 계산 (--dev, --sealed) |
| `test_portfolio19.py` | `7e46a05630956e47a9e4dd332bfa9f9a89e99a67af3b5df60fd521b0df9255ea` | portfolio19 합성 테스트 (가짜 입력) |
| `results/portfolio19_dev.json` | `7c41b5eec79a110933cabaaf1759e051fad7100779ff761fde3625721cd74b5b` | 개발 구간 묶음 결과와 고정 비중 |
| `results/portfolio19_tests.json` | `382ca11895a320d3c3243147f80c2d901c3e55e4027bb39f6e56598f02631893` | portfolio19 합성 테스트 17/17 통과 |

- **바깥 의존성:** numpy(a Python virtual environment). pandas와 scikit-learn은 쓰지 않는다.
- **자료:** `$DATA_DIR/research19_data/`의 build, sealed, candidates/build. 파일별 sha256은 위 매니페스트에 있고, run_test19.py는 읽은 파일을 모두 대조한다.
- **이 표에서 뺀 문서:** PREDICTIONS19.md와 DEVIATIONS19.md는 Kyle이 확정 전에 고칠 수 있다. 태그 커밋에는 함께 넣는다.
- **이 표에서 뺀 확인용 파일:** 봉인 실행이 쓰지 않는 파일이다.
  - `compare_dev19.py`와 results/dev19_rerun_compare.json: 재실행과 중단 전 결과의 비교
  - `crosscheck19.py`와 results/crosscheck19.json: 엔진을 쓰지 않는 별도 계산
  - 이름에 `_interrupted`, `_before_rerun`, `_run1`이 붙은 결과 파일: 이전 실행의 출력
- **옵션 칸의 입력:** `research13/results/dev_daily.csv`(`b6822a0a…`)와 `research13/study14/results14/holdout_daily14.csv`(`93843c74…`). 연구 13·14의 저장된 일별 결과다.
  - 두 파일의 sha256은 portfolio19.py 안에 적혀 있고, 실행할 때마다 대조한다.
  - 옵션 엔진과 홀드아웃 스크립트는 다시 돌리지 않는다.
- **주의:** 테스트(test_engine19.py, test_run19.py, test_portfolio19.py)나 a1_19.py를 다시 돌리면 결과 JSON의 작성 시각이 바뀌어 sha256이 달라진다. 그러면 guard가 멈추므로 이 표를 고쳐야 한다.

## 봉인 실행의 순서
1. guard: 결과 파일 유무, 잠금 문구, 위 sha256, A*·엔진·A1 경로 일치를 본다. 자료는 읽지 않는다.
2. build와 sealed 파일을 읽고 sha256을 매니페스트와 대조한다.
3. 날짜와 결측만으로 계획을 확인한다: 158개월(2013-01 ~ 2026-02) 모두 포지션과 청산일이 있어야 한다.
4. 개발 월 1990-01 ~ 2012-11을 전체 자료로 다시 계산해 results/dev19_monthly.csv와 같은지 본다.
5. 봉인 통계와 판정을 계산하고 결과를 쓴다.

- 2 ~ 4에서 멈추면 결과를 쓰지 않는다. 그 경우 봉인 수익은 계산되지 않은 상태다.

## 실행 명령 (4단계, 한 번)
Kyle이 `research19/UNLOCK19.txt`에 `연구19 테스트를 한 번 연다`를 직접 적은 뒤:

```
cd research19 && ../research15/py15.sh run_test19.py --go
```

- 출력은 results/test19_summary.json과 results/test19_report.md다. 화면 출력은 results/test19_stdout.log로 남긴다.

봉인 실행이 끝난 뒤, 전략 묶음의 봉인 구간 계산(판정 아님):

```
cd research19 && ../research15/py15.sh portfolio19.py --sealed
```

- 출력은 results/portfolio19_sealed.json과 results/portfolio19_sealed_report.md다. test19 결과 파일이 없으면 아무것도 읽지 않고 멈춘다.
