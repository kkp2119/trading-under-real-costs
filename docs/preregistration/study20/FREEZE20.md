(Korean)

# 연구 20 · 동결 기록 (FREEZE20, 2단계 · 2단계 보완 · 3단계에서 작성)

3단계에서 Kyle의 지시로 커밋하고 태그 `code-freeze-20`을 단다. run_test20.py는 봉인 실행 전에 아래 표의 sha256을 하나씩 확인하고, 하나라도 다르면 멈춘다.

## A*와 선택 근거
- **개발에서 수익 없음:** 후보 16개 모두 개발 구간(쌍 2015–2020, 1,511일, 쌍·일 5,831개)에서 평균 일 손익이 음수다. 그래서 H2를 없애고, 봉인 검정은 H1 하나만 α 0.05로 한다(SPEC20 4·5절).
- **A* = W30_z3_k1_P (기계적 선택, SPEC20 4절):** L3D1 평균 일 손익이 가장 높은 후보다.
  - 평균 일 손익은 -976.8달러 [-1,090.3, -876.7]이고, 거래는 19,824건이다.
  - 5% 안의 동률은 없다. 다음 후보는 W30_z3_k1_M(-1,097.9달러)다. 가장 나쁜 후보는 W10_z2_k0_M(-24,206.2달러)다.
  - 거래당 손익은 -74.45달러다: 중간값 기준 +34.56 − 체결 비용 94.48 − 수수료 14.53.
  - 지연 1초 점검(L3D1): -968.8달러 [-1,080.0, -870.5]
- **H1:** 봉인 구간에서 신호만의 시간표로 두 다리를 중간값에 거래한 손익(비용 0)의 일 합 평균 > 0.
  - 신호 설정은 기본값이다(W = 30, z_in = 2, κ = 0). H2가 없기 때문이다.
  - 개발 값: 거래당 +24.77달러 [+23.46, +26.30], 일 합 평균 +3,400.7달러 [+3,096.4, +3,734.5], 거래 207,420건.
- **H1 검정력:** 봉인 1,470일 기준 MDE는 408.5달러/일(α 0.05), 460.3달러/일(α 0.025)이다. 개발 효과 +3,400.7달러/일보다 작아서 "검정력 충분"이다(VIF 13.64).
- **H2:** 없음. 결론 문장은 SPEC20 6절의 "개발에서 수익 없음" 줄을 쓴다.
- **보조(판정 아님):**
  - 봉인 실행은 기본 신호 설정과 A*의 신호 설정(W30_z3_k1)을 M·P, L3D1·L0D0로 함께 돌려 SPEC20 5절의 보조 항목을 낸다.
  - SPEC20_ADDENDUM B2의 지연 중간값 손익(1·5·30초)을 H1 기본 설정으로 내고, 5초 문장을 B2의 표에서 고른다.
  - 개발 값(5초 지연): 일 합 평균 +1,780.0달러 [+1,663.7, +1,904.1], 지연 0초의 52.3%.
- **호가가 비는 구간 (SPEC20_ADDENDUM B1):** 엔진과 H1 시간표가 같은 시점 규칙을 쓴다. B1의 문구를 A3에 맞췄고, 다른 곳은 없다.
- **Kyle의 결정 (2026-11-11):** 구현 해석 23개(DEVIATIONS20 1절)를 모두 승인했다. P 후보의 늦어진 30분 청산은 엔진을 그대로 두는 (가)로 정했다. 2단계 보완의 수정과 추가(DEVIATIONS20 3절 10번)를 승인했다.
- **시행 수:** 지금까지 누적 793이다(개발 후보 16개). 지연 중간값 점검은 세지 않는다. 봉인 실행에서 H1 1개를 더해 794가 된다(SPEC20 8절).

## 문서 sha256
| 파일 | sha256 |
|---|---|
| `SPEC20.md` | `5ba1c05b36f59378dd3b9e3aa530f35dd90ca06343c26283663594998eb1ab41` |
| `SPEC20_ANSWERS.md` | `a64fe06f29542fd9a5099db4427ff537c5612d8cb1488b9e57984f0e383f636c` |
| `SPEC20_ADDENDUM.md` | `37e7e43a6f80a9e517d19bd8f673c87c6856dd95008ed99b2d748b9aa2785460` |

- 우선순위는 SPEC20_ADDENDUM > SPEC20_ANSWERS > SPEC20이다.
- SPEC20_ADDENDUM.md는 2026-11-11의 수정본이다. 이전 판(sha256 `f7b05a2d…`)은 superseded_step2/SPEC20_ADDENDUM_step2b.md에 있다.

## 동결 파일 (경로는 research20/ 기준)
| 파일 | sha256 | 역할 |
|---|---|---|
| `run_test20.py` | `605eb3049b159b787fe9c60a18917519cdfe95b5d88d8dc58357302cca6eb1c3` | 봉인 실행 스크립트 (상수: A* = W30_z3_k1_P, H2 없음, H1 신호 W30_z2_k0) |
| `core20.py` | `35887641113e9b3ca581e986a3c0b06d67724438275e0f2dfff2de4469aa3b9c` | 모의 실행과 표: 개발 실행, 리허설, 봉인 실행이 함께 쓰는 경로 |
| `pair20.py` | `27623a242fd03fbb843ea71853f1d2409793c0b78664466fc5d4ae1b20f5e5d4` | 쌍 엔진: 진입·청산 판단, 두 다리 주문, 손익 |
| `signal20.py` | `e5d1127ccad05c9980cfc6c79dbffbfc2fb19c5fef40d60674af05cf33bad224` | 신호: 5초 격자, z, 진입·청산 표시, 신호만의 시간표(H1), 지연 중간값(B2) |
| `stats20.py` | `45f16c9e7b18e05082b8d4e401fefc401ab1685601b8f59626ea9ed3d16942f0` | 통계: 일 손익, 부트스트랩(seed 20), A* 규칙, MDE, Holm |
| `loader20.py` | `b6fc30b560e66897fe3a502a4e1dccc3a0328e2118a2d62ae110466d5e0cce30` | 막대 로더 (기본은 개발 구간만) |
| `../research17/engine17.py` | `2a9ddcb7db7b96201cbd63b62e44c7c00140447a947e883aeeebf6a542d8d42a` | 체결 커널 (연구 17 원본 그대로, 고치지 않음) |
| `dev20.py` | `d56c055404f61bee016865ecc7f09e475ce438aac9dec9c115555a98944b83ae` | 개발 실행: 후보 16개, A* 선택, 검정력, 지연 점검 (2단계에서 한 번 실행) |
| `delay20.py` | `fce53f827094e4441a9ca39f82668b42a2c7cbdcdba90c0720d89ed6d6954759` | 2단계 보완: 지연 중간값 손익의 개발 표, 2단계 숫자 재확인 |
| `gapcheck20.py` | `b99ad183daff7cc253640af599558c117b7191691eaeb2da83b66f0ddbcb500a` | 2단계 보완: 호가가 비는 상황의 개발 구간 발생 횟수 |
| `gridcheck20.py` | `50e966314b940447e34cd3c718003966739a9bd0841d4b029b748f5a17478577` | 2단계: 격자 유효성 개수 (DEVIATIONS20 1절 3번의 근거) |
| `test20.py` | `388cf5bad6dbedb812c0323bfe0ccee06121262b2436141ff127bc5a9912f1fa` | 합성 테스트 |
| `regress20.py` | `184d5beb95d1fd5a2484533ace7fa14d385232efeeca3e9aa96c4009dfd084ee` | 회귀 테스트 (연구 17 값 재현) |
| `../research15/loader15.py` | `d5d2beb11cea016265ab2af7828e0e8db8a4c2d9faa19e71cb589dff99b07fd6` | 회귀 테스트의 로더 (원본 그대로, run_test20은 쓰지 않음) |
| `SEALED_MANIFEST20.tsv` | `3e7f178852fdd69b93696391e4cf9abed8a2d64ffcf59b7f7a87363e2ef7fa36` | 봉인 막대 11,594개의 sha256 |
| `DATA_MANIFEST20.tsv` | `db178fb22f6c12440a2210f065be552247fd78cd0336a10fb70f2a2694764d00` | 개발 막대 11,676개의 sha256 (리허설용) |
| `calendar20.csv` | `ccdc0db7e2b98db958d2acc7a50f6b6b7f4eb2708fb41ff6ab22cfd49de65541` | 거래일과 마감 시각 |
| `universe20_2015.tsv` | `76093a7c961d4d46b65a7588470d8d7060448755e9c3009d1075b421cf92c39c` | 쌍 명단 |
| `universe20_2016.tsv` | `62e7ef4962e7260a756c6e4f82257bdfca2e6c4d9d54882256a75311d0c9cc70` | 〃 |
| `universe20_2017.tsv` | `9d3839aeb96bd867c567eb81774c7a0f805b19d3d471ef1a3b0ce6de362386b2` | 〃 |
| `universe20_2018.tsv` | `16ec8438019e45c577b8b8d80f0b3ef64662f37848975ecac75c5cb219b1b616` | 〃 |
| `universe20_2019.tsv` | `8d675572ecc31d080c40e4e4d16fe4e37ee44e257d239f661e58de2bdbb6ad6f` | 〃 |
| `universe20_2020.tsv` | `b84592afb22cb9062eb52b05436d95dc484a6615ec3423b58c8aa5e4feffa47c` | 〃 |
| `universe20_2021.tsv` | `ec075fda09b4bdacd6d571c183785f70e11f5d85ed95f4f93584a20a3c38c697` | 〃 |
| `universe20_2022.tsv` | `a494cf25bf358ce3bec8aadb2ca9b8ae3bcd412adaa3098e0d6a1e407ec8d032` | 〃 |
| `universe20_2023.tsv` | `3dc1900964416ff42042ed3668fa7f806da3b1dc6a0e6c5f839a88b1cd24ff5e` | 〃 |
| `universe20_2024.tsv` | `9f571ced5f68f284b68c91dea49a5c67b928ac7b7deebae8a032a241c363396a` | 〃 |
| `universe20_2025.tsv` | `1d81c80aad73765b1fc341ac67112566243b84e276558383ffc57a6310818c1b` | 〃 |
| `universe20_2026.tsv` | `3ba6dbb1fd5e67d5bb1539343c71d471ce887e96abe34bd38889e1759a3532cd` | 〃 |
| `results/dev20_summary.json` | `5831ff12981d7f755770ea042934cdbc9c8a27bd82c2892a29276ebe60796e04` | 개발 결과와 A* 선택 (2단계, 다시 돌리지 않음) |
| `results/delay20_summary.json` | `df3835e35d7f5326f32a8bafddb479bd06f0d3e6df3ad84dd23dc8c2266d5fb5` | 지연 중간값 손익의 개발 표 (리허설이 대조) |
| `results/gapcheck20.json` | `b1d0300cd6e7514553a870d1ff525b30aafd09e93ec0441d53944b69ef667b20` | 호가가 비는 상황의 개발 구간 발생 횟수 |
| `results/gridcheck20.json` | `ffa2e40e5ce9a73b3dca7ad668e3b9ad85e7d126f88455525cc82250887932d6` | 격자 유효성 개수 (DEVIATIONS20 1절 2·3번의 근거) |
| `results/engine20_tests.json` | `fc74cdc5796f4cf8a67e290e4d95a9a3ad644078cfeaaf27b3d8b25970258f92` | 합성 테스트 117/117 통과 |
| `results/regress20.json` | `e05d03977352a07c891092a700f6e35d920b3a33c7938313913f770394a33cb2` | 회귀 테스트 통과 |
| `results/rehearsal20_summary.json` | `a08d14ae8ba861f11ed44399f6f7aaff1a8768e5f5d223cf16f1e5a537dfd237` | 리허설 결과 (2단계 숫자와 일치) |

- **바깥 의존성:** numpy, pandas, numba, scipy(a Python virtual environment). scipy는 dev20.py의 검정력 계산과 test20.py에서만 쓰고, run_test20.py는 쓰지 않는다.
- **동결 커밋에 꼭 넣을 미추적 파일:** 위 표의 2단계 파일 전부, SPEC20_ADDENDUM.md, 그리고 guard가 읽는 SPEC20_ANSWERS.sha256과 SPEC20_ADDENDUM.sha256.
- **이 표에서 뺀 문서:** PREDICTIONS20.md와 DEVIATIONS20.md는 Kyle이 확정 전에 고칠 수 있다. 태그 커밋에는 함께 넣는다.
- **2단계 보완에서 바뀐 파일:** signal20.py, pair20.py, core20.py, run_test20.py, test20.py. 바뀌기 전 판은 superseded_step2/에 있다.
  - 개발 표 40줄(후보 16개 × 가정 2개, 중간값 8줄)을 바뀐 코드로 다시 계산해 2단계 값과 대조했다. 최대 차이는 0이다(results/delay20_report.md).
- **3단계에서 바뀐 파일:** SPEC20_ADDENDUM.md(수정본), 문구만 고친 test20.py와 gapcheck20.py, 다시 만든 results/engine20_tests.json과 results/rehearsal20_summary.json. 엔진, 신호, 통계 코드는 2단계 보완과 같은 파일이다. 2단계 보완 판은 superseded_step2/에 `_step2b` 이름으로 있다.
- **리허설:** run_test20.py --rehearsal을 3단계에서 개발 데이터에 다시 돌렸다. 2단계 개발 숫자, 지연 중간값 표와 최대 차이는 0이고, 봉인 경로는 읽지 않았다(results/rehearsal20_report.md).

## 실행 명령 (4단계, 한 번)
Kyle이 `research20/UNLOCK20.txt`에 `연구20 테스트를 한 번 연다`를 직접 적은 뒤:

```
cd research20 && ../research15/py15.sh run_test20.py --go
```

- 결과는 results/test20_summary.json, test20_report.md, test20_daily.csv에 한 번만 쓴다. 결과 파일이 있으면 다시 돌지 않는다.
