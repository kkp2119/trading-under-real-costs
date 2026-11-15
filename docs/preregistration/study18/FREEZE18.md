(Korean)

# 연구 18 · 동결 기록 (FREEZE18, 2단계 작성, 커밋 전)

3단계에서 Kyle의 "커밋"으로 태그 `code-freeze-18`을 단다. run_test18.py는 봉인 실행 전에 아래 표의 sha256을 하나씩 확인하고, 하나라도 다르면 멈춘다.

## A*와 선택 근거
- **A* = R0 (E* 그대로):** "개발에서 개선 없음"이다. 봉인 검정은 H1 하나만, α 0.05로 한다(SPEC18 4절, ANSWERS A-22).
- **근거 (SPEC18 4절):** 검증 구간(기존 유니버스 2019–2020, 주문 99,080건, 505일, L3D1)에서 고른다.
  - 평균 IS가 가장 낮은 후보는 R2_0.40(3.216¢)이다. R0는 3.217¢이고, 차이는 −0.001¢ [−0.004, +0.002]이다.
  - 0.02¢ 안의 동률은 8개다: R0, R1_0.75, R2_0.30, R2_0.35, R2_0.40, R3_0.75_0.30, R3_0.75_0.35, R3_0.75_0.40.
  - 8개 모두 p95가 18.20¢로 같았다. 그래서 발동 비율이 가장 낮은 R0(0)를 골랐다.
  - 조기 체결(R1)은 모든 기준값에서 R0보다 비쌌다: p_hi 0.60 +0.350¢, 0.65 +0.114¢, 0.70 +0.029¢, 0.75 +0.009¢.
- **H1:** 봉인 AUC > 0.5.
  - 모델은 model18.json이다. 학습 구간(2015–2018)에서 한 번 적합했고, 봉인 실행은 이 파일을 읽기만 한다.
  - 검증 AUC는 0.5687 [0.5655, 0.5717]이다.
- **H1 검정력:** 봉인 규모(20종목 × 1,468일)로 환산한 MDE(AUC − 0.5)는 α 0.05에서 0.0016이다(주문 수 비례, 일수만 비례하면 0.0023). 검증 효과 0.069보다 훨씬 작아서 "검정력 충분"이다.
- **H2·H3:** 없음(A* = R0).
- **시행 수:**
  - 지금까지 누적 750이다(모델 1개 + 후보 19개).
  - 봉인 실행에서는 비교가 1개(H1)뿐이라, 실행하면 751이 된다(Kyle 결정, DEVIATIONS18 1절 13).
  - SPEC18 8절의 753은 가설 3개를 전제로 한 숫자다.

## 문서 sha256
| 파일 | sha256 |
|---|---|
| `SPEC18.md` | `5a6abcfc2a11c109e0dadaff23cb1a8fc8268be7d080e5c9a6621d27820d2a36` |
| `SPEC18_ANSWERS.md` | `2d73d69d810d7ba70b44c6cdbfbfa92e701f47f97b291a236f11b94555562735` |
| `model18.json` | `8dff572ef84ce73d9e2d7fbc6cb0fa7eb5bd2baf856ba90039da56bbbd46e439` |

## 동결 파일 (경로는 research18/ 기준)
| 파일 | sha256 | 역할 |
|---|---|---|
| `engine18.py` | `21d9459edbc7cfa9c5c8077225b6db9ab0c2bc5a3535ab81eaaa57292a707065` | 엔진: 연구 17 커널 + R1/R2 + 지연, 격자 입력·y, 모델 적용, 통계 |
| `run_test18.py` | `eefdf782a83a7afb8181c0937fe34c531e45a5228450ba4916b8cbdbe315da78` | 봉인 실행 스크립트 (ASTAR = "R0") |
| `model18.json` | `8dff572ef84ce73d9e2d7fbc6cb0fa7eb5bd2baf856ba90039da56bbbd46e439` | 로지스틱 계수, 절편, 표준화 값, vol_pre 대체값 |
| `dev18.py` | `6fe99aa79707df8550d2a1d4d1f5a82c0f82e179ab8685571ac3bd7cf617b9d2` | 개발 실행: 회귀, 적합, 검증 구간 규칙, A* 선택 |
| `test_engine18.py` | `23336a2cb63babde6075b4eb14e086d7c8d72c13894f864326bac1a102b6516a` | 합성 테스트 |
| `../research15/loader15.py` | `d5d2beb11cea016265ab2af7828e0e8db8a4c2d9faa19e71cb589dff99b07fd6` | 개발 막대 로더 (리허설용, 원본 그대로) |
| `../research17/engine17.py` | `2a9ddcb7db7b96201cbd63b62e44c7c00140447a947e883aeeebf6a542d8d42a` | 합성 테스트의 비교 대상 (원본 그대로, run_test18은 쓰지 않음) |
| `SEALED_MANIFEST18.tsv` | `4b8e974fa3246e3d8ff2453c2c87dcd0e0eaaa537ab850380c435e11be23b47e` | 봉인 막대 28,727개의 sha256 |
| `EXCLUDE18.tsv` | `8e9aba3f4764337bdad9f1f8e3cb2e308232a7067e6e4c04572852cdfa01bf04` | 제외 140개 (GRUB 2021) |
| `calendar18.csv` | `934d06aaff1e9ba74ba4ba4431e085b4ebd0814819d7e2d12a793533da9b2477` | 거래일과 마감 시각 |
| `universe18_2021.tsv` | `af2e790e0f87c5b06519defe537bc99d7eed5a7b1a188b8190c193065f7f6f3b` | 새 유니버스 B |
| `universe18_2022.tsv` | `ad9fe6264c9c75ea10433a75aa2085ba7848eeaeda2e53a311d8c95401962db6` | 〃 |
| `universe18_2023.tsv` | `0da1dccbfa59ef0fe11e4251467fb86fb743b9af816aff785d0dbcaa6bb8fa8e` | 〃 |
| `universe18_2024.tsv` | `c2cd7582fe7cb463b8d9b0c17ea939341de0b3b65ddf2fa47dda74b71cc0656e` | 〃 |
| `universe18_2025.tsv` | `3a976e5d381e74d17cf61b479f7acf23fd49434ff94728c7137f483b3d5f4d20` | 〃 |
| `universe18_2026.tsv` | `4a0342f60fb1937aa906debd337343b0c07a840bc971f60327e492e5355eb417` | 〃 |
| `results/dev18_summary.json` | `81af8356e45e7c776eaf9544d6eaef4af28c1337f52da9de5a6cc398289d0f85` | 개발 결과와 A* 선택 |
| `results/engine18_tests.json` | `193d370052b6da327715c62d8f7da336ade679b4ab1aaa1927ac6500f40c8679` | 합성 테스트 78/78 통과 |

- **바깥 의존성:** numpy, pandas, numba(a Python virtual environment). scikit-learn은 dev18.py의 적합과 test_engine18.py에서만 쓰고, run_test18.py는 쓰지 않는다.
- **동결 커밋에 꼭 넣을 미추적 파일:** guard가 읽는 calendar18.csv, SPEC18.sha256, SPEC18_ANSWERS.sha256, model18.sha256, 그리고 model18.json과 위 결과 파일.
- **이 표에서 뺀 문서:** PREDICTIONS18.md와 DEVIATIONS18.md는 Kyle이 확정 전에 고칠 수 있다. 태그 커밋에는 함께 넣는다.

## 실행 명령 (4단계, 한 번)
Kyle이 `research18/UNLOCK18.txt`에 `연구18 테스트를 한 번 연다`를 직접 적은 뒤:

```
cd research18 && ../research15/py15.sh run_test18.py --go
```
