(Korean)

# 연구 23 · 동결 기록 (FREEZE23, 2단계 3절에서 작성)

**상태 (2026-11-13 14:40):** Kyle의 지시(2026-11-13)로 커밋하고 태그 `code-freeze-23`을 단다. 봉인 테스트는 아직 돌리지 않았다. run_test23.py는 봉인 실행 전에 아래 표의 sha256을 하나씩 확인하고, 하나라도 다르면 멈춘다.

**4절 (2026-11-13 14:35 ~ 14:36):** 유니버스 D 수집(14:35:07 종료, 파일 28,072개, 실패 0건, 2.39 GiB)과 품질 확인, 두 가지 단위 확인이 모두 통과했다. `SEALED_MANIFEST23.tsv`의 sha256을 "동결 파일" 표에 넣었다. 내용은 DEVIATIONS23.md 2절에 있다.
- **호가 잔량의 단위:** 2025-10-31까지는 모든 종목·일의 최대공약수가 100이고, 2025-11-03부터는 모두 10,000이다. 로더의 보정(100으로 나눔)이 모든 종목에 맞는다.
- **체결 수량의 단위 (답변 A4):** 체결이 있는 모든 종목·일의 최대공약수가 앞뒤 모두 1이다(주 단위). 체결 한 건당 수량의 앞뒤 비율은 0.73 ~ 1.82다. 100배 차이는 없다. 체결 수량은 보정하지 않는다.
- **종일 호가가 없는 파일 6개:** AQUA 2023-05-24, GRA 2021-09-22, MDLA 2021-10-29, PPBI 2025-09-02, SKYT 2026-07-31, SOVO 2024-03-12. 단위가 다른 것으로 세지 않고 따로 적었다. 봉인 실행에서 이 종목·일은 주문 0건이 된다(연구 22의 TPH, KALV와 같은 처리).

**동결 전 확인 (2026-11-13 14:36 ~ 14:39, Kyle의 지시):**
- **합성 테스트:** 71/71 통과(연구 18의 78/78, 연구 20의 117/117, 연구 22의 49/49 포함).
- **리허설:** 통과. 검증 실행의 숫자와 최대 차이 0이다(숫자 329개). 봉인 경로는 읽지 않았다.
- **guard:** `run_test23.py --go`는 "refused: no unlock phrase in UNLOCK23.txt" 하나로 멈춘다.
- **sha256 기록:** 두 결과 파일을 새로 써서 표의 두 줄(results/engine23_tests.json, results/rehearsal23_summary.json)과 기록만 하는 줄(리허설 보고서와 일 단위 파일)을 고쳤다. 앞선 결과는 results/superseded/에 있다.

## A*와 선택 근거 (검증 구간: 기존 유니버스 2019–2020, 505일, 주문 99,080건)
- **A* = R0 (E* 그대로)다. H2는 없다.**
  - L3D1 평균 IS가 가장 낮은 후보는 R2_0.40(3.211¢)다. R0는 3.217¢이고 차이는 0.006¢다.
  - 0.02¢ 넘게 낮지 않으므로 SPEC23 4절과 답변 A6대로 A* = R0다. 후보끼리의 동률 규칙은 쓰이지 않았다.
  - SPEC23 6절의 문장: "신호를 합친 예측은 검증 구간에서 E*를 이기지 못했다. 그래서 개선 가설은 봉인 데이터로 시험하지 않았다"
- **봉인 검정:** H1(유니버스 D에서 입력 9개 모델의 AUC > 0.5) 하나다. Holm 수준은 0.05다.
- **H1의 검증 값:** AUC 0.5745, 95% CI [0.5707, 0.5783]. 연구 18 모델은 0.5687 [0.5656, 0.5718]이고, 차이는 +0.0059 [+0.0047, +0.0071]다.
  - 적중률은 0.5533(연구 18 모델 0.5464), Brier skill score는 +0.0154(연구 18 모델 +0.0139)다.
- **검정력 (참고, 유니버스 D: 연 20개, 1,471일, 단측 α 0.05, 80%):** AUC − 0.5의 MDE는 주문 수로 환산하면 0.0020, 일수만으로 환산하면 0.0028다. 검증 값 0.0745보다 작다.

### 동결 모델 (학습 2015–2018, 시점 10,567,048개, 한 번 적합)
| 입력 | 표준화 계수 | 그 입력을 뺐을 때 줄어드는 검증 AUC (참고) |
|---|---|---|
| 1 잔량 불균형 | +0.2943 | +0.0418 |
| 2 스프레드 | +0.0041 | -0.0000 |
| 3 vol_pre | +0.0018 | -0.0000 |
| 4 직전 5초 중간값 변화 | +0.0255 | -0.0003 |
| 5 a_t (SPY 기준) | +0.0642 | +0.0011 |
| 6 a_t (자기 가격 기준) | -0.0564 | +0.0005 |
| 7 체결 불균형 30초 | +0.1514 | +0.0015 |
| 8 체결 강도 60초 | +0.0009 | -0.0000 |
| 9 7번 × 8번 | +0.0035 | +0.0000 |

- 절편 -0.0006. 채움값: vol_pre 5.4808 bp, 체결 강도 0.4970. model23.json sha256 `61b6610038345f60e088df7260a84a380c2aa5665bdd43d03a17b0be86c5bc04`.

### 후보 (L3D1, 일부. 전체 표는 results/dev23_report.md)
| 후보 | 평균 IS (¢) | − R0 [95% CI] | 발동: 주문 |
|---|---|---|---|
| R0 (E*) | 3.217 | — | — |
| R1_0.60 | 3.514 | +0.297 [+0.238, +0.356] | 50.7% |
| R1_0.65 | 3.309 | +0.092 [+0.061, +0.123] | 22.5% |
| R1_0.70 | 3.242 | +0.025 [+0.010, +0.042] | 2.8% |
| R1_0.75 | 3.221 | +0.004 [+0.001, +0.007] | 0.1% |
| R2_0.30 | 3.217 | -0.000 [-0.001, +0.000] | 0.2% |
| R2_0.35 | 3.216 | -0.001 [-0.002, +0.001] | 1.4% |
| R2_0.40 | 3.211 | -0.006 [-0.010, -0.002] | 5.1% |
| R3_0.60_0.40 | 3.510 | +0.293 [+0.234, +0.353] | 55.0% |
| R3_0.65_0.40 | 3.304 | +0.087 [+0.056, +0.118] | 27.3% |
| R3_0.70_0.40 | 3.236 | +0.019 [+0.003, +0.038] | 7.8% |
| R3_0.75_0.40 | 3.215 | -0.002 [-0.007, +0.004] | 5.2% |

- 조기 체결(R1)은 문턱이 낮을수록 비용이 늘었다. 대기 유지(R2)는 주문의 0.2 ~ 5.1%에서만 발동했고 R0와의 차이가 0.006¢ 이하다.
- **비교 (참고):** 연구 18의 A*는 R0였다. 연구 18 모델로 돌린 같은 19개 규칙 가운데 평균이 가장 낮은 것은 R2_0.40(3.216¢, R0와의 차이 -0.001¢)다.
- 대조 실행(1 − p)과 검증 구간의 지연 점검은 A* = R0라서 돌리지 않았다.
- **봉인 실행이 돌리는 것:** S0, S2i, R0와 C3(S2i와 R0에서 고름), 그리고 R0의 지연 실행이다. 후보는 돌리지 않는다. 두 모델(입력 9개 모델, 연구 18의 동결 모델)을 모든 결정 격자 시점에서 채점한다.
- **선택의 전달:** run_test23.py는 A*와 H2 유무를 상수로 적지 않고 results/dev23_summary.json에서 읽는다. 이 표가 그 파일과 model23.json의 sha256을 고정한다. 검증 결과 파일에 적힌 모델 해시가 model23.json과 다르면 멈춘다.
- **Kyle의 승인:** DEVIATIONS23 1절의 31개 전부(2026-11-13 14:03, 학습 전. 개수는 14:30에 Kyle이 31개가 맞다고 확인했다).
- **시행 수:** 모델 1개와 후보 19개로 826 → 846. 봉인 판정 1개로 847이 된다.

## 문서 sha256
| 파일 | sha256 |
|---|---|
| SPEC23.md | 85a81015b759e5a606031bb098065bc89e4b82edc7d5a844d9d8b1d84cc18809 |
| SPEC23_ANSWERS.md | 52eb4b18d6c979ab7c6931c6d3ad3f4f0d53f9140c75a6d4d277e51df44a12d1 |

- guard는 두 문서를 각각의 .sha256 파일과 대조한다.
- PREDICTIONS23.md(sha256 `d5a0ca79…`, 검증 결과를 보기 전인 14:06에 SPEC23 7절을 그대로 옮김)와 DEVIATIONS23.md는 Kyle이 확정 전에 고칠 수 있어서 표에 넣지 않았다.

## 동결 파일 (경로는 research23/ 기준)
| 파일 | sha256 | 역할 |
|---|---|---|
| `run_test23.py` | `fe29e06510b169c3c43b8a6555b155204ecf8418650048802f488aa9cdda8a30` | 봉인 실행과 리허설 |
| `core23.py` | `09ce9662bb1ed2222151393974b25fe6a76d2cb47b3a82526675d4c6cc073669` | 실행 묶음과 통계 |
| `feat23.py` | `e2667f7356225d5c89c80e3085a8bdaf1a3e87a3ef5672e6ffe36ea462dd055d` | 입력 9개 |
| `model23.py` | `9bab7a3d95e293036018246844e9dd91191d014ba15995f95a53247c35eb3fb4` | 학습과 채점 |
| `dev23.py` | `f9c8912e1061125301519dda29cd99344e5a97d6dc07807bcef95009258359dd` | 학습과 검증 실행 (각각 한 번 돌렸다) |
| `test23.py` | `90791a1b5c668a772fdaddfdfab3293a1edfaf9d4c42ebc5c0fbca1dd4edeb6c` | 합성 테스트 |
| `regress23.py` | `69a10baf43e9714c35c7a479cf37b35f0625a77f84c17e075d0054c272b653dc` | 회귀 테스트 |
| `model23.json` | `61b6610038345f60e088df7260a84a380c2aa5665bdd43d03a17b0be86c5bc04` | 동결 모델 (입력 9개, 2026-11-13 14:05 학습) |
| `model23.sha256` | `30ceaa83bd5a94a5a1abc8cafa993f72452eb217d6a1fe68733e9f6af56bfabd` | 동결 모델의 sha256 기록 |
| `../research22/core22.py` | `954d6a22cc69df72f30ba109380e3434efbed4dabb6ea4a9f37ac361a3cac9b0` | 연구 22의 실행 묶음 (태그 code-freeze-22와 같음) |
| `../research22/loader22.py` | `370a2e138a1715b8f868f2e715179b80f6f14eb7f729c106768cb9f8c97117f7` | 막대 읽기와 잔량 보정 (태그 code-freeze-22와 같음) |
| `../research22/signal22.py` | `22f17a2c5deca70214050a597b03a6c8c79421910993ced1965c622923cb7845` | 기준 가격 신호 (태그 code-freeze-22와 같음) |
| `../research18/engine18.py` | `21d9459edbc7cfa9c5c8077225b6db9ab0c2bc5a3535ab81eaaa57292a707065` | 연구 18의 엔진 (태그 test-run-18과 같음) |
| `../research18/model18.json` | `8dff572ef84ce73d9e2d7fbc6cb0fa7eb5bd2baf856ba90039da56bbbd46e439` | 연구 18의 동결 모델 (태그 test-run-18과 같음) |
| `../research20/signal20.py` | `e5d1127ccad05c9980cfc6c79dbffbfc2fb19c5fef40d60674af05cf33bad224` | 연구 20의 신호 (태그 code-freeze-20과 같음) |
| `../research15/loader15.py` | `d5d2beb11cea016265ab2af7828e0e8db8a4c2d9faa19e71cb589dff99b07fd6` | 개발 막대 로더 (태그 test-run-18과 같음) |
| `../research17/engine17.py` | `2a9ddcb7db7b96201cbd63b62e44c7c00140447a947e883aeeebf6a542d8d42a` | 연구 17의 엔진 (합성 테스트가 대조에 씀) |
| `calendar23.csv` | `039717891922028624b9c2bfdb937b5bbbb55f063f4c04d95756363bff0d58c2` | 봉인 구간의 거래일과 마감 시각 |
| `EXCLUDE23.tsv` | `7fdcac7a71dec065f3606141cae73219e8066d90fbcd2f496dab14213063c4ec` | 빼는 종목·일 (FWRD 2025의 138일, 답변 A2) |
| `SEALED_MANIFEST23.tsv` | `34b1167335c11b8818576dbb3e0ebc5a8a19f0714f2344f108e6df0741ceb6c9` | 봉인 막대 28,072개의 sha256 (4절, 2026-11-13) |
| `universe23_2021.tsv` | `92747cc4fdb29d8ee11ad332e312a8a90aa41948c760f9cad0ec52c33fe185f5` | 유니버스 D 2021 |
| `universe23_2022.tsv` | `5cd898c9fe30225407891af5baec95cf788953d90c25420b30d2299e2bf3b413` | 유니버스 D 2022 |
| `universe23_2023.tsv` | `23ccfa4ba37ffe17a4626bf9147f618e3e296a890d4af1fdc67ea7765350f4f6` | 유니버스 D 2023 |
| `universe23_2024.tsv` | `f100ad8428eb1b71457acc68294a9027508c64527945dfb52a9c1e5217ef2e0e` | 유니버스 D 2024 |
| `universe23_2025.tsv` | `e53ae6568e6347895a77cd888b6f59a076fa07093a2ff37d843bd0ad8b95a71f` | 유니버스 D 2025 |
| `universe23_2026.tsv` | `447f7f9a1592bf0857f6c36c9171448613313fa37f4237eab7d273688b0b0e5e` | 유니버스 D 2026 |
| `../research21/SEALED_MANIFEST21.tsv` | `3fc12267a567e08f58ff75aaa00020a2d5f6438c2c7c9f23abf4cc5627d8c93f` | 봉인 구간 SPY 막대의 sha256 (연구 21) |
| `../research21/DATA_MANIFEST21.tsv` | `1eb4086a45eaf7cd88082173c1e40f39230fcd2191f27ba0e5134dd9cb9451d6` | 개발 구간 SPY 막대의 sha256 (리허설용) |
| `results/dev23_summary.json` | `52dede47640f979b3301975f4ce5a0d9d58f541e7f35b126a2ea40873dfb6b90` | 검증 결과 (A*와 H2 유무를 여기서 읽는다) |
| `results/fit23_summary.json` | `80fc715fe635eb8d47d4bbacf24741b6ec94de3c2432527d60b2e5a53d9e2384` | 학습 요약 (입력을 하나씩 뺀 참고 모델 포함) |
| `results/engine23_tests.json` | `25bc8ef624eb888d8877cce9a5f2320f39ec1f8c5eb1ec31e4414209c6b87cf3` | 합성 테스트 71/71 통과 (연구 18의 78/78, 연구 20의 117/117, 연구 22의 49/49 포함, 2026-11-13 14:38에 다시 돌린 것) |
| `results/regress23.json` | `7b89e8bc29ec12cdbb9ff29de9379541511d0451b390a0240dafc69e9cb6b673` | 회귀 테스트 통과 |
| `results/rehearsal23_summary.json` | `67ee134a0bc463e559777ef0c8ad6431743b17caf06cf188c19303a71a738f6e` | 리허설 결과 (검증 숫자와 일치, 2026-11-13 14:37에 다시 돌린 것) |

- **기록만 하는 파일 (guard가 확인하지 않는다. 봉인 실행이 읽지 않는 출력이다):**
  - results/dev23_rows.csv: sha256 9955e2da4ce17b22c5705272a955c60fdaa97cbf7829e1c27e15abbbf2fecbfe
  - results/dev23_report.md: sha256 6dea2bfc2812f70e36108589552923146784eb6e66d4c1cff3eb2837f8f959a0
  - results/dev23_daily.csv: sha256 4fc285be325c933166bedab2af654fed953e515aee8ee25dd47c38bdd0914aa1
  - results/dev23_daily_pred.csv: sha256 0923849592e9faf10c3a03d7a9c734ce4a68958b7eb90f848ad20b03903b01f3
  - results/rehearsal23_report.md: sha256 e8c97daf9d31010640f9d910c1fddc807630381c7dc2ed901829fe55ed16879b
  - results/rehearsal23_daily.csv: sha256 a1fd380bd0b82b43237fa021d082eb4f0d468c5fa1070220647c1ac53af5f7d0
  - results/rehearsal23_daily_pred.csv: sha256 0923849592e9faf10c3a03d7a9c734ce4a68958b7eb90f848ad20b03903b01f3
  - 주문별 검증 출력은 저장소 밖 `$DATA_DIR/research23_data/dev_results/dev23_valid_parents.npz`에 있다.
- **바깥 의존성:** numpy, pandas, numba, scipy, scikit-learn 1.7.2(a Python virtual environment). scikit-learn은 학습에만 쓰고 봉인 실행은 쓰지 않는다. scipy는 검정력 계산에만 쓴다.
- **리허설:** run_test23.py --rehearsal을 검증 데이터에 돌렸다(2026-11-13 14:08 ~ 14:09). 검증 실행의 숫자와 최대 차이는 0이고(숫자 329개: 표 224개, 예측 지표 105개), 봉인 경로는 읽지 않았다(results/rehearsal23_report.md). 동결 커밋 전에 한 번 더 돌렸다(14:36 ~ 14:37). 결과는 같고, 요약과 보고서는 실행 시간과 기록 시각만 다르며 일 단위 파일 두 개는 바이트까지 같다.
- **4절에서 넣은 줄:** `SEALED_MANIFEST23.tsv`(봉인 막대의 sha256). check23.py가 2026-11-13 14:35에 만들었다. 봉인 실행은 막대를 읽을 때 이 목록의 sha256과 대조한다. EXCLUDE23.tsv의 138줄을 뺀 27,934 종목·일을 쓴다.
- **동결 커밋에 넣는 것 (Kyle의 목록, 2026-11-13):** 위 표의 research23 파일 전부(model23.json 포함), 이 파일과 PREDICTIONS23.md, DEVIATIONS23.md, sha256 파일들, 테스트·회귀·학습·검증·리허설의 결과와 보고서(results/superseded/ 포함), 수집 코드(bars1s/download.py, bars.py)와 SEALED_MANIFEST23.tsv, EXCLUDE23.tsv, 품질 확인 스크립트(check23.py, check23_sizes.py, check23_trades.py)와 그 결과.
- **넣지 않는 것:** 데이터($DATA_DIR)와 UNLOCK23.txt. 목록에 없는 기록(bars1s/logs/, logs/)도 이 커밋에는 넣지 않았다.

## 실행 명령 (한 번)
Kyle이 `research23/UNLOCK23.txt`에 `연구23 테스트를 한 번 연다`를 직접 적은 뒤:

```
cd research23 && ../research15/py15.sh run_test23.py --go
```

- 결과는 results/test23_summary.json, test23_report.md, test23_daily.csv, test23_daily_pred.csv에 한 번만 쓴다. 결과 파일이 있으면 다시 돌지 않는다.
- 봉인 구간은 2021-01-04 ~ 2026-11-10이다. 기준 가격은 연구 21의 봉인 구간 SPY 막대이고, 이 실행에서만 읽는다.
- UNLOCK23.txt는 Kyle만 만든다.
