(Korean)

# 연구 22 · 동결 기록 (FREEZE22, 2단계 3절에서 작성)

Kyle의 지시(2026-11-13)로 커밋하고 태그 `code-freeze-22`를 단다. 봉인 테스트는 아직 돌리지 않았다. run_test22.py는 봉인 실행 전에 아래 표의 sha256을 하나씩 확인하고, 하나라도 다르면 멈춘다.

**4절 (2026-11-13 12:09 ~ 12:20, 해시는 12:53):** 봉인 구간 수집(12:09:08 종료, 파일 29,021개, 실패 0건)과 품질 확인이 끝났다. `SEALED_MANIFEST22.tsv`의 sha256을 "동결 파일" 표에 넣었다. 품질 확인의 내용은 DEVIATIONS22.md 2절 9번과 10번에 있다.
- **종일 호가와 체결이 없는 파일 2개 (Kyle의 결정, 2026-11-13):** TPH 2026-05-14와 KALV 2026-06-11은 그대로 둔다. EXCLUDE22.tsv에 넣지 않는다. 봉인 실행에서 이 두 종목·일은 주문 0건이 된다.

**동결 전 확인 (2026-11-13 12:52 ~ 12:56, Kyle의 지시):**
- **합성 테스트:** 51/51 통과(연구 18의 78/78, 연구 20의 117/117 포함).
- **리허설:** 통과. 개발 실행의 숫자와 최대 차이 0이다(숫자 450개). 봉인 경로는 읽지 않았다.
- **guard:** `run_test22.py --go`는 "refused: no unlock phrase in UNLOCK22.txt" 하나로 멈춘다.
- **sha256 기록:** 두 결과 파일을 새로 써서 표의 두 줄(results/engine22_tests.json, results/rehearsal22_summary.json)과 기록만 하는 한 줄(rehearsal22_report.md)을 고쳤다. 앞선 결과는 results/superseded/에 있다.

## A*와 선택 근거 (개발 구간: 기존 유니버스 2015–2020, 1,511일, 주문 295,960건)
- **A* = G0 (E* 그대로)다. H2는 없다.**
  - L3D1 평균 IS가 가장 낮은 후보는 spy_z1_G2(3.142¢)다. G0는 3.148¢이고 차이는 0.006¢다.
  - 0.02¢ 넘게 낮지 않으므로 SPEC22 4절과 답변 A4대로 A* = G0다. 후보끼리의 동률 규칙은 쓰이지 않았다.
  - SPEC22 6절의 문장: "기준 가격 신호는 개발 구간에서 E*를 이기지 못했다. 그래서 개선 가설은 봉인 데이터로 시험하지 않았다"
- **봉인 검정:** H1(E* − S0 < 0, L3D1) 하나다. Holm 수준은 0.05다.
- **H1의 개발 값:** E* − S0 = −1.431¢/주, 95% CI [−1.543, −1.331].
- **검정력 (유니버스 C: 연 20개, 1,471일, 단측, 80%):** H1의 MDE는 0.125¢다(α 0.05, VIF 3.71). 개발 효과 1.431¢보다 작으므로 "검정력 충분"이다.

| 후보 (L3D1) | 평균 IS (¢) | − G0 [95% CI] | 발동: 주문 |
|---|---|---|---|
| G0 (E*) | 3.148 | — | — |
| spy_z1_G1 / G2 / G3 | 3.508 / 3.142 / 3.502 | +0.361 / −0.006 [−0.013, +0.000] / +0.355 | 28.8% / 12.2% / 40.8% |
| spy_z2_G1 / G2 / G3 | 3.263 / 3.148 / 3.263 | +0.115 / +0.000 / +0.115 | 10.4% / 3.2% / 13.7% |
| own_z1_G1 / G2 / G3 | 3.544 / 3.150 / 3.546 | +0.396 / +0.002 / +0.399 | 27.4% / 12.2% / 39.5% |
| own_z2_G1 / G2 / G3 | 3.289 / 3.151 / 3.293 | +0.141 / +0.004 / +0.145 | 9.7% / 3.2% / 12.9% |

- 전체 표(L0D0, 연도별, 95번째 백분위, CI)는 results/dev22_report.md에 있다.
- **보조 (판정 아님):**
  - 신호의 예측력: SPY 기준은 조기 체결 조건(a_t ≥ 1) 뒤 60초에 +0.150bp [+0.097, +0.207], 대기 유지 조건(a_t ≤ −1) 뒤에 −0.111bp [−0.156, −0.066]다. 자기 가격 기준은 0과 구분되지 않는다(+0.042bp [−0.022, +0.101]).
  - 대조 실행(추가 문서 D1)과 지연 점검은 A* = G0라서 돌리지 않았다.
- **봉인 실행이 돌리는 것:** S0, S2i, G0와 C3(S2i와 G0에서 고름), 그리고 G0의 지연 실행이다. 후보는 돌리지 않는다.
  - C3 − E*를 보조로 낸다(추가 문서 D2).
  - 예측력 표는 두 기준 전부를 낸다(추가 문서 D3).
  - 예측표 다섯째 줄은 개발 결과 파일의 `prediction_setting`(SPY 기준, 기준값 1: 개발 평균이 가장 낮았던 후보의 설정)에서 조기 체결 조건의 60초 값으로 본다(DEVIATIONS22 1절 12번).
- **선택의 전달:** run_test22.py는 A*와 H2 유무를 상수로 적지 않고 results/dev22_summary.json에서 읽는다. 이 표가 그 파일의 sha256을 고정한다.
- **Kyle의 승인:** DEVIATIONS22 1절 1 ~ 20번(2026-11-13). 21 ~ 23번은 추가 문서를 넣으며 정한 세부다.
- **시행 수:** 개발 후보 12개로 813 → 825. 봉인 판정 1개로 826이 된다.

## 문서 sha256
| 파일 | sha256 |
|---|---|
| SPEC22.md | 0158a08643274c28f6fdf7851ed8672f1ec62cf5714ef1a0c6128d503d987807 |
| SPEC22_ANSWERS.md | 94f897b1ec9f354541a803379d6541dcb4b14f917ea054ea6dbbe7df60d38da9 |
| SPEC22_ADDENDUM.md | 5d498f5acd350f8936e329358a5ba8b277d62d6a98cf998cbce4e7d7ac1b8df0 |

- guard는 세 문서를 각각의 .sha256 파일과 대조한다.
- PREDICTIONS22.md(sha256 `3118cd38…`, 개발 실행 전에 작성)와 DEVIATIONS22.md는 Kyle이 확정 전에 고칠 수 있어서 표에 넣지 않았다. 태그 커밋에는 함께 넣는다.

## 동결 파일 (경로는 research22/ 기준)
| 파일 | sha256 | 역할 |
|---|---|---|
| `run_test22.py` | `95feef44b31cef8b553706dc0a910a1c8a86a5ed1a742399dcca7f8240236b77` | 봉인 실행과 리허설 |
| `core22.py` | `954d6a22cc69df72f30ba109380e3434efbed4dabb6ea4a9f37ac361a3cac9b0` | 실행 묶음과 통계 |
| `loader22.py` | `370a2e138a1715b8f868f2e715179b80f6f14eb7f729c106768cb9f8c97117f7` | 막대 읽기와 잔량 보정 |
| `signal22.py` | `22f17a2c5deca70214050a597b03a6c8c79421910993ced1965c622923cb7845` | 신호와 발동 표시 |
| `dev22.py` | `ce44b1cc34c51e5060bc338828bee6681d4c6b931b1b30c62e8c5f29d0dc898d` | 개발 실행 (한 번 돌렸다) |
| `test22.py` | `06a6a2e01d5492a4c7044466b7f74be9b406bd672936ab6417bc86970aa4b058` | 합성 테스트 |
| `regress22.py` | `441a510b251128b08a7bd09a8487129474868eee7489b2133e91b4ae3c2c654b` | 회귀 테스트 |
| `../research18/engine18.py` | `21d9459edbc7cfa9c5c8077225b6db9ab0c2bc5a3535ab81eaaa57292a707065` | 연구 18의 엔진 (태그 test-run-18과 같음) |
| `../research20/signal20.py` | `e5d1127ccad05c9980cfc6c79dbffbfc2fb19c5fef40d60674af05cf33bad224` | 연구 20의 신호 (태그 code-freeze-20과 같음) |
| `../research15/loader15.py` | `d5d2beb11cea016265ab2af7828e0e8db8a4c2d9faa19e71cb589dff99b07fd6` | 개발 막대 로더 (태그 test-run-18과 같음) |
| `../research17/engine17.py` | `2a9ddcb7db7b96201cbd63b62e44c7c00140447a947e883aeeebf6a542d8d42a` | 연구 17의 엔진 (합성 테스트가 대조에 씀) |
| `calendar22.csv` | `039717891922028624b9c2bfdb937b5bbbb55f063f4c04d95756363bff0d58c2` | 봉인 구간의 거래일과 마감 시각 |
| `EXCLUDE22.tsv` | `45e5097e3d38d651922ce289512b97cb72be0409776b604f07bcf35335be77d5` | 빼는 종목·일 (TSE 2021의 58일, 답변 A2) |
| `SEALED_MANIFEST22.tsv` | `9442f9aa85df4564bf33c36aa174798c5ba72b914a77e5e52608acc161674f69` | 봉인 막대 29,021개의 sha256 (4절, 2026-11-13) |
| `universe22_2021.tsv` | `a89e0c7fcefe3fc6aead09c8a389604e4709c4fb75875554321b9dfd47a715d7` | 유니버스 C 2021 |
| `universe22_2022.tsv` | `ee9903febe8bc1af936810754836d8bf973efbbe1880e728100f7ace5cd71346` | 유니버스 C 2022 |
| `universe22_2023.tsv` | `34d62959f4e331a9028484a42b05b1e3425081ff796a4412184d1e0919281e8c` | 유니버스 C 2023 |
| `universe22_2024.tsv` | `88743f2a163ba5512016ec68ec1fd6a47423927044ba15f4dbccd3992c56d9ef` | 유니버스 C 2024 |
| `universe22_2025.tsv` | `f95b9a74d89a804170a1c78e45a90e3c2eff91b717ee364cda32ba1b19798ac2` | 유니버스 C 2025 |
| `universe22_2026.tsv` | `2521e62ab113689146b3bdb69c71168881980247def17f76c34d9d835346d827` | 유니버스 C 2026 |
| `../research21/SEALED_MANIFEST21.tsv` | `3fc12267a567e08f58ff75aaa00020a2d5f6438c2c7c9f23abf4cc5627d8c93f` | 봉인 구간 SPY 막대의 sha256 (연구 21) |
| `../research21/DATA_MANIFEST21.tsv` | `1eb4086a45eaf7cd88082173c1e40f39230fcd2191f27ba0e5134dd9cb9451d6` | 개발 구간 SPY 막대의 sha256 (리허설용) |
| `results/dev22_summary.json` | `14cefdd2dc3e4de4bea27ad0c99e157fc3a250c133ad0988b795f2da0b0c8f23` | 개발 결과 (A*와 H2 유무를 여기서 읽는다) |
| `results/engine22_tests.json` | `e71d32cbd07bfef3077210510343081efc71d1a4ecf7bc2802b3cfab6a8ace50` | 합성 테스트 51/51 통과 (연구 18의 78/78, 연구 20의 117/117 포함, 2026-11-13 12:54에 다시 돌린 것) |
| `results/regress22.json` | `75e2da31e34542882c13c1cb876e848671473e1f13bf296c1976b43e61439aa2` | 회귀 테스트 통과 |
| `results/rehearsal22_summary.json` | `4df3d120e1e83dca17eed6b926e3044eb00a7dc250cd79f4dd1d3022ab44f423` | 리허설 결과 (개발 숫자와 일치, 2026-11-13 12:54에 다시 돌린 것) |

- **4절에서 넣은 줄:** `SEALED_MANIFEST22.tsv`(봉인 막대의 sha256). check22.py가 2026-11-13 12:14에 만들었다. run_test22.py가 꼭 있어야 한다고 보는 파일이다. 봉인 실행은 막대를 읽을 때 이 목록의 sha256과 대조한다. EXCLUDE22.tsv의 58줄을 뺀 28,963 종목·일을 쓴다.
- **기록만 하는 파일 (guard가 확인하지 않는다. 봉인 실행이 읽지 않는 출력이다):**
  - results/dev22_rows.csv: sha256 ac8fa7987e1faa28f5990ea83d6b7c850ed468f5bceef2e652e937b6ce7683a3
  - results/dev22_report.md: sha256 b461861d12887b5d54f9dab117d66feb109da09e352d4eb5177bb06f024c956a
  - results/dev22_daily.csv: sha256 db78354f1fde7d849b1b4f186dd02e74fd30a99830502168f4c9d6d1f07444b7 (48,352줄, 약 1.8 MB)
  - results/rehearsal22_report.md: sha256 46c26d221332183056aa6f767ce5d4cb898cdea2be022c86e3cc1ad2848f3726 (12:54의 리허설. 11:05의 보고서와 실행 시간 한 줄만 다르다)
  - results/rehearsal22_daily.csv: sha256 b615a5dff376ad34355a2236fd37cda20ed9f3d1230f8fd1f15bf71d583abfb6 (12:54의 리허설. 11:05의 파일과 바이트까지 같다)
  - 주문별 개발 출력은 저장소 밖 `$DATA_DIR/research22_data/dev_results/dev22_parents.npz`에 있다.
- **동결 커밋에 넣는 것 (Kyle의 목록, 2026-11-13):** 위 표의 research22 파일 전부, SPEC22_ADDENDUM.md와 sha256 파일들, 이 파일과 PREDICTIONS22.md, DEVIATIONS22.md, 테스트·회귀·개발·리허설의 결과와 보고서(results/superseded/ 포함), 수집 코드(bars1s/download.py, bars.py), EXCLUDE22.tsv, 품질 확인 스크립트(check22.py, check22_sizes.py)와 그 결과.
- **넣지 않는 것:** 데이터($DATA_DIR)와 UNLOCK22.txt. 목록에 없는 기록(bars1s/logs/, logs/)도 이 커밋에는 넣지 않았다.
- **바깥 의존성:** numpy, pandas, numba, scipy(a Python virtual environment). scipy는 검정력 계산에만 쓴다.
- **리허설:** run_test22.py --rehearsal을 개발 데이터에 돌렸다(2026-11-13 11:03 ~ 11:05). 개발 실행의 숫자와 최대 차이는 0이고(숫자 450개), 봉인 경로는 읽지 않았다(results/rehearsal22_report.md). 동결 커밋 전에 한 번 더 돌렸다(12:52 ~ 12:54). 결과는 같고, 요약 파일은 실행 시간과 기록 시각만 다르다.

## 실행 명령 (한 번)
Kyle이 `research22/UNLOCK22.txt`에 `연구22 테스트를 한 번 연다`를 직접 적은 뒤:

```
cd research22 && ../research15/py15.sh run_test22.py --go
```

- 결과는 results/test22_summary.json, test22_report.md, test22_daily.csv에 한 번만 쓴다. 결과 파일이 있으면 다시 돌지 않는다.
- 봉인 구간은 2021-01-04 ~ 2026-11-10이다. 기준 가격은 연구 21의 봉인 구간 SPY 막대이고, 이 실행에서만 읽는다.
- UNLOCK22.txt는 Kyle만 만든다.
