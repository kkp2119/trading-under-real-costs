(Korean)

# 연구 21 · 동결 기록 (FREEZE21, 2단계 4절에서 작성)

Kyle의 지시(2026-11-13)로 커밋하고 태그 `code-freeze-21`을 단다. 봉인 테스트는 아직 돌리지 않았다. run_test21.py는 봉인 실행 전에 아래 표의 sha256을 하나씩 확인하고, 하나라도 다르면 멈춘다.

**5절 (2026-11-13 02:07 ~ 02:12):** 봉인 구간 수집(02:06:40 종료, 파일 23,520개, 실패 0건)과 품질 확인이 끝났다. `SEALED_MANIFEST21.tsv`의 sha256을 "동결 파일" 표에 넣었다. 이제 guard가 멈추는 이유는 UNLOCK21.txt의 문구가 없다는 것 하나다. 품질 확인의 내용은 DEVIATIONS21.md 2절 11번에 있다.

**동결 전 확인 (2026-11-13 09:12 ~ 09:17, Kyle의 지시):**
- **합성 테스트:** 47/47 통과(연구 20의 117/117 포함).
- **리허설:** 통과. 개발 실행의 숫자와 최대 차이 0이다.
- **guard:** `run_test21.py --go`는 "refused: no unlock phrase in UNLOCK21.txt" 하나로 멈춘다.
- **sha256 기록:** 두 결과 파일을 새로 써서 표의 두 줄(results/engine21_tests.json, results/rehearsal21_summary.json)을 고쳤다. 앞선 결과는 results/superseded/에 있다. 자세한 내용은 DEVIATIONS21.md 2절 12번에 있다.
- **Kyle의 결정 (2026-11-13):** 2026-11-02의 라운드 로트 변화는 그대로 두고 진행한다(라운드 로트는 해마다 5월과 11월의 첫 영업일에 다시 배정되고 100배 단위는 그대로다). results/dev21_daily.csv는 동결 커밋에 넣는다. 예측표(PREDICTIONS21.md)는 고치지 않는다.

## A*와 선택 근거 (개발 구간: 쌍 2015–2020, 1,511일, 쌍·일 E 6,040개 · S 6,042개)
- **가족 E (거의 같은 ETF 쌍): A* = W10_z3_k2_M (기계적 선택, SPEC21 4절)**
  - L3D1 평균 일 손익이 가장 높은 후보다: +1.79달러 [-1.67, +8.11], 거래 328건(거래가 있는 날 171일).
  - 5% 안의 동률은 없다. 다음 후보는 W30_z3_k2_M(-2.2달러)이고, 가장 나쁜 후보는 W30_z2_k1_M(-88.6달러)이다.
  - 거래당 손익은 +8.23달러다: 중간값 기준 +73.87 − 체결 비용 57.21 − 수수료 8.43.
  - L0D0: +4.24달러 [-0.83, +13.81]
  - 지연 1초 점검(참고): L3D1 -1.8달러 [-3.3, -0.6], L0D0 +0.6달러 [-1.7, +4.5]
  - 평균이 0보다 크므로 **H2-E를 둔다.**
- **가족 S (같은 업종 주식 쌍): 개발에서 수익 없음**
  - 후보 8개 모두 L3D1 평균 일 손익이 음수다. 가장 높은 후보는 W30_z3_k2_M(-545.39달러 [-577.61, -516.56], 거래 27,557건)이고, 가장 나쁜 후보는 W10_z2_k1_M(-5,899.9달러)이다.
  - 거래당 손익은 -29.90달러다: 중간값 기준 +15.28 − 체결 비용 35.64 − 수수료 9.54.
  - 0 이하이므로 **H2-S를 없앤다.** 결론 문장은 SPEC21 6절의 "개발에서 수익 없음" 줄을 쓴다: "같은 업종 주식 쌍은 개발 구간에서 이미 비용을 넘지 못했다. 그래서 수익 가설은 봉인 데이터로 시험하지 않았다"
- **봉인 검정의 가설:** H1-E, H1-S, H2-E 세 개다. Holm 수준은 0.0167, 0.025, 0.05다(전체 α 0.05).
  - **H1-E, H1-S:** 봉인 구간에서 신호만의 시간표로 두 다리를 중간값에 거래한 손익(비용 0)의 일 합 평균 > 0. 신호 설정은 W30_z2_k0다.
  - **H2-E:** 봉인 구간에서 W10_z3_k2_M의 L3D1 일 손익 평균 > 0.
  - 판정은 일 블록 부트스트랩(21일, 10,000회, seed 21)의 단측 p에 Holm을 적용한다.
- **H1의 개발 값 (W30_z2_k0, 중간값):**

| 가족 | 거래당 [95% CI] | 일 합 평균 [95% CI] | 거래 수 |
|---|---|---|---|
| E | +13.71달러 [+11.90, +16.39] | +946.6달러 [+693.6, +1,292.1] | 104,300 |
| S | +13.83달러 [+12.57, +15.42] | +967.3달러 [+870.9, +1,082.8] | 105,694 |

- **검정력 (봉인 1,471일, 단측, 80%):**

| 가설 | 개발 효과 (달러/일) | MDE α 0.0167 | MDE α 0.05 | VIF | 판정 |
|---|---|---|---|---|---|
| H1-E | +946.6 | 463.0 | 387.6 | 13.79 | 검정력 충분 |
| H1-S | +967.3 | 164.9 | 138.1 | 5.42 | 검정력 충분 |
| H2-E | +1.8 | 8.3 | 6.9 | 1.89 | 검정력 부족 |

  - H2-E는 MDE가 개발 효과보다 크다. "확인 안 됨"이 나와도 효과가 없다는 증거로는 약하다.
- **보조 (판정 아님):**
  - 봉인 실행은 가족마다 W30_z2_k0(H1, 시장가 실행은 비교용 C2)와 그 가족 A*의 신호 설정을 돌린다. E는 W10_z3_k2, S는 W30_z3_k2다(S의 A* 값은 참고).
  - 지연 중간값 손익(1·5·30초)과 5초 문장을 가족마다 낸다. 개발 값은 5초 뒤 남는 비율이 E 28.7%, S 56.9%다.
  - 대조 쌍(SPEC21_ADDENDUM C1): 중간값 손익과 지연 표만 낸다. 개발 값은 거래당 +0.32달러 [-1.24, +2.02]다(쌍·일 3,274개, 거래 45,707건).
  - 모든 실행을 2025-10-31까지와 2025-11-03부터로 나눠 낸다. 2025-11-03부터의 호가 잔량은 100으로 나눠 읽는다(SPEC21 3절).
- **선택의 전달:** run_test21.py는 A*와 H2 유무를 상수로 적지 않고 `results/dev21_summary.json`에서 읽는다. 아래 표가 이 파일의 sha256을 고정한다(DEVIATIONS21 1절 19번).
- **Kyle의 결정:** 구현 해석 22개(DEVIATIONS21 1절)를 2026-11-11에 모두 승인했다.
- **A1의 사실 (보고에 적는다):** SPY와 QQQ는 연구 1과 LastHalfHour에서 쓴 종목이다. 연구 21의 규칙은 그 데이터를 보고 정한 것이 아니다. 가족 E의 쌍에는 SPY/IVV가 해마다 들어 있다.
- **시행 수:** 지금까지 누적 810이다(개발 후보 16개). 비교용 실행, 대조 쌍, 지연 점검, 진단은 세지 않는다. 봉인 실행에서 판정 3개를 더해 813이 된다(SPEC21 8절).

## 문서 sha256
| 파일 | sha256 |
|---|---|
| `SPEC21.md` | `284078f17c0e8c786070c638caab096a32490e1f9f069f2917b70f230438f220` |
| `SPEC21_ANSWERS.md` | `29b2ba6c6d14ab674c2c1d32c0c49119be9b532a450b98af3bd859a3b4d05c16` |
| `SPEC21_ADDENDUM.md` | `0e32355aee9e10a698538e2ac4a95cc743bbfa5b95645823f7e7142dfd1d12dd` |

- 우선순위는 SPEC21_ADDENDUM > SPEC21_ANSWERS > SPEC21이다.
- guard는 `SPEC21.sha256`, `SPEC21_ANSWERS.sha256`, `SPEC21_ADDENDUM.sha256`도 읽어 세 문서와 대조한다.

## 동결 파일 (경로는 research21/ 기준)
| 파일 | sha256 | 역할 |
|---|---|---|
| `run_test21.py` | `574f063a893d616d662d577f57cd7389f93d9cce04821062876ae967c2ab12a5` | 봉인 실행 스크립트 (A*와 H2 유무는 results/dev21_summary.json에서 읽는다) |
| `core21.py` | `73f92ca92285ed7f288a784bbc0ff0ef457517014da19291e4fe1f5541128695` | 모의 실행과 표: 개발 실행, 리허설, 봉인 실행이 함께 쓰는 경로 |
| `loader21.py` | `00729cb26b57abe6a97e028caba02cf5c894754f8be7bf789e5e833874d5a5a5` | 막대 로더 (기본은 개발 구간만, 2025-11-03부터 잔량 ÷ 100) |
| `stats21.py` | `0d156eea551a3c320fe8fda70636084d71a5fe520da00a0f8893199dec27d589` | 통계: 후보 격자, 부트스트랩(seed 21), A* 규칙, Holm, 문장 |
| `../research20/core20.py` | `35887641113e9b3ca581e986a3c0b06d67724438275e0f2dfff2de4469aa3b9c` | 연구 20의 모의 실행 (태그 code-freeze-20 그대로, 고치지 않음) |
| `../research20/pair20.py` | `27623a242fd03fbb843ea71853f1d2409793c0b78664466fc5d4ae1b20f5e5d4` | 쌍 엔진: 진입·청산 판단, 두 다리 주문, 손익 (〃) |
| `../research20/signal20.py` | `e5d1127ccad05c9980cfc6c79dbffbfc2fb19c5fef40d60674af05cf33bad224` | 신호: 5초 격자, z, 신호만의 시간표, 지연 중간값 (〃) |
| `../research20/stats20.py` | `45f16c9e7b18e05082b8d4e401fefc401ab1685601b8f59626ea9ed3d16942f0` | 통계: 일 손익, 부트스트랩, A* 규칙, MDE (〃) |
| `../research20/loader20.py` | `b6fc30b560e66897fe3a502a4e1dccc3a0328e2118a2d62ae110466d5e0cce30` | 연구 20의 로더 (〃, 회귀 테스트가 쓴다) |
| `../research17/engine17.py` | `2a9ddcb7db7b96201cbd63b62e44c7c00140447a947e883aeeebf6a542d8d42a` | 체결 커널 (연구 17 원본 그대로, 고치지 않음) |
| `dev21.py` | `b1545333962b9819f7186a6444b4ae267d6bd5d9342c4c6377e3755afde7b115` | 개발 실행: 후보 16개, A* 선택, 검정력, 지연 점검, 진단 (한 번 실행) |
| `test21.py` | `ba1969b24cabe316b3b7b4b5fdd6d4244393bd0000be782f9671022d045241bc` | 합성 테스트 |
| `regress21.py` | `f0c49d14034f3544e7a8c93b206e54070e535386222c9f9e2d519ee9e234bc05` | 회귀 테스트 (연구 20 개발 값 재현) |
| `calendar21.csv` | `8e3eb3de93f91842b06084afc095644e5f54e723a233a08ebc434ca7665c346f` | 거래일과 마감 시각 |
| `DATA_MANIFEST21.tsv` | `1eb4086a45eaf7cd88082173c1e40f39230fcd2191f27ba0e5134dd9cb9451d6` | 개발 막대 24,165개의 sha256 (리허설용) |
| `SEALED_MANIFEST21.tsv` | `3fc12267a567e08f58ff75aaa00020a2d5f6438c2c7c9f23abf4cc5627d8c93f` | 봉인 막대 23,520개의 sha256 (5절, 2026-11-13) |
| `results/universe21_pairs.tsv` | `b53b282013ac6d42cd7dc7e70582affd8dc14dad2a500c69157b7839a97982f2` | 쌍 명단 전체 (대조 쌍을 만드는 데 쓴다) |
| `universe21_2015.tsv` | `0cd507a99ec7682430cf74277cf039297586a41fce3070a8b7f375c8389820eb` | 쌍 명단 |
| `universe21_2016.tsv` | `2cfecbbf03e38dd099f0104405179cbd0f36aae09ad38288e4e90245a035b930` | 〃 |
| `universe21_2017.tsv` | `cb2b6337e858b357d10b83c9af7a1ea24a9d689abfa137b1500ee8f6df540251` | 〃 |
| `universe21_2018.tsv` | `2ceec193945c438787f5eb2fea26594ad9cd52d028c7cd30423a9c85e8fe68a0` | 〃 |
| `universe21_2019.tsv` | `ea3844e94b40c87b64d2091c6fef1fc01356565830c23900a6013a891cb84649` | 〃 |
| `universe21_2020.tsv` | `a6a521b4850f73c86e82bfc0766c0696025b7ae88ee95d434ac696b470d45df0` | 〃 |
| `universe21_2021.tsv` | `e0adc3ea7fbb008265d4a13b7ee679bab36eaa362387b664900dd3d50f5c2e18` | 〃 |
| `universe21_2022.tsv` | `f4ad9cf0aa1b60f0e8e0b5a0ce5cf8fbd68a56959145afdca0b0d63d4e89dffb` | 〃 |
| `universe21_2023.tsv` | `3c217deddc5044bdb1c7ecb8247bafdc608a5a1561051bc0695f61845f728752` | 〃 |
| `universe21_2024.tsv` | `8328dc4471fc1135ee54b6222d38aabc9ccc008363bf59b71eef2130d8f10648` | 〃 |
| `universe21_2025.tsv` | `a65d3467d589134d58c6b2a16b80993978134dde30726b4d86b1c19b02e2b5d2` | 〃 |
| `universe21_2026.tsv` | `92d478b5747bc6b3b4fe9b61f9c9b1e16c55989793c2b3f4d3975dfc49b71d0d` | 〃 |
| `results/dev21_summary.json` | `5288b7aceeb0feeb247ca6bbe0f9c20a87ff0bc1e14f7763feab14c0a944f7a2` | 개발 결과와 A* 선택 (봉인 실행이 A*와 H2 유무를 여기서 읽는다. 다시 돌리지 않음) |
| `../research20/results/test20_summary.json` | `2b80241e13dbdf01ec0e71b3a684f0878317d3af8a1972d4d87bda5a076f0701` | 연구 20의 봉인 결과 (나란히 놓은 표의 한 줄, 읽기만) |
| `../research20/results/dev20_summary.json` | `5831ff12981d7f755770ea042934cdbc9c8a27bd82c2892a29276ebe60796e04` | 연구 20의 개발 결과 (개발 실행과 리허설의 나란히 놓은 표, 읽기만) |
| `../research20/results/delay20_summary.json` | `df3835e35d7f5326f32a8bafddb479bd06f0d3e6df3ad84dd23dc8c2266d5fb5` | 연구 20의 지연 중간값 표 (〃) |
| `results/engine21_tests.json` | `9bdc46b2a570761dfdc17a07aef32ec97fde83fecd90293143022e067f21f6c6` | 합성 테스트 47/47 통과 (연구 20의 117/117 포함, 2026-11-13 09:16에 다시 돌린 것) |
| `results/regress21.json` | `2719dce0410218ec556d3a4df70f5a11d4f4e2d5e010566a70973450ca9e46b5` | 회귀 테스트 통과 (숫자 1,544개, 최대 차이 0) |
| `results/rehearsal21_summary.json` | `82ebaf8af856f05c92ba36ea9403fd6beb47cee9b55080879dd7581d7e476c3c` | 리허설 결과 (개발 숫자와 일치, 2026-11-13 09:14에 다시 돌린 것) |

- **5절에서 넣은 줄:** `SEALED_MANIFEST21.tsv`(봉인 막대의 sha256). check21.py --part sealed가 2026-11-13 02:07에 만들었다. run_test21.py가 꼭 있어야 한다고 보는 파일이다. 봉인 실행은 막대를 읽을 때 이 목록의 sha256과 대조한다.
- **기록만 하는 파일 (guard가 확인하지 않는다. 봉인 실행이 읽지 않는 출력이다):**
  - results/dev21_daily.csv: sha256 83a2e7716c70d46d9bcfea09f64a3f51d850723a58ff0be6d955b08affa3dc19 (112,012줄, 약 9.6 MB. Kyle의 결정으로 동결 커밋에 넣는다)
  - results/dev21_rows.csv: sha256 d3a29bbbc32d5c80c89947df98f49de903dbf0d2528403bd74c108af778bd617
  - results/dev21_report.md: sha256 b5ac871afc8b7b71742b1a6edd5473769526c21c34b7b140a31e8762c9bdf66d
  - results/rehearsal21_report.md: sha256 454cd9ba4c2ec006b1b57c72888b1198ab1e2e8c4de1c123a47f8cc69d4446a2 (2026-11-13 09:14의 리허설. 11-12의 보고서와 실행 시간 한 줄만 다르다)
  - results/rehearsal21_daily.csv: sha256 4fa573a6afc0fb9632185e72b65e9568720a6dc0b917aae437a94b23db6320a4 (2026-11-13 09:14의 리허설. 11-12의 파일과 바이트까지 같다)
- **바깥 의존성:** numpy, pandas, numba, scipy(a Python virtual environment). scipy는 dev21.py의 검정력 계산과 테스트에서만 쓴다.
- **동결 커밋에 넣는 것 (Kyle의 목록, 2026-11-13):** 위 표의 research21 파일 전부, SPEC21_ADDENDUM.md와 sha256 파일들, SPEC21_QUESTIONS.md, 이 파일과 PREDICTIONS21.md, DEVIATIONS21.md, 테스트·회귀·개발·리허설의 결과와 보고서, 명단과 MANIFEST, 품질 확인 스크립트(check21*.py)와 그 결과, results/superseded/.
- **넣지 않는 것:** 데이터($DATA_DIR)와 UNLOCK21.txt. 목록에 없는 수집용 파일(bars1s/, calendar21.py, plan21.py, progress21.py, logs/, results/estimate21_first_days.json, results/plan21_presence.json)도 이 커밋에는 넣지 않았다.
- **이 표에서 뺀 문서:** PREDICTIONS21.md와 DEVIATIONS21.md는 Kyle이 확정 전에 고칠 수 있다. 태그 커밋에는 함께 넣는다. Kyle은 예측표를 고치지 않기로 했다(2026-11-13).
- **리허설:** run_test21.py --rehearsal을 개발 데이터에 돌렸다(2026-11-12 01:07). 개발 실행의 숫자와 최대 차이는 0이고(표 12줄, 지연 표, 대조 쌍, 나란히 놓은 표), 봉인 경로는 읽지 않았다(results/rehearsal21_report.md). 동결 커밋 전에 한 번 더 돌렸다(2026-11-13 09:14). 결과는 같고, 요약 파일은 실행 시간과 기록 시각만 다르다.

## 실행 명령 (한 번)
Kyle이 `research21/UNLOCK21.txt`에 `연구21 테스트를 한 번 연다`를 직접 적은 뒤:

```
cd research21 && ../research15/py15.sh run_test21.py --go
```

- 결과는 results/test21_summary.json, test21_report.md, test21_daily.csv에 한 번만 쓴다. 결과 파일이 있으면 다시 돌지 않는다.
- 봉인 구간은 2021-01-04 ~ 2026-11-10이다(수집된 마지막 날로 고정).
- UNLOCK21.txt는 Kyle만 만든다.
