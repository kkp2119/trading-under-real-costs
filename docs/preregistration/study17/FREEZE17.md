(Korean)

# 연구 17 · 동결 기록 (FREEZE17, 2단계 작성, 커밋 전)

3단계에서 Kyle의 "커밋"으로 태그 `code-freeze-17`을 단다. run_test17.py는 봉인 실행 전에 아래 표의 sha256을 하나씩 확인하고, 하나라도 다르면 멈춘다.

## A*와 선택 근거
- **A* = C3 (스프레드 전환):** 도착 스프레드 ≤ 4¢이면 S2i, 아니면 E*.
- **근거 (SPEC17 3절):** 개발 데이터(기존 유니버스 2015–2020, 주문 295,960건, L3D1)에서 평균 IS가 가장 낮다.
  - C3 3.071¢, C2 3.105¢, C1 3.112¢, C0 3.148¢, C5 3.506¢, C4 3.532¢
  - 2위 C2와 0.034¢ 차이로 0.02¢ 안의 동률이 없어서, p95와 규칙 수는 쓰지 않았다.
  - C3 − C0 = −0.076¢, 95% CI [−0.094, −0.059] (일 블록 21, 10,000회, seed 17)
- **H2:** 새 유니버스에서 C3 − C0 < 0 (L3D1).
- **H2 검정력:** 개발 효과 0.076¢가 1단계 MDE 0.023¢(Holm 0.025)보다 크므로 "검정력 충분"이다.
- **시행 수:** 누적 728(후보 6개 추가). 봉인 실행 때 비교 2개를 더해 730이 된다.

## 문서 sha256
| 파일 | sha256 |
|---|---|
| `SPEC17.md` | `b0c05a8491c2428e41243d93c74aaf2657ae4906c34bd749268354fe58eb79c0` |
| `SPEC17_ANSWERS.md` | `c8204c19ef0783dcac3cb6a7bafc103da578a73222ffdd63a93ba0535b76ec80` |

## 동결 파일 (경로는 research17/ 기준)
| 파일 | sha256 | 역할 |
|---|---|---|
| `engine17.py` | `2a9ddcb7db7b96201cbd63b62e44c7c00140447a947e883aeeebf6a542d8d42a` | 엔진: 16-D 커널 + 손절, C0–C5 분기, vol_pre, 통계 |
| `run_test17.py` | `de9557cebe6bf28bba3e65a737e42a1000053c278148cd29acfd99ed9bd730d0` | 봉인 실행 스크립트 (ASTAR = "C3") |
| `dev17.py` | `b0bf8f45a88edafa16bc8243a2b3e7ad68e538951b3dce7152ba12786e33115b` | 개발 실행과 A* 선택 |
| `test_engine17.py` | `c42371038a911a4c15472c24482772b27b944bacccbd8bf14e75c44fd7063d9c` | 합성 테스트 |
| `../research15/loader15.py` | `d5d2beb11cea016265ab2af7828e0e8db8a4c2d9faa19e71cb589dff99b07fd6` | 개발 막대 로더 (리허설용, 원본 그대로) |
| `SEALED_MANIFEST.tsv` | `b81e43031177379ed2153736afec300d13954e38962a3eb9d1c29be1796b0698` | 봉인 막대 28,019개의 sha256 |
| `EXCLUDE17.tsv` | `501b8c74cfff9486aa1da25030aea6852fdaf8ded5d23999ed5b7240c795c9d1` | 제외 163개 (B 2025) |
| `calendar17.csv` | `48b5acdadac63ba80f51ce3ae11c27e5f24e40ac979d59ef22659cb6bba81c8d` | 거래일과 마감 시각 |
| `universe17_2021.tsv` | `12e2592b7bb1e8b398157a316f190045b855ad00532a17bbd7fe3aa11d98e800` | 새 유니버스 |
| `universe17_2022.tsv` | `8ef8831674625fc4219fcffebe92deb6d93edf736ac317c8593df51b9cb440f0` | 〃 |
| `universe17_2023.tsv` | `39c376837075b44f401b4aee770f061ac29302729d431562c2fafa70faef1bdc` | 〃 |
| `universe17_2024.tsv` | `94097b7c61e3eb6301dd322e715ecb6e6b94a7ea5907aeb14456058f3dd9e4e7` | 〃 |
| `universe17_2025.tsv` | `b342f6d8b18eca361670dfa5e2508fe42fc38d2f3a74827324fb795840adeeb9` | 〃 |
| `universe17_2026.tsv` | `f115bd10f16e5a8e9c0c25223812b199a2b550ceaf5b35093d7a5de40994b989` | 〃 |
| `results/dev17_summary.json` | `7705e3ee21bcfd12522f5f3751c4a67e41fd622761993a95c86faaa468a676d6` | 개발 결과와 A* 선택 |
| `results/engine17_tests.json` | `851b0344cd43ed93fec6d2ecb723bf1173f076c4f45c2d6321d930019da2052d` | 합성 테스트 40/40 통과 |

- 바깥 의존성: numpy, pandas, numba(~/.venvs/research15). 연구 12·15·16의 다른 파일은 run_test17이 쓰지 않는다.
- PREDICTIONS17.md와 DEVIATIONS17.md는 Kyle이 확정 전에 고칠 수 있어서 이 표에 넣지 않았다. 태그 커밋에 함께 들어간다.

## 실행 명령 (4단계, 한 번)
Kyle이 `research17/UNLOCK17.txt`에 `연구17 테스트를 한 번 연다`를 직접 적은 뒤:

```
cd research17 && ../research15/py15.sh run_test17.py --go
```
