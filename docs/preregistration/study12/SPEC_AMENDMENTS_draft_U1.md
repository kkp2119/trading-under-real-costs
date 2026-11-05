(Korean)
# U1 (초안): WRDS complete_nbbo 최우선 호가 수량의 단위 — 2026-10-27

**상태: 미확정.** 문서와 열 메타데이터만 확인했고, 데이터 값은 읽지 않았다.

## 확인한 열

| 테이블 (WRDS) | 수량 열 | 형식 | WRDS 열 설명 (코멘트) |
|---|---|---|---|
| `taqm_YYYY.complete_nbbo_YYYYMMDD` (2019-01-02, 2023-05-16, 2026-10-23 확인) | `best_bidsizeshares`, `best_asksizeshares` | integer | 없음 (NULL) |
| `taqm_2023.nbbom_20230516` | `best_bidsiz`, `best_asksiz`, `bidsiz`, `asksiz` | integer | 없음 (NULL) |
| `taqm_2023.cqm_20230516` | `bidsiz`, `asksiz` | integer | 없음 (NULL) |
| `taqm_2023.mastm_20230516` | `round_lot`, `uot` (unit of trade) | integer | 확인 안 함 (열 이름만, 2026-10-27 Q4) |

- complete_nbbo에는 `best_bidsiz`·`best_asksiz` 열이 없고 `…sizeshares` 열만 있다. nbbom과 cqm에는 `…siz` 열만 있고 `…sizeshares` 열은 없다.
- 테이블 설명:
  - `taqmsec.complete_nbbo_2023`: "WRDS Complete NBBO - Supplements NYSE Official NBBOs"
  - `taqm_2023.complete_nbbo_20230516`, `taqm_2023.nbbom_20230516`: 설명 없음
- 조회 방법: `information_schema.columns`, `col_description()`, `obj_description()` (메타데이터만, 2026-10-27)

## 문서

| 문서 | 필드 | 단위 설명 |
|---|---|---|
| NYSE Daily TAQ Client Specification v3.3 (2020-05-20), 4절 Quotes File | Bid Size, Offer Size | "The maximum number of shares … **in round lots**" |
| 같은 문서, 5절 NBBO File | Best Bid Size, Best Offer Size | "The Bid Size associated with the Best Bid Price on a consolidated basis, **in units of trade**" |
| 같은 문서, 2절 Master File | Unit of Trade, Round Lot | "Minimum incremental size of a trade, in shares", "The number of shares the security quotes at" |
| 같은 문서, 부록 I (69쪽) | Complete NBBO 구성법 | Quotes 파일의 NBBO 지시자 G(CTA)·4(UTP) 호가를 NBBO 파일과 합친다고만 하고, 수량 변환은 언급하지 않는다 |
| NYSE Daily TAQ Client Specification v4.0 (2022-11-02) | 위와 같은 필드 | v3.3과 같은 문구 (round lots / units of trade) |
| WRDS 웹 문서 | `best_bidsizeshares` | 검색으로 찾지 못함 (WRDS 로그인 페이지는 확인 안 함) |

## 결론
- **NYSE 원자료**: 거래소별 호가 수량(`bidsiz`)은 라운드랏 단위, NBBO 파일의 최우선 수량(`best_bidsiz`)은 "units of trade" 단위다(문서 근거, 확신 높음).
- **WRDS complete_nbbo의 `best_bidsizeshares`·`best_asksizeshares`**: 열 이름으로는 주식 수(라운드랏 × 라운드랏당 주식 수)로 바꾼 값으로 보인다. 하지만 **이를 정한 문서를 찾지 못했다 → 미확정**(확신 낮음~중간, 근거는 이름뿐).
- "units of trade"와 라운드랏의 관계도 설명서에 명시되어 있지 않다. 대부분의 ETF는 Round Lot = 100주일 것으로 보이지만 확인하지 않았다.

## 확정하는 방법 (Kyle 허락 후에만, 데이터 값 확인)
1. **WRDS 문서**: WRDS 웹사이트(로그인)의 TAQ Millisecond "Complete NBBO" 변수 설명 페이지에서 `best_bidsizeshares` 정의를 확인한다(값을 읽지 않는 방법).
2. **같은 행 대조** (1종목·1일, 건수와 비율만 출력):
   - 한 종목·하루에서 complete_nbbo의 nbbom 출처 행(natbbo_ind U 또는 2)과 같은 시각·순번의 `nbbom.best_bidsiz`를 맞춘다.
   - `best_bidsizeshares / best_bidsiz`의 분포(1인지 100인지)를 센다.
   - 같은 날 `mastm.round_lot`도 함께 본다.
3. **나머지 분포** (보조): `best_bidsizeshares`가 100의 배수인 행의 비율, 그리고 nbbom 출처 행과 cqm 출처 행(G·4)의 비율 차이를 센다.

## 막대 데이터에 주는 영향
- 이미 받은 `bars_1s_dev`의 `bsz`·`asz`는 `best_bidsizeshares`·`best_asksizeshares`를 그대로 int32로 저장한 것이다. 단위가 확정되면 SPEC에 적기만 하면 되고, 파일을 다시 받을 필요는 없다.
- 체결 수량(ctm `size`)은 주식 수다(Trades File "Trade Volume: The number of shares traded").

---

## 확인 결과 (2026-10-27, Kyle 허락으로 WRDS 접속, 서버 집계 요약만)

**방법**
- 날짜: 2018-06-13 (연구 기간 밖), 종목: TLT, DIA
- `taqm_2018.complete_nbbo_20180613`과 `taqm_2018.nbbom_20180613`을 서버에서 같은 종목·시각(time_m, time_m_nano)·최우선 가격·natbbo_ind·qu_source로 맞췄다.
- 비율 `best_bidsizeshares / best_bidsiz`, `best_asksizeshares / best_asksiz`를 집계했다.
- 가격과 시계열은 출력하거나 저장하지 않았고, 로컬 파일도 만들지 않았다.

| 항목 | TLT (UTP, qu_source N) | DIA (CTA, qu_source C) |
|---|---|---|
| complete_nbbo 행 / nbbom 행 | 395,247 / 293,221 | 146,666 / 148,040 |
| complete_nbbo 코드 구성 | natbbo_ind 2: 234,503, 4: 160,744 | U: 119,076, G: 27,590 |
| 맞춘 쌍 / 맞춘 complete_nbbo 행 | 234,502 / 234,502 | 119,660 / 119,076 |
| 매수 비율: 최소 / 중앙값 / 최대 | 100 / 100 / 100 | 5.88 / 100 / 1,600 |
| 매수 비율: 정확히 100 / 정확히 1 / 그 밖 | 100% / 0% / 0% | 99.85% / 0% / 0.15% |
| 매도 비율: 최소 / 중앙값 / 최대 | 100 / 100 / 100 | 3.33 / 100 / 3,000 |
| 매도 비율: 정확히 100 / 정확히 1 / 그 밖 | 100% / 0% / 0% | 99.83% / 0% / 0.17% |
| 분모 0 또는 NULL | 0 | 0 |
| mastm `round_lot` / `uot` (2018-06-13) | 100 / 1 | 100 / 1 |

**해석**
- 맞춘 행은 nbbom에서 온 행(TLT 코드 2, DIA 코드 U)이다. 거래소 호가 파일에서 온 행(G, 4)은 nbbom에 짝이 없어 맞춰지지 않았는데, 설명서 부록 I의 구성법과 맞는다.
- DIA의 "그 밖" 비율 0.15~0.17%는 같은 시각·같은 가격의 nbbom 행이 여럿이라 생긴 중복 짝으로 보인다(맞춘 쌍이 맞춘 행보다 584개 많음). 확인하지는 않았다.

## 결론 (U1 확정)
- **`best_bidsizeshares`·`best_asksizeshares`는 주식 수다.** 확인한 두 종목·하루에서 nbbom의 최우선 수량(라운드랏 단위) × 100과 같고, round_lot = 100이다.
- **확신: 높음** (round_lot 100인 종목). 연구 대상 10종목의 round_lot은 이 확인에서 보지 않았다.
- **확인하지 않은 범위**: round_lot이 100이 아닌 종목, 그리고 라운드랏 정의가 바뀐 이후 날짜. WRDS가 ×100으로 바꾸는지 ×round_lot으로 바꾸는지는 이 표본(둘 다 100)으로는 구별할 수 없다.
  - 2025년 이후 날짜는 mastm의 round_lot을 한 번 더 보는 편이 안전하다.
- **LOT_FACTOR**: 막대의 bsz·asz는 complete_nbbo의 `best_bidsizeshares`·`best_asksizeshares`를 그대로 저장한 것이므로 **1**이다(이미 주식 수). 100은 nbbom·cqm의 `…siz` 열(라운드랏 단위)을 쓸 때만 해당한다.
