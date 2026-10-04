"""학교 근처 전월세 적정가 예측 웹앱 (Streamlit).

    streamlit run app.py
"""
import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from src.predictor import ROOM_TYPES, RentPredictor

st.set_page_config(page_title="우리 학교 근처 전월세 적정가", page_icon="🏠", layout="centered")


@st.cache_resource(show_spinner="모델을 불러오는 중…")
def load_predictor() -> RentPredictor:
    return RentPredictor()


@st.cache_data(show_spinner=False, max_entries=200)
def run_prediction(school, rent_type, months, room, deposit, price, housing):
    cats = ("연립다세대", "오피스텔") if housing == "전체" else (housing,)
    p = load_predictor().predict(school, rent_type, months, room, deposit, price, categories=cats)
    return p.to_dict(include_samples=True)


def won(v: float) -> str:
    """만원 단위 숫자를 읽기 쉬운 금액으로 (예: 25136 → 2억 5,136만원)."""
    v = round(v)
    if v >= 10000:
        eok, man = divmod(v, 10000)
        return f"{eok}억" + (f" {man:,}만원" if man else "원")
    return f"{v:,}만원"


P = load_predictor()
schools = P.schools["school_name"].tolist()

st.title("🏠 우리 학교 근처 전월세, 얼마가 적정할까?")
st.caption(f"국토교통부 실거래가(연립다세대·오피스텔, 2021-01 ~ {P.asof.date()}) 기반 예측 · 서울 동작·노원·서대문·동대문·성북구")

# ------------------------------------------------------------------ 입력
# 계약 형태는 폼 밖에 둬야 바꾸는 즉시 보증금/가격 입력칸이 바뀐다 (폼 안 위젯은 제출 시에만 반영)
rent_type = st.radio("계약 형태", ["월세", "전세"], horizontal=True)

with st.form("query"):
    school = st.selectbox("학교", schools, index=schools.index("고려대학교 서울캠퍼스") if "고려대학교 서울캠퍼스" in schools else 0)

    c3, c4, c5 = st.columns(3)
    months = c3.select_slider("언제 계약하나요?", options=list(range(7)), value=0,
                              format_func=lambda m: "지금" if m == 0 else f"{m}개월 뒤")
    room = c4.selectbox("방 크기", list(ROOM_TYPES), index=0,
                        help="원룸: 전용 25㎡ 미만 · 투룸: 25~45㎡ · 쓰리룸+: 45㎡ 이상")
    housing = c5.selectbox("주택 유형", ["전체", "연립다세대", "오피스텔"],
                           format_func=lambda h: {"연립다세대": "빌라(연립·다세대)"}.get(h, h))

    c6, c7 = st.columns(2)
    deposit = None
    if rent_type == "월세":
        deposit = c6.number_input("보증금 (만원)", min_value=0, max_value=50000, value=1000, step=100)
    price_label = "보고 있는 매물 월세 (만원, 선택)" if rent_type == "월세" else "보고 있는 매물 전세금 (만원, 선택)"
    price = c7.number_input(price_label, min_value=0, value=0, step=1 if rent_type == "월세" else 500,
                            help="입력하면 근처 시세 대비 저렴/적정/비쌈을 알려드려요. 0이면 생략.")
    submitted = st.form_submit_button("예측하기", type="primary", width="stretch")

if not submitted:
    st.info("조건을 고르고 **예측하기**를 눌러주세요. 한 번 계산에 10초 정도 걸려요.")
    st.stop()

try:
    with st.spinner("근처 건물들의 과거 거래로 예측하는 중… (약 10초)"):
        r = run_prediction(school, rent_type, months, room, float(deposit) if deposit is not None else None,
                           float(price) if price else None, housing)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ------------------------------------------------------------------ 결과
what = "전세 보증금" if rent_type == "전세" else f"월세 (보증금 {won(r['deposit_10k'])} 기준)"
st.subheader(f"{r['school']} · {room} {what}")
st.caption(f"계약 예정일 {r['contract_date']} · 학교 반경 {r['radius_m']:,}m 안 건물 {r['n_buildings']}곳 기준")

m1, m2, m3 = st.columns(3)
m1.metric("예상 가격", won(r["expected"]))
m2.metric("적정 범위 (80%)", f"{won(r['range_low'])} ~ {won(r['range_high'])}")
m3.metric("지금 계약하면", won(r["now_expected"]), f"{r['change_vs_now_pct']:+.1f}% (예정일 대비)",
          delta_color="off")

if r["price_check"]:
    pc = r["price_check"]
    msg = (f"입력한 **{won(pc['listing_price'])}** 은(는) 근처 예상 가격 중 하위 **{pc['percentile']:.0f}%** 위치예요 "
           f"→ **{pc['verdict']}**")
    {"저렴": st.success, "적정": st.info, "다소 비쌈": st.warning, "비쌈": st.error}[pc["verdict"]](msg)

st.write(f"⏱️ {r['timing_message']}")

# 분포 차트
samples = pd.DataFrame({"price": r["samples"]})
lo_clip, hi_clip = samples["price"].quantile([0.01, 0.99])
samples = samples[samples["price"].between(lo_clip, hi_clip)]
base = alt.Chart(samples).mark_bar(opacity=0.75).encode(
    alt.X("price:Q", bin=alt.Bin(maxbins=40), title="예상 가격 (만원)"),
    alt.Y("count():Q", title="상대 빈도", axis=alt.Axis(labels=False, ticks=False)),
)
band = alt.Chart(pd.DataFrame({"lo": [r["range_low"]], "hi": [r["range_high"]]})).mark_rect(
    opacity=0.12, color="#2e7d32").encode(x="lo:Q", x2="hi:Q")
rules = [alt.Chart(pd.DataFrame({"x": [r["expected"]]})).mark_rule(strokeWidth=2).encode(x="x:Q")]
if r["price_check"]:
    rules.append(alt.Chart(pd.DataFrame({"x": [r["price_check"]["listing_price"]]}))
                 .mark_rule(color="#d32f2f", strokeDash=[6, 3], strokeWidth=2).encode(x="x:Q"))
st.altair_chart(alt.layer(band, base, *rules).properties(height=220), width="stretch")
st.caption("초록 영역 = 적정 범위(80%) · 검은 선 = 예상 가격" + (" · 빨간 점선 = 입력한 매물" if r["price_check"] else ""))

# 지도
with st.expander("🗺️ 예측에 사용한 근처 건물 지도", expanded=False):
    b = pd.DataFrame(r["buildings"])
    if not b.empty:
        b["label"] = b.apply(lambda x: f"{x['building']} ({x['dong']})\n예상 {won(x['expected'])}", axis=1)
        layers = [
            pdk.Layer("ScatterplotLayer", b, get_position="[lon, lat]", get_radius=25,
                      get_fill_color=[30, 136, 229, 160], pickable=True),
            pdk.Layer("ScatterplotLayer", pd.DataFrame([{"lat": r["school_lat"], "lon": r["school_lon"],
                                                         "label": r["school"]}]),
                      get_position="[lon, lat]", get_radius=60, get_fill_color=[211, 47, 47, 220], pickable=True),
        ]
        st.pydeck_chart(pdk.Deck(layers=layers, map_style=None, tooltip={"text": "{label}"},
                                 initial_view_state=pdk.ViewState(latitude=r["school_lat"], longitude=r["school_lon"],
                                                                  zoom=14)))
        st.caption("빨간 점 = 학교 · 파란 점 = 예측에 사용한 건물 (점에 마우스를 올리면 예상 가격)")

# 실거래 사례
if r["recent_examples"]:
    st.markdown("**근처 최근 실거래**")
    ex = pd.DataFrame(r["recent_examples"])
    ex["가격"] = ex.apply(lambda x: won(x["deposit_10k"]) if rent_type == "전세"
                        else f"{x['deposit_10k']:,.0f} / {x['monthly_rent_10k']:,.0f}", axis=1)
    ex = ex.rename(columns={"date": "계약일", "building": "건물", "dong": "동", "area_m2": "전용(㎡)",
                            "floor": "층", "distance_m": "학교까지(m)"})
    st.dataframe(ex[["계약일", "건물", "동", "전용(㎡)", "층", "학교까지(m)", "가격"]], hide_index=True,
                 width="stretch")

for n in r["notes"]:
    st.caption(f"※ {n}")

with st.expander("이 예측은 어떻게 만들어졌나요?"):
    st.markdown(f"""
- **데이터**: 국토교통부 실거래가 공개시스템의 연립·다세대, 오피스텔 전월세 신고 자료(2021-01 ~ {P.asof.date()}).
  실거래가라서 부동산 앱의 **호가와 다를 수 있어요.**
- **방법**: 학교 반경 안 건물마다 '예정일에 새로 계약한다면'의 가격을 LightGBM 분위수 모델로 예측하고,
  건물별 결과를 최근 거래량 비율로 합쳐 분포를 만들었어요. 같은 건물의 과거 거래가, 면적, 층, 연식, 역·학교 거리,
  동네의 최근 시세와 추세를 사용해요.
- **정확도** (2026년 거래로 검증): 예상 가격 오차 중앙값 전세 약 8.5%, 월세 약 11%.
  적정 범위 안에 실제 거래가 들어온 비율은 약 80%예요.
- **'지금 vs 나중'**: 6개월 이내의 시장 변화는 대부분 ±1~3% 수준이라, 언제 계약하느냐보다 **어떤 매물을 고르느냐**가
  가격에 훨씬 큰 영향을 줘요.
- 참고용 정보이며 실제 계약 전에는 등기부등본, 선순위 채권 등을 꼭 확인하세요.
""")
