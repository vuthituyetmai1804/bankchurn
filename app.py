ash
cat > /mnt/user-data/outputs/app.py << 'PYEOF'
import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load("bidv_churn_modeltuning.pkl")

# =========================================================
# CUSTOM CSS  — dark forest theme
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Be Vietnam Pro', sans-serif; box-sizing: border-box; }

/* ── GLOBAL BG: deep forest green ── */
.stApp, .main {
    background-color: #0b1f18 !important;
}
.block-container {
    padding: 2.5rem 2rem 4rem 2rem !important;
    max-width: 1160px !important;
}

/* ── HIDE streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── PAGE HEADER ── */
.page-header {
    margin-bottom: 8px;
}
.page-header h1 {
    font-size: 36px;
    font-weight: 800;
    color: #e8f5e9;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.page-header p {
    font-size: 14px;
    color: #6aad7a;
    margin: 0;
}
.divider {
    border: none;
    border-top: 1px solid #1e3a2e;
    margin: 18px 0 28px 0;
}

/* ── FOREST BACKGROUND SVG STRIPE ── */
.forest-stripe {
    position: relative;
    overflow: hidden;
    border-radius: 20px;
    margin-bottom: 24px;
}

/* ── INPUT CARD ── */
.input-card {
    background: linear-gradient(145deg, #122b20, #0f2318);
    border: 1px solid #1e4030;
    border-radius: 20px;
    padding: 28px 28px 8px 28px;
    margin-bottom: 0;
}
.card-title-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 22px;
}
.card-icon {
    width: 38px;
    height: 38px;
    background: #1a5c38;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.card-title {
    font-size: 15px;
    font-weight: 700;
    color: #c8e6c9;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 0;
}

/* ── RESULT PANEL ── */
.result-panel {
    background: linear-gradient(160deg, #0f2318 0%, #0b1f18 100%);
    border: 1px solid #1e4030;
    border-radius: 20px;
    padding: 32px 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.result-panel-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4caf50;
    margin: 0 0 28px 0;
}
.score-wrap {
    display: flex;
    justify-content: center;
    margin-bottom: 16px;
}
.risk-verdict {
    font-size: 20px;
    font-weight: 800;
    margin: 12px 0 6px 0;
    letter-spacing: 0.02em;
}
.risk-sub {
    font-size: 13px;
    color: #6aad7a;
    margin: 0 0 24px 0;
}
.risk-bar-wrap {
    background: #1a3326;
    border-radius: 999px;
    height: 10px;
    width: 100%;
    margin: 0 0 8px 0;
    overflow: hidden;
}
.risk-bar-fill {
    height: 10px;
    border-radius: 999px;
    transition: width 0.6s ease;
}
.bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #4a7a5a;
    margin-bottom: 24px;
}
.rec-box {
    background: #0d2419;
    border: 1px solid #1e4030;
    border-radius: 14px;
    padding: 16px 18px;
    text-align: left;
    margin-top: 8px;
}
.rec-box .rec-icon { font-size: 16px; margin-right: 6px; }
.rec-box .rec-head {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 0 0 6px 0;
}
.rec-box .rec-text {
    font-size: 13px;
    color: #a5c9af;
    line-height: 1.6;
    margin: 0;
}

/* ── STREAMLIT OVERRIDES ── */
label, .stSlider label, .stNumberInput label,
.stRadio label, .stSelectbox label {
    color: #8dbf9a !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #4caf50 !important;
}
.stSlider [data-testid="stSliderTrack"] > div:first-child {
    background: #1e4030 !important;
}
.stSlider [data-testid="stSliderTrack"] > div:nth-child(2) {
    background: #4caf50 !important;
}
div[data-baseweb="input"] {
    background: #0b1f18 !important;
    border-color: #1e4030 !important;
}
div[data-baseweb="input"] input {
    background: #0b1f18 !important;
    color: #e8f5e9 !important;
}
div[data-baseweb="radio"] label {
    color: #8dbf9a !important;
}
[data-testid="stNumberInputContainer"] {
    background: #0d2419 !important;
    border: 1px solid #1e4030 !important;
    border-radius: 10px !important;
}
[data-testid="stNumberInputContainer"] input {
    color: #e8f5e9 !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    width: 100% !important;
    height: 56px !important;
    background: linear-gradient(90deg, #2e7d32, #43a047) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    border: none !important;
    letter-spacing: 0.05em !important;
    margin-top: 8px !important;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #388e3c, #4caf50) !important;
}

/* ── FOREST DECO SVG (bg) ── */
.forest-bg-svg {
    position: absolute;
    bottom: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    opacity: 0.12;
    z-index: 0;
}
.result-panel > *:not(.forest-bg-svg) {
    position: relative;
    z-index: 1;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="page-header">
  <h1>🏦 DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ</h1>
  <p>Nhập thông tin khách hàng để dự đoán khả năng rời bỏ dịch vụ · BIDV 2025</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# =========================================================
# LAYOUT: left (inputs) | right (result placeholder)
# =========================================================
col_input, col_result = st.columns([1.15, 1], gap="large")

# ── LEFT: INPUT CARD ──
with col_input:
    st.markdown("""
    <div class="input-card">
      <div class="card-title-row">
        <div class="card-icon">👤</div>
        <p class="card-title">Thông tin khách hàng</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        age = st.slider("🎂 Tuổi", 20, 80, 35)
        balance = st.number_input(
            "💰 Số dư TK (VND)",
            min_value=0, value=50_000_000, step=1_000_000, format="%d"
        )
        credit_sco = st.slider("💳 Điểm tín dụng", 495, 800, 650)
        nums_service = st.slider("🏦 Số dịch vụ", 1, 8, 3)

    with c2:
        monthly_ir = st.number_input(
            "💵 Thu nhập/tháng (VND)",
            min_value=0, value=15_000_000, step=1_000_000, format="%d"
        )
        engagement_score = st.slider("📱 Điểm tương tác app", 0, 100, 50)
        active_text = st.radio(
            "⚡ Hoạt động gần đây",
            ["Có", "Không"],
            horizontal=True
        )

    predict_btn = st.button("🔍  Phân tích & Dự đoán")

active_member = 1 if active_text == "Có" else 0

# ── RIGHT: RESULT PANEL ──
with col_result:
    if not predict_btn:
        # Idle state
        st.markdown("""
        <div class="result-panel">
          <svg class="forest-bg-svg" viewBox="0 0 400 260" xmlns="http://www.w3.org/2000/svg">
            <polygon points="60,260 100,160 140,260" fill="#4caf50"/>
            <polygon points="40,260 90,130 140,260" fill="#388e3c"/>
            <polygon points="130,260 175,150 220,260" fill="#4caf50"/>
            <polygon points="110,260 165,110 220,260" fill="#2e7d32"/>
            <polygon points="200,260 245,170 290,260" fill="#4caf50"/>
            <polygon points="185,260 240,125 295,260" fill="#388e3c"/>
            <polygon points="270,260 315,155 360,260" fill="#4caf50"/>
            <polygon points="255,260 310,105 365,260" fill="#2e7d32"/>
            <rect x="88" y="230" width="8" height="30" fill="#1b5e20"/>
            <rect x="163" y="235" width="8" height="25" fill="#1b5e20"/>
            <rect x="238" y="232" width="8" height="28" fill="#1b5e20"/>
            <rect x="308" y="234" width="8" height="26" fill="#1b5e20"/>
          </svg>
          <p class="result-panel-title">KẾT QUẢ DỰ ĐOÁN</p>
          <div style="margin: 40px 0 16px 0;">
            <svg width="90" height="90" viewBox="0 0 90 90">
              <circle cx="45" cy="45" r="40" fill="none" stroke="#1e4030" stroke-width="6"/>
              <circle cx="45" cy="45" r="40" fill="none" stroke="#2e7d32" stroke-width="6"
                stroke-dasharray="50 201" transform="rotate(-90 45 45)" opacity="0.4"/>
              <text x="45" y="50" text-anchor="middle"
                font-family="Be Vietnam Pro,sans-serif" font-size="13"
                fill="#4a7a5a">Chờ nhập</text>
            </svg>
          </div>
          <p style="color:#2e5c3a; font-size:13px; margin-top:8px;">
            Điền thông tin và nhấn<br><strong style="color:#4caf50;">Phân tích & Dự đoán</strong>
          </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── PREDICT ──
        features_order = [
            'monthly_ir','credit_sco','nums_service',
            'engagement_score','balance','age','active_member'
        ]
        input_df = pd.DataFrame([{
            'monthly_ir': monthly_ir, 'credit_sco': credit_sco,
            'nums_service': nums_service, 'engagement_score': engagement_score,
            'balance': balance, 'age': age, 'active_member': active_member
        }])
        final_input = input_df[features_order]
        risk_score   = model.predict_proba(final_input)[0][1]
        risk_percent = round(risk_score * 100, 1)

        # ── Risk config ──
        if risk_percent < 30:
            ring_color  = "#4caf50"
            bar_color   = "#4caf50"
            verdict     = "KHẢ NĂNG RỜI BỎ THẤP"
            verdict_col = "#4caf50"
            rec_head    = "✅ Duy trì & Chăm sóc"
            rec_col     = "#4caf50"
            rec_text    = "Khách hàng trung thành. Tiếp tục chăm sóc định kỳ và gợi ý sản phẩm phù hợp."
        elif risk_percent <= 70:
            ring_color  = "#ffc107"
            bar_color   = "#ffc107"
            verdict     = "KHẢ NĂNG RỜI BỎ TRUNG BÌNH"
            verdict_col = "#ffc107"
            rec_head    = "📞 Chủ động tiếp cận"
            rec_col     = "#ffc107"
            rec_text    = "Gọi điện tư vấn trong 48h, tặng voucher ưu đãi lãi suất hoặc miễn phí dịch vụ."
        else:
            ring_color  = "#ef5350"
            bar_color   = "#ef5350"
            verdict     = "KHẢ NĂNG RỜI BỎ CAO"
            verdict_col = "#ef5350"
            rec_head    = "🚨 Hành động khẩn cấp"
            rec_col     = "#ef5350"
            rec_text    = "Liên hệ khẩn cấp trong 24h. Đề xuất gói ưu đãi đặc biệt và ưu tiên xử lý vướng mắc."

        # Ring SVG
        radius = 72
        import math
        circ = 2 * math.pi * radius
        dash = circ * risk_percent / 100
        gap  = circ - dash

        ring_svg = f"""
        <svg width="190" height="190" viewBox="0 0 190 190">
          <circle cx="95" cy="95" r="{radius}"
            fill="none" stroke="#1a3326" stroke-width="14"/>
          <circle cx="95" cy="95" r="{radius}"
            fill="none" stroke="{ring_color}" stroke-width="14"
            stroke-linecap="round"
            stroke-dasharray="{dash:.1f} {gap:.1f}"
            transform="rotate(-90 95 95)"/>
          <text x="95" y="89" text-anchor="middle"
            font-family="Be Vietnam Pro,sans-serif"
            font-size="34" font-weight="800" fill="{ring_color}">{risk_percent}%</text>
          <text x="95" y="112" text-anchor="middle"
            font-family="Be Vietnam Pro,sans-serif"
            font-size="11" fill="#4a7a5a">Xác suất rời bỏ</text>
        </svg>
        """

        st.markdown(f"""
        <div class="result-panel">
          <svg class="forest-bg-svg" viewBox="0 0 400 260" xmlns="http://www.w3.org/2000/svg">
            <polygon points="60,260 100,160 140,260" fill="#4caf50"/>
            <polygon points="40,260 90,130 140,260" fill="#388e3c"/>
            <polygon points="130,260 175,150 220,260" fill="#4caf50"/>
            <polygon points="110,260 165,110 220,260" fill="#2e7d32"/>
            <polygon points="200,260 245,170 290,260" fill="#4caf50"/>
            <polygon points="185,260 240,125 295,260" fill="#388e3c"/>
            <polygon points="270,260 315,155 360,260" fill="#4caf50"/>
            <polygon points="255,260 310,105 365,260" fill="#2e7d32"/>
            <rect x="88" y="230" width="8" height="30" fill="#1b5e20"/>
            <rect x="163" y="235" width="8" height="25" fill="#1b5e20"/>
            <rect x="238" y="232" width="8" height="28" fill="#1b5e20"/>
            <rect x="308" y="234" width="8" height="26" fill="#1b5e20"/>
          </svg>
          <p class="result-panel-title">KẾT QUẢ DỰ ĐOÁN</p>
          <div class="score-wrap">{ring_svg}</div>
          <p class="risk-verdict" style="color:{verdict_col};">{verdict}</p>
          <p class="risk-sub">Khách hàng có khả năng rời bỏ dịch vụ</p>
          <div class="risk-bar-wrap">
            <div class="risk-bar-fill"
              style="width:{risk_percent}%; background:{bar_color};"></div>
          </div>
          <div class="bar-labels"><span>0%</span><span>100%</span></div>
          <div class="rec-box">
            <p class="rec-head" style="color:{rec_col};">{rec_head}</p>
            <p class="rec-text">{rec_text}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
PYEOF
echo "done"
