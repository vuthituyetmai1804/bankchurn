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
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&display=swap');
 
* {
    font-family: 'Be Vietnam Pro', sans-serif;
}
 
.main {
    background-color: #f0f4f8;
}
 
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}
 
/* ---- HEADER ---- */
.header-wrap {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 36px;
    padding-bottom: 24px;
    border-bottom: 2px solid #e2e8f0;
}
.header-logo {
    width: 56px;
    height: 56px;
    background: #1a3c78;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    flex-shrink: 0;
}
.header-text-title {
    font-size: 26px;
    font-weight: 700;
    color: #1a3c78;
    margin: 0;
    line-height: 1.2;
}
.header-text-sub {
    font-size: 14px;
    color: #64748b;
    margin: 4px 0 0 0;
}
 
/* ---- SECTION CARD ---- */
.section-card {
    background: white;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    border: 1px solid #e2e8f0;
}
.section-title {
    font-size: 15px;
    font-weight: 600;
    color: #1a3c78;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0 0 20px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid #f1f5f9;
}
 
/* ---- PREDICT BUTTON ---- */
.stButton > button {
    width: 100%;
    height: 52px;
    background: #1a3c78;
    color: white;
    font-size: 16px;
    font-weight: 600;
    border-radius: 12px;
    border: none;
    letter-spacing: 0.04em;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #14306a;
}
 
/* ---- RESULT HEADER ---- */
.result-header {
    font-size: 17px;
    font-weight: 600;
    color: #1e293b;
    margin: 0 0 20px 0;
    padding-bottom: 14px;
    border-bottom: 1px solid #f1f5f9;
}
 
/* ---- SCORE RING WRAPPER ---- */
.score-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 16px 0 8px 0;
}
 
/* ---- BADGE ---- */
.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
    margin-top: 10px;
    letter-spacing: 0.03em;
}
 
/* ---- RECOMMENDATION ---- */
.rec-card {
    border-radius: 12px;
    padding: 20px 22px;
    border-left: 5px solid;
    margin-top: 16px;
}
.rec-card p {
    margin: 6px 0 0 0;
    font-size: 15px;
    color: #374151;
    line-height: 1.6;
}
.rec-card .rec-title {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 0;
}
 
/* ---- FACTOR ROWS ---- */
.factor-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #f8fafc;
}
.factor-row:last-child {
    border-bottom: none;
}
.factor-label {
    font-size: 14px;
    color: #64748b;
}
.factor-value {
    font-size: 14px;
    font-weight: 600;
    color: #1e293b;
}
 
/* Override Streamlit widget labels */
.stSlider label, .stNumberInput label, .stRadio label {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #374151 !important;
}
</style>
""", unsafe_allow_html=True)
 
# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="header-wrap">
    <div class="header-logo">🏦</div>
    <div>
        <p class="header-text-title">Hệ thống dự đoán khách hàng rời bỏ</p>
        <p class="header-text-sub">Mô hình Decision Tree · Quản trị rủi ro BIDV · Phiên bản 2025</p>
    </div>
</div>
""", unsafe_allow_html=True)
 
# =========================================================
# INPUT SECTION — 2 CARDS
# =========================================================
col_left, col_right = st.columns([1, 1], gap="large")
 
with col_left:
    st.markdown('<div class="section-card"><p class="section-title">Thông tin tài chính</p>', unsafe_allow_html=True)
 
    balance = st.number_input(
        "Số dư tài khoản (VND)",
        min_value=0,
        value=50_000_000,
        step=1_000_000,
        format="%d"
    )
    monthly_ir = st.number_input(
        "Thu nhập hàng tháng (VND)",
        min_value=0,
        value=15_000_000,
        step=1_000_000,
        format="%d"
    )
    credit_sco = st.slider("Điểm tín dụng", 495, 800, 650)
 
    st.markdown('</div>', unsafe_allow_html=True)
 
with col_right:
    st.markdown('<div class="section-card"><p class="section-title">Thông tin hành vi</p>', unsafe_allow_html=True)
 
    age = st.slider("Tuổi", 20, 80, 35)
    nums_service = st.slider("Số dịch vụ đang sử dụng", 1, 8, 3)
    engagement_score = st.slider("Điểm tương tác ứng dụng", 0, 100, 50)
    active_text = st.radio(
        "Hoạt động gần đây",
        ["Có", "Không"],
        horizontal=True
    )
 
    st.markdown('</div>', unsafe_allow_html=True)
 
active_member = 1 if active_text == "Có" else 0
 
st.markdown("<br>", unsafe_allow_html=True)
 
# =========================================================
# PREDICT BUTTON
# =========================================================
predict_btn = st.button("🔍  Phân tích & Dự đoán")
 
# =========================================================
# PREDICTION LOGIC
# =========================================================
if predict_btn:
 
    features_order = [
        'monthly_ir', 'credit_sco', 'nums_service',
        'engagement_score', 'balance', 'age', 'active_member'
    ]
 
    input_df = pd.DataFrame([{
        'monthly_ir': monthly_ir,
        'credit_sco': credit_sco,
        'nums_service': nums_service,
        'engagement_score': engagement_score,
        'balance': balance,
        'age': age,
        'active_member': active_member
    }])
 
    final_input = input_df[features_order]
    risk_score = model.predict_proba(final_input)[0][1]
    risk_percent = round(risk_score * 100, 1)
 
    # ---- Risk levels ----
    if risk_percent < 30:
        risk_label   = "Rủi ro thấp"
        badge_style  = "background:#dcfce7; color:#166534;"
        border_color = "#16a34a"
        bar_color    = "#22c55e"
        rec_bg       = "#f0fdf4"
        rec_label    = "✅ Duy trì & Chăm sóc"
        rec_text     = "Khách hàng có xu hướng trung thành. Tiếp tục chăm sóc định kỳ, gợi ý sản phẩm phù hợp và tặng ưu đãi khách hàng thân thiết."
        verdict      = "Khách hàng có khả năng tiếp tục sử dụng dịch vụ"
        verdict_icon = "✅"
    elif risk_percent <= 70:
        risk_label   = "Rủi ro trung bình"
        badge_style  = "background:#fef9c3; color:#854d0e;"
        border_color = "#eab308"
        bar_color    = "#facc15"
        rec_bg       = "#fefce8"
        rec_label    = "📞 Chủ động tiếp cận"
        rec_text     = "Gọi điện tư vấn trong 48h, tặng voucher ưu đãi lãi suất hoặc miễn phí dịch vụ. Tìm hiểu lý do chưa tích cực sử dụng app."
        verdict      = "Khách hàng có nguy cơ rời bỏ vừa phải"
        verdict_icon = "⚠️"
    else:
        risk_label   = "Rủi ro cao"
        badge_style  = "background:#fee2e2; color:#991b1b;"
        border_color = "#ef4444"
        bar_color    = "#ef4444"
        rec_bg       = "#fff1f2"
        rec_label    = "🚨 Hành động khẩn cấp"
        rec_text     = "Cần liên hệ khẩn cấp trong vòng 24h. Phân công chuyên viên khách hàng cá nhân, đề xuất gói ưu đãi đặc biệt và ưu tiên xử lý mọi vướng mắc."
        verdict      = "Khách hàng có nguy cơ rời bỏ cao"
        verdict_icon = "🔴"
 
    st.markdown("---")
 
    # ---- Result layout ----
    res_left, res_right = st.columns([1, 1.6], gap="large")
 
    # Left: Score ring (SVG) + badge
    with res_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="result-header">Chỉ số rủi ro</p>', unsafe_allow_html=True)
 
        # Draw SVG ring
        radius = 70
        circ = 2 * 3.14159 * radius
        dash = circ * risk_percent / 100
        gap  = circ - dash
 
        ring_svg = f"""
        <div class="score-ring-wrap">
          <svg width="180" height="180" viewBox="0 0 180 180">
            <circle cx="90" cy="90" r="{radius}"
              fill="none" stroke="#e2e8f0" stroke-width="14"/>
            <circle cx="90" cy="90" r="{radius}"
              fill="none" stroke="{bar_color}" stroke-width="14"
              stroke-linecap="round"
              stroke-dasharray="{dash:.1f} {gap:.1f}"
              transform="rotate(-90 90 90)"/>
            <text x="90" y="84" text-anchor="middle"
              font-family="Be Vietnam Pro, sans-serif"
              font-size="32" font-weight="700" fill="#1e293b">{risk_percent}%</text>
            <text x="90" y="106" text-anchor="middle"
              font-family="Be Vietnam Pro, sans-serif"
              font-size="12" fill="#94a3b8">Xác suất rời bỏ</text>
          </svg>
          <span class="risk-badge" style="{badge_style}">{risk_label}</span>
        </div>
        """
        st.markdown(ring_svg, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    # Right: Summary + Recommendation
    with res_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="result-header">Kết quả phân tích</p>', unsafe_allow_html=True)
 
        st.markdown(f"""
        <div style="font-size:17px; font-weight:600; color:#1e293b; margin-bottom:20px;">
            {verdict_icon} {verdict}
        </div>
        """, unsafe_allow_html=True)
 
        # Factor summary rows
        st.markdown(f"""
        <div class="factor-row">
            <span class="factor-label">Điểm tín dụng</span>
            <span class="factor-value">{credit_sco}</span>
        </div>
        <div class="factor-row">
            <span class="factor-label">Số dư tài khoản</span>
            <span class="factor-value">{balance:,.0f} VND</span>
        </div>
        <div class="factor-row">
            <span class="factor-label">Thu nhập hàng tháng</span>
            <span class="factor-value">{monthly_ir:,.0f} VND</span>
        </div>
        <div class="factor-row">
            <span class="factor-label">Số dịch vụ sử dụng</span>
            <span class="factor-value">{nums_service} dịch vụ</span>
        </div>
        <div class="factor-row">
            <span class="factor-label">Điểm tương tác app</span>
            <span class="factor-value">{engagement_score}/100</span>
        </div>
        <div class="factor-row">
            <span class="factor-label">Hoạt động gần đây</span>
            <span class="factor-value">{"Có ✓" if active_member else "Không ✗"}</span>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown("</div>", unsafe_allow_html=True)
 
    # Recommendation card full width
    st.markdown(f"""
    <div class="rec-card" style="background:{rec_bg}; border-left-color:{border_color};">
        <p class="rec-title" style="color:{border_color};">🎯 Khuyến nghị · {rec_label}</p>
        <p>{rec_text}</p>
    </div>
    """, unsafe_allow_html=True)
 
