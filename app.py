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
# LOAD MODEL (Đã bỏ hoàn toàn Scaler)
# =========================================================
model = joblib.load("bidv_churn_modeltuning.pkl")

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* =========================
BACKGROUND
========================= */

.stApp {
    background: linear-gradient(135deg, #e8fff3, #dffbf2, #f4fffb);
    font-family: 'Poppins', sans-serif;
}

/* =========================
MAIN CONTAINER
========================= */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1350px;
}

/* =========================
HEADER
========================= */

.header-box {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #00b978, #00d084);
    padding: 45px;
    border-radius: 30px;
    margin-bottom: 35px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

/* Wave animation */

.header-box::before {
    content: "";
    position: absolute;
    width: 200%;
    height: 200px;
    left: -50%;
    bottom: -120px;

    background: rgba(255,255,255,0.18);

    border-radius: 45%;

    animation: waveMove 8s linear infinite;
}

@keyframes waveMove {
    0% {
        transform: translateX(0) rotate(0deg);
    }
    100% {
        transform: translateX(25%) rotate(360deg);
    }
}

.header-title {
    position: relative;
    color: white;
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    z-index: 2;
}

.header-sub {
    position: relative;
    color: rgba(255,255,255,0.95);
    font-size: 20px;
    text-align: center;
    margin-top: 12px;
    z-index: 2;
}

/* =========================
CARD
========================= */

[data-testid="stVerticalBlock"] > div:has(.modern-card) {
    background: rgba(255,255,255,0.72);
    border-radius: 24px;
    padding: 25px;
    backdrop-filter: blur(10px);
}

/* =========================
INPUT LABEL
========================= */

label {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #106b4b !important;
}

/* =========================
INPUT BOX
========================= */

.stNumberInput input,
.stTextInput input {
    border-radius: 14px !important;
    border: 2px solid #d8f5e6 !important;
    height: 52px !important;
    background-color: white !important;
}

/* =========================
SLIDER
========================= */

.stSlider > div > div {
    color: #00b978 !important;
}

/* =========================
RADIO
========================= */

.stRadio label {
    font-size: 17px !important;
}

/* =========================
BUTTON
========================= */

.stButton > button {
    width: 100%;
    height: 68px;

    border: none;
    border-radius: 18px;

    background: linear-gradient(135deg, #00b978, #00d084);

    color: white;
    font-size: 24px;
    font-weight: 700;

    transition: 0.3s ease;

    box-shadow: 0 10px 25px rgba(0,185,120,0.25);
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(0,185,120,0.35);
}

/* =========================
RESULT BOX
========================= */

.result-box {
    background: rgba(255,255,255,0.9);

    padding: 35px;

    border-radius: 24px;

    box-shadow: 0 10px 30px rgba(0,0,0,0.08);

    margin-top: 20px;

    text-align: center;
}

/* =========================
RECOMMEND BOX
========================= */

.recommend-box {
    background: rgba(255,255,255,0.9);

    padding: 25px;

    border-radius: 20px;

    font-size: 18px;

    box-shadow: 0 10px 30px rgba(0,0,0,0.08);

    margin-top: 20px;
}

/* =========================
METRIC
========================= */

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.9);

    border-radius: 20px;

    padding: 20px;

    box-shadow: 0 8px 20px rgba(0,0,0,0.05);

    border: 1px solid rgba(255,255,255,0.5);
}

/* =========================
PROGRESS BAR
========================= */

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00b978, #00d084);
}

/* =========================
SECTION TITLE
========================= */

h2, h3 {
    color: #106b4b;
}

/* =========================
FLOATING BLUR CIRCLES
========================= */

.stApp::before {
    content: "";

    position: fixed;

    width: 350px;
    height: 350px;

    background: rgba(0,208,132,0.15);

    border-radius: 50%;

    top: -120px;
    right: -100px;

    filter: blur(60px);

    z-index: -1;
}

.stApp::after {
    content: "";

    position: fixed;

    width: 300px;
    height: 300px;

    background: rgba(0,185,120,0.12);

    border-radius: 50%;

    bottom: -100px;
    left: -80px;

    filter: blur(60px);

    z-index: -1;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="header-box">
    <div class="header-title">
        🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ
    </div>

    <div class="header-sub">
        AI Banking Analytics • BIDV Churn Prediction 2026
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("""
<div class="modern-card">
<h2>📋 Thông tin khách hàng</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# =========================================================
# LEFT COLUMN
# =========================================================
with col1:
    age = st.slider(
        "🎂 Tuổi",
        20, 80, 35
    )

    credit_sco = st.slider(
        "💳 Điểm tín dụng",
        495, 800, 650
    )

    balance = st.number_input(
        "💰 Số dư tài khoản (VND)",
        min_value=0,
        value=50000000,
        step=1000000
    )

# =========================================================
# RIGHT COLUMN
# =========================================================
with col2:
    monthly_ir = st.number_input(
        "💵 Thu nhập hàng tháng (VND)",
        min_value=0,
        value=15000000,
        step=1000000
    )

    nums_service = st.slider(
        "🏦 Số lượng dịch vụ sử dụng",
        1, 8, 3
    )

    engagement_score = st.slider(
        "🤝 Điểm tương tác app",
        0, 100, 50
    )

    active_text = st.radio(
        "📱 Hoạt động gần đây",
        ["Có", "Không"]
    )

# =========================================================
# ENCODE INPUT
# =========================================================
active_member = 1 if active_text == "Có" else 0

# =========================================================
# PREDICT BUTTON
# =========================================================
predict_btn = st.button("🔍 DỰ ĐOÁN NGAY")

# =========================================================
# PREDICTION LOGIC
# =========================================================
if predict_btn:

    # =====================================================
    # 1. TẠO DATAFRAME VỚI ĐÚNG 7 CỘT THEO ĐÚNG THỨ TỰ YÊU CẦU
    # =====================================================
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
    
    # Đảm bảo thứ tự cột gửi vào mô hình chuẩn xác 100%
    final_input = input_df[features_order]

    # =====================================================
    # 2. DỰ ĐOÁN TRỰC TIẾP KHÔNG QUA SCALER
    # =====================================================
    risk_score = model.predict_proba(final_input)[0][1]
    risk_percent = round(risk_score * 100, 2)

    # =====================================================
    # RISK LEVEL
    # =====================================================
    if risk_percent < 30:
        risk_level = "🟢 LOW RISK"
        prediction_text = "✅ Khách hàng có khả năng tiếp tục sử dụng dịch vụ"
        recommendation = "✅ Duy trì mối quan hệ tốt và tiếp tục chăm sóc định kỳ."
        color = "green"
    elif risk_percent <= 70:
        risk_level = "🟡 MEDIUM RISK"
        prediction_text = "⚠️ Khách hàng có nguy cơ rời bỏ"
        recommendation = "📞 Nên chăm sóc chủ động: Gọi điện tư vấn, tặng ưu đãi lãi suất, voucher."
        color = "orange"
    else:
        risk_level = "🔴 HIGH RISK"
        prediction_text = "⚠️ Khách hàng có nguy cơ rời bỏ"
        recommendation = "🚨 Cần liên hệ khẩn cấp trong 24h để giữ chân khách hàng."
        color = "red"

    # =====================================================
    # OUTPUT GRAPHICS
    # =====================================================
    st.markdown("---")
    st.markdown("# 📊 KẾT QUẢ PHÂN TÍCH")

    # =====================================================
    # METRICS
    # =====================================================
    colA, colB, colC = st.columns(3)

    with colA:
        st.metric(
            label="RISK SCORE",
            value=f"{risk_percent}%"
        )

    with colB:
        st.metric(
            label="RISK LEVEL",
            value=risk_level
        )

    with colC:
        st.metric(
            label="PREDICTION",
            value="CHURN" if risk_percent >= 50 else "STAY"
        )

    # =====================================================
    # PROGRESS BAR
    # =====================================================
    st.progress(int(risk_percent))

    # =====================================================
    # PREDICTION RESULT BOX
    # =====================================================
    st.markdown(f"""
    <div class="result-box">
        <h2 style="color:{color}; text-align: center; margin: 0;">
            {prediction_text}
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # RECOMMENDATION BOX
    # =====================================================
    st.markdown(f"""
    <div class="recommend-box"
    style="
        background-color:white;
        border-left:8px solid {color};
        margin-top:20px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.08);
    ">
    <h3 style="margin-top: 0;">🎯 Khuyến nghị hành động:</h3>
    <p style="margin-bottom: 0;">{recommendation}</p>
    </div>
    """, unsafe_allow_html=True)
