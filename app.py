import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

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
MAIN BACKGROUND
========================= */

.stApp {
    background: linear-gradient(135deg, #f7fbfb, #eef8f7);
    font-family: 'Poppins', sans-serif;
    overflow-x: hidden;
}

/* =========================
WAVE BACKGROUND
========================= */

.stApp::before {
    content: "";

    position: fixed;

    top: -20%;
    right: -10%;

    width: 1200px;
    height: 1200px;

    background:
        repeating-radial-gradient(
            circle at center,
            transparent 0px,
            transparent 90px,
            rgba(0,107,104,0.03) 92px,
            transparent 95px
        );

    animation: rotateWave 35s linear infinite;

    z-index: -3;
}

@keyframes rotateWave {
    0% {
        transform: rotate(0deg);
    }

    100% {
        transform: rotate(360deg);
    }
}

/* =========================
FLOATING PARTICLES
========================= */

.particle {
    position: fixed;

    width: 7px;
    height: 7px;

    background: rgba(0,107,104,0.18);

    border-radius: 50%;

    animation: floatParticle 18s linear infinite;

    z-index: -2;
}

@keyframes floatParticle {

    0% {
        transform: translateY(100vh);
        opacity: 0;
    }

    20% {
        opacity: 1;
    }

    100% {
        transform: translateY(-100vh);
        opacity: 0;
    }
}

/* =========================
HEADER
========================= */

.header-box {

    position: relative;

    overflow: hidden;

    background: linear-gradient(135deg, #006B68, #008B87);

    padding: 45px;

    border-radius: 28px;

    margin-bottom: 30px;

    box-shadow: 0 10px 35px rgba(0,0,0,0.08);
}

.header-box::before {

    content: "";

    position: absolute;

    width: 200%;
    height: 250px;

    background: rgba(255,255,255,0.08);

    left: -40%;
    bottom: -180px;

    border-radius: 45%;

    animation: waveMove 12s linear infinite;
}

@keyframes waveMove {

    0% {
        transform: translateX(0) rotate(0deg);
    }

    100% {
        transform: translateX(20%) rotate(360deg);
    }
}

.header-title {

    position: relative;

    color: white;

    font-size: 52px;

    font-weight: 800;

    z-index: 2;
}

.header-sub {

    position: relative;

    color: rgba(255,255,255,0.95);

    font-size: 20px;

    margin-top: 10px;

    z-index: 2;
}

/* =========================
CARD
========================= */

.custom-card {

    background: rgba(255,255,255,0.82);

    border-radius: 24px;

    padding: 28px;

    box-shadow: 0 10px 30px rgba(0,0,0,0.05);

    backdrop-filter: blur(10px);

    border: 1px solid rgba(255,255,255,0.4);

    margin-bottom: 25px;
}

/* =========================
BUTTON
========================= */

.stButton > button {

    width: 100%;

    height: 70px;

    border: none;

    border-radius: 18px;

    background: linear-gradient(135deg, #006B68, #00A19D);

    color: white;

    font-size: 24px;

    font-weight: 700;

    transition: 0.3s ease;

    box-shadow: 0 10px 25px rgba(0,107,104,0.25);
}

.stButton > button:hover {

    transform: translateY(-3px);

    box-shadow: 0 15px 35px rgba(0,107,104,0.35);
}

/* =========================
METRIC
========================= */

[data-testid="metric-container"] {

    background: white;

    border-radius: 20px;

    padding: 20px;

    box-shadow: 0 5px 20px rgba(0,0,0,0.05);
}

/* =========================
RESULT BOX
========================= */

.result-box {

    background: white;

    border-radius: 24px;

    padding: 30px;

    box-shadow: 0 10px 30px rgba(0,0,0,0.06);

    margin-top: 20px;
}

/* =========================
RECOMMENDATION
========================= */

.recommend-box {

    background: white;

    border-radius: 22px;

    padding: 25px;

    box-shadow: 0 10px 25px rgba(0,0,0,0.05);

    margin-top: 20px;
}

/* =========================
SLIDER
========================= */

.stSlider > div > div {

    color: #006B68 !important;
}

/* =========================
PROGRESS
========================= */

.stProgress > div > div > div > div {

    background: linear-gradient(90deg, #006B68, #00A19D);
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
        Ứng dụng Mô hình Cây quyết định trong Quản trị Rủi ro Ngân hàng BIDV
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

    with st.spinner("🤖 AI đang phân tích hành vi khách hàng..."):

        scan = st.progress(0)

        for percent in range(100):
            time.sleep(0.01)
            scan.progress(percent + 1)

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
left, right = st.columns([1.1, 1])

with left:

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",

        value = risk_percent,

        number = {
            'suffix': "%",
            'font': {'size': 44}
        },

        title = {
            'text': "CHURN RISK SCORE"
        },

        gauge = {

            'axis': {'range': [0,100]},

            'bar': {'color': "#006B68"},

            'steps': [

                {'range': [0,30], 'color': "#d7f5ea"},
                {'range': [30,70], 'color': "#ffe7a0"},
                {'range': [70,100], 'color': "#ffb0b0"}
            ],

            'threshold': {

                'line': {'color': "red", 'width': 5},

                'thickness': 0.8,

                'value': risk_percent
            }
        }
    ))

    fig.update_layout(

        height=380,

        margin=dict(l=20,r=20,t=60,b=20),

        paper_bgcolor="rgba(0,0,0,0)",

        font={'color': "#006B68"}
    )

    st.plotly_chart(fig, use_container_width=True)

with right:

    st.metric(
        "RISK LEVEL",
        risk_level
    )

    st.metric(
        "PREDICTION",
        "CHURN" if risk_percent >= 50 else "STAY"
    )

    st.metric(
        "SUCCESS RETENTION",
        f"{100-risk_percent}%"
    )

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
