import streamlit as st
import pandas as pd
import joblib

# =========================================
# CONFIG
# =========================================
st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================================
# LOAD MODEL
# =========================================
model = joblib.load("bidv_churn_model.pkl")

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.big-title {
    font-size: 40px;
    font-weight: 800;
    color: white;
    text-align: center;
}

.sub-title {
    text-align: center;
    color: white;
    font-size: 18px;
}

.header-box {
    background: linear-gradient(90deg,#005bea,#00c6fb);
    padding: 35px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.block-container {
    padding-top: 2rem;
}

.stButton>button {
    width: 100%;
    height: 60px;
    font-size: 22px;
    font-weight: bold;
    border-radius: 15px;
    background-color: #005bea;
    color: white;
}

.result-box {
    background-color: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 0 15px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.recommend-box {
    padding: 20px;
    border-radius: 15px;
    font-size: 18px;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================
st.markdown("""
<div class="header-box">
    <div class="big-title">
        🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ
    </div>

    <div class="sub-title">
        AI-Powered Customer Churn Prediction System
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================
# INPUT
# =========================================
st.markdown("## 📋 Nhập thông tin khách hàng")

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "🎂 Tuổi",
        18, 80, 35
    )

    credit_sco = st.slider(
        "💳 Điểm tín dụng",
        300, 900, 650
    )

    balance = st.number_input(
        "💰 Số dư tài khoản (VND)",
        min_value=0,
        value=50000000,
        step=1000000
    )

    monthly_ir = st.number_input(
        "💵 Thu nhập hàng tháng (VND)",
        min_value=0,
        value=15000000,
        step=1000000
    )

with col2:

    active_text = st.radio(
        "📱 Hoạt động gần đây",
        ["Có", "Không"]
    )

    loyalty_level = st.selectbox(
        "👑 Hạng khách hàng",
        ["Bronze", "Silver", "Gold"]
    )

# =========================================
# ENCODE INPUT
# =========================================
active_member = 1 if active_text == "Có" else 0

loyalty_map = {
    "Bronze": 1,
    "Silver": 2,
    "Gold": 3
}

loyalty_encoded = loyalty_map[loyalty_level]

# =========================================
# PREDICT BUTTON
# =========================================
predict_btn = st.button("🔍 DỰ ĐOÁN NGAY")

# =========================================
# PREDICTION
# =========================================
if predict_btn:

    # Feature Engineering đơn giản
    nums_service = 3

    engagement_score = (
        active_member * 30
        + loyalty_encoded * 20
        + (credit_sco / 20)
    )

    input_df = pd.DataFrame([{
        'monthly_ir': monthly_ir,
        'credit_sco': credit_sco,
        'nums_service': nums_service,
        'engagement_score': engagement_score,
        'balance': balance,
        'age': age,
        'active_member': active_member
    }])

    # Predict Probability
    risk_score = model.predict_proba(input_df)[0][1]

    risk_percent = round(risk_score * 100, 2)

    # =====================================
    # RISK LEVEL
    # =====================================
    if risk_percent < 30:
        risk_level = "🟢 LOW RISK"
        recommendation = "✅ Duy trì mối quan hệ tốt và tiếp tục chăm sóc định kỳ."
        color = "green"

    elif risk_percent <= 70:
        risk_level = "🟡 MEDIUM RISK"
        recommendation = "📞 Nên chăm sóc chủ động: Gọi điện tư vấn, tặng ưu đãi lãi suất, voucher."
        color = "orange"

    else:
        risk_level = "🔴 HIGH RISK"
        recommendation = "🚨 Cần liên hệ khẩn cấp trong 24h để giữ chân khách hàng."
        color = "red"

    # =====================================
    # PREDICTION LABEL
    # =====================================
    if risk_percent >= 50:
        prediction_text = "⚠️ Khách hàng có nguy cơ rời bỏ"
    else:
        prediction_text = "✅ Khách hàng có khả năng tiếp tục sử dụng dịch vụ"

    # =====================================
    # OUTPUT UI
    # =====================================
    st.markdown("---")
    st.markdown("# 📊 KẾT QUẢ PHÂN TÍCH")

    # Metric
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

    # Progress bar
    st.progress(int(risk_percent))

    # Prediction
    st.markdown(f"""
    <div class="result-box">
        <h2 style="color:{color};">
            {prediction_text}
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation
    st.markdown(f"""
    <div class="recommend-box"
    style="background-color:#ffffff;
    border-left: 8px solid {color};">

    <h3>🎯 Recommendation</h3>

    <p>{recommendation}</p>

    </div>
    """, unsafe_allow_html=True)
