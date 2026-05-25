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
model = joblib.load("bidv_churn_model.pkl")
model = joblib.load("scaler_bidv_model.pkl")
# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #f4f6f9;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.header-box {
    background: linear-gradient(90deg, #005bea, #00c6fb);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 30px;
}

.header-title {
    color: white;
    font-size: 42px;
    font-weight: bold;
}

.header-sub {
    color: white;
    font-size: 18px;
}

.stButton > button {
    width: 100%;
    height: 65px;
    background-color: #005bea;
    color: white;
    font-size: 24px;
    font-weight: bold;
    border-radius: 15px;
    border: none;
}

.result-box {
    background-color: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.08);
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

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="header-box">
    <div class="header-title">
        🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ
    </div>

    <div class="header-sub">
        AI-Powered Bank Customer Churn Prediction
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("## 📋 Nhập thông tin khách hàng")

col1, col2 = st.columns(2)

# =========================================================
# LEFT COLUMN
# =========================================================
with col1:

    age = st.slider(
        "🎂 Tuổi",
        18,
        80,
        35
    )

    credit_sco = st.slider(
        "💳 Điểm tín dụng",
        300,
        900,
        650
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

# =========================================================
# RIGHT COLUMN
# =========================================================
with col2:

    nums_service = st.slider(
        "🏦 Số lượng dịch vụ",
        1,
        10,
        3
    )

    active_text = st.radio(
        "📱 Hoạt động gần đây",
        ["Có", "Không"]
    )

    engagement_score = st.slider(
        "🤝 Engagement Score",
        0,
        100,
        50
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
# PREDICTION
# =========================================================
if predict_btn:

    # =====================================================
    # CREATE INPUT DATAFRAME
    # =====================================================
    input_df = pd.DataFrame([{
        'monthly_ir': monthly_ir,
        'credit_sco': credit_sco,
        'nums_service': nums_service,
        'engagement_score': engagement_score,
        'balance': balance,
        'age': age,
        'active_member': active_member
    }])

    # =====================================================
    # PREDICT RISK SCORE
    # =====================================================
    risk_score = model.predict_proba(input_df)[0][1]

    risk_percent = round(risk_score * 100, 2)

    # =====================================================
    # RISK LEVEL & RECOMMENDATION
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
