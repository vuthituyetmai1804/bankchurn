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
# LOAD MODEL & SCALER
# =========================================================
model = joblib.load("bidv_churn_model.pkl")
scaler = joblib.load("scaler_bidv_model.pkl")

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
.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.05);
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

# =========================================================
# RIGHT COLUMN
# =========================================================
with col2:
    nums_card = st.slider(
        "💳 Số lượng thẻ sở hữu",
        1, 5, 1
    )

    nums_service = st.slider(
        "🏦 Số lượng dịch vụ sử dụng",
        1, 10, 3
    )

    tenure_ye = st.slider(
        "⏳ Số năm gắn bó
