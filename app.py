import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("bank_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("BIDV CHURN PREDICTION")

st.subheader("📋 Thông tin khách hàng")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Tuổi", 18, 80, 35)

    credit_sco = st.slider("Điểm tín dụng", 300, 850, 650)

    balance = st.number_input(
        "Số dư tài khoản",
        min_value=0,
        value=50000000
    )

    monthly_ir = st.number_input(
        "Thu nhập hàng tháng",
        min_value=0,
        value=15000000
    )

with col2:

    tenure_ye = st.slider(
        "Số năm gắn bó",
        0, 20, 3
    )

    active_member = st.selectbox(
        "Hoạt động gần đây",
        [0,1],
        format_func=lambda x:
        "Có" if x == 1 else "Không"
    )

    engagement_score = st.slider(
        "Điểm tương tác App",
        0, 100, 60
    )

    loyalty_level = st.selectbox(
        "Hạng khách hàng",
        [0,1,2,3],
        format_func=lambda x:
        ["Bronze","Silver","Gold","Platinum"][x]
    )

# Predict
if st.button("📊 DỰ ĐOÁN"):

    input_data = pd.DataFrame([[
        age,
        credit_sco,
        balance,
        monthly_ir,
        tenure_ye,
        active_member,
        engagement_score,
        loyalty_level
    ]], columns=[
        'age',
        'credit_sco',
        'balance',
        'monthly_ir',
        'tenure_ye',
        'active_member',
        'engagement_score',
        'loyalty_level'
    ])

    input_scaled = scaler.transform(input_data)

    risk_proba = model.predict_proba(input_scaled)[0][1]

    prediction = model.predict(input_scaled)[0]

    st.subheader("📈 Kết quả phân tích")

    st.metric(
        "Risk Score",
        f"{risk_proba:.2%}"
    )

    if risk_proba < 0.3:

        st.success("✅ LOW RISK")

    elif risk_proba < 0.7:

        st.warning("⚠️ MEDIUM RISK")

    else:

        st.error("🚨 HIGH RISK")

    if prediction == 1:
        st.error("Khách hàng có nguy cơ rời bỏ")
    else:
        st.success("Khách hàng có khả năng tiếp tục sử dụng dịch vụ")
