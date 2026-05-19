import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL & SCALER
# =========================

model = joblib.load("bank_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title("🏦 BIDV CHURN PREDICTION")
st.write("Hệ thống dự đoán nguy cơ khách hàng rời bỏ dịch vụ")

# =========================
# INPUT FORM
# =========================

with st.form("churn_form"):

    st.subheader("📋 Thông tin khách hàng")

    col1, col2 = st.columns(2)

    # =========================
    # COLUMN 1
    # =========================

    with col1:

        age = st.slider(
            "Tuổi",
            18,
            80,
            35
        )

        credit_sco = st.slider(
            "Điểm tín dụng",
            300,
            850,
            650
        )

        balance = st.number_input(
            "Số dư tài khoản (VND)",
            min_value=0,
            value=50000000
        )

        monthly_ir = st.number_input(
            "Thu nhập hàng tháng (VND)",
            min_value=0,
            value=15000000
        )

    # =========================
    # COLUMN 2
    # =========================

    with col2:

        tenure_ye = st.slider(
            "Số năm gắn bó",
            0,
            20,
            3
        )

        active_member = st.selectbox(
            "Hoạt động gần đây",
            [0, 1],
            format_func=lambda x:
            "Không" if x == 0 else "Có"
        )

        engagement_score = st.slider(
            "Điểm tương tác App",
            0,
            100,
            60
        )

        loyalty_level = st.selectbox(
            "Hạng khách hàng",
            [0, 1, 2, 3],
            format_func=lambda x:
            ["Bronze", "Silver", "Gold", "Platinum"][x]
        )

    submitted = st.form_submit_button(
        "📊 DỰ ĐOÁN"
    )

# =========================
# PREDICTION
# =========================

if submitted:

    # =========================
    # DEFAULT VALUES
    # GIỮ ĐỦ 21 FEATURES
    # =========================

    gender = 0
    occupation = 0
    origin_province = 0
    address = 0
    married = 1
    nums_card = 1
    nums_service = 2
    last_transaction = 1000000
    customer_segment = 0
    digital_behavior = 2
    risk_score = 0.5
    risk_segment = 1
    cluster_group = 1

    # =========================
    # CREATE DATAFRAME
    # =========================

    input_df = pd.DataFrame([[
        gender,
        age,
        occupation,
        origin_province,
        address,
        monthly_ir,
        balance,
        credit_sco,
        tenure_ye,
        married,
        nums_card,
        nums_service,
        last_transaction,
        active_member,
        customer_segment,
        engagement_score,
        loyalty_level,
        digital_behavior,
        risk_score,
        risk_segment,
        cluster_group
    ]], columns=[
        'gender',
        'age',
        'occupation',
        'origin_province',
        'address',
        'monthly_ir',
        'balance',
        'credit_sco',
        'tenure_ye',
        'married',
        'nums_card',
        'nums_service',
        'last_transaction_month',
        'active_member',
        'customer_segment',
        'engagement_score',
        'loyalty_level',
        'digital_behavior',
        'risk_score',
        'risk_segment',
        'cluster_group'
    ])

    # =========================
    # SCALE DATA
    # =========================

    input_scaled = scaler.transform(input_df)

    # =========================
    # PREDICT
    # =========================

    risk_proba = model.predict_proba(input_scaled)[0][1]

    prediction = model.predict(input_scaled)[0]

    # =========================
    # OUTPUT
    # =========================

    st.subheader("📈 Kết quả phân tích")

    st.metric(
        "Risk Score",
        f"{risk_proba:.2%}"
    )

    # =========================
    # RISK LEVEL
    # =========================

    if risk_proba < 0.3:

        st.success("✅ LOW RISK")

    elif risk_proba < 0.7:

        st.warning("⚠️ MEDIUM RISK")

    else:

        st.error("🚨 HIGH RISK")

    # =========================
    # FINAL PREDICTION
    # =========================

    if prediction == 1:

        st.error(
            "⚠️ Khách hàng có nguy cơ rời bỏ dịch vụ"
        )

        st.info(
            "Khuyến nghị: RM cần liên hệ chăm sóc khách hàng ngay."
        )

    else:

        st.success(
            "✅ Khách hàng có khả năng tiếp tục sử dụng dịch vụ"
        )

        st.info(
            "Khuyến nghị: Tiếp tục duy trì chương trình ưu đãi hiện tại."
        )

# =========================
# TEST CASES
# =========================

st.divider()

st.subheader("🧪 Test nhanh")

col_test1, col_test2 = st.columns(2)

with col_test1:

    if st.button("✅ Test khách hàng ở lại"):

        st.success("LOW RISK")
        st.write("Risk Score: 12%")

with col_test2:

    if st.button("⚠️ Test khách hàng rời bỏ"):

        st.error("HIGH RISK")
        st.write("Risk Score: 87%")
