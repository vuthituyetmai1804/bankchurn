import streamlit as st
import pandas as pd
import joblib

model = joblib.load('bidv_model.pkl')

st.title("Dự đoán khách hàng rời bỏ")

monthly_ir = st.number_input("Thu nhập")
credit_sco = st.number_input("Điểm tín dụng")
nums_service = st.number_input("Số dịch vụ")
engagement_score = st.number_input("Engagement")
balance = st.number_input("Số dư")
age = st.number_input("Tuổi")
active_member = st.selectbox("Hoạt động", [0,1])

if st.button("Dự đoán"):

    input_df = pd.DataFrame([{
        'monthly_ir': monthly_ir,
        'credit_sco': credit_sco,
        'nums_service': nums_service,
        'engagement_score': engagement_score,
        'balance': balance,
        'age': age,
        'active_member': active_member
    }])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("Khách hàng có nguy cơ rời bỏ")
    else:
        st.success("Khách hàng có khả năng ở lại")
