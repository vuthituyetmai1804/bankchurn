import streamlit as st
import joblib
import pandas as pd

# Load model và scaler
@st.cache_resource
def load_model():
    model = joblib.load('bidv_churn_model.pkl')
    scaler = joblib.load('scaler_bidv_model.pkl')
    return model, scaler

model, scaler = load_model()

st.title("Dự đoán Khách hàng Rời bỏ (Churn) - BIDV")

# Input từ user
col1, col2 = st.columns(2)

with col1:
    monthly_ir = st.number_input("Thu nhập hàng tháng (triệu VND)", min_value=5.0, max_value=200.0, value=25.0)
    credit_sco = st.number_input("Điểm tín dụng", min_value=300, max_value=850, value=650)
    nums_service = st.number_input("Số dịch vụ đang dùng", min_value=1, max_value=10, value=3)
    engagement_score = st.slider("Điểm gắn kết", 0.0, 100.0, 65.0)

with col2:
    balance = st.number_input("Số dư tài khoản (triệu VND)", min_value=0.0, max_value=5000.0, value=120.0)
    age = st.number_input("Tuổi", min_value=18, max_value=80, value=35)
    active_member = st.selectbox("Có phải thành viên tích cực?", [1, 0])

# Tạo DataFrame
input_data = pd.DataFrame({
    'monthly_ir': [monthly_ir],
    'credit_sco': [credit_sco],
    'nums_service': [nums_service],
    'engagement_score': [engagement_score],
    'balance': [balance],
    'age': [age],
    'active_member': [active_member]
})

# Scale và Predict
if st.button("Dự đoán"):
    cols_to_scale = ['monthly_ir', 'credit_sco', 'nums_service', 'engagement_score', 'balance', 'age']
    input_scaled = scaler.transform(input_data[cols_to_scale])
    
    # Ghép lại với cột không scale
    input_final = input_data.copy()
    input_final[cols_to_scale] = input_scaled
    
    pred = model.predict(input_final)
    prob = model.predict_proba(input_final)

    if pred[0] == 1:
        st.error(f"**KHÁCH HÀNG CÓ NGUY CƠ RỜI BỎ CAO** ({prob[0][1]:.1%})")
    else:
        st.success(f"**KHÁCH HÀNG CÓ XÁC SUẤT Ở LẠI CAO** ({prob[0][0]:.1%})")
