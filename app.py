import streamlit as st
import numpy as np
import pickle

# Load model
@st.cache_resource
def load_assets():
    with open('decision_tree_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

st.title("🏦 HỆ THỐNG DỰ BÁO RỦI RO CHURN - BIDV")
st.markdown("### Thông tin khách hàng")

# Giao diện Input
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 45)
    credit_sco = st.number_input("Credit Score", 300, 900, 650)
    balance = st.number_input("Balance", 0, 1000000000, 20000000)
    monthly_ir = st.number_input("Monthly Income", 0, 500000000, 15000000)
with col2:
    tenure_ye = st.number_input("Tenure (years)", 0, 20, 3)
    active_member = st.selectbox("Active Member", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    engagement_sc = st.number_input("Engagement Score", 0, 100, 75)
    loyalty_level = st.selectbox("Loyalty Level", [0, 1, 2], format_func=lambda x: ["Low", "Medium", "High"][x])

if st.button("DỰ ĐOÁN NGUY CƠ"):
    # Map vào 22 features (giữ đúng vị trí)
    input_array = np.zeros(22)
    input_array[0] = credit_sco
    input_array[1] = age
    input_array[2] = balance
    input_array[3] = monthly_ir
    input_array[4] = tenure_ye
    input_array[8] = active_member
    input_array[10] = engagement_sc
    input_array[11] = loyalty_level
    
    # Dự đoán
    prob = model.predict_proba(scaler.transform(input_array.reshape(1, -1)))[0][1]
    risk_score = prob * 100
    
    st.markdown("---")
    st.subheader("📊 KẾT QUẢ DỰ ĐOÁN")
    
    # 1. Prediction
    if risk_score < 70:
        st.success("✅ KHÁCH HÀNG Ở LẠI")
    else:
        st.error("🚨 KHÁCH HÀNG CÓ NGUY CƠ RỜI BỎ")
        
    # 2. Risk Score
    st.write(f"### Risk Score: {risk_score:.1f}%")
    
    # 3. Risk Level
    if risk_score < 30:
        st.info("Risk Level: LOW")
    elif risk_score < 70:
        st.warning("Risk Level: MEDIUM")
    else:
        st.error("Risk Level: HIGH")
