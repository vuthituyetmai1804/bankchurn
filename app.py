import streamlit as st
import numpy as np
import pickle

@st.cache_resource
def load_assets():
    with open('decision_tree_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

st.title("🏦 DỰ ĐOÁN RỦI RO CHURN - BIDV")

# Input
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 45)
    credit = st.number_input("Credit Score", 300, 900, 650)
    bal = st.number_input("Balance", 0, 1000000000, 20000000)
    inc = st.number_input("Monthly Income", 0, 500000000, 15000000)
with col2:
    tenure = st.number_input("Tenure", 0, 20, 3)
    act = st.selectbox("Active Member", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    eng = st.number_input("Engagement Score", 0, 100, 75)
    loy = st.selectbox("Loyalty Level", [0, 1, 2], format_func=lambda x: ["Low", "Medium", "High"][x])

if st.button("DỰ ĐOÁN NGUY CƠ"):
    # TẠO MẢNG 27 SỐ 0 (Đúng theo yêu cầu của scaler)
    input_array = np.zeros(27)
    
    # Gán vào 8 vị trí đầu tiên (đây là giả định thứ tự features của bạn)
    input_array[0] = credit
    input_array[1] = age
    input_array[2] = bal
    input_array[3] = inc
    input_array[4] = tenure
    input_array[8] = act
    input_array[10] = eng
    input_array[11] = loy
    
    # Dự đoán
    scaled = scaler.transform(input_array.reshape(1, -1))
    prob = model.predict_proba(scaled)[0][1]
    
    st.markdown("---")
    if prob * 100 < 70: st.success(f"✅ KHÁCH HÀNG Ở LẠI (Risk: {prob*100:.1f}%)")
    else: st.error(f"🚨 KHÁCH HÀNG CÓ NGUY CƠ RỜI BỎ (Risk: {prob*100:.1f}%)")
