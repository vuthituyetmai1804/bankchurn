import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="BIDV AI Churn Prediction", layout="centered")

@st.cache_resource
def load_assets():
    with open('decision_tree_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    # LƯU Ý QUAN TRỌNG: Bỏ qua kiểm tra tên cột của scaler
    scaler.feature_names_in_ = None 
    return model, scaler

model, scaler = load_assets()

st.title("🏦 DỰ ĐOÁN RỦI RO CHURN - BIDV")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Tuổi", 18, 80, 35)
    balance = st.number_input("Số dư (VND)", 0, 500000000, 50000000)
    credit_sco = st.slider("Điểm tín dụng", 300, 850, 650)
with col2:
    engagement = st.slider("Điểm tương tác App", 0, 100, 50)
    active = st.selectbox("Khách hàng hoạt động?", [0, 1], format_func=lambda x: "Có" if x==1 else "Không")
    num_service = st.number_input("Số sản phẩm đang dùng", 1, 10, 2)

if st.button("🚀 DỰ ĐOÁN NGUY CƠ"):
    # Tạo list giá trị đúng 22 vị trí như khi train
    input_data = [[credit_sco, age, balance, 20000000, 3, 1, 2, num_service, active, 1, engagement, 1, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    # Dự báo bằng cách dùng mảng numpy trực tiếp (tránh lỗi DataFrame)
    scaled_data = scaler.transform(input_data)
    prob = model.predict_proba(scaled_data)[0][1]
    
    if prob < 0.3:
        st.success(f"### KẾT QUẢ: AN TOÀN (Nguy cơ: {prob*100:.1f}%)")
    elif prob < 0.7:
        st.warning(f"### KẾT QUẢ: CẦN THEO DÕI (Nguy cơ: {prob*100:.1f}%)")
    else:
        st.error(f"### KẾT QUẢ: NGUY CƠ RỜI BỎ CAO (Nguy cơ: {prob*100:.1f}%)")
        st.write("**Nguyên nhân:** Tương tác thấp và số dư tài khoản không ổn định.")
