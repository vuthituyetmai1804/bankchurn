import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦", layout="wide")

st.title("🏦 DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ")
st.markdown("### Hệ thống dự báo churn - Ngân hàng")

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    model = joblib.load('bank_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()
feature_names = scaler.feature_names_in_

# ====================== INPUT ======================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Thông tin cá nhân")
    age = st.slider("Tuổi", 18, 90, 45)
    credit_score = st.slider("Điểm tín dụng", 495, 800, 680)
    balance = st.number_input("Số dư tài khoản (VND)", min_value=0, value=35_000_000, step=1_000_000)
    monthly_ir = st.number_input("Thu nhập hàng tháng (VND)", min_value=0, value=25_000_000, step=1_000_000)

with col2:
    st.subheader("Hành vi khách hàng")
    tenure = st.slider("Số năm gắn bó", 0, 4, 2)
    active_member = st.radio("Hoạt động gần đây", ["Có", "Không"], horizontal=True)
    engagement_score = st.slider("Điểm tương tác App", 7, 100, 35)
    loyalty_level = st.selectbox("Hạng khách hàng", ["Bronze", "Silver", "Gold", "Platinum"])

# ====================== DỰ ĐOÁN ======================
if st.button("🔍 DỰ ĐOÁN NGAY", type="primary", use_container_width=True):
    
    loyalty_map = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}

    input_dict = {
        'credit_sco': credit_score,
        'age': age,
        'balance': balance,
        'monthly_ir': monthly_ir,
        'tenure_ye': tenure,
        'married': 1,
        'nums_card': 3,
        'nums_service': 4,
        'active_member': 1 if active_member == "Có" else 0,
        'last_transaction_month': 0,
        'engagement_score': engagement_score,
        'loyalty_level': loyalty_map[loyalty_level],
