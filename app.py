import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦", layout="centered")

st.title("🏦 Dự Đoán Khách Hàng Rời Bỏ Ngân Hàng")
st.markdown("### Model Logistic Regression")

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('bank_churn_model.pkl')      # ← Đường dẫn tương đối
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Không tìm thấy file model. Vui lòng upload `bank_churn_model.pkl` và `scaler.pkl` cùng thư mục với app.py")
        st.stop()

model, scaler = load_model()

# ====================== INPUT ======================
st.sidebar.header("Nhập thông tin khách hàng")

age = st.sidebar.slider("Tuổi", 18, 90, 45)
credit_score = st.sidebar.slider("Điểm tín dụng", 495, 800, 680)
balance = st.sidebar.number_input("Số dư tài khoản (VND)", min_value=0, value=50000000, step=1000000)
tenure = st.sidebar.slider("Thời gian gắn bó (năm)", 0, 4, 2)
engagement_score = st.sidebar.slider("Engagement Score", 7, 100, 30)
risk_score = st.sidebar.slider("Risk Score", 0.01, 0.55, 0.27, step=0.01)

active_member = st.sidebar.radio("Hội viên hoạt động?", ["Có", "Không"])
loyalty_level = st.sidebar.selectbox("Mức độ trung thành", ["Bronze", "Silver", "Gold", "Platinum"])

if st.button("🔍 Dự đoán", type="primary", use_container_width=True):
    input_dict = {
        'credit_sco': credit_score,
        'age': age,
        'balance': balance,
        'tenure_ye': tenure,
        'married': 1,
        'nums_service': 3,
        'active_member': 1 if active_member == "Có" else 0,
        'engagement_score': engagement_score,
        'risk_score': risk_score,
        'monthly_ir': 25000000,
        'nums_card': 3,
        'last_transaction_month': 0,
        'cluster_group': 2,
        'loyalty_level': {"Bronze":0, "Silver":1, "Gold":2, "Platinum":3}.get(loyalty_level, 0)
    }

    df_input = pd.DataFrame([input_dict])
    df_scaled = scaler.transform(df_input)

    prob = model.predict_proba(df_scaled)[0][1]

    if prob > 0.5:
        st.error(f"**CÓ NGUY CƠ RỜI BỎ CAO** ({prob:.1%})")
    else:
        st.success(f"**XÁC SUẤT Ở LẠI CAO** ({1-prob:.1%})")

    col1, col2 = st.columns(2)
    col1.metric("Xác suất rời bỏ", f"{prob:.1%}")
    col2.metric("Xác suất ở lại", f"{(1-prob):.1%}")
