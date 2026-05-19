import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦", layout="centered")

st.title("🏦 Dự Đoán Khách Hàng Rời Bỏ")
st.markdown("### Mô hình Logistic Regression")

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    model = joblib.load('bank_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Lấy danh sách feature đúng thứ tự mà scaler mong đợi
feature_names = scaler.feature_names_in_   # Quan trọng nhất!

st.sidebar.header("Nhập thông tin khách hàng")

# ==================== INPUT FEATURES ====================
col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.slider("Tuổi", 18, 90, 45)
    credit_score = st.slider("Điểm tín dụng", 495, 800, 680)
    balance = st.number_input("Số dư (VND)", 0, 1000000000, 50000000, step=1000000)
    tenure = st.slider("Thời gian gắn bó (năm)", 0, 4, 2)
    engagement_score = st.slider("Engagement Score", 7, 100, 30)

with col2:
    risk_score = st.slider("Risk Score", 0.01, 0.55, 0.27, step=0.001)
    active_member = st.radio("Hội viên hoạt động?", ["Có", "Không"])
    loyalty_level = st.selectbox("Mức độ trung thành", ["Bronze", "Silver", "Gold", "Platinum"])
    nums_service = st.slider("Số dịch vụ sử dụng", 1, 8, 3)
    married = st.radio("Tình trạng hôn nhân", ["Đã kết hôn", "Độc thân"])

# ==================== DỰ ĐOÁN ====================
if st.button("🔍 Dự đoán", type="primary", use_container_width=True):
    
    # Tạo dictionary với đầy đủ 21 features (theo đúng thứ tự)
    input_dict = {
        'credit_sco': credit_score,
        'age': age,
        'balance': balance,
        'monthly_ir': 25000000,           # giá trị mặc định
        'tenure_ye': tenure,
        'married': 1 if married == "Đã kết hôn" else 0,
        'nums_card': 3,
        'nums_service': nums_service,
        'active_member': 1 if active_member == "Có" else 0,
        'last_transaction_month': 0,
        'engagement_score': engagement_score,
        'loyalty_level': {"Bronze":0, "Silver":1, "Gold":2, "Platinum":3}.get(loyalty_level, 0),
        'risk_score': risk_score,
        'cluster_group': 2,
        # Các cột categorical đã được encode ở training
        'gender': 0,
        'occupation': 5,
        'origin_province': 10,
        'address': 50,
        'customer_segment': 2,
        'digital_behavior': 1,
        'risk_segment': 1
    }

    # Tạo DataFrame đúng thứ tự features
    df_input = pd.DataFrame([input_dict])[feature_names]   # ← Quan trọng!

    # Scale và dự đoán
    df_scaled = scaler.transform(df_input)
    
    prob = model.predict_proba(df_scaled)[0][1]

    # Hiển thị kết quả
    st.subheader("Kết quả dự đoán")
    
    if prob >= 0.5:
        st.error(f"🚨 KHÁCH HÀNG CÓ NGUY CƠ RỜI BỎ CAO ({prob:.1%})")
    else:
        st.success(f"✅ KHÁCH HÀNG CÓ XÁC SUẤT Ở LẠI CAO ({(1-prob):.1%})")

    col1, col2 = st.columns(2)
    col1.metric("Xác suất rời bỏ", f"{prob:.1%}")
    col2.metric("Xác suất ở lại", f"{1-prob:.1%}")

    st.progress(prob)
