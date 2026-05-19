import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Dự Đoán Khách Hàng Rời Bỏ",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Churn Prediction")
st.markdown("### Dự đoán xác suất khách hàng rời bỏ ngân hàng")

# ====================== LOAD MODEL & SCALER ======================
@st.cache_resource
def load_model():
    model = joblib.load('/content/drive/MyDrive/bank_churn_model.pkl')
    scaler = joblib.load('/content/drive/MyDrive/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ====================== DANH SÁCH FEATURE QUAN TRỌNG ======================
st.sidebar.header("Nhập thông tin khách hàng")

age = st.sidebar.slider("Tuổi", 18, 90, 45)
credit_score = st.sidebar.slider("Điểm tín dụng", 495, 800, 680)
balance = st.sidebar.number_input("Số dư tài khoản (VND)", min_value=0, value=50_000_000, step=1_000_000)
tenure = st.sidebar.slider("Thời gian gắn bó (năm)", 0, 4, 2)
engagement_score = st.sidebar.slider("Engagement Score", 7, 100, 30)
risk_score = st.sidebar.slider("Risk Score", 0.0, 0.55, 0.25, step=0.01)

active_member = st.sidebar.radio("Là hội viên hoạt động?", ["Có", "Không"])
married = st.sidebar.radio("Tình trạng hôn nhân", ["Đã kết hôn", "Độc thân", "Khác"])

nums_service = st.sidebar.slider("Số dịch vụ đang sử dụng", 1, 8, 3)
loyalty_level = st.sidebar.selectbox("Mức độ trung thành", ["Bronze", "Silver", "Gold", "Platinum"])

# ====================== XỬ LÝ DỮ LIỆU ======================
if st.button("🔍 Dự đoán", type="primary", use_container_width=True):
    # Tạo dict dữ liệu
    input_data = {
        'credit_sco': credit_score,
        'age': age,
        'balance': balance,
        'tenure_ye': tenure,
        'married': 1 if married == "Đã kết hôn" else 0,
        'nums_service': nums_service,
        'active_member': 1 if active_member == "Có" else 0,
        'engagement_score': engagement_score,
        'risk_score': risk_score,
        # Các feature khác dùng giá trị trung bình (hoặc mode)
        'monthly_ir': 25000000,
        'nums_card': 3,
        'last_transaction_month': 0,
        'cluster_group': 2,
    }

    # Xử lý categorical
    loyalty_map = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
    input_data['loyalty_level'] = loyalty_map.get(loyalty_level, 0)

    # Tạo DataFrame
    df_input = pd.DataFrame([input_data])

    # Scale
    df_scaled = scaler.transform(df_input)

    # Predict
    prob = model.predict_proba(df_scaled)[0][1]   # Xác suất rời bỏ
    prediction = model.predict(df_scaled)[0]

    # ====================== HIỂN THỊ KẾT QUẢ ======================
    st.subheader("Kết quả dự đoán")

    if prediction:
        st.error(f"**KHÁCH HÀNG CÓ NGUY CƠ RỜI BỎ CAO** ({prob:.1%})")
        st.progress(prob)
    else:
        st.success(f"**KHÁCH HÀNG CÓ XÁC SUẤT Ở LẠI CAO** ({1-prob:.1%})")
        st.progress(1 - prob)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Xác suất rời bỏ", f"{prob:.1%}")
    with col2:
        st.metric("Xác suất ở lại", f"{1-prob:.1%}")

    # Gợi ý hành động
    st.markdown("### Gợi ý hành động:")
    if prob > 0.7:
        st.warning("⚠️ Cần liên hệ khẩn cấp + ưu đãi đặc biệt")
    elif prob > 0.4:
        st.info("📞 Nên chăm sóc chủ động (gọi điện, tặng quà, ưu đãi lãi suất)")
    else:
        st.success("✅ Khách hàng ổn định, chỉ cần duy trì")

# ====================== THÔNG TIN ======================
st.caption("Ứng dụng sử dụng mô hình Logistic Regression + SMOTE")
