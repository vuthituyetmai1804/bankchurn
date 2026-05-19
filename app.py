import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦", layout="wide")

st.title("🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ")
st.markdown("### Ngân hàng - Churn Prediction Model")

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    model = joblib.load('bank_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()
feature_names = list(scaler.feature_names_in_)

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
    risk_score = st.slider("Risk Score (%)", 0.0, 100.0, 45.0, step=0.1)

# ====================== DỰ ĐOÁN ======================
if st.button("🔍 Dự Đoán Khách Hàng", type="primary", use_container_width=True):
    
    loyalty_map = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}

    # Tạo input với đầy đủ 21 features
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
        'risk_score': risk_score / 100,           # Chuyển % về 0-1
        'cluster_group': 2,
        'gender': 0,
        'occupation': 5,
        'origin_province': 15,
        'address': 100,
        'customer_segment': 2,
        'digital_behavior': 1,
        'risk_segment': 1
    }

    df_input = pd.DataFrame([input_dict])[feature_names]
    df_scaled = scaler.transform(df_input)

    prob = model.predict_proba(df_scaled)[0][1]   # Xác suất rời bỏ

    # ====================== HIỂN THỊ KẾT QUẢ ======================
    st.markdown("---")
    st.subheader("📊 KẾT QUẢ DỰ ĐOÁN")

    risk_percent = prob * 100

    # Risk Level
    if risk_percent < 30:
        risk_level = "LOW RISK"
        color = "🟢"
    elif risk_percent <= 70:
        risk_level = "MEDIUM RISK"
        color = "🟡"
    else:
        risk_level = "HIGH RISK"
        color = "🔴"

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Xác suất rời bỏ", f"{risk_percent:.1f}%")
    with col_b:
        st.metric("Mức độ rủi ro", f"{color} {risk_level}")
    with col_c:
        st.metric("Risk Score Input", f"{risk_score:.1f}%")

    # Prediction & Recommendation
    if prob >= 0.5:
        st.error("⚠️ KHÁCH HÀNG CÓ NGUY CƠ RỜI BỎ CAO")
        st.markdown("**Khuyến nghị:** RM cần liên hệ chăm sóc khách hàng ngay. Có thể xem xét ưu đãi lãi suất, quà tặng hoặc gói dịch vụ đặc biệt.")
    else:
        st.success("✅ KHÁCH HÀNG CÓ KHẢ NĂNG TIẾP TỤC SỬ DỤNG DỊCH VỤ")
        st.markdown("**Khuyến nghị:** Duy trì chăm sóc định kỳ. Khách hàng thuộc nhóm ổn định.")

    # Progress bar
    st.progress(float(prob))
    
    with st.expander("Chi tiết dự đoán"):
        st.write(f"Xác suất rời bỏ: **{prob:.4f}**")
        st.write(f"Risk Level: **{risk_level}**")
