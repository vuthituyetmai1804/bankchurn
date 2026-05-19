import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦", layout="wide")

st.title("🏦 DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ NGÂN HÀNG")
st.markdown("### Hệ thống dự báo Churn")

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
    balance = st.number_input("Số dư tài khoản (VND)", min_value=0, value=35000000, step=1000000)
    monthly_ir = st.number_input("Thu nhập hàng tháng (VND)", min_value=0, value=25000000, step=1000000)

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
        'risk_score': 0.25,
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
    
    prob = model.predict_proba(df_scaled)[0][1]

    # Risk Level
    if prob < 0.30:
        risk_level = "LOW RISK"
        color = "🟢"
    elif prob < 0.70:
        risk_level = "MEDIUM RISK"
        color = "🟡"
    else:
        risk_level = "HIGH RISK"
        color = "🔴"

    # ====================== OUTPUT ======================
    st.markdown("---")
    st.subheader("📊 KẾT QUẢ DỰ ĐOÁN")

    st.metric(label="**RISK SCORE** (Xác suất rời bỏ)", value=f"{prob:.1%}")

    st.markdown(f"### {color} **{risk_level}**")

    if prob >= 0.5:
        st.error("⚠️ **Khách hàng có nguy cơ rời bỏ**")
    else:
        st.success("✅ **Khách hàng có khả năng tiếp tục sử dụng dịch vụ**")

    st.markdown("### 💡 Khuyến nghị")
    if prob >= 0.70:
        st.error("🚨 Cần liên hệ khẩn cấp trong 24h. Ưu đãi đặc biệt, quà tặng cao cấp.")
    elif prob >= 0.40:
        st.warning("📞 Nên chăm sóc chủ động: Gọi điện tư vấn, tặng ưu đãi lãi suất, voucher.")
    else:
        st.success("✅ Duy trì mối quan hệ tốt. Có thể gửi chương trình khách hàng thân thiết.")

    st.progress(float(prob))

    with st.expander("Chi tiết dự đoán"):
        col_a, col_b = st.columns(2)
        col_a.metric("Xác suất rời bỏ", f"{prob:.1%}")
        col_b.metric("Xác suất ở lại", f"{(1 - prob):.1%}")
