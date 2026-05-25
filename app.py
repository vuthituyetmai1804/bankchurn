import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="BIDV Churn Predict", page_icon="🔍", layout="wide")

# CSS đẹp
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stButton>button {
        width: 100%; height: 3.8rem; font-size: 1.3rem; font-weight: bold;
        background: linear-gradient(90deg, #FF4B4B, #FF6B6B); color: white;
    }
    .metric {font-size: 2.5rem !important;}
</style>
""", unsafe_allow_html=True)

st.title("🔍 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ - BIDV")
st.markdown("**Machine Learning Model** | Phân tích rủi ro churn")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('bidv_churnn_model.pkl')
        scaler = joblib.load('scaler_bidv_model.pkl')
        return model, scaler
    except:
        st.error("❌ Chưa tìm thấy file model. Vui lòng upload `bidv_churnn_model.pkl` và `scaler_bidv_model.pkl`")
        return None, None

model, scaler = load_model()

if model is None:
    st.stop()

# ==================== INPUT FORM ====================
st.header("📋 Nhập Thông Tin Khách Hàng")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("**Tuổi**", 18, 100, 45)
    credit_score = st.slider("**Điểm tín dụng**", 300, 850, 700)
    balance = st.number_input("**Số dư tài khoản (VND)**", 0, 1000000000, 50000000, step=1000000)
    monthly_income = st.number_input("**Thu nhập hàng tháng (VND)**", 0, 500000000, 25000000, step=1000000)

with col2:
    tenure = st.slider("**Số năm gắn bó**", 0, 10, 3)
    active_member = st.radio("**Hoạt động gần đây**", ["Có", "Không"], horizontal=True)
    loyalty_level = st.selectbox("**Hạng khách hàng**", ["Bronze", "Silver", "Gold"])

# Bổ sung
st.subheader("Thông tin bổ sung")
col3, col4 = st.columns(2)
with col3:
    engagement_score = st.slider("Điểm tương tác (0-100)", 0, 100, 65)
    nums_card = st.slider("Số thẻ tín dụng", 1, 5, 2)
with col4:
    nums_service = st.slider("Số dịch vụ đang dùng", 1, 6, 3)

# ==================== DỰ ĐOÁN ====================
if st.button("🔍 DỰ ĐOÁN NGAY", type="primary"):
    # Chuẩn bị dữ liệu
    input_dict = {
        'credit_sco': credit_score,
        'age': age,
        'balance': balance,
        'monthly_ir': monthly_income,
        'tenure_ye': tenure,
        'nums_card': nums_card,
        'nums_service': nums_service,
        'engagement_score': engagement_score,
        'active_member': 1 if active_member == "Có" else 0,
        'married': 1,
        'last_transaction_month': 3,
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Scale
    cols_to_scale = ['credit_sco', 'age', 'balance', 'monthly_ir', 'tenure_ye',
                    'nums_card', 'nums_service', 'engagement_score']
    
    input_scaled = input_df.copy()
    input_scaled[cols_to_scale] = scaler.transform(input_scaled[cols_to_scale])
    
    # Predict
    proba = model.predict_proba(input_scaled)[0][1]
    risk_score = proba * 100
    
    # Hiển thị kết quả
    st.success("**DỰ ĐOÁN HOÀN TẤT**")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.metric("**RISK SCORE**", f"{risk_score:.1f}%")
    
    with col_res2:
        if risk_score < 30:
            level = "🟢 LOW RISK"
        elif risk_score <= 70:
            level = "🟡 MEDIUM RISK"
        else:
            level = "🔴 HIGH RISK"
        st.markdown(f"### {level}")
    
    if risk_score >= 50:
        st.error("⚠️ **Khách hàng có nguy cơ rời bỏ**")
    else:
        st.success("✅ **Khách hàng có khả năng tiếp tục**")
    
    # Khuyến nghị
    st.markdown("### 💡 Khuyến nghị hành động")
    if risk_score >= 70:
        st.error("🚨 **Liên hệ khẩn cấp trong 24h** - Ưu đãi đặc biệt, gọi tư vấn cá nhân hóa")
    elif risk_score >= 40:
        st.warning("⚠️ **Chăm sóc chủ động**: Gọi điện, tặng voucher, nâng cấp dịch vụ")
    else:
        st.success("✅ Duy trì mối quan hệ tốt, theo dõi định kỳ")

st.markdown("---")
st.caption("BIDV Churn Prediction System • Powered by Streamlit")
