import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="BIDV Churn Predict", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .main {padding: 2rem;}
    .stButton>button {width: 100%; height: 3.8rem; font-size: 1.3rem; font-weight: bold;
        background: linear-gradient(90deg, #FF4B4B, #FF6B6B); color: white;}
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
        st.sidebar.success("✅ Model loaded!")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Lỗi load: {e}")
        return None, None

model, scaler = load_model()
if model is None:
    st.stop()

# ==================== INPUT ====================
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

st.subheader("Thông tin bổ sung")
col3, col4 = st.columns(2)
with col3:
    engagement_score = st.slider("Điểm tương tác (0-100)", 0, 100, 65)
    nums_card = st.slider("Số thẻ tín dụng", 1, 5, 2)
with col4:
    nums_service = st.slider("Số dịch vụ đang dùng", 1, 6, 3)

# ==================== DỰ ĐOÁN ====================
if st.button("🔍 DỰ ĐOÁN NGAY", type="primary"):
    try:
        # Chỉ tạo các cột số + các cột đã encode sẵn
        input_data = {
            'credit_sco': [credit_score],
            'age': [age],
            'balance': [balance],
            'monthly_ir': [monthly_income],
            'tenure_ye': [tenure],
            'nums_card': [nums_card],
            'nums_service': [nums_service],
            'engagement_score': [engagement_score],
            'active_member': [1 if active_member == "Có" else 0],
            'married': [1],
            'last_transaction_month': [3],
            'risk_score': [0.15],
            
            # One-hot & encoded columns (đã là số)
            'digital_behavior_offline': [0],
            'origin_province_TP. Hồ Chí Minh': [1],
            'origin_province_Hà Nội': [0],
            'origin_province_Đồng Nai': [0],
            'origin_province_Bình Dương': [0],
            'origin_province_Cần Thơ': [0],
            'origin_province_Long An': [0],
            'origin_province_Tiền Giang': [0],
            'origin_province_Tỉnh khác': [0],
            'occupation_Giáo viên/Giảng viên': [0],
            'occupation_Hưu trí': [0],
            'occupation_Kinh doanh/Bán hàng': [0],
            'occupation_Kế toán/Tài chính': [0],
            'occupation_Kỹ sư/Chuyên viên IT': [0],
            'occupation_Lao động phổ thông': [0],
            'occupation_Nhân viên văn phòng/Công chức': [0],
            'occupation_Nội trợ/Sinh viên': [0],
            'occupation_Quản lý/Lãnh đạo': [0],
            'customer_segment_Mass': [1],           # Giả sử
            'customer_segment_Priority': [0],
            'customer_segment_Emerging': [0],
            'loyalty_level_Bronze': [1 if loyalty_level == "Bronze" else 0],
            'loyalty_level_Silver': [1 if loyalty_level == "Silver" else 0],
            'loyalty_level_Gold': [1 if loyalty_level == "Gold" else 0],
            'gender_male': [1],
            'gender_female': [0],
            'risk_segment_Low': [1],
            'risk_segment_Medium': [0],
            'risk_segment_High': [0],
            'cluster_group': [4]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Chỉ scale các cột số
        cols_to_scale = ['credit_sco', 'age', 'balance', 'monthly_ir', 'nums_card', 
                        'nums_service', 'engagement_score', 'tenure_ye', 'risk_score']
        
        input_scaled = input_df.copy()
        input_scaled[cols_to_scale] = scaler.transform(input_scaled[cols_to_scale])
        
        # Dự đoán
        proba = model.predict_proba(input_scaled)[0][1]
        risk_score = proba * 100
        
        # Hiển thị
        st.success("**DỰ ĐOÁN HOÀN TẤT**")
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric("**RISK SCORE**", f"{risk_score:.1f}%")
        
        with col_res2:
            if risk_score < 30: level = "🟢 LOW RISK"
            elif risk_score <= 70: level = "🟡 MEDIUM RISK"
            else: level = "🔴 HIGH RISK"
            st.markdown(f"### {level}")
        
        if risk_score >= 50:
            st.error("⚠️ **Khách hàng có nguy cơ rời bỏ**")
        else:
            st.success("✅ **Khách hàng có khả năng tiếp tục**")
        
        st.markdown("### 💡 Khuyến nghị")
        if risk_score >= 70:
            st.error("🚨 Liên hệ khẩn cấp trong 24h")
        elif risk_score >= 40:
            st.warning("⚠️ Chăm sóc chủ động")
        else:
            st.success("✅ Duy trì tốt")

    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")

st.markdown("---")
st.caption("BIDV Churn Prediction System")
