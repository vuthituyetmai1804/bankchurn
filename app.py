import streamlit as st
import pandas as pd
import joblib

# Cache model
@st.cache_resource
def load_model():
    return joblib.load("bidv_churn_modeltuning.pkl")

model = load_model()

# Page Config
st.set_page_config(page_title="BIDV Churn Prediction", page_icon="🏦", layout="wide")

# =========================================================
# REFINED CSS - "Premium Bank Interface"
# =========================================================
st.markdown("""
<style>
    /* Tổng thể */
    .stApp { background-color: #f8fafc; }
    
    /* Header chuyên nghiệp hơn */
    .header-box {
        background: linear-gradient(135deg, #007353 0%, #005a42 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(0, 115, 83, 0.2);
        margin-bottom: 30px;
    }
    
    /* Card nội dung sáng sủa, đổ bóng mềm */
    .content-box {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }

    /* Input Fields */
    .stNumberInput, .stSlider { margin-bottom: 15px; }

    /* Button "Xịn" */
    .stButton > button {
        width: 100%;
        height: 60px;
        background-color: #FFCC00 !important;
        color: #007353 !important;
        font-weight: 800 !important;
        font-size: 18px !important;
        border-radius: 12px !important;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover { transform: translateY(-3px); box-shadow: 0 6px 12px rgba(255, 204, 0, 0.3); }

    /* Kết quả */
    .metric-card {
        background: #f1f5f9;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# UI
st.markdown('<div class="header-box"><h1>🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ</h1><p>Giải pháp Quản trị Rủi ro chuẩn BIDV</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    age = st.slider("🎂 Tuổi", 20, 80, 35)
    credit_sco = st.slider("💳 Điểm tín dụng", 495, 800, 650)
    balance = st.number_input("💰 Số dư tài khoản (VND)", value=50000000)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    monthly_ir = st.number_input("💵 Thu nhập hàng tháng (VND)", value=15000000)
    nums_service = st.slider("🏦 Số lượng dịch vụ", 1, 8, 3)
    engagement_score = st.slider("🤝 Điểm tương tác app", 0, 100, 50)
    active_text = st.radio("📱 Hoạt động gần đây", ["Có", "Không"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🔍 DỰ ĐOÁN NGAY"):
    # Logic
    active_member = 1 if active_text == "Có" else 0
    input_df = pd.DataFrame([{
        'monthly_ir': monthly_ir, 'credit_sco': credit_sco, 'nums_service': nums_service,
        'engagement_score': engagement_score, 'balance': balance, 'age': age, 'active_member': active_member
    }])
    
    risk_score = model.predict_proba(input_df)[0][1]
    risk_percent = round(risk_score * 100, 2)

    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Risk Score", f"{risk_percent}%")
    res_col2.metric("Level", "HIGH" if risk_percent > 70 else "LOW")
    res_col3.metric("Action", "URGENT" if risk_percent > 70 else "KEEP")
    
    st.progress(int(risk_percent))
