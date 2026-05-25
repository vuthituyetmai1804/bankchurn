import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("bidv_churn_modeltuning.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Không thể tải model: {e}")
    st.stop()

# ====================== CONFIG ======================
st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ====================== CUSTOM CSS - PHIÊN BẢN SÁNG CAO CẤP ======================
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fc;
    }

    /* Wave Container - Giữ nguyên sóng vàng của bạn */
    .wave-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 220px;
        z-index: 0;
        pointer-events: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%23FFCC00' fill-opacity='0.45' d='M0,192L48,176C96,160,192,128,288,133.3C384,139,480,181,576,197.3C672,213,768,203,864,170.7C960,139,1056,85,1152,80C1248,75,1344,117,1392,138.7L1440,160L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
        background-size: cover;
        background-repeat: no-repeat;
    }

    /* Glass Card sáng */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 115, 83, 0.15);
        border-radius: 24px;
        padding: 2.2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    }

    .header-box {
        background: linear-gradient(90deg, #007353, #00a67e);
        padding: 45px 20px;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 30px;
        color: white;
        box-shadow: 0 15px 25px rgba(0, 115, 83, 0.25);
    }

    .header-title { 
        font-size: 48px; 
        font-weight: 900; 
        margin-bottom: 10px;
    }
    .header-sub { 
        font-size: 19px; 
        opacity: 0.95;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 68px;
        background: linear-gradient(90deg, #007353, #00a67e) !important;
        color: white !important;
        font-size: 26px;
        font-weight: bold;
        border-radius: 50px !important;
        border: none;
        box-shadow: 0 8px 25px rgba(0, 115, 83, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0, 115, 83, 0.4);
    }

    .input-card {
        background: white;
        border-radius: 20px;
        padding: 24px;
        border: 1px solid #e0e7e0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="wave-container"></div>', unsafe_allow_html=True)

# ====================== HEADER (GIỮ NGUYÊN CHỮ CỦA BẠN) ======================
st.markdown("""
<div class="header-box">
    <div class="header-title">🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ</div>
    <div class="header-sub">Ứng dụng Mô hình Cây quyết định trong Quản trị Rủi ro Ngân hàng BIDV</div>
</div>
""", unsafe_allow_html=True)

# ====================== INPUT ======================
st.markdown("## 📋 Nhập thông tin khách hàng")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    age = st.slider("🎂 Tuổi", 20, 80, 35)
    credit_sco = st.slider("💳 Điểm tín dụng", 495, 800, 650)
    balance = st.number_input("💰 Số dư tài khoản (VND)", min_value=0, value=50_000_000, step=1_000_000)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    monthly_ir = st.number_input("💵 Thu nhập hàng tháng (VND)", min_value=0, value=15_000_000, step=1_000_000)
    nums_service = st.slider("🏦 Số lượng dịch vụ sử dụng", 1, 8, 3)
    engagement_score = st.slider("🤝 Điểm tương tác app", 0, 100, 50)
    active_text = st.radio("📱 Hoạt động gần đây", ["Có", "Không"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

active_member = 1 if active_text == "Có" else 0

# ====================== BUTTON ======================
predict_btn = st.button("🔍 DỰ ĐOÁN NGAY")

# ====================== PREDICTION ======================
if predict_btn:
    features_order = ['monthly_ir', 'credit_sco', 'nums_service', 'engagement_score', 'balance', 'age', 'active_member']
    
    input_df = pd.DataFrame([{
        'monthly_ir': monthly_ir, 'credit_sco': credit_sco, 'nums_service': nums_service,
        'engagement_score': engagement_score, 'balance': balance, 'age': age, 'active_member': active_member
    }])
    
    final_input = input_df[features_order]
    risk_score = model.predict_proba(final_input)[0][1]
    risk_percent = round(risk_score * 100, 2)

    if risk_percent < 30:
        risk_level = "🟢 LOW RISK"
        prediction_text = "✅ Khách hàng có khả năng tiếp tục sử dụng dịch vụ"
        recommendation = "✅ Duy trì mối quan hệ tốt và tiếp tục chăm sóc định kỳ."
        color = "#22c55e"
    elif risk_percent <= 70:
        risk_level = "🟡 MEDIUM RISK"
        prediction_text = "⚠️ Khách hàng có nguy cơ rời bỏ"
        recommendation = "📞 Nên chăm sóc chủ động: Gọi điện tư vấn, tặng ưu đãi lãi suất, voucher."
        color = "#f59e0b"
    else:
        risk_level = "🔴 HIGH RISK"
        prediction_text = "⚠️ Khách hàng có nguy cơ rời bỏ"
        recommendation = "🚨 Cần liên hệ khẩn cấp trong 24h để giữ chân khách hàng."
        color = "#ef4444"

    st.markdown("---")
    st.markdown("### 📊 KẾT QUẢ PHÂN TÍCH")

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("RISK SCORE", f"{risk_percent}%")
    with colB:
        st.metric("RISK LEVEL", risk_level)
    with colC:
        st.metric("DỰ ĐOÁN", "CHURN" if risk_percent >= 50 else "STAY")

    st.progress(int(risk_percent))

    st.markdown(f"""
    <div style="background: white; padding: 28px; border-radius: 20px; border-left: 8px solid {color}; margin: 20px 0; box-shadow: 0 5px 20px rgba(0,0,0,0.08);">
        <h2 style="color:{color}; text-align:center; margin:0;">{prediction_text}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:white; padding:25px; border-radius:20px; box-shadow:0 5px 20px rgba(0,0,0,0.06);">
        <h3>🎯 Khuyến nghị hành động:</h3>
        <p>{recommendation}</p>
    </div>
    """, unsafe_allow_html=True)
