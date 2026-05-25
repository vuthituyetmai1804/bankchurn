import streamlit as st
import pandas as pd
import joblib

# =========================================================
# CONFIG & MODEL
# =========================================================
st.set_page_config(page_title="BIDV Churn Prediction", page_icon="🏦", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("bidv_churn_modeltuning.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Không thể tải model: {e}")
    st.stop()

# =========================================================
# CUSTOM CSS (Sợi sóng chéo & Giao diện chuẩn)
# =========================================================
st.markdown("""
<style>
.stApp {
    background-color: #f4f6f9;
    background-image: 
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 800'%3E%3Cpath fill='none' stroke='%23FFCC00' stroke-width='1' stroke-opacity='0.2' d='M-100,800 C200,600 400,200 800,300 C1200,400 1400,0 1600,-100'/%3E%3Cpath fill='none' stroke='%23FFCC00' stroke-width='1' stroke-opacity='0.2' d='M-100,850 C200,650 400,250 800,350 C1200,450 1400,50 1600,-50'/%3E%3Cpath fill='none' stroke='%23FFCC00' stroke-width='1' stroke-opacity='0.2' d='M-100,900 C200,700 400,300 800,400 C1200,500 1400,100 1600,0'/%3E%3Cpath fill='none' stroke='%23FFCC00' stroke-width='1' stroke-opacity='0.2' d='M-100,950 C200,750 400,350 800,450 C1200,550 1400,150 1600,50'/%3E%3C/svg%3E");
    background-attachment: fixed;
    background-size: cover;
}

.block-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 30px;
    padding: 3rem !important;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
    z-index: 1;
    position: relative;
}

.header-box {
    background: #007353;
    padding: 40px;
    border-radius: 30px;
    text-align: center;
    margin-bottom: 30px;
    color: white;
    box-shadow: 0px 10px 20px rgba(0, 115, 83, 0.3);
}
.header-title { font-size: 50px; font-weight: 900; margin-bottom: 10px; color: white; }
.header-sub { font-size: 18px; color: rgba(255,255,255,0.9); }

.stButton > button {
    width: 100%;
    height: 65px;
    background-color: #007353 !important;
    color: white !important;
    font-size: 24px;
    font-weight: bold;
    border-radius: 50px !important;
    border: none;
    transition: all 0.3s ease;
}
.stButton > button:hover { filter: brightness(1.2); transform: translateY(-2px); }

.result-box, .recommend-box {
    background-color: white;
    padding: 25px;
    border-radius: 25px !important;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    border: 1px solid #e1e1e1;
    margin-bottom: 20px;
}
h2 { color: #007353 !important; }
[data-testid="stMetricValue"] { color: #007353; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# UI LAYOUT
# =========================================================
st.markdown("""
<div class="header-box">
    <div class="header-title">🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ</div>
    <div class="header-sub">Ứng dụng Mô hình Cây quyết định trong Quản trị Rủi ro Ngân hàng BIDV</div>
</div>
""", unsafe_allow_html=True)

st.markdown("## 📋 Nhập thông tin khách hàng")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Tuổi", 20, 80, 35)
    credit_sco = st.slider("💳 Điểm tín dụng", 495, 800, 650)
    balance = st.number_input("💰 Số dư tài khoản (VND)", min_value=0, value=50000000, step=1000000)

with col2:
    monthly_ir = st.number_input("💵 Thu nhập hàng tháng (VND)", min_value=0, value=15000000, step=1000000)
    nums_service = st.slider("🏦 Số lượng dịch vụ sử dụng", 1, 8, 3)
    engagement_score = st.slider("🤝 Điểm tương tác app", 0, 100, 50)
    active_text = st.radio("📱 Hoạt động gần đây", ["Có", "Không"])

active_member = 1 if active_text == "Có" else 0

if st.button("🔍 DỰ ĐOÁN NGAY"):
    features_order = ['monthly_ir', 'credit_sco', 'nums_service', 'engagement_score', 'balance', 'age', 'active_member']
    input_df = pd.DataFrame([{
        'monthly_ir': monthly_ir, 'credit_sco': credit_sco, 'nums_service': nums_service,
        'engagement_score': engagement_score, 'balance': balance, 'age': age, 'active_member': active_member
    }])[features_order]

    risk_score = model.predict_proba(input_df)[0][1]
    risk_percent = round(risk_score * 100, 2)

    # Logic kết quả
    if risk_percent < 30:
        risk_level, color = "🟢 LOW RISK", "green"
        prediction_text, recommendation = "✅ Khách hàng ổn định", "Duy trì chăm sóc định kỳ."
    elif risk_percent <= 70:
        risk_level, color = "🟡 MEDIUM RISK", "orange"
        prediction_text, recommendation = "⚠️ Khách hàng có nguy cơ rời bỏ", "Gọi điện tư vấn, tặng ưu đãi/voucher."
    else:
        risk_level, color = "🔴 HIGH RISK", "red"
        prediction_text, recommendation = "🚨 Cần liên hệ khẩn cấp trong 24h để giữ chân khách hàng.", "Gửi ưu đãi đặc biệt ngay."

    st.markdown("---")
    st.markdown("# 📊 KẾT QUẢ PHÂN TÍCH")
    cA, cB, cC = st.columns(3)
    cA.metric("RISK SCORE", f"{risk_percent}%")
    cB.metric("RISK LEVEL", risk_level)
    cC.metric("PREDICTION", "CHURN" if risk_percent >= 50 else "STAY")
    st.progress(int(risk_percent))
    
    st.markdown(f'<div class="result-box"><h2 style="color:{color}; text-align:center;">{prediction_text}</h2></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="recommend-box" style="border-left:8px solid {color};"><h3>🎯 Khuyến nghị:</h3><p>{recommendation}</p></div>', unsafe_allow_html=True)
