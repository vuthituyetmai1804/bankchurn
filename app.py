import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="BIDV Risk Intelligence", 
    page_icon="🏦", 
    layout="wide"
)

@st.cache_resource
def load_model():
    # Đảm bảo file .pkl nằm cùng thư mục với app.py
    return joblib.load("bidv_churn_modeltuning.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Không thể tải model: {e}")
    st.stop()

# =========================================================
# CUSTOM CSS - GLASSMORPHISM STYLE
# =========================================================
st.markdown("""
<style>
    /* Tổng thể */
    .stApp { 
        background: linear-gradient(135deg, #007353 0%, #004d38 100%); 
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Hiệu ứng Kính mờ */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    h1, h2, h3 { color: #FFCC00 !important; }
    
    /* Input Style */
    .stNumberInput, .stSlider { color: white !important; }
    
    /* Button Style */
    .stButton > button {
        background: #FFCC00 !important;
        color: #007353 !important;
        font-weight: 800 !important;
        border-radius: 50px !important;
        width: 100% !important;
        height: 60px !important;
        border: none !important;
        font-size: 18px !important;
        transition: transform 0.2s;
    }
    .stButton > button:hover { transform: scale(1.02); }
    
    [data-testid="stMetricValue"] { color: white !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="glass-card" style="text-align: center;"><h1>🏦 HỆ THỐNG PHÂN TÍCH RỦI RO BIDV</h1><p>Ứng dụng AI dự báo rời bỏ khách hàng - Phiên bản cao cấp</p></div>', unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📋 Thông tin khách hàng")
    age = st.slider("🎂 Tuổi", 20, 80, 35)
    credit_sco = st.slider("💳 Điểm tín dụng", 495, 800, 650)
    balance = st.number_input("💰 Số dư tài khoản (VND)", value=50000000, step=1000000)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Thông tin dịch vụ")
    monthly_ir = st.number_input("💵 Thu nhập hàng tháng (VND)", value=15000000, step=1000000)
    nums_service = st.slider("🏦 Số lượng dịch vụ sử dụng", 1, 8, 3)
    engagement_score = st.slider("🤝 Điểm tương tác app", 0, 100, 50)
    active_text = st.radio("📱 Hoạt động gần đây", ["Có", "Không"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

predict_btn = st.button("🚀 PHÂN TÍCH DỮ LIỆU NGAY")

# =========================================================
# PREDICTION LOGIC
# =========================================================
if predict_btn:
    active_member = 1 if active_text == "Có" else 0
    
    features_order = ['monthly_ir', 'credit_sco', 'nums_service', 'engagement_score', 'balance', 'age', 'active_member']
    input_df = pd.DataFrame([{
        'monthly_ir': monthly_ir, 'credit_sco': credit_sco, 'nums_service': nums_service,
        'engagement_score': engagement_score, 'balance': balance, 'age': age, 'active_member': active_member
    }])
    
    risk_score = model.predict_proba(input_df[features_order])[0][1]
    risk_percent = round(risk_score * 100, 2)

    st.markdown("---")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Kết quả phân tích")
    
    # Gauge Chart cho chuyên nghiệp
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_percent,
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#FFCC00"},
            'steps': [
                {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.3)"},
                {'range': [30, 70], 'color': "rgba(255, 165, 0, 0.3)"},
                {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.3)"}
            ]
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Lời khuyên
    if risk_percent < 30:
        msg = "Khách hàng đang rất hài lòng. Hãy tiếp tục duy trì các ưu đãi hiện tại."
        color = "#00FF00"
    elif risk_percent <= 70:
        msg = "Khách hàng có nguy cơ rời bỏ trung bình. Nên gửi ưu đãi lãi suất hoặc Voucher."
        color = "#FFA500"
    else:
        msg = "CẢNH BÁO: Khách hàng có khả năng rời bỏ cao. Cần liên hệ CSKH ngay trong 24h."
        color = "#FF0000"
        
    st.markdown(f"<h3 style='color: {color} !important;'>🎯 Khuyến nghị: {msg}</h3>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
