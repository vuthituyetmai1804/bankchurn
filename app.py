import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(page_title="BIDV Premium", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("bidv_churn_modeltuning.pkl")

model = load_model()

# =========================================================
# CLEAN & MODERN CSS (Light Theme)
# =========================================================
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    
    /* Card thiết kế dạng card trắng, đổ bóng nhẹ */
    .clean-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    
    h1, h2 { color: #007353 !important; font-weight: 700; }
    
    .stButton > button {
        background: #007353 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold;
        width: 100%;
        border: none;
    }
    
    .stMetric { background: #f1f3f5; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# UI LAYOUT
# =========================================================
st.title("🏦 BIDV Risk Intelligence")
st.markdown("Hệ thống dự báo rời bỏ khách hàng - Giao diện chuyên nghiệp")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.subheader("📋 Thông tin khách hàng")
    age = st.slider("🎂 Tuổi", 20, 80, 35)
    credit_sco = st.slider("💳 Điểm tín dụng", 495, 800, 650)
    balance = st.number_input("💰 Số dư tài khoản (VND)", value=50000000)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.subheader("📊 Thông tin sử dụng")
    monthly_ir = st.number_input("💵 Thu nhập hàng tháng (VND)", value=15000000)
    nums_service = st.slider("🏦 Số lượng dịch vụ", 1, 8, 3)
    engagement_score = st.slider("🤝 Điểm tương tác app", 0, 100, 50)
    active_text = st.radio("📱 Hoạt động gần đây", ["Có", "Không"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🚀 XEM KẾT QUẢ PHÂN TÍCH"):
    # Xử lý logic
    active_member = 1 if active_text == "Có" else 0
    features = pd.DataFrame([{'monthly_ir': monthly_ir, 'credit_sco': credit_sco, 'nums_service': nums_service, 
                              'engagement_score': engagement_score, 'balance': balance, 'age': age, 'active_member': active_member}])
    
    risk_score = model.predict_proba(features)[0][1]
    risk_percent = round(risk_score * 100, 2)

    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.metric("Risk Score", f"{risk_percent}%")
    
    with c2:
        # Gauge chart nhưng nền trắng
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = risk_percent,
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#007353"}}
        ))
        fig.update_layout(height=200, margin={'t':0, 'b':0, 'l':0, 'r':0})
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
