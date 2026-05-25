import streamlit as st
import pandas as pd
import joblib

# Cache model
@st.cache_resource
def load_model():
    return joblib.load("bidv_churn_modeltuning.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Không thể tải model: {e}")
    st.stop()

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="BIDV | Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================== GLASSMORPHISM CSS ======================
st.markdown("""
<style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e2937 100%);
        color: #e2e8f0;
    }

    /* Glassmorphism Container */
    .glass-container {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Header */
    .main-header {
        background: linear-gradient(90deg, #007353, #00a67e);
        padding: 2.5rem 0;
        border-radius: 0 0 30px 30px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 115, 83, 0.4);
        position: relative;
        overflow: hidden;
    }

    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        text-align: center;
        color: white;
    }

    .logo {
        font-size: 42px;
        font-weight: 900;
        letter-spacing: -2px;
        margin-bottom: 8px;
    }

    /* Input Cards */
    .input-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.8rem;
        transition: all 0.3s ease;
    }

    .input-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 115, 83, 0.5);
    }

    /* Custom Button */
    .predict-btn {
        background: linear-gradient(90deg, #007353, #00c48c);
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        padding: 18px 60px;
        border: none;
        border-radius: 50px;
        width: 100%;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0, 115, 83, 0.4);
        transition: all 0.3s ease;
    }

    .predict-btn:hover {
        transform: scale(1.03);
        box-shadow: 0 15px 40px rgba(0, 115, 83, 0.5);
    }

    /* Result Cards */
    .result-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
    }

    .risk-high { border-left: 6px solid #ef4444; }
    .risk-medium { border-left: 6px solid #f59e0b; }
    .risk-low { border-left: 6px solid #22c55e; }
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown("""
<div class="main-header">
    <div class="header-content">
        <div style="font-size: 3.2rem; font-weight: 800; letter-spacing: -3px;">
            BIDV <span style="color:#fff;">INTELLIGENCE</span>
        </div>
        <h1 style="font-size: 2.1rem; margin: 12px 0 8px 0; font-weight: 600;">
            HỆ THỐNG DỰ ĐOÁN RỜI BỎ KHÁCH HÀNG
        </h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">
            Mô hình Machine Learning • Độ chính xác cao • Hỗ trợ quyết định nhanh chóng
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================== MAIN CONTENT ======================
col1, col2 = st.columns([1, 1.1])

with col1:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📋 Thông tin khách hàng")

    c1, c2 = st.columns(2)
    
    with c1:
        age = st.slider("🎂 Tuổi", 20, 80, 35, label_visibility="collapsed")
        st.caption("Tuổi")
        
        credit_sco = st.slider("💳 Điểm tín dụng", 495, 800, 650, label_visibility="collapsed")
        st.caption("Điểm tín dụng")
        
        balance = st.number_input("💰 Số dư (VND)", 
                                min_value=0, 
                                value=50_000_000, 
                                step=1_000_000,
                                label_visibility="collapsed")
        st.caption("Số dư tài khoản")

    with c2:
        monthly_ir = st.number_input("💵 Thu nhập hàng tháng (VND)", 
                                   min_value=0, 
                                   value=15_000_000, 
                                   step=1_000_000,
                                   label_visibility="collapsed")
        st.caption("Thu nhập hàng tháng")
        
        nums_service = st.slider("🏦 Số dịch vụ đang sử dụng", 1, 8, 3, label_visibility="collapsed")
        st.caption("Số dịch vụ")
        
        engagement_score = st.slider("📱 Điểm tương tác ứng dụng", 0, 100, 50, label_visibility="collapsed")
        st.caption("Điểm tương tác")

    active_text = st.radio("📱 Hoạt động gần đây", ["Có", "Không"], horizontal=True)
    active_member = 1 if active_text == "Có" else 0

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    st.markdown("### 📊 Kết quả dự đoán")
    
    predict_btn = st.button("🔍 DỰ ĐOÁN NGAY", 
                          type="primary", 
                          use_container_width=True,
                          key="predict")

    if predict_btn:
        # Prepare data
        features_order = ['monthly_ir', 'credit_sco', 'nums_service', 
                         'engagement_score', 'balance', 'age', 'active_member']
        
        input_df = pd.DataFrame([{
            'monthly_ir': monthly_ir,
            'credit_sco': credit_sco,
            'nums_service': nums_service,
            'engagement_score': engagement_score,
            'balance': balance,
            'age': age,
            'active_member': active_member
        }])
        
        final_input = input_df[features_order]
        risk_score = model.predict_proba(final_input)[0][1]
        risk_percent = round(risk_score * 100, 2)

        # Risk Level
        if risk_percent < 30:
            level = "LOW RISK"
            color = "#22c55e"
            status = "🟢 AN TOÀN"
        elif risk_percent <= 70:
            level = "MEDIUM RISK"
            color = "#f59e0b"
            status = "🟡 CẢNH BÁO"
        else:
            level = "HIGH RISK"
            color = "#ef4444"
            status = "🔴 NGUY HIỂM"

        # Display Results
        st.markdown(f"""
        <div class="result-card risk-{'low' if risk_percent < 30 else 'medium' if risk_percent <= 70 else 'high'}">
            <h2 style="color: {color}; margin: 0; font-size: 3.5rem; font-weight: 700;">
                {risk_percent}%
            </h2>
            <h3 style="margin: 8px 0 20px 0;">{status}</h3>
            <p style="font-size: 1.3rem; margin-bottom: 20px;">{level}</p>
            
            <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 10px; overflow: hidden;">
                <div style="width: {risk_percent}%; height: 100%; background: linear-gradient(90deg, {color}, #fff);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommendation
        if risk_percent < 30:
            rec = "Khách hàng rất trung thành. Nên duy trì và mở rộng mối quan hệ."
        elif risk_percent <= 70:
            rec = "Cần chăm sóc chủ động: Ưu đãi lãi suất, quà tặng, tư vấn cá nhân hóa."
        else:
            rec = "Nguy cơ cao rời bỏ. Nên liên hệ khẩn cấp trong 24h."

        st.info(f"**Khuyến nghị:** {rec}")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 4rem; opacity: 0.6; font-size: 0.9rem;">
    © 2026 BIDV • Hệ thống dự đoán churn bằng Machine Learning
</div>
""", unsafe_allow_html=True)
