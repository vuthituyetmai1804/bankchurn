import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load("bidv_churn_modeltuning.pkl")

# =========================================================
# CUSTOM CSS - Thiết kế giống ảnh
# =========================================================
st.markdown("""
<style>
    .main {
        background-color: #f8f9fc;
    }
    .header-box {
        background: linear-gradient(90deg, #005bea, #00c6fb);
        padding: 35px 20px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 20px rgba(0,91,234,0.3);
    }
    .header-title {
        color: white;
        font-size: 38px;
        font-weight: bold;
        margin: 0;
    }
    .header-sub {
        color: #e0f0ff;
        font-size: 18px;
        margin-top: 8px;
    }

    .stButton > button {
        width: 100%;
        height: 68px;
        background: linear-gradient(90deg, #005bea, #00aaff);
        color: white;
        font-size: 26px;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 15px rgba(0,91,234,0.4);
    }

    .input-section {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    .result-box {
        background: white;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        text-align: center;
    }

    .gauge {
        font-size: 72px;
        font-weight: bold;
        background: linear-gradient(90deg, #00c6fb, #005bea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .status-box {
        background: linear-gradient(90deg, #005bea, #003399);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="header-box">
    <h1 class="header-title">🏦 BIDV - PHÂN TÍCH VÀ DỰ ĐOÁN RỦI RO RỜI BỎ KHÁCH HÀNG</h1>
    <p class="header-sub">HỆ THỐNG AI-POWERED ĐÁNH GIÁ CHURN TRỰC TUYẾN</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LAYOUT 2 CỘT
# =========================================================
col_input, col_result = st.columns([1.1, 1])

# ====================== INPUT ======================
with col_input:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("1. NHẬP THÔNG TIN TEST CASE")

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Tuổi", min_value=18, max_value=80, value=38)
        balance = st.number_input("Số dư tài khoản (VND)", min_value=0, value=150000000, step=1000000)
        credit_sco = st.slider("Điểm tín dụng (CIC)", 300, 850, 785)
        
    with c2:
        monthly_ir = st.number_input("Thu nhập tháng (VND)", min_value=0, value=25000000, step=500000)
        nums_service = st.slider("Số dịch vụ", 1, 8, 4)
        engagement_score = st.slider("Engagement Score", 0, 100, 92)
        active_text = st.selectbox("Hoạt động gần đây", ["Có", "Không"])

    st.markdown('</div>', unsafe_allow_html=True)

# ====================== RESULT ======================
with col_result:
    predict_btn = st.button("🔍 CHẠY TEST CASE MÔ HÌNH")

    if predict_btn:
        active_member = 1 if active_text == "Có" else 0

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
            risk_level = "LOW RISK"
            color = "#00c853"
            status = "Ở lại (STAY)"
        elif risk_percent <= 70:
            risk_level = "MEDIUM RISK"
            color = "#ffb300"
            status = "TRUNG BÌNH (Medium Risk)"
        else:
            risk_level = "HIGH RISK"
            color = "#f44336"
            status = "RỜI BỎ (CHURN)"

        # ====================== KẾT QUẢ ======================
        st.markdown(f"""
        <div class="result-box">
            <h3>RISK SCORE</h3>
            <h1 class="gauge">{risk_percent}%</h1>
            <p style="color:{color}; font-size:22px; font-weight:bold;">{risk_level}</p>
            
            <div class="status-box">
                <h4>TRẠNG THÁI: CHURN (RISK LEVEL: {risk_level})</h4>
                <h3 style="margin:10px 0;">{status}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar
        st.progress(int(risk_percent))

        # Recommendation
        if risk_percent >= 50:
            st.error("⚠️ Khách hàng có nguy cơ rời bỏ cao. Nên có hành động giữ chân khẩn cấp.")
        else:
            st.success("✅ Khách hàng có xu hướng ở lại. Tiếp tục duy trì chăm sóc.")
