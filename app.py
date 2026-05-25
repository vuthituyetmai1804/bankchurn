import streamlit as st
import pandas as pd
import joblib

# =========================================================
# CONFIG & LOAD MODEL
# =========================================================
st.set_page_config(page_title="BIDV Churn Prediction", page_icon="🏦", layout="wide")

@st.cache_resource
def load_model():
    # Đảm bảo file .pkl ở cùng thư mục
    return joblib.load("bidv_churn_modeltuning.pkl")

try:
    model = load_model()
except:
    model = None

# =========================================================
# CSS HOÀN CHỈNH
# =========================================================
st.markdown("""
<style>
    /* 1. Sóng uốn lượn liền mạch */
    .wave-container {
        position: fixed; bottom: 0; left: 0; width: 100%; height: 120px;
        z-index: 0; pointer-events: none; opacity: 0.5;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%23FFCC00' d='M0,192L48,176C96,160,192,128,288,133.3C384,139,480,181,576,197.3C672,213,768,203,864,170.7C960,139,1056,85,1152,80C1248,75,1344,117,1392,138.7L1440,160L1440,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
        background-repeat: repeat-x; background-size: 1440px 100%; background-position: bottom;
    }
    
    /* 2. Header & Container */
    .header-box { background: #007353; padding: 10px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px; }
    div[data-testid="column"] { background: rgba(255, 255, 255, 0.95); padding: 20px !important; border-radius: 15px; border: 1px solid #eee; }
    
    /* 3. Chữ nhỏ gọn */
    .small-text { font-size: 14px !important; }
    .risk-label { font-size: 18px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="wave-container"></div>', unsafe_allow_html=True)

# =========================================================
# UI
# =========================================================
st.markdown('<div class="header-box"><h5>🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ (BIDV)</h5></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    st.markdown("##### 📋 Thông tin cơ bản")
    age = st.slider("Tuổi", 20, 80, 35)
    credit_sco = st.slider("Điểm tín dụng", 495, 800, 650)
    balance = st.number_input("Số dư (VND)", min_value=0, value=50000000, step=1000000)
    active_text = st.radio("Hoạt động gần đây", ["Có", "Không"])

with col2:
    st.markdown("##### 📊 Thông tin tài chính")
    monthly_ir = st.number_input("Thu nhập (VND)", min_value=0, value=15000000, step=1000000)
    nums_service = st.slider("Số lượng dịch vụ", 1, 8, 3)
    engagement_score = st.slider("Điểm tương tác app", 0, 100, 50)
    st.write("---")
    predict_btn = st.button("🚀 DỰ ĐOÁN NGAY", use_container_width=True)

with col3:
    st.markdown("##### 🎯 Kết quả dự báo")
    if predict_btn and model:
        input_data = pd.DataFrame([{'monthly_ir': monthly_ir, 'credit_sco': credit_sco, 'nums_service': nums_service, 
                                    'engagement_score': engagement_score, 'balance': balance, 'age': age, 
                                    'active_member': 1 if active_text == "Có" else 0}])
        
        risk_score = model.predict_proba(input_data)[0][1]
        risk_percent = round(risk_score * 100, 2)
        
        if risk_percent < 30:
            level, color, desc = "LOW RISK", "green", "Khách hàng ổn định."
        elif risk_percent <= 70:
            level, color, desc = "MEDIUM RISK", "orange", "Cần chăm sóc chủ động."
        else:
            level, color, desc = "HIGH RISK", "red", "Cần liên hệ khẩn cấp."

        # Hiển thị nhỏ gọn
        c1, c2 = st.columns(2)
        c1.metric("SCORE", f"{risk_percent}%")
        c2.markdown(f"**LEVEL**<br><span style='color:{color}; font-size:18px;'>{level}</span>", unsafe_allow_html=True)
        
        st.progress(int(risk_percent))
        st.markdown(f"<div class='small-text' style='color:{color}; margin-top:10px;'>{desc}</div>", unsafe_allow_html=True)
    else:
        st.info("Nhập thông tin và nhấn nút để xem kết quả.")
