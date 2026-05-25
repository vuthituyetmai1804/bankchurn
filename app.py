import streamlit as st
import pandas as pd
import joblib

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="BIDV Churn Prediction", page_icon="🏦", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("bidv_churn_modeltuning.pkl")

try:
    model = load_model()
except Exception:
    model = None

# =========================================================
# CSS TỐI ƯU: Header cố định, sóng ở background
# =========================================================
st.markdown("""
<style>
    /* 1. Sóng ở nền */
    .wave-container {
        position: fixed; bottom: 0; left: 0; width: 100%; height: 150px;
        z-index: 0; pointer-events: none; opacity: 0.4;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%23FFCC00' d='M0,192L48,176C96,160,192,128,288,133.3C384,139,480,181,576,197.3C672,213,768,203,864,170.7C960,139,1056,85,1152,80C1248,75,1344,117,1392,138.7L1440,160L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
    }
    /* 2. Đẩy nội dung xuống để không bị khuất header */
    .main { padding-top: 0px !important; }
    .header-box { background: #007353; padding: 15px; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px; z-index: 10; position: relative; }
    /* 3. Đảm bảo các cột nằm trên lớp nền */
    div[data-testid="column"] { z-index: 1; background: rgba(255,255,255,0.8); border-radius: 15px; padding: 15px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="wave-container"></div>', unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="header-box"><h3>🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ (BIDV)</h3></div>', unsafe_allow_html=True)

# =========================================================
# LAYOUT 3 CỘT
# =========================================================
col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    st.subheader("📋 Thông tin cơ bản")
    age = st.slider("🎂 Tuổi", 20, 80, 35)
    credit_sco = st.slider("💳 Điểm tín dụng", 495, 800, 650)
    balance = st.number_input("💰 Số dư (VND)", min_value=0, value=50000000, step=1000000)
    active_text = st.radio("📱 Hoạt động gần đây", ["Có", "Không"])

with col2:
    st.subheader("📊 Thông tin tài chính")
    monthly_ir = st.number_input("💵 Thu nhập (VND)", min_value=0, value=15000000, step=1000000)
    nums_service = st.slider("🏦 Số lượng dịch vụ", 1, 8, 3)
    engagement_score = st.slider("🤝 Điểm tương tác app", 0, 100, 50)
    st.write("")
    predict_btn = st.button("🚀 DỰ ĐOÁN NGAY", use_container_width=True)

with col3:
    st.subheader("🎯 Kết quả dự báo")
    if predict_btn and model:
        active_member = 1 if active_text == "Có" else 0
        input_df = pd.DataFrame([{'monthly_ir': monthly_ir, 'credit_sco': credit_sco, 'nums_service': nums_service, 
                                  'engagement_score': engagement_score, 'balance': balance, 'age': age, 'active_member': active_member}])
        
        risk_score = model.predict_proba(input_df)[0][1]
        risk_percent = round(risk_score * 100, 2)
        
        color = "green" if risk_percent < 30 else ("orange" if risk_percent <= 70 else "red")
        
        st.metric("Mức độ rủi ro", f"{risk_percent}%")
        st.progress(int(risk_percent))
        st.markdown(f'<div style="border-left: 5px solid {color}; padding: 10px; background: #fff;"><b>Kết quả:</b> {'Rủi ro cao' if risk_percent > 70 else 'Cần theo dõi'}</div>', unsafe_allow_html=True)
    elif not model:
        st.error("Model chưa được load!")
    else:
        st.info("Nhấn 'DỰ ĐOÁN NGAY' để xem kết quả.")
