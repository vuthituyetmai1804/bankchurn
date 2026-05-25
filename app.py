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
# CSS TỐI ƯU GIAO DIỆN
# =========================================================
st.markdown("""
<style>
    /* Cố định Sóng ở dưới cùng, trải dài toàn màn hình */
    .wave-container {
        position: fixed;
        bottom: 0; left: 0;
        width: 100%; height: 150px;
        z-index: 0; pointer-events: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%23FFCC00' fill-opacity='0.4' d='M0,192L48,176C96,160,192,128,288,133.3C384,139,480,181,576,197.3C672,213,768,203,864,170.7C960,139,1056,85,1152,80C1248,75,1344,117,1392,138.7L1440,160L1440,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
        background-repeat: repeat-x;
        background-position: bottom;
    }
    
    /* Header nhỏ gọn */
    .header-box { 
        background: #007353; padding: 15px; border-radius: 10px; 
        color: white; text-align: center; margin-bottom: 20px; 
    }
    
    /* Định dạng cột để chứa nội dung trên nền sóng */
    div[data-testid="column"] { 
        background: rgba(255, 255, 255, 0.9); 
        padding: 20px !important; 
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Hiển thị sóng
st.markdown('<div class="wave-container"></div>', unsafe_allow_html=True)

# =========================================================
# UI CHÍNH
# =========================================================
st.markdown('<div class="header-box"><h3>🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ (BIDV)</h3></div>', unsafe_allow_html=True)

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
    st.write("---")
    predict_btn = st.button("🚀 DỰ ĐOÁN NGAY", use_container_width=True)

with col3:
    st.subheader("🎯 Kết quả dự báo")
    if predict_btn and model:
        active_member = 1 if active_text == "Có" else 0
        input_data = pd.DataFrame([{'monthly_ir': monthly_ir, 'credit_sco': credit_sco, 'nums_service': nums_service, 
                                    'engagement_score': engagement_score, 'balance': balance, 'age': age, 'active_member': active_member}])
        
        risk_score = model.predict_proba(input_data)[0][1]
        risk_percent = round(risk_score * 100, 2)
        
        color = "green" if risk_percent < 30 else ("orange" if risk_percent <= 70 else "red")
        
        st.metric("Tỷ lệ rủi ro", f"{risk_percent}%")
        st.progress(int(risk_percent))
        st.markdown(f"**Trạng thái:** <span style='color:{color}; font-weight:bold;'>{'Rủi ro CAO' if risk_percent > 70 else ('Trung bình' if risk_percent > 30 else 'Thấp')}</span>", unsafe_allow_html=True)
    elif predict_btn and not model:
        st.error("Lỗi: Không tìm thấy model!")
    else:
        st.info("Nhập thông tin và nhấn nút DỰ ĐOÁN để xem kết quả.")
