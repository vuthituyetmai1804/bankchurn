import streamlit as st
import pandas as pd
import joblib

# =========================================================
# CONFIG & LOAD MODEL
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
# CSS TỐI ƯU KHÔNG GIAN
# =========================================================
st.markdown("""
<style>
    /* Giảm khoảng cách mặc định của Streamlit */
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
    
    /* Header thu nhỏ */
    .header-box { background: #007353; padding: 15px; border-radius: 15px; color: white; margin-bottom: 20px; text-align: center; }
    .header-title { font-size: 24px; font-weight: bold; }
    
    /* Căn chỉnh các thành phần input gọn hơn */
    .stSlider, .stNumberInput, .stRadio { margin-bottom: -5px !important; }
    
    /* Khung kết quả trong cột 3 */
    .result-box { background-color: #f9f9f9; padding: 15px; border-radius: 15px; border: 1px solid #ddd; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="header-box">
    <div class="header-title">🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ (BIDV)</div>
</div>
""", unsafe_allow_html=True)

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
    
    st.write("") # Tạo khoảng cách
    predict_btn = st.button("🚀 DỰ ĐOÁN NGAY", use_container_width=True)

# =========================================================
# CỘT 3: KẾT QUẢ
# =========================================================
with col3:
    st.subheader("🎯 Kết quả dự báo")
    
    if predict_btn:
        active_member = 1 if active_text == "Có" else 0
        features_order = ['monthly_ir', 'credit_sco', 'nums_service', 'engagement_score', 'balance', 'age', 'active_member']
        
        input_df = pd.DataFrame([{
            'monthly_ir': monthly_ir, 'credit_sco': credit_sco, 'nums_service': nums_service, 
            'engagement_score': engagement_score, 'balance': balance, 'age': age, 'active_member': active_member
        }])
        
        # Dự đoán
        risk_score = model.predict_proba(input_df[features_order])[0][1]
        risk_percent = round(risk_score * 100, 2)

        # Logic hiển thị
        if risk_percent < 30:
            color, level = "green", "🟢 THẤP"
            recom = "Khách hàng hài lòng, tiếp tục duy trì chăm sóc."
        elif risk_percent <= 70:
            color, level = "orange", "🟡 TRUNG BÌNH"
            recom = "Cần gửi ưu đãi lãi suất hoặc voucher để giữ chân."
        else:
            color, level = "red", "🔴 CAO"
            recom = "Cảnh báo khẩn cấp: Liên hệ CSKH ngay trong 24h."

        # Hiển thị kết quả
        st.metric(label="Mức độ rủi ro", value=f"{risk_percent}%")
        st.progress(int(risk_percent))
        
        st.markdown(f"""
        <div class="result-box">
            <h4 style="color:{color}; margin:0;">Phân loại: {level}</h4>
            <p style="margin-top:10px;"><b>Khuyến nghị:</b> {recom}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Nhập thông tin và nhấn 'DỰ ĐOÁN NGAY' để xem kết quả tại đây.")

# Chân trang (tùy chọn)
st.markdown("---")
st.caption("Ứng dụng hỗ trợ ra quyết định kinh doanh - BIDV Banking © 2026")
