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
except Exception:
    model = None

# =========================================================
# CSS TỐI ƯU KÍCH THƯỚC CHỮ
# =========================================================
st.markdown("""
<style>
    .wave-container {
        position: fixed; bottom: 0; left: 0; width: 100%; height: 150px;
        z-index: 0; pointer-events: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%23FFCC00' fill-opacity='0.4' d='M0,192L48,176C96,160,192,128,288,133.3C384,139,480,181,576,197.3C672,213,768,203,864,170.7C960,139,1056,85,1152,80C1248,75,1344,117,1392,138.7L1440,160L1440,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
        background-repeat: repeat-x; background-position: bottom;
    }
    .header-box { background: #007353; padding: 15px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px; }
    div[data-testid="column"] { background: rgba(255, 255, 255, 0.9); padding: 20px !important; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    
    /* Điều chỉnh kích thước chữ trong kết quả */
    .result-text { font-size: 16px; font-weight: 600; text-align: center; padding: 10px; }
    .recommend-title { font-size: 15px; font-weight: bold; margin-bottom: 5px; }
    .recommend-body { font-size: 14px; color: #444; }
</style>
""", unsafe_allow_html=True)

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
        features_order = ['monthly_ir', 'credit_sco', 'nums_service', 'engagement_score', 'balance', 'age', 'active_member']
        input_df = pd.DataFrame([{'monthly_ir': monthly_ir, 'credit_sco': credit_sco, 'nums_service': nums_service, 
                                  'engagement_score': engagement_score, 'balance': balance, 'age': age, 'active_member': active_member}])
        
        risk_score = model.predict_proba(input_df[features_order])[0][1]
        risk_percent = round(risk_score * 100, 2)

        if risk_percent < 30:
            risk_level, prediction_text, recommendation, color = "🟢 LOW RISK", "✅ Khách hàng ổn định", "Duy trì mối quan hệ tốt và tiếp tục chăm sóc định kỳ.", "green"
        elif risk_percent <= 70:
            risk_level, prediction_text, recommendation, color = "🟡 MEDIUM RISK", "⚠️ Khách hàng có nguy cơ rời bỏ", "Nên chăm sóc chủ động: Gọi điện tư vấn, tặng ưu đãi lãi suất, voucher.", "orange"
        else:
            risk_level, prediction_text, recommendation, color = "🔴 HIGH RISK", "⚠️ Khách hàng có nguy cơ rời bỏ", "Cần liên hệ khẩn cấp trong 24h để giữ chân khách hàng.", "red"

        # Hiển thị metrics với font size chuẩn
        cA, cB = st.columns(2)
        cA.metric("RISK SCORE", f"{risk_percent}%")
        cB.metric("LEVEL", risk_level.split(" ")[1]) # Chỉ lấy chữ (LOW/MEDIUM/HIGH) để gọn
        
        st.progress(int(risk_percent))
        
        # Kết quả text tinh tế hơn
        st.markdown(f'<div class="result-text" style="color:{color};">{prediction_text}</div>', unsafe_allow_html=True)
        
        # Khuyến nghị tinh tế
        st.markdown(f"""<div style="border-left: 5px solid {color}; padding-left: 10px; background: #fff; padding: 10px;">
        <div class="recommend-title">🎯 Khuyến nghị hành động:</div>
        <div class="recommend-body">{recommendation}</div></div>""", unsafe_allow_html=True)
        
    elif predict_btn and not model:
        st.error("Lỗi: Không tìm thấy model!")
    else:
        st.info("Nhập thông tin và nhấn nút DỰ ĐOÁN để xem kết quả.")
