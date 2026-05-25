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
except:
    model = None

# =========================================================

# CSS TỐI ƯU
# =========================================================
st.markdown("""
<style>
    
    /* 1. Header to và đẹp hơn */
    .header-box { 
        background: #007353; padding: 25px; border-radius: 20px; 
        color: white; text-align: center; margin-bottom: 30px; 
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    .header-title { font-size: 32px; font-weight: 800; margin: 0; }
    
    /* 2. Container nội dung */
    div[data-testid="column"] { 
        background: rgba(255, 255, 255, 0.95); 
        padding: 25px !important; border-radius: 20px; 
        border: 1px solid #ddd; z-index: 1; position: relative;
    }
    /* 3. Lớp sóng uốn lượn cố định (Wave Container) - Màu Vàng hoa mai #FFCC00 */
    .wave-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 200px;
        z-index: 0;
        pointer-events: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%23FFCC00' fill-opacity='0.4' d='M0,192L48,176C96,160,192,128,288,133.3C384,139,480,181,576,197.3C672,213,768,203,864,170.7C960,139,1056,85,1152,80C1248,75,1344,117,1392,138.7L1440,160L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
        background-size: cover;
        background-repeat: no-repeat;
    }
    /* 5. Nút bấm bo tròn - Màu Xanh ngọc lục bảo #007353 */
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

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="wave-container"></div>', unsafe_allow_html=True)

# =========================================================
# UI HEADER
# =========================================================
# Tăng tỷ lệ cột bên trái và phải (ví dụ: 1:1:1)
col_logo_l, col_logo_m, col_logo_r = st.columns([1, 1, 1]) 

with col_logo_m:
    # Tăng width để logo to hơn như bạn muốn
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/f4/Logo_Bidv_m%E1%BB%9Bi.svg", width=500)
st.markdown("""  
<div style="text-align: center; margin-bottom: 30px;">
    <h2 style="color: #007353; font-weight: 800; margin-bottom: 5px;">🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ</h2>
    <p style="font-size: 17px; color: #555; font-style: italic; font-weight: 300;">
        Giải pháp phân tích hành vi khách hàng, hỗ trợ chiến lược giữ chân chủ động tại BIDV
    </p>
</div>
""", unsafe_allow_html=True)

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
    predict_btn = st.button("🔍 DỰ ĐOÁN NGAY", use_container_width=True)
with col3:
    st.markdown("##### 🎯 Kết quả dự báo")
    if predict_btn and model:
        # Dự đoán
        input_data = pd.DataFrame([{'monthly_ir': monthly_ir, 'credit_sco': credit_sco, 'nums_service': nums_service, 
                                    'engagement_score': engagement_score, 'balance': balance, 'age': age, 
                                    'active_member': 1 if active_text == "Có" else 0}])
        
        risk_score = model.predict_proba(input_data)[0][1]
        risk_percent = round(risk_score * 100, 2)
        
        # LOGIC ĐẦY ĐỦ CỦA BẠN
        if risk_percent < 30:
            risk_level = "🟢 LOW RISK"
            prediction_text = "✅ Khách hàng có khả năng tiếp tục sử dụng dịch vụ"
            recommendation = "✅ Duy trì mối quan hệ tốt và tiếp tục chăm sóc định kỳ."
            color = "green"
        elif risk_percent <= 70:
            risk_level = "🟡 MEDIUM RISK"
            prediction_text = "⚠️ Khách hàng có nguy cơ rời bỏ"
            recommendation = "📞 Nên chăm sóc chủ động: Gọi điện tư vấn, tặng ưu đãi lãi suất, voucher."
            color = "orange"
        else:
            risk_level = "🔴 HIGH RISK"
            prediction_text = "⚠️ Khách hàng có nguy cơ rời bỏ"
            recommendation = "🚨 Cần liên hệ khẩn cấp trong 24h để giữ chân khách hàng."
            color = "red"

        # 1. METRICS (SCORE & LEVEL)
        c1, c2 = st.columns(2)
        c1.metric("SCORE", f"{risk_percent}%")
        # Dùng markdown để chữ Level nhỏ gọn, không bị vỡ
        c2.markdown(f"**LEVEL**<br><span style='color:{color}; font-size:16px; font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
        
        # 2. PROGRESS BAR
        st.progress(int(risk_percent))
        
        # 3. PREDICTION TEXT BOX
        st.markdown(f"""
        <div style="background-color:#f9f9f9; padding:10px; border-radius:10px; border:1px solid #ddd; margin-top:10px;">
            <p style="color:{color}; font-weight:bold; margin:0;">{prediction_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 4. RECOMMENDATION BOX
        st.markdown(f"""
        <div style="background-color:#ffffff; padding:10px; border-radius:10px; border-left:5px solid {color}; margin-top:10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.05);">
            <p style="font-weight:bold; margin-bottom:5px;">🎯 Khuyến nghị:</p>
            <p style="font-size:14px; margin:0;">{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
    elif predict_btn and not model:
        st.error("Model chưa được tải!")
    else:
        st.info("Nhập thông tin và nhấn nút 🔍 DỰ ĐOÁN NGAY để xem kết quả.")
