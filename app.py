import streamlit as st
import pandas as pd
import joblib

# Sử dụng decorator để cache model
@st.cache_resource
def load_model():
    # Đảm bảo file .pkl nằm cùng thư mục với app.py
    return joblib.load("bidv_churn_modeltuning.pkl")

# Gọi hàm load model
try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Không thể tải model: {e}")
    st.stop()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"
)
# =========================================================
# CUSTOM CSS 
st.markdown("""
<style>
/* 1. Thiết lập chung cho nền trang */
.stApp { background-color: #f4f6f9; }

/* 2. Lớp sóng uốn lượn (Wave Container) */
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

/* 3. Container nội dung chính */
.block-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 30px;
    padding: 3rem !important;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
    z-index: 1;
    position: relative;
}

/* 4. Header chuyên nghiệp - Kích thước lớn */
.header-box {
    background: #007353;
    padding: 50px 40px;
    border-radius: 30px;
    text-align: center;
    margin-bottom: 30px;
    color: white;
    box-shadow: 0px 10px 20px rgba(0, 115, 83, 0.3);
}
.header-title { 
    font-size: 55px !important; 
    font-weight: 900 !important; 
    margin-bottom: 15px !important; 
    color: white !important;
    text-transform: uppercase;
    line-height: 1.1 !important;
}
.header-sub { 
    font-size: 22px !important; 
    color: rgba(255,255,255,0.95) !important;
    font-weight: 400 !important;
}

/* 5. Cấu hình màu sắc Xám đậm cho Slider */
.stSlider [data-baseweb="slider"] [data-testid="stThumb"] {
    background-color: #4f4f4f !important;
    border: 2px solid #4f4f4f !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #4f4f4f !important;
}
.stSlider [data-baseweb="slider"] > div > div > div > div {
    background: #4f4f4f !important;
}

/* 6. Làm đẹp ô nhập liệu (Number Input) */
.stNumberInput input {
    border: 2px solid #e1e1e1 !important;
    border-radius: 12px !important;
    padding: 12px 15px !important;
    background-color: #ffffff !important;
    transition: all 0.3s ease;
}
.stNumberInput input:focus {
    border-color: #007353 !important;
    box-shadow: 0 0 5px rgba(0, 115, 83, 0.2);
}

/* 7. Nút bấm bo tròn */
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

/* 8. Khung kết quả */
.result-box, .recommend-box {
    background-color: white;
    padding: 25px;
    border-radius: 25px !important;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    border: 1px solid #e1e1e1;
    margin-bottom: 20px;
}
h2 { color: #007353 !important; }

/* 9. Metric */
[data-testid="stMetricValue"] { color: #007353; font-size: 40px !important; }
</style>

st.markdown('<div class="wave-container"></div>', unsafe_allow_html=True) 
# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="header-box">
    <div class="header-title">🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ</div>
    <div class="header-sub">Ứng dụng Mô hình Cây quyết định trong Quản trị Rủi ro Ngân hàng BIDV</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("## 📋 Nhập thông tin khách hàng")

col1, col2 = st.columns(2)

# =========================================================
# LEFT COLUMN
# =========================================================
with col1:
    age = st.slider(
        "🎂 Tuổi",
        20, 80, 35
    )

    credit_sco = st.slider(
        "💳 Điểm tín dụng",
        495, 800, 650
    )

    balance = st.number_input(
        "💰 Số dư tài khoản (VND)",
        min_value=0,
        value=50000000,
        step=1000000
    )

# =========================================================
# RIGHT COLUMN
# =========================================================
with col2:
    monthly_ir = st.number_input(
        "💵 Thu nhập hàng tháng (VND)",
        min_value=0,
        value=15000000,
        step=1000000
    )

    nums_service = st.slider(
        "🏦 Số lượng dịch vụ sử dụng",
        1, 8, 3
    )

    engagement_score = st.slider(
        "🤝 Điểm tương tác app",
        0, 100, 50
    )

    active_text = st.radio(
        "📱 Hoạt động gần đây",
        ["Có", "Không"]
    )

# =========================================================
# ENCODE INPUT
# =========================================================
active_member = 1 if active_text == "Có" else 0

# =========================================================
# PREDICT BUTTON
# =========================================================
predict_btn = st.button("🔍 DỰ ĐOÁN NGAY")

# =========================================================
# PREDICTION LOGIC
# =========================================================
if predict_btn:

    # =====================================================
    # 1. TẠO DATAFRAME VỚI ĐÚNG 7 CỘT THEO ĐÚNG THỨ TỰ YÊU CẦU
    # =====================================================
    features_order = [
        'monthly_ir', 'credit_sco', 'nums_service', 
        'engagement_score', 'balance', 'age', 'active_member'
    ]
    
    input_df = pd.DataFrame([{
        'monthly_ir': monthly_ir,
        'credit_sco': credit_sco,
        'nums_service': nums_service,
        'engagement_score': engagement_score,
        'balance': balance,
        'age': age,
        'active_member': active_member
    }])
    
    # Đảm bảo thứ tự cột gửi vào mô hình chuẩn xác 100%
    final_input = input_df[features_order]

    # =====================================================
    # 2. DỰ ĐOÁN TRỰC TIẾP KHÔNG QUA SCALER
    # =====================================================
    risk_score = model.predict_proba(final_input)[0][1]
    risk_percent = round(risk_score * 100, 2)

    # =====================================================
    # RISK LEVEL
    # =====================================================
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

    # =====================================================
    # OUTPUT GRAPHICS
    # =====================================================
    st.markdown("---")
    st.markdown("# 📊 KẾT QUẢ PHÂN TÍCH")

    # =====================================================
    # METRICS
    # =====================================================
    colA, colB, colC = st.columns(3)

    with colA:
        st.metric(
            label="RISK SCORE",
            value=f"{risk_percent}%"
        )

    with colB:
        st.metric(
            label="RISK LEVEL",
            value=risk_level
        )

    with colC:
        st.metric(
            label="PREDICTION",
            value="CHURN" if risk_percent >= 50 else "STAY"
        )

    # =====================================================
    # PROGRESS BAR
    # =====================================================
    st.progress(int(risk_percent))

    # =====================================================
    # PREDICTION RESULT BOX
    # =====================================================
    st.markdown(f"""
    <div class="result-box">
        <h2 style="color:{color}; text-align: center; margin: 0;">
            {prediction_text}
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # RECOMMENDATION BOX
    # =====================================================
    st.markdown(f"""
    <div class="recommend-box"
    style="
        background-color:white;
        border-left:8px solid {color};
        margin-top:20px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.08);
    ">
    <h3 style="margin-top: 0;">🎯 Khuyến nghị hành động:</h3>
    <p style="margin-bottom: 0;">{recommendation}</p>
    </div>
    """, unsafe_allow_html=True)
