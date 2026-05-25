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
    /* 1. Đổi font chữ toàn trang và nền */
    @import url('https://googleapis.com');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8F9FA;
    }

    /* 2. Làm mới Thanh Banner Đầu Trang (Header) */
    .stMarkdown div[data-testid="stMarkdownContainer"] h1 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* 3. Thiết kế lại các Khối Nhập Liệu (Cards) */
    [data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] {
        background: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid #EAEAEA;
        margin-bottom: 20px;
    }

    /* 4. Thay đổi màu Đỏ của Slider sang Xanh Ngọc Lục Bảo sang trọng */
    div[data-baseweb="slider"] [role="slider"] {
        background-color: #006A4E !important; /* Xanh BIDV */
        border: 2px solid #ffffff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    div[data-baseweb="slider"] [aria-valuenow] {
        background-color: #006A4E !important;
    }
    /* Thanh ray của slider */
    div[data-baseweb="slider"] > div > div {
        background: linear-gradient(to right, #006A4E 0%, #006A4E var(--progress), #EAEAEA var(--progress), #EAEAEA 100%) !important;
    }

    /* 5. Nâng cấp các ô Nhập Số (Input Number) */
    div[data-baseweb="input"] {
        border-radius: 10px !important;
        border: 1px solid #DCE1E6 !important;
        background-color: #FAFAFA !important;
        transition: all 0.3s ease;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #006A4E !important;
        box-shadow: 0 0 0 3px rgba(0, 106, 78, 0.15) !important;
    }

    /* 6. Nút Dự Đoán Đẳng Cấp */
    div.stButton > button {
        background: linear-gradient(135deg, #006A4E 0%, #004D38 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        font-weight: 600 !important;
        border: none !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(0, 106, 78, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 106, 78, 0.4) !important;
    }

    /* 7. Tinh chỉnh khu vực Radio Button */
    div[data-testid="stRadio"] label {
        font-size: 15px;
        color: #4A4A4A;
    }
</style>
""", unsafe_allow_html=True)
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
