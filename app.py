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
# =========================================================
st.markdown("""
<style>
/* 1. Nền tổng thể: 40% Vàng ban mai (#FFC62F) */
.stApp {
    background-color: rgba(255, 198, 47, 0.4) !important;
}

/* 2. Lớp sóng sợi trắng uốn lượn */
.wave-container {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 300px;
    z-index: 0;
    pointer-events: none;
    opacity: 0.6;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='none' stroke='white' stroke-width='1.5' d='M0,160C240,320,480,0,720,160C960,320,1200,0,1440,160'/%3E%3Cpath fill='none' stroke='white' stroke-width='1' d='M0,200C240,360,480,40,720,200C960,360,1200,40,1440,200'/%3E%3C/svg%3E");
    background-repeat: repeat-x;
}

/* 3. Container nội dung */
.block-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 30px;
    padding: 3rem !important;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
    z-index: 1;
    position: relative;
}

/* 4. Header BIDV */
.header-box {
    background: #006B68;
    padding: 40px;
    border-radius: 30px;
    text-align: center;
    margin-bottom: 30px;
    color: white;
}
.header-title { font-size: 50px !important; font-weight: 900 !important; color: white !important; }
.header-sub { font-size: 18px !important; color: rgba(255,255,255,0.9) !important; }

/* 5. Nút bấm BIDV */
.stButton > button {
    width: 100%;
    height: 65px !important;
    background-color: #006B68 !important;
    color: white !important;
    font-size: 24px !important;
    font-weight: bold !important;
    border-radius: 50px !important;
    border: none;
}

/* 6. Cỡ chữ nhãn và input */
label { font-size: 18px !important; font-weight: 600 !important; }
h2 { font-size: 32px !important; color: #006B68 !important; }

/* 7. Định dạng chữ trong các ô kết quả */
.result-box, .recommend-box {
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

# Kích hoạt lớp sóng
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
