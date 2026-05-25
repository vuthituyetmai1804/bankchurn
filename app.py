import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load('/content/drive/MyDrive/bidv_churnn_model.pkl')
scaler = joblib.load('/content/drive/MyDrive/scaler_bidv_model.pkl')

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    color: #005BAC;
    text-align: center;
    font-weight: bold;
}

.stButton>button {
    width: 100%;
    height: 60px;
    font-size: 22px;
    font-weight: bold;
    border-radius: 12px;
    background-color: #005BAC;
    color: white;
}

.stButton>button:hover {
    background-color: #004080;
    color: white;
}

.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

st.title("🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ")

st.markdown("---")

# =========================
# INPUT AREA
# =========================

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Tuổi",
        min_value=18,
        max_value=90,
        value=35
    )

    credit_score = st.slider(
        "Điểm tín dụng",
        min_value=300,
        max_value=900,
        value=650
    )

    balance = st.number_input(
        "Số dư tài khoản (VND)",
        min_value=0,
        value=50000000,
        step=1000000
    )

    monthly_income = st.number_input(
        "Thu nhập hàng tháng (VND)",
        min_value=0,
        value=15000000,
        step=1000000
    )

with col2:

    tenure = st.slider(
        "Số năm gắn bó",
        min_value=0,
        max_value=4,
        value=2
    )

    active_member = st.radio(
        "Hoạt động gần đây",
        ["Có", "Không"]
    )

    loyalty_level = st.selectbox(
        "Hạng khách hàng",
        ["Bronze", "Silver", "Gold"]
    )

# =========================
# BUTTON
# =========================

predict_button = st.button("🔍 DỰ ĐOÁN NGAY")

# =========================
# PREDICTION
# =========================

if predict_button:

    # =========================
    # ENCODE INPUT
    # =========================

    active_member_value = 1 if active_member == "Có" else 0

    loyalty_bronze = 1 if loyalty_level == "Bronze" else 0
    loyalty_silver = 1 if loyalty_level == "Silver" else 0

    # =========================
    # TẠO DATAFRAME
    # =========================

    input_data = pd.DataFrame({

        'credit_sco': [credit_score],
        'age': [age],
        'balance': [balance],
        'monthly_ir': [monthly_income],
        'tenure_ye': [tenure],
        'active_member': [active_member_value],

        # GIÁ TRỊ GIẢ ĐỊNH
        'married': [1],
        'nums_card': [2],
        'nums_service': [2],
        'engagement_score': [50],
        'risk_score': [0.2],
        'cluster_group': [2],

        'customer_segment': [1],
        'loyalty_level': [1],

        'risk_segment': [1],

        'gender': [1],

        'digital_behavior_offline': [0],

        # OCCUPATION
        'occupation_Giáo viên/Giảng viên': [0],
        'occupation_Hưu trí': [0],
        'occupation_Kinh doanh/Bán hàng': [0],
        'occupation_Kế toán/Tài chính': [0],
        'occupation_Kỹ sư/Chuyên viên IT': [0],
        'occupation_Lao động phổ thông': [0],
        'occupation_Nhân viên văn phòng/Công chức': [1],
        'occupation_Nội trợ/Sinh viên': [0],
        'occupation_Quản lý/Lãnh đạo': [0],

        # PROVINCE
        'origin_province_Bình Dương': [0],
        'origin_province_Cần Thơ': [0],
        'origin_province_Hà Nội': [1],
        'origin_province_Long An': [0],
        'origin_province_TP. Hồ Chí Minh': [0],
        'origin_province_Tiền Giang': [0],
        'origin_province_Tỉnh khác': [0],
        'origin_province_Đồng Nai': [0],

        'last_transaction_month': [1000000]

    })

    # =========================
    # SCALE
    # =========================

    cols_to_scale = [
        'credit_sco',
        'age',
        'balance',
        'monthly_ir',
        'nums_card',
        'nums_service',
        'engagement_score',
        'tenure_ye',
        'risk_score'
    ]

    input_data[cols_to_scale] = scaler.transform(
        input_data[cols_to_scale]
    )

    # =========================
    # PREDICT
    # =========================

    probability = model.predict_proba(input_data)[0][1]

    risk_percent = round(probability * 100, 2)

    prediction = 1 if probability >= 0.5 else 0

    st.markdown("---")

    # =========================
    # RISK SCORE
    # =========================

    st.metric(
        label="🎯 RISK SCORE",
        value=f"{risk_percent}%"
    )

    # =========================
    # RISK LEVEL
    # =========================

    if risk_percent < 30:
        st.success("🟢 LOW RISK")

    elif risk_percent < 70:
        st.warning("🟡 MEDIUM RISK")

    else:
        st.error("🔴 HIGH RISK")

    # =========================
    # PREDICTION
    # =========================

    if prediction == 1:
        st.error("⚠️ Khách hàng có nguy cơ rời bỏ")

    else:
        st.success("✅ Khách hàng có khả năng tiếp tục sử dụng dịch vụ")

    # =========================
    # RECOMMENDATION
    # =========================

    st.subheader("📌 Recommendation")

    if risk_percent >= 70:

        st.error(
            "🚨 Cần liên hệ khẩn cấp trong 24h để giữ chân khách hàng."
        )

    elif risk_percent >= 40:

        st.warning(
            "📞 Nên chăm sóc chủ động: Gọi điện tư vấn, ưu đãi lãi suất, voucher."
        )

    else:

        st.success(
            "✅ Duy trì mối quan hệ tốt và tiếp tục chăm sóc định kỳ."
        )
