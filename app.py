import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("bidv_churnn_model.pkl")
scaler = joblib.load("scaler_bidv_model.pkl")

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
    padding-bottom: 2rem;
}

h1 {
    color: #005BAC;
    text-align: center;
    font-weight: bold;
}

.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 12px;
    border: none;
    background-color: #005BAC;
    color: white;
    font-size: 22px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #003f7d;
    color: white;
}

.metric-card {
    background: white;
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
# INPUT FORM
# =========================

col1, col2 = st.columns(2)

# LEFT COLUMN
with col1:

    age = st.slider(
        "🎂 Tuổi",
        18,
        90,
        35
    )

    credit_score = st.slider(
        "💳 Điểm tín dụng",
        300,
        900,
        650
    )

    balance = st.number_input(
        "💰 Số dư tài khoản (VND)",
        min_value=0,
        value=50000000,
        step=1000000
    )

    monthly_income = st.number_input(
        "📈 Thu nhập hàng tháng (VND)",
        min_value=0,
        value=15000000,
        step=1000000
    )

# RIGHT COLUMN
with col2:

    tenure = st.slider(
        "⏳ Số năm gắn bó",
        0,
        4,
        2
    )

    active_member = st.radio(
        "📱 Hoạt động gần đây",
        ["Có", "Không"]
    )

    loyalty_level = st.selectbox(
        "🏅 Hạng khách hàng",
        ["Bronze", "Silver", "Gold"]
    )

st.markdown("")

# =========================
# BUTTON
# =========================

predict_btn = st.button("🔍 DỰ ĐOÁN NGAY")

# =========================
# PREDICT
# =========================

if predict_btn:

    # =========================
    # ALL FEATURES
    # =========================

    all_columns = [
        'credit_sco',
        'gender',
        'age',
        'balance',
        'monthly_ir',
        'tenure_ye',
        'married',
        'nums_card',
        'nums_service',
        'active_member',
        'last_transaction_month',
        'customer_segment',
        'engagement_score',
        'loyalty_level',
        'risk_score',
        'risk_segment',
        'cluster_group',
        'occupation_Giáo viên/Giảng viên',
        'occupation_Hưu trí',
        'occupation_Kinh doanh/Bán hàng',
        'occupation_Kế toán/Tài chính',
        'occupation_Kỹ sư/Chuyên viên IT',
        'occupation_Lao động phổ thông',
        'occupation_Nhân viên văn phòng/Công chức',
        'occupation_Nội trợ/Sinh viên',
        'occupation_Quản lý/Lãnh đạo',
        'origin_province_Bình Dương',
        'origin_province_Cần Thơ',
        'origin_province_Hà Nội',
        'origin_province_Long An',
        'origin_province_TP. Hồ Chí Minh',
        'origin_province_Tiền Giang',
        'origin_province_Tỉnh khác',
        'origin_province_Đồng Nai',
        'digital_behavior_offline'
    ]

    # =========================
    # CREATE DATAFRAME
    # =========================

    input_data = pd.DataFrame(
        np.zeros((1, len(all_columns))),
        columns=all_columns
    )

    # =========================
    # USER INPUT
    # =========================

    input_data['credit_sco'] = credit_score
    input_data['age'] = age
    input_data['balance'] = balance
    input_data['monthly_ir'] = monthly_income
    input_data['tenure_ye'] = tenure

    input_data['active_member'] = 1 if active_member == "Có" else 0

    # =========================
    # DEFAULT VALUES
    # =========================

    input_data['gender'] = 1
    input_data['married'] = 1
    input_data['nums_card'] = 2
    input_data['nums_service'] = 2
    input_data['last_transaction_month'] = 1000000
    input_data['customer_segment'] = 1
    input_data['engagement_score'] = 50
    input_data['risk_score'] = 0.2
    input_data['risk_segment'] = 1
    input_data['cluster_group'] = 2

    # =========================
    # LOYALTY LEVEL
    # =========================

    if loyalty_level == "Bronze":
        input_data['loyalty_level'] = 0

    elif loyalty_level == "Silver":
        input_data['loyalty_level'] = 1

    else:
        input_data['loyalty_level'] = 2

    # =========================
    # DEFAULT OCCUPATION
    # =========================

    input_data['occupation_Nhân viên văn phòng/Công chức'] = 1

    # =========================
    # DEFAULT PROVINCE
    # =========================

    input_data['origin_province_Hà Nội'] = 1

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

    # =========================
    # OUTPUT
    # =========================

    st.markdown("---")

    # RISK SCORE
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

    if risk_percent >= 50:

        st.error(
            "⚠️ Khách hàng có nguy cơ rời bỏ"
        )

    else:

        st.success(
            "✅ Khách hàng có khả năng tiếp tục sử dụng dịch vụ"
        )

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

    # =========================
    # FEATURE IMPORTANCE
    # =========================

    st.markdown("---")

    st.subheader("📊 Các yếu tố ảnh hưởng")

    importance_data = pd.DataFrame({
        'Yếu tố': [
            'Điểm tín dụng',
            'Số dư tài khoản',
            'Thu nhập',
            'Mức độ hoạt động',
            'Số năm gắn bó'
        ],
        'Giá trị': [
            credit_score,
            balance,
            monthly_income,
            active_member,
            tenure
        ]
    })

    st.dataframe(
        importance_data,
        use_container_width=True
    )
