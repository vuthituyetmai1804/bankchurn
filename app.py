import streamlit as st
import pandas as pd
import joblib

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("bidv_churn_model.pkl")
scaler = joblib.load("scaler_bidv_model.pkl")

# =========================================================
# TITLE
# =========================================================

st.markdown("""
<h1 style='text-align: center; color: #0B5ED7;'>
🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; font-size:18px; color:gray;'>
Ứng dụng AI hỗ trợ phát hiện khách hàng có nguy cơ rời bỏ dịch vụ ngân hàng
</p>
""", unsafe_allow_html=True)

st.divider()

# =========================================================
# INPUT FORM
# =========================================================

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "🎂 Tuổi",
        18,
        80,
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
        "💵 Thu nhập hàng tháng (VND)",
        min_value=0,
        value=15000000,
        step=1000000
    )

with col2:

    tenure = st.slider(
        "📆 Số năm gắn bó",
        0,
        10,
        3
    )

    active_member_text = st.radio(
        "📱 Hoạt động gần đây",
        ["Có", "Không"]
    )

    loyalty = st.selectbox(
        "🏅 Hạng khách hàng",
        ["Bronze", "Silver", "Gold"]
    )

# =========================================================
# ENCODE INPUT
# =========================================================

active_member = 1 if active_member_text == "Có" else 0

loyalty_map = {
    "Bronze": 1,
    "Silver": 2,
    "Gold": 3
}

loyalty_level = loyalty_map[loyalty]

# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown("")

predict_btn = st.button(
    "🔍 DỰ ĐOÁN NGAY",
    use_container_width=True,
    type="primary"
)

# =========================================================
# PREDICTION
# =========================================================

if predict_btn:

    # Tạo dataframe đúng thứ tự feature
    input_data = pd.DataFrame({
        'credit_sco': [credit_score],
        'age': [age],
        'balance': [balance],
        'monthly_ir': [monthly_income],
        'tenure_ye': [tenure],
        'active_member': [active_member],
        'loyalty_level': [loyalty_level]
    })

    # Scale đúng các cột số
    cols_to_scale = [
        'credit_sco',
        'age',
        'balance',
        'monthly_ir',
        'tenure_ye'
    ]

    input_data[cols_to_scale] = scaler.transform(
        input_data[cols_to_scale]
    )

    # Predict probability
    probability = model.predict_proba(input_data)[0][1]

    risk_percent = round(probability * 100, 2)

    st.divider()

    # =====================================================
    # RISK SCORE
    # =====================================================

    st.subheader("📊 RISK SCORE")

    st.metric(
        label="Nguy cơ rời bỏ",
        value=f"{risk_percent}%"
    )

    # =====================================================
    # RISK LEVEL
    # =====================================================

    st.subheader("🚦 Risk Level")

    if risk_percent < 30:
        st.success("🟢 LOW RISK")

    elif risk_percent < 70:
        st.warning("🟡 MEDIUM RISK")

    else:
        st.error("🔴 HIGH RISK")

    # =====================================================
    # PREDICTION
    # =====================================================

    st.subheader("🤖 Prediction")

    if risk_percent >= 50:
        st.error(
            "⚠️ Khách hàng có nguy cơ rời bỏ dịch vụ."
        )
    else:
        st.success(
            "✅ Khách hàng có khả năng tiếp tục sử dụng dịch vụ."
        )

    # =====================================================
    # RECOMMENDATION
    # =====================================================

    st.subheader("💡 Recommendation")

    if risk_percent >= 70:

        st.error("""
🚨 Cần liên hệ khách hàng trong vòng 24h.

Đề xuất:
- Chăm sóc ưu tiên
- Tặng voucher
- Ưu đãi lãi suất
- RM gọi điện trực tiếp
""")

    elif risk_percent >= 40:

        st.warning("""
📞 Nên chăm sóc chủ động.

Đề xuất:
- Gọi điện tư vấn
- Gửi ưu đãi cá nhân hóa
- Khuyến khích sử dụng thêm dịch vụ
""")

    else:

        st.success("""
✅ Duy trì mối quan hệ tốt với khách hàng.

Đề xuất:
- Chăm sóc định kỳ
- Giới thiệu sản phẩm mới
- Tăng loyalty
""")

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.subheader("📌 Các yếu tố ảnh hưởng")

    importance_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Value": input_data.iloc[0].values
    })

    st.dataframe(
        importance_df,
        use_container_width=True
    )
