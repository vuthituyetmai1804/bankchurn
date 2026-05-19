import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Tải mô hình và bộ chuẩn hóa (Đảm bảo file nằm cùng thư mục trên GitHub)
try:
    model = joblib.load('bank_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("Không tìm thấy file mô hình hoặc scaler. Hãy kiểm tra lại thư mục lưu trữ.")

st.set_page_config(page_title="BIDV Churn Prediction", layout="wide")
st.title("🏦 Hệ thống Dự báo Nguy cơ Rời bỏ Khách hàng - BIDV")
st.markdown("---")
with st.form("churn_form"):

    st.subheader("📋 Thông tin khách hàng")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Tuổi", 18, 80, 35)

        credit_sco = st.slider("Điểm tín dụng", 300, 850, 650)

        balance = st.number_input(
            "Số dư tài khoản",
            min_value=0,
            value=50000000
        )

        monthly_ir = st.number_input(
            "Thu nhập hàng tháng",
            min_value=0,
            value=15000000
        )

    with col2:

        tenure_ye = st.slider(
            "Số năm gắn bó",
            0, 20, 3
        )

        active_member = st.selectbox(
            "Hoạt động gần đây",
            [0,1],
            format_func=lambda x:
            "Có" if x == 1 else "Không"
        )

        engagement_score = st.slider(
            "Điểm tương tác App",
            0, 100, 60
        )

        loyalty_level = st.selectbox(
            "Hạng khách hàng",
            ["Bronze","Silver","Gold","Platinum"]
        )

    submitted = st.form_submit_button("📊 DỰ ĐOÁN")

# This block must be indented under the 'if submitted:' statement
if submitted:
    # Define all 21 feature names that the model expects
    # IMPORTANT: The current form only collects 8 features. You need to add more inputs
    # or handle the missing features (e.g., with default/imputed values) in the correct order.
    feature_names = [
        'gender', 'age', 'occupation', 'origin_province', 'address', 'monthly_ir',
        'balance', 'credit_sco', 'tenure_ye', 'married', 'nums_card', 'nums_service',
        'last_transaction_month', 'active_member', 'customer_segment', 'engagement_score',
        'loyalty_level', 'digital_behavior', 'risk_score', 'risk_segment', 'cluster_group'
    ]

    # Placeholder for missing features. In a real app, you'd collect these or impute them.
    # For demonstration, filling with zeros or default values. Ensure these match the data types used during training.
    # The order MUST match 'feature_names'.
    # Based on the previous notebook context, some encoded values were ints.
    gender_val = 0 # Example: Female (assuming 0 for female, 1 for male from previous LabelEncoder)
    occupation_val = 0 # Example: First occupation category
    origin_province_val = 0 # Example: First province category
    address_val = 0 # Example: First address category
    married_val = 1 # Example: Kết hôn (assuming 1 from previous LabelEncoder)
    nums_card_val = 1 # Example: Default 1 card
    nums_service_val = 2 # Example: Default 2 services
    last_transaction_month_val = 0 # Example: No recent transaction
    customer_segment_val = 0 # Example: Mass
    digital_behavior_val = 2 # Example: Mobile
    risk_score_val = 0.5 # Example: Default risk score
    risk_segment_val = 0 # Example: Thấp
    cluster_group_val = 1 # Example: First cluster

    input_values = [
        gender_val, age, occupation_val, origin_province_val, address_val, monthly_ir,
        balance, credit_sco, tenure_ye, married_val, nums_card_val, nums_service_val,
        last_transaction_month_val, active_member, customer_segment_val, engagement_score,
        loyalty_level, digital_behavior_val, risk_score_val, risk_segment_val, cluster_group_val
    ]
    
    # Map loyalty_level to numerical value (0-3)
    loyalty_level_map = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
    input_values[feature_names.index('loyalty_level')] = loyalty_level_map.get(loyalty_level, 0)

    input_df = pd.DataFrame([input_values], columns=feature_names)

    st.subheader("📈 Kết quả phân tích")

    # Ensure scaler and model are loaded (handled in RcRNhj-4JEV_ cell)
    if 'scaler' in globals() and 'model' in globals():
        try:
            input_scaled = scaler.transform(input_df)

            risk_proba = model.predict_proba(input_scaled)[0][1] # Get probability of positive class

            prediction = model.predict(input_scaled)[0]

            st.metric(
                "Risk Score",
                f"{risk_proba:.2%}"
            )

            if risk_proba < 0.3:
                st.success("✅ LOW RISK")
            elif risk_proba < 0.7:
                st.warning("⚠️ MEDIUM RISK")
            else:
                st.error("🚨 HIGH RISK")

            if risk_proba > 0.7:
                st.info("Khuyến nghị: RM cần liên hệ chăm sóc khách hàng ngay.")
            else:
                st.info("Khuyến nghị: Tiếp tục duy trì chương trình ưu đãi.")
        except ValueError as e:
            st.error(f"Lỗi: Dữ liệu đầu vào không khớp với các đặc trưng mong đợi của mô hình. Chi tiết: {e}")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi không mong muốn trong quá trình dự đoán: {e}")
    else:
        st.error("Lỗi: Mô hình hoặc bộ chuẩn hóa chưa được tải. Vui lòng kiểm tra lại.")
