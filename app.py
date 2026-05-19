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
    # Define all 21 feature names that the model expects, IN THE EXACT ORDER OF TRAINING DATA
    # Order from X_resampled: 'credit_sco', 'gender', 'age', 'occupation', 'balance', 'monthly_ir',
    # 'address', 'origin_province', 'tenure_ye', 'married', 'nums_card', 'nums_service',
    # 'last_transaction_month', 'active_member', 'customer_segment', 'engagement_score',
    # 'loyalty_level', 'digital_behavior', 'risk_score', 'risk_segment', 'cluster_group'
    feature_names = [
        'credit_sco', 'gender', 'age', 'occupation', 'balance', 'monthly_ir',
        'address', 'origin_province', 'tenure_ye', 'married', 'nums_card', 'nums_service',
        'last_transaction_month', 'active_member', 'customer_segment', 'engagement_score',
        'loyalty_level', 'digital_behavior', 'risk_score', 'risk_segment', 'cluster_group'
    ]

    # Placeholder for missing features. Ensure these match the data types used during training.
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

    # Construct input_values in the EXACT order of feature_names
    input_values = [
        credit_sco,         # 1. credit_sco (user input)
        gender_val,         # 2. gender (placeholder)
        age,                # 3. age (user input)
        occupation_val,     # 4. occupation (placeholder)
        balance,            # 5. balance (user input)
        monthly_ir,         # 6. monthly_ir (user input)
        address_val,        # 7. address (placeholder)
        origin_province_val,# 8. origin_province (placeholder)
        tenure_ye,          # 9. tenure_ye (user input)
        married_val,        # 10. married (placeholder)
        nums_card_val,      # 11. nums_card (placeholder)
        nums_service_val,   # 12. nums_service (placeholder)
        last_transaction_month_val, # 13. last_transaction_month (placeholder)
        active_member,      # 14. active_member (user input)
        customer_segment_val, # 15. customer_segment (placeholder)
        engagement_score,   # 16. engagement_score (user input)
        # loyalty_level will be mapped below
        digital_behavior_val, # 18. digital_behavior (placeholder)
        risk_score_val,     # 19. risk_score (placeholder)
        risk_segment_val,   # 20. risk_segment (placeholder)
        cluster_group_val   # 21. cluster_group (placeholder)
    ]

    # Map loyalty_level to numerical value (0-3) and insert into the correct position
    loyalty_level_map = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
    loyalty_level_encoded = loyalty_level_map.get(loyalty_level, 0)
    # Insert at the correct index for 'loyalty_level' (index 16 in feature_names)
    input_values.insert(feature_names.index('loyalty_level'), loyalty_level_encoded)

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
