import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Cấu hình trang & Tải mô hình ---
st.set_page_config(page_title="BIDV Churn Prediction", layout="wide")

@st.cache_resource
def load_assets():
    # Đảm bảo 2 file này đã được upload lên GitHub cùng thư mục với app.py
    model = joblib.load('bank_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Lỗi tải mô hình: {e}. Vui lòng kiểm tra file .pkl trên GitHub.")

# --- 2. Giao diện tiêu đề ---
st.title("🏦 BIDV - Hệ thống Dự báo Nguy cơ Rời bỏ Khách hàng")
st.markdown("Nhập thông tin khách hàng để AI phân tích xác suất rời bỏ (Churn) và đưa ra giải pháp chăm sóc chủ động.")

# --- 3. Form nhập liệu (Đủ 21 đặc trưng theo Sprint 4) ---
with st.form("customer_input_form"):
    st.subheader("📋 Thông tin chi tiết khách hàng")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Giới tính", options=[0, 1], format_func=lambda x: "Nữ" if x == 0 else "Nam")
        age = st.number_input("Tuổi", 18, 100, 35)
        occupation = st.number_input("Nghề nghiệp (Mã hóa số)", 0, 10, 1)
        origin_province = st.number_input("Tỉnh thành (Mã hóa số)", 0, 63, 0)
        address = st.number_input("Quận/Huyện (Mã hóa số)", 0, 20, 0)
        monthly_ir = st.number_input("Thu nhập hàng tháng (VND)", min_value=0, value=20000000)
        balance = st.number_input("Số dư hiện tại (VND)", min_value=0, value=100000000)

    with col2:
        credit_sco = st.slider("Điểm tín dụng (300-800)", 300, 800, 650)
        tenure_ye = st.number_input("Số năm gắn bó", 0, 50, 5)
        married = st.selectbox("Tình trạng hôn nhân", options=[0, 1, 2, 3],
                              format_func=lambda x: ["Độc thân", "Kết hôn", "Ly hôn", "Góa"][x])
        nums_card = st.number_input("Số thẻ sở hữu", 0, 10, 1)
        nums_service = st.number_input("Số dịch vụ đang dùng", 0, 20, 3)
        last_transaction_month = st.number_input("Giao dịch tháng cuối (VND)", 0, value=5000000)
        active_member = st.selectbox("Hội viên hoạt động", options=[0, 1], format_func=lambda x: "Không" if x==0 else "Có")

    with col3:
        customer_segment = st.selectbox("Phân khúc", options=[0, 1, 2, 3],
                                      format_func=lambda x: ["Mass", "Emerging", "Affluent", "Priority"][x])
        engagement_score = st.slider("Điểm tương tác app (0-100)", 0, 100, 75)
        loyalty_level = st.selectbox("Hạng thành viên", options=[0, 1, 2, 3],
                                   format_func=lambda x: ["Bronze", "Silver", "Gold", "Platinum"][x])
        digital_behavior = st.selectbox("Hành vi số", options=[0, 1, 2],
                                      format_func=lambda x: ["Offline", "Hybrid", "Mobile"][x])
        risk_score = st.slider("Điểm rủi ro nội bộ", 0.0, 1.0, 0.2)
        risk_segment = st.selectbox("Nhóm rủi ro", options=[0, 1, 2], format_func=lambda x: ["Thấp", "Trung bình", "Cao"][x])
        cluster_group = st.selectbox("Nhóm hành vi (K-Means)", options=[1, 2, 3, 4])

    submitted = st.form_submit_button("📊 PHÂN TÍCH RỦI RO")

# --- 4. Xử lý dự đoán và Hiển thị kết quả ---
if submitted:
    # Sắp xếp đúng thứ tự 21 cột như lúc huấn luyện
    # Order from X_resampled.columns: 'credit_sco', 'gender', 'age', 'occupation', 'balance', 'monthly_ir',
    # 'address', 'origin_province', 'tenure_ye', 'married', 'nums_card', 'nums_service',
    # 'last_transaction_month', 'active_member', 'customer_segment', 'engagement_score',
    # 'loyalty_level', 'digital_behavior', 'risk_score', 'risk_segment', 'cluster_group'
    features = [
        credit_sco,
        gender,
        age,
        occupation,
        balance,
        monthly_ir,
        address,
        origin_province,
        tenure_ye,
        married,
        nums_card,
        nums_service,
        last_transaction_month,
        active_member,
        customer_segment,
        engagement_score,
        loyalty_level,
        digital_behavior,
        risk_score,
        risk_segment,
        cluster_group
    ]
    
    # Define feature names in the correct order for input_df (matching X_resampled columns)
    feature_names_for_df = [
        'credit_sco', 'gender', 'age', 'occupation', 'balance', 'monthly_ir',
        'address', 'origin_province', 'tenure_ye', 'married', 'nums_card', 'nums_service',
        'last_transaction_month', 'active_member', 'customer_segment', 'engagement_score',
        'loyalty_level', 'digital_behavior', 'risk_score', 'risk_segment', 'cluster_group'
    ]

    # Chuyển thành DataFrame và chuẩn hóa
    input_df = pd.DataFrame([features], columns=feature_names_for_df)
    input_scaled = scaler.transform(input_df)

    # Dự đoán
    risk_proba = model.predict_proba(input_scaled)[0][1] # Get probability of positive class (churn)
    prediction = model.predict(input_scaled)[0]

    # --- 5. Dashboard tổng quan ---
    st.subheader("📈 Kết quả phân tích Dashboard")
    m1, m2, m3 = st.columns(3)
    
    m1.metric("Xác suất rời bỏ (Risk Score)", f"{risk_proba:.2%}")
    
    if prediction == 1:
        m2.error("Trạng thái: NGUY CƠ CAO")
        st.warning("🚨 Khuyến nghị: RM cần gọi điện chăm sóc và tặng voucher phí thường niên ngay.")
    else:
        m2.success("Trạng thái: AN TOÀN")
        st.info("✅ Khuyến nghị: Tiếp tục duy trì ưu đãi hiện tại.")
    
    m3.metric("Mức độ tương tác", f"{engagement_score}/100")

    # --- 6. Giải thích lý do Churn (Feature Importance) ---
    st.subheader("🔍 Tại sao khách hàng có nguy cơ này?")
    # Lấy hệ số từ Logistic Regression
    importance = model.coef_[0] # Access the first (and only) row of coefficients
    feature_names_for_importance = [
        "Điểm tín dụng", "Giới tính", "Tuổi", "Nghề nghiệp", "Số dư", "Thu nhập", "Địa chỉ",
        "Tỉnh thành", "Năm gắn bó", "Hôn nhân", "Số thẻ", "Số dịch vụ", "GD tháng cuối",
        "Hoạt động", "Phân khúc", "Điểm App", "Hạng TV", "Hành vi số", "Rủi ro NB", "Nhóm RR", "Cluster"
    ]
    
    feat_importances = pd.Series(importance, index=feature_names_for_importance).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feat_importances.head(5).values, y=feat_importances.head(5).index, palette="Reds_r", ax=ax)
    plt.title("Top 5 yếu tố làm tăng nguy cơ rời bỏ")
    st.pyplot(fig)
