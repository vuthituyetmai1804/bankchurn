import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Tải mô hình và bộ chuẩn hóa ---
@st.cache_resource
def load_assets():
    model = joblib.load('bank_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("🏦 BIDV - Dự báo Churn (Giao diện Rút gọn)")
st.info("Hệ thống tập trung vào 8 chỉ số quan trọng nhất ảnh hưởng đến rủi ro khách hàng.")

# --- 2. Giao diện nhập liệu (Chỉ 8 đặc trưng chính) ---
with st.form("short_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        balance = st.number_input("Số dư hiện tại (VND)", min_value=0, value=50000000)
        age = st.number_input("Tuổi", 18, 100, 35)
        monthly_ir = st.number_input("Thu nhập hàng tháng (VND)", min_value=0, value=20000000)
        engagement_score = st.slider("Điểm tương tác app (0-100)", 0, 100, 70)

    with col2:
        tenure_ye = st.number_input("Số năm gắn bó", 0, 50, 3)
        credit_sco = st.slider("Điểm tín dụng", 300, 800, 650)
        active_member = st.selectbox("Hội viên hoạt động", [1], format_func=lambda x: "Có" if x==1 else "Không")
        customer_segment = st.selectbox("Phân khúc", [1-3], format_func=lambda x: ["Mass", "Emerging", "Affluent", "Priority"][x])

    submitted = st.form_submit_button("📊 PHÂN TÍCH NGAY")

# --- 3. Xử lý logic: Bù đắp 13 đặc trưng còn lại ---
if submitted:
    # Thứ tự 21 cột mô hình yêu cầu (phải khớp hoàn toàn với Sprint 4)
    # 8 giá trị lấy từ Form, 13 giá trị lấy mặc định (Default)
    input_data = {
        'gender': 0, # Mặc định Nữ
        'age': age,
        'occupation': 1, # Mặc định nhân viên văn phòng
        'origin_province': 0,
        'address': 0,
        'monthly_ir': monthly_ir,
        'balance': balance,
        'credit_sco': credit_sco,
        'tenure_ye': tenure_ye,
        'married': 1, # Mặc định đã kết hôn
        'nums_card': 1,
        'nums_service': 2,
        'last_transaction_month': 1000000,
        'active_member': active_member,
        'customer_segment': customer_segment,
        'engagement_score': engagement_score,
        'loyalty_level': 1, # Mặc định Silver
        'digital_behavior': 2, # Mặc định Mobile
        'risk_score': 0.2,
        'risk_segment': 0,
        'cluster_group': 1
    }
    
    # Chuyển thành DataFrame theo đúng thứ tự cột
    df_input = pd.DataFrame([input_data])
    
    # Chuẩn hóa (Sẽ không lỗi vì đã đủ 21 cột)
    input_scaled = scaler.transform(df_input)
    
    # Dự đoán
    risk_proba = model.predict_proba(input_scaled)[1]
    prediction = model.predict(input_scaled)

    # --- 4. Hiển thị Dashboard kết quả ---
    st.subheader("📈 Kết quả phân tích")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Xác suất rời bỏ", f"{risk_proba:.2%}")
    
    if prediction == 1:
        res_col2.error("NGUY CƠ: RỜI BỎ CAO")
    else:
        res_col2.success("AN TOÀN: Ở LẠI")
