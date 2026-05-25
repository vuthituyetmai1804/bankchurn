import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Cấu hình trang
st.set_page_config(page_title="BIDV Churn Prediction", layout="wide")
st.title("🏦 Hệ thống dự đoán khách hàng rời bỏ")

# 2. Load model và scaler
@st.cache_resource
def load_models():
    model = joblib.load('bidv_churnn_model.pkl')
    scaler = joblib.load('scaler_bidv_model.pkl')
    return model, scaler

model, scaler = load_models()

# 3. Giao diện Input
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Tuổi", 18, 90, 30)
    credit_sco = st.slider("Điểm tín dụng", 300, 850, 600)
    balance = st.number_input("Số dư tài khoản (VND)", min_value=0, value=10000000)
    monthly_ir = st.number_input("Thu nhập hàng tháng (VND)", min_value=0, value=20000000)

with col2:
    tenure_ye = st.slider("Số năm gắn bó", 0, 10, 2)
    active_member = st.radio("Hoạt động gần đây", ["Có", "Không"])
    loyalty_level = st.selectbox("Hạng khách hàng", ["Bronze", "Silver", "Gold"])
    # Bạn có thể thêm các input khác như occupation, origin_province... ở đây

# 4. Xử lý logic dự đoán
if st.button("🔍 DỰ ĐOÁN NGAY", use_container_width=True):
    # Tạo DataFrame với 0 cho tất cả các cột (giả lập encoding)
    # Lưu ý: 'feature_names' phải là danh sách 35 cột trong quá trình train
    feature_names = [...] # ĐIỀN DANH SÁCH 35 CỘT CỦA BẠN VÀO ĐÂY
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Gán giá trị thực tế
    input_df['age'] = age
    input_df['credit_sco'] = credit_sco
    input_df['balance'] = balance
    input_df['monthly_ir'] = monthly_ir
    input_df['tenure_ye'] = tenure_ye
    input_df['active_member'] = 1 if active_member == "Có" else 0
    # ... gán giá trị cho các cột đã encode (ví dụ: loyalty_level_Gold = 1) ...

    # Scale dữ liệu
    cols_to_scale = ['credit_sco', 'age', 'balance', 'monthly_ir', 'tenure_ye']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    # Dự đoán
    risk_score = model.predict_proba(input_df)[0][1] * 100
    
    # 5. Hiển thị Output
    st.markdown("---")
    st.subheader("📊 Kết quả dự đoán")
    st.metric(label="RISK SCORE", value=f"{risk_score:.2f}%")
    
    if risk_score < 30:
        st.success(f"Risk Level: LOW RISK (🟢)")
        st.write("✅ Khuyến nghị: Duy trì mối quan hệ tốt với khách hàng.")
    elif 30 <= risk_score <= 70:
        st.warning(f"Risk Level: MEDIUM RISK (🟡)")
        st.write("Nên chăm sóc chủ động: Gọi điện tư vấn, tặng ưu đãi lãi suất, voucher.")
    else:
        st.error(f"Risk Level: HIGH RISK (🔴)")
        st.write("🚨 Khuyến nghị: Cần liên hệ khẩn cấp trong 24h...")
