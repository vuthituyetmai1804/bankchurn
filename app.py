import streamlit as st
import joblib
import pandas as pd

# Load model + scaler
@st.cache_resource
def load_model():
    model = joblib.load('bidv_churn_model.pkl')      # Đường dẫn đúng
    scaler = joblib.load('scaler_bidv_model.pkl')
    return model, scaler

model, scaler = load_model()

st.title("🚀 Dự đoán Churn - BIDV")

# ==================== INPUT ====================
col1, col2 = st.columns(2)

with col1:
    monthly_ir = st.number_input("Thu nhập hàng tháng (triệu VND)", min_value=0.0, value=25.0)
    credit_sco = st.number_input("Điểm tín dụng", min_value=300, max_value=850, value=680)
    nums_service = st.number_input("Số dịch vụ đang dùng", min_value=0, max_value=10, value=3)
    engagement_score = st.slider("Điểm gắn kết khách hàng", 0.0, 100.0, 65.0)

with col2:
    balance = st.number_input("Số dư tài khoản (triệu VND)", min_value=0.0, value=150.0)
    age = st.number_input("Tuổi", min_value=18, max_value=80, value=38)
    active_member = st.selectbox("Thành viên tích cực?", options=[1, 0], format_func=lambda x: "Có" if x == 1 else "Không")
    risk_score = st.slider("Risk Score", 0.0, 100.0, 45.0)   # ← Thêm cột này

# Tạo DataFrame theo đúng thứ tự scaler được fit
input_data = pd.DataFrame([{
    'risk_score': risk_score,
    'monthly_ir': monthly_ir,
    'credit_sco': credit_sco,
    'nums_service': nums_service,
    'engagement_score': engagement_score,
    'balance': balance,
    'age': age,
    'active_member': active_member
}])

# ==================== PHẦN DỰ ĐOÁN ====================
if st.button("🔍 Dự đoán khả năng rời bỏ", type="primary"):
    
    # Định nghĩa CHÍNH XÁC danh sách cột theo thứ tự scaler được fit
    cols_to_scale = ['risk_score', 'monthly_ir', 'credit_sco', 'nums_service', 
                     'engagement_score', 'balance', 'age']
    
    # Tạo DataFrame với đúng tên cột và thứ tự
    input_dict = {
        'risk_score': risk_score,
        'monthly_ir': monthly_ir,
        'credit_sco': credit_sco,
        'nums_service': nums_service,
        'engagement_score': engagement_score,
        'balance': balance,
        'age': age,
        'active_member': active_member
    }
    
    input_data = pd.DataFrame([input_dict])
    
    # Debug (bạn có thể comment sau khi chạy ổn)
    st.write("**Debug - Cột input:**", input_data.columns.tolist())
    
    # Scale chỉ các cột cần scale
    try:
        input_scaled = scaler.transform(input_data[cols_to_scale])
        
        # Gán lại giá trị đã scale
        input_final = input_data.copy()
        input_final[cols_to_scale] = input_scaled
        
        # Predict
        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0][1]
        
        if prediction == 1:
            st.error(f"**KHÁCH HÀNG CÓ NGUY CƠ RỜI BỎ CAO** ({probability:.1%})")
        else:
            st.success(f"**KHÁCH HÀNG CÓ XÁC SUẤT Ở LẠI TỐT** ({(1-probability):.1%})")
            
        st.info(f"Xác suất churn: **{probability:.1%}**")
        
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")
        st.write("Debug - Input DataFrame:")
        st.write(input_data)
