import streamlit as st
import pandas as pd
import pickle

# Load model
@st.cache_resource
def load_assets():
    with open('decision_tree_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

st.title("🎯 Dự đoán khách hàng")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Tuổi", 18, 100, 35)
    balance = st.number_input("Số dư (VND)", 0, 1000000000, 50000000)
    credit_sco = st.slider("Điểm tín dụng", 300, 800, 650)
    salary = st.number_input("Thu nhập", 0, 100000000, 20000000)

with col2:
    active = st.selectbox("Hoạt động", [0, 1], format_func=lambda x: "Có" if x==1 else "Không")
    engagement = st.slider("Điểm tương tác", 0, 100, 50)
    tenure = st.slider("Số năm gắn bó", 0, 10, 3)

if st.button("Dự đoán ngay"):
    # Tạo input vector (cần khớp 22 feature của mô hình cũ)
    # Lưu ý: Đây là ví dụ demo, bạn cần map đầy đủ 22 cột như Sprint 5 đã làm
    input_data = pd.DataFrame([[credit_sco, age, balance, 20000000, tenure, 1, 2, 2, active, 1, engagement, 1, 1, 0.3, 0, 0,0,0,0,0,0,0]], 
                              columns=['credit_sco', 'age', 'balance', 'monthly_ir', 'tenure_ye', 'married', 'nums_card', 'nums_service', 'active_member', 'customer_segment', 'engagement_score', 'loyalty_level', 'digital_behavior', 'risk_score', 'risk_segment', 'occupation_Giáo viên/Giảng viên', 'occupation_Kinh doanh tự do', 'occupation_Kỹ sư/Chuyên viên IT', 'occupation_Kế toán/Tài chính', 'occupation_Nội trợ/Sinh viên', 'occupation_Nông dân/Lao động tự do', 'occupation_Y sĩ/Bác sĩ/Nghành y'])
    
    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)[0]
    
    if pred == 1:
        st.error("🚨 Nguy cơ rời bỏ: CAO")
        st.write("💡 Nguyên nhân: Điểm tương tác thấp, Số dư giảm.")
    else:
        st.success("✅ Khách hàng trung thành")
