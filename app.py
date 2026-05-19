import streamlit as st
import pandas as pd
import pickle

# --- CẤU HÌNH ---
st.set_page_config(page_title="BIDV AI Churn Prediction", layout="centered")

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    with open('decision_tree_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# --- GIAO DIỆN CHÍNH ---
st.title("🏦 DỰ ĐOÁN RỦI RO CHURN - BIDV")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Tuổi", 18, 80, 35)
    balance = st.number_input("Số dư (VND)", 0, 500000000, 50000000)
    credit_sco = st.slider("Điểm tín dụng", 300, 850, 650)
with col2:
    engagement = st.slider("Điểm tương tác App", 0, 100, 50)
    active = st.selectbox("Khách hàng hoạt động?", [0, 1], format_func=lambda x: "Có" if x==1 else "Không")
    num_service = st.number_input("Số sản phẩm đang dùng", 1, 10, 2)

# --- XỬ LÝ DỰ ĐOÁN ---
if st.button("🚀 DỰ ĐOÁN NGUY CƠ"):
    # Tạo vector input chuẩn
    feature_cols = ['credit_sco', 'age', 'balance', 'monthly_ir', 'tenure_ye', 'married', 
                    'nums_card', 'nums_service', 'active_member', 'customer_segment', 
                    'engagement_score', 'loyalty_level', 'digital_behavior', 'risk_score', 
                    'risk_segment', 'occupation_Giáo viên/Giảng viên', 'occupation_Kinh doanh tự do', 
                    'occupation_Kỹ sư/Chuyên viên IT', 'occupation_Kế toán/Tài chính', 
                    'occupation_Nội trợ/Sinh viên', 'occupation_Nông dân/Lao động tự do', 
                    'occupation_Y sĩ/Bác sĩ/Nghành y']
    
    data = {col: 0 for col in feature_cols}
    data.update({'credit_sco': credit_sco, 'age': age, 'balance': balance, 
                 'active_member': active, 'engagement_score': engagement, 'nums_service': num_service})
    
    input_df = pd.DataFrame([data])[feature_cols]
    
    # Dự báo
    prob = model.predict_proba(scaler.transform(input_df))[0][1]
    
    # Hiển thị kết quả (Dùng màu sắc để nhấn mạnh)
    if prob < 0.3:
        st.success(f"### KẾT QUẢ: AN TOÀN (Nguy cơ: {prob*100:.1f}%)")
    elif prob < 0.7:
        st.warning(f"### KẾT QUẢ: CẦN THEO DÕI (Nguy cơ: {prob*100:.1f}%)")
    else:
        st.error(f"### KẾT QUẢ: NGUY CƠ RỜI BỎ CAO (Nguy cơ: {prob*100:.1f}%)")
        st.write("**Nguyên nhân:** Tương tác thấp và số dư tài khoản không ổn định.")

st.markdown("---")
st.caption("Ứng dụng hỗ trợ Quản trị rủi ro BIDV | Team AI")
