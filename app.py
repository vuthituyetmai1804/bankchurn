import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Cau hinh
st.set_page_config(page_title="BIDV Churn Prediction", layout="wide")

@st.cache_resource
def load_assets():
    with open('decision_tree_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

st.title("🏦 HE THONG AI DU BAO RUI RO KHACH HANG - BIDV")

# --- FORM NHAP LIEU ---
st.sidebar.header("📝 NHAP THONG TIN")
age = st.sidebar.number_input("Do tuoi", 18, 100, 35)
balance = st.sidebar.number_input("So du tai khoan", 0, 1000000000, 50000000)
credit_sco = st.sidebar.slider("Diem tin dung", 300, 800, 650)
engagement_score = st.sidebar.slider("Diem tuong tac", 0, 100, 50)
monthly_ir = st.sidebar.number_input("Thu nhap hang thang", 0, 100000000, 20000000)
# Them cac bien mac dinh cho cac cot con thieu
tenure_ye, married, nums_card, nums_service, active_member = 3, 1, 2, 2, 1
customer_segment, loyalty_level, digital_behavior = 1, 1, 1
risk_score, risk_segment = 0.3, 0
occ_teacher, occ_free_biz, occ_it, occ_finance, occ_student, occ_farmer, occ_doctor = 0,0,0,0,0,0,0

# DataFrame dung thu tu cot khi Train (22 cot)
input_data = pd.DataFrame([[
    credit_sco, age, balance, monthly_ir, tenure_ye, married, nums_card, nums_service,
    active_member, customer_segment, engagement_score, loyalty_level, digital_behavior,
    risk_score, risk_segment, occ_teacher, occ_free_biz, occ_it, occ_finance, occ_student,
    occ_farmer, occ_doctor
]], columns=['credit_sco', 'age', 'balance', 'monthly_ir', 'tenure_ye', 'married', 
            'nums_card', 'nums_service', 'active_member', 'customer_segment', 
            'engagement_score', 'loyalty_level', 'digital_behavior', 'risk_score', 
            'risk_segment', 'occupation_Giáo viên/Giảng viên', 'occupation_Kinh doanh tự do', 
            'occupation_Kỹ sư/Chuyên viên IT', 'occupation_Kế toán/Tài chính', 
            'occupation_Nội trợ/Sinh viên', 'occupation_Nông dân/Lao động tự do', 
            'occupation_Y sĩ/Bác sĩ/Nghành y'])

# --- DU BAO ---
if st.button("🚀 Xac nhan du bao"):
    scaled_data = scaler.transform(input_data)
    pred = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]
    
    if pred == 1:
        st.error(f"🚨 CANH BAO: Kha nang roi bo: {prob*100:.2f}% - Can cham soc ngay!")
    else:
        st.success(f"✅ AN TOAN: Kha nang roi bo chi: {prob*100:.2f}% - Khach hang than thiet.")

    # Bieu do Feature Importance
    st.subheader("🎯 Cac yeu to chi phoi quyet dinh")
    importances = pd.Series(model.feature_importances_, index=input_data.columns)
    fig, ax = plt.subplots()
    importances.nlargest(5).plot(kind='barh', ax=ax)
    st.pyplot(fig)
