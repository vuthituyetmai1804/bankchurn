import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Cau hinh giao dien trang web ngan hang
st.set_page_config(page_title="BIDV Churn Prediction System", layout="wide", page_icon="🏦")

# --- HAM TAI MO HINH VA THANG DO (SCALER) ---
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'decision_tree_model.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    
    with open(model_path, 'rb') as f_model:
        model = pickle.load(f_model)
    with open(scaler_path, 'rb') as f_scaler:
        scaler = pickle.load(f_scaler)
    return model, scaler

try:
    model, scaler = load_assets()
    status_load = True
except Exception as e:
    status_load = False

# Tieu de chinh
st.title("🏦 HE THONG AI DU BAO VA QUAN TRI RUI RO KHACH HANG ROI BO - BIDV")
st.markdown("---")

if not status_load:
    st.error("❌ Loi he thong: Chua tim thay file `decision_tree_model.pkl` hoac `scaler.pkl` tren GitHub.")
else:
    feature_names = [
        'credit_sco', 'age', 'balance', 'monthly_ir', 'tenure_ye', 'married', 
        'nums_card', 'nums_service', 'active_member', 'customer_segment', 
        'engagement_score', 'loyalty_level', 'digital_behavior', 'risk_score', 
        'risk_segment', 'occupation_Giáo viên/Giảng viên', 'occupation_Kinh doanh tự do', 
        'occupation_Kỹ sư/Chuyên viên IT', 'occupation_Kế toán/Tài chính', 
        'occupation_Nội trợ/Sinh viên', 'occupation_Nông dân/Lao động tự do', 
        'occupation_Y sĩ/Bác sĩ/Nghành y'
    ]

    feature_display_names = {
        'credit_sco': 'Diem tin dung', 'age': 'Do tuoi', 'balance': 'So du tai khoan',
        'monthly_ir': 'Thu nhap hang thang', 'tenure_ye': 'Tham nien gan ket',
        'married': 'Tinh trang hon nhan', 'nums_card': 'So luong the',
        'nums_service': 'So san pham dang dung', 'active_member': 'Trang thai hoat dong',
        'customer_segment': 'Phan khuc khach hang', 'engagement_score': 'Diem tuong tac',
        'loyalty_level': 'Hang thanh vien', 'digital_behavior': 'Hanh vi so hoa',
        'risk_score': 'Diem rui ro tong hop', 'risk_segment': 'Phan lop rui ro'
    }

    # --- DASHBOARD TONG QUAN ---
    col_db1, col_db2, col_db3 = st.columns(3)
    col_db1.metric("Tong quy mo khach hang", "80,000")
    col_db2.metric("Ty le canh bao (Recall)", "69.0 %")
    col_db3.metric("Rui ro trung binh", "0.28")
    st.markdown("---")

    # --- FORM NHAP LIEU ---
    st.sidebar.header("📝 FORM NHAP THONG TIN")
    age = st.sidebar.number_input("Do tuoi", min_value=18, value=35)
    balance = st.sidebar.number_input("So du tai khoan", min_value=0, value=50000000)
    credit_sco = st.sidebar.slider("Diem tin dung", 300, 800, 650)
    engagement_score = st.sidebar.slider("Diem tuong tac", 0, 100, 50)
    
    # Placeholder cho cac input khac de code chay gon (them cac input vao day neu can)
    # ... (ban co the copy them cac dong sidebar.selectbox tu code cu vao day)

    # --- KET QUA ---
    if st.button("Du bao ngay"):
        # Logic xu ly input vao dataframe...
        st.success("He thong da ghi nhan du lieu!")
        
    st.subheader("🎯 CAY QUYET DINH VS THANG DO")
    fig, ax = plt.subplots()
    st.pyplot(fig)
