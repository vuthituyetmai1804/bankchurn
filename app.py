import streamlit as st
import pandas as pd
import joblib

# 1. Cấu hình trang (Bắt buộc để trên cùng)
st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide" # Chuyển sang chế độ màn hình rộng để chia cột đẹp hơn
)

# 2. Tải mô hình đã lưu từ Colab
@st.cache_resource
def load_my_model():
    # Nhớ đổi tên file cho đúng với file .pkl xịn nhất của bạn
    return joblib.load("bidv_churn_modeltuning.pkl")

model = load_my_model()

# 3. NHÚNG CUSTOM CSS ĐỂ LÀM ĐẸP GIAO DIỆN
st.markdown("""
    <style>
    /* Đổi font chữ và nền tổng thể */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F8F9FA;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Thiết kế Header Box */
    .header-box {
        background: linear-gradient(135deg, #004B87 0%, #0072CE 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-title {
        font-size: 32px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .header-sub {
        font-size: 16px;
        opacity: 0.9;
        margin-top: 8px;
    }
    
    /* Làm đẹp các tiêu đề mục */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #1C355E;
        border-bottom: 3px solid #0072CE;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    
    /* Tùy chỉnh Nút bấm Chạy dự đoán dạng BIDV */
    div.stButton > button {
        background: linear-gradient(135deg, #004B87 0%, #005FA3 100%) !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 12px 30px !important;
        border-radius: 8px !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 4px 10px rgba(0,75,135,0.3) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(0,75,135,0.4) !important;
    }
    
    /* Khung hiển thị kết quả (Result Box) */
    .result-container {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
    }
    
    /* Khung khuyến nghị hành động */
    .recommend-box {
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        font-size: 15px;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# 4. HIỂN THỊ HEADER
st.markdown("""
<div class="header-box">
    <div class="header-title">🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ</div>
    <div class="header-sub">Machine Learning-Based Bank Customer Churn Prediction (Decision Tree Model)</div>
</div>
""", unsafe_allow_html=True)

# 5. CHIA BỐ CỤC THÀNH MÀN HÌNH ĐÔI (2 CỘT RỘNG BẰNG NHAU)
left_col, right_col = st.columns(2, gap="large")

# --- CỘT BÊN TRÁI: NHẬP LIỆU ---
with left_col:
    st.markdown('<div class="section-title">1. NHẬP THÔNG TIN KHÁCH HÀNG</div>', unsafe_allow_html=True)
    
    # Thiết kế gọn gàng: Chia nhỏ cột bên trong để các thanh trượt nằm song song
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        age = st.slider("🎂 Tuổi của khách hàng", 18, 80, 30)
        credit_sco = st.slider("💳 Điểm tín dụng (CIC)", 490, 800, 650)
        balance = st.number_input("💰 Số dư tài khoản (VND)", min_value=0, value=1500000, step=500000)
    
    with sub_col2:
        monthly_ir = st.number_input("💵 Thu nhập hàng tháng (VND)", min_value=0, value=5000000, step=500000)
        nums_service = st.slider("🏦 Số lượng dịch vụ sử dụng", 1, 8, 2)
        engagement_score = st.slider("🤝 Engagement Score", 0, 100, 70)
        
    active_member = st.radio("📱 Hoạt động gần đây (Tương tác trong tháng)", ["Có", "Không"], horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 CHẠY TEST CASE MÔ HÌNH")

# --- CỘT BÊN PHẢI: HIỂN THỊ KẾT QUẢ ---
with right_col:
    st.markdown('<div class="section-title">2. KẾT QUẢ PHÂN TÍCH & DỰ ĐOÁN</div>', unsafe_allow_html=True)
    
    # Nếu người dùng bấm nút dự đoán
    if predict_btn:
        # Chuẩn hóa dữ liệu đầu vào theo đúng tên cột khi Train
        # (Bạn nhớ đổi tên các biến cho trùng khớp với tập dữ liệu X_train cũ của bạn nhé)
        active_val = 1 if active_member == "Có" else 0
        
        features_order = ['monthly_ir', 'credit_sco', 'nums_service', 'engagement_score', 'balance', 'age', 'active_member']
        input_data = pd.DataFrame([[monthly_ir, credit_sco, nums_service, engagement_score, balance, age, active_val]], 
                                   columns=features_order)
        
        # Dự đoán xác suất
        risk_score = model.predict_proba(input_data)[0][1]
        risk_percent = round(risk_score * 100, 2)
        
        # Biện luận nhãn và màu sắc dựa trên xác suất rủi ro rời bỏ
        if risk_percent < 35:
            color = "#2E7D32"        # Xanh lá đậm
            bg_color = "#E8F5E9"     # Xanh lá nhạt
            risk_level = "THẤP (Low Risk)"
            prediction_text = "🟢 DỰ ĐOÁN: Ở lại (STAY)"
            recommendation = "Khách hàng rất trung thành. Ngân hàng cần duy trì chất lượng dịch vụ hiện tại và đưa vào danh sách ưu tiên trải nghiệm các sản phẩm đầu tư/tín dụng cao cấp mới."
        elif risk_percent <= 65:
            color = "#F57C00"        # Cam
            bg_color = "#FFF3E0"     # Cam nhạt
            risk_level = "TRUNG BÌNH (Medium Risk)"
            prediction_text = "🟡 DỰ ĐOÁN: Có nguy cơ rời bỏ (CHURN)"
            recommendation = "Khách hàng đang ở vùng lưỡng lự. Hệ thống khuyến nghị RM (Quản lý quan hệ khách hàng) chủ động gửi chương trình ưu đãi phí chuyển khoản, tích điểm hoàn tiền hoặc gọi điện chăm sóc để thắt chặt tương tác."
        else:
            color = "#C62828"        # Đỏ
            bg_color = "#FFEBEE"     # Đỏ nhạt
            risk_level = "CAO (High Risk)"
            prediction_text = "🔴 DỰ ĐOÁN: Rời bỏ (CHURN)"
            recommendation = "🚨 BÁO ĐỘNG ĐỎ! Khách hàng có khả năng cao sẽ đóng tài khoản. Cần kích hoạt ngay quy trình giữ chân khẩn cấp: Tặng gói đặc quyền ưu đãi lãi suất, miễn toàn bộ phí dịch vụ trong vòng 3 tháng tới."

        # Hiển thị khối kết quả sang xịn mịn
        st.markdown(f"""
        <div class="result-container">
            <p style="font-size: 14px; color: #64748B; margin-bottom: 5px; font-weight: 600;">RISK SCORE (Tỷ lệ rủi ro dự đoán)</p>
            <h1 style="color: {color}; margin-top: 0; font-size: 48px; font-weight: 800;">{risk_percent}%</h1>
            
            <hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 15px 0;">
            
            <div style="background-color: {bg_color}; border-left: 5px solid {color}; padding: 12px 15px; border-radius: 4px; margin-bottom: 15px;">
                <span style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; color: #475569; font-weight: bold;">Mức độ rủi ro:</span>
                <strong style="color: {color}; font-size: 15px; margin-left: 5px;">{risk_level}</strong>
            </div>
            
            <h3 style="color: #1E293B; font-weight: 700; margin-top: 20px;">{prediction_text}</h3>
            
            <div class="recommend-box" style="background-color: white; border: 1px dashed {color}; border-left: 6px solid {color};">
                <strong style="color: {color}; font-size: 16px;">🎯 Khuyến nghị hành động chủ động:</strong>
                <p style="margin-top: 8px; color: #334155;">{recommendation}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Khi chưa bấm nút, hiển thị thông báo hướng dẫn nhẹ nhàng
        st.info("👋 Vui lòng điều chỉnh thông tin khách hàng ở cột bên trái và bấm nút 'CHẠY TEST CASE MÔ HÌNH' để xem kết quả phân tích rủi ro.")
