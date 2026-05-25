import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. CẤU HÌNH TRANG (BẮT BUỘC ĐỂ TRÊN CÙNG)
# ==========================================
st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"  # Chuyển sang chế độ màn hình rộng để chia cột Trái - Phải
)

# ==========================================
# 2. KHỞI TẠO VÀ TẢI MÔ HÌNH ĐÃ TUNING
# ==========================================
@st.cache_resource
def load_my_model():
    # Đảm bảo tên file này trùng với file .pkl xịn nhất bạn vừa tải từ Colab xuống
    return joblib.load("bidv_churn_modeltuning.pkl")

try:
    model = load_my_model()
except:
    st.error("🚨 Không tìm thấy file 'bidv_churn_model.pkl'. Hãy đảm bảo bạn đã upload file model mới lên cùng thư mục với app.py!")

# ==========================================
# 3. NHÚNG CUSTOM CSS ĐỂ LÀM ĐẸP GIAO DIỆN
# ==========================================
st.markdown("""
    <style>
    /* Nền tổng thể của toàn bộ ứng dụng */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F8F9FA;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Thiết kế Header Box phía trên cùng */
    .header-box {
        background: linear-gradient(135deg, #004B87 0%, #0072CE 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 35px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-title {
        font-size: 30px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .header-sub {
        font-size: 15px;
        opacity: 0.9;
        margin-top: 8px;
    }
    
    /* Làm đẹp các thanh tiêu đề mục (1. Nhập thông tin / 2. Kết quả) */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #1C355E;
        border-bottom: 3px solid #0072CE;
        padding-bottom: 8px;
        margin-bottom: 25px;
    }
    
    /* Tùy chỉnh nút bấm "CHẠY TEST CASE MÔ HÌNH" chuẩn UI BIDV */
    div.stButton > button {
        background: linear-gradient(135deg, #004B87 0%, #005FA3 100%) !important;
        color: white !important;
        font-size: 17px !important;
        font-weight: bold !important;
        padding: 12px 30px !important;
        border-radius: 8px !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 4px 10px rgba(0,75,135,0.3) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(0,75,135,0.4) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. HIỂN THỊ HEADER CHÍNH
# ==========================================
st.markdown("""
<div class="header-box">
    <div class="header-title">🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ</div>
    <div class="header-sub">Machine Learning-Based Bank Customer Churn Prediction (Decision Tree Model)</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 5. CHIA BỐ CỤC THÀNH MÀN HÌNH ĐÔI (2 CỘT)
# ==========================================
left_col, right_col = st.columns(2, gap="large")

# ------------------------------------------
# --- CỘT BÊN TRÁI: KHU VỰC NHẬP LIỆU ---
# ------------------------------------------
with left_col:
    st.markdown('<div class="section-title">1. NHẬP THÔNG TIN KHÁCH HÀNG</div>', unsafe_allow_html=True)
    
    # Sử dụng cột phụ bên trong để chia hàng song song cho gọn gàng
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        age = st.slider("🎂 Tuổi của khách hàng", 18, 80, 21)
        credit_sco = st.slider("💳 Điểm tín dụng (CIC)", 490, 800, 600)
        balance = st.number_input("💰 Số dư tài khoản (VND)", min_value=0, value=1500000, step=500000)
    
    with sub_col2:
        monthly_ir = st.number_input("💵 Thu nhập hàng tháng (VND)", min_value=0, value=3000000, step=500000)
        nums_service = st.slider("🏦 Số lượng dịch vụ sử dụng", 1, 8, 3)
        engagement_score = st.slider("🤝 Engagement Score", 0, 100, 71)
        
    active_member = st.radio("📱 Hoạt động gần đây (Tương tác trong tháng)", ["Có", "Không"], horizontal=True)

    # Khoảng cách dòng và Nút bấm kích hoạt
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 CHẠY TEST CASE MÔ HÌNH")

# ------------------------------------------
# --- CỘT BÊN PHẢI: KHU VỰC KẾT QUẢ ---
# ------------------------------------------
with right_col:
    st.markdown('<div class="section-title">2. KẾT QUẢ PHÂN TÍCH & DỰ ĐOÁN</div>', unsafe_allow_html=True)
    
    if predict_btn:
        # Chuyển đổi trạng thái radio sang định dạng nhị phân 0/1 giống tập dữ liệu train
        active_val = 1 if active_member == "Có" else 0
        
        # Sắp xếp đúng thứ tự các đặc trưng đầu vào của mô hình
        features_order = ['monthly_ir', 'credit_sco', 'nums_service', 'engagement_score', 'balance', 'age', 'active_member']
        input_data = pd.DataFrame([[monthly_ir, credit_sco, nums_service, engagement_score, balance, age, active_val]], 
                                   columns=features_order)
        
        # Thực hiện dự đoán xác suất rủi ro lớp Churn (mục số 1)
        risk_score = model.predict_proba(input_data)[0][1]
        risk_percent = round(risk_score * 100, 2)
        
        # Biện luận động nhãn hiển thị, màu sắc chủ đạo, và khuyến nghị dựa trên Risk Percent
        if risk_percent < 35:
            color = "#2E7D32"        # Xanh lá đậm
            bg_color = "#E8F5E9"     # Xanh lá nhạt
            risk_level = "THẤP (Low Risk)"
            prediction_text = "🟢 DỰ ĐOÁN: Ở lại (STAY)"
            recommendation = "Khách hàng rất trung thành. Ngân hàng cần duy trì chất lượng dịch vụ hiện tại và đưa vào danh sách ưu tiên trải nghiệm các sản phẩm đầu tư/tín dụng cao cấp mới."
        elif risk_percent <= 65:
            color = "#F57C00"        # Màu cam
            bg_color = "#FFF3E0"     # Màu cam nhạt
            risk_level = "TRUNG BÌNH (Medium Risk)"
            prediction_text = "🟡 DỰ ĐOÁN: Có nguy cơ rời bỏ (CHURN)"
            recommendation = "Khách hàng đang ở vùng lưỡng lự. Hệ thống khuyến nghị RM (Quản lý quan hệ khách hàng) chủ động gửi chương trình ưu đãi phí chuyển khoản, tích điểm hoàn tiền hoặc gọi điện chăm sóc để thắt chặt tương tác."
        else:
            color = "#C62828"        # Màu đỏ
            bg_color = "#FFEBEE"     # Màu đỏ nhạt
            risk_level = "CAO (High Risk)"
            prediction_text = "🔴 DỰ ĐOÁN: Rời bỏ (CHURN)"
            recommendation = "🚨 BÁO ĐỘNG ĐỎ! Khách hàng có khả năng rất cao sẽ đóng tài khoản. Kích hoạt quy trình giữ chân khẩn cấp: Tặng gói đặc quyền ưu đãi lãi suất, miễn toàn bộ các loại phí dịch vụ trong vòng 3 tháng tới."

        # Hiển thị khối Card đồ họa đổ bóng mờ sang trọng, an toàn 100% nhờ có unsafe_allow_html=True
        st.markdown(f"""
        <div style="background-color: white; padding: 28px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); border: 1px solid #EAF2F8; margin-top: 10px;">
            
            <p style="font-size: 13px; color: #64748B; margin-bottom: 3px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;">RISK SCORE (Tỷ lệ rủi ro)</p>
            <h1 style="color: {color}; margin-top: 0; font-size: 52px; font-weight: 800; line-height: 1;">{risk_percent}%</h1>
            
            <div style="display: inline-block; background-color: {bg_color}; padding: 6px 16px; border-radius: 20px; margin-top: 10px; margin-bottom: 20px;">
                <span style="font-size: 13px; color: {color}; font-weight: 700;">● Mức độ rủi ro: {risk_level}</span>
            </div>
            
            <h3 style="color: #1E293B; font-weight: 700; margin-top: 5px; font-size: 20px;">
                {prediction_text}
            </h3>
            
            <div style="background-color: {bg_color}33; border-left: 5px solid {color}; padding: 20px; border-radius: 8px; margin-top: 20px; border-top: 1px solid {color}15; border-right: 1px solid {color}15; border-bottom: 1px solid {color}15;">
                <div style="display: flex; align-items: center; gap: 8px; color: {color}; font-weight: 700; font-size: 15px;">
                    <span>🎯</span> Khuyến nghị hành động chủ động
                </div>
                <p style="margin-top: 10px; color: #475569; font-size: 14px; line-height: 1.6; font-weight: 500; margin-bottom: 0;">
                    {recommendation}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Trạng thái ban đầu khi vừa khởi động ứng dụng (Giao diện trống chờ nhập liệu)
        st.info("👋 Vui lòng điều chỉnh thông tin khách hàng ở khối bên trái và bấm nút '🔍 CHẠY TEST CASE MÔ HÌNH' để xem kết quả phân tích rủi ro.")
