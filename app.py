import streamlit as st
import pandas as pd
import joblib

# =========================================================
# 1. PAGE CONFIG (Bắt buộc đặt trên cùng)
# =========================================================
st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"  # Kích hoạt màn hình rộng để chia 2 cột Trái - Phải như thiết kế mẫu
)

# =========================================================
# 2. LOAD MODEL (Giữ nguyên logic nạp file của bạn)
# =========================================================
@st.cache_resource
def load_my_model():
    return joblib.load("bidv_churn_modeltuning.pkl")

try:
    model = load_my_model()
except Exception as e:
    st.error("🚨 Không tìm thấy file 'bidv_churn_modeltuning.pkl'. Hãy kiểm tra lại kho GitHub!")
    st.stop()

# =========================================================
# 3. CUSTOM CSS (Lột xác màu sắc và hiệu ứng chuẩn BIDV)
# =========================================================
st.markdown("""
<style>
/* Nền xám nhạt cao cấp cho toàn bộ ứng dụng */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F8F9FA;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

/* Đè lại padding mặc định của Streamlit để giao diện khít đẹp */
[data-testid="stSidebarCollapse"] {
    display: none;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* Thiết kế Header Box dải màu Gradient chuẩn BIDV */
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

/* Thanh tiêu đề phân chia khu vực */
.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #1C355E;
    border-bottom: 3px solid #0072CE;
    padding-bottom: 8px;
    margin-bottom: 25px;
}

/* Tùy chỉnh Nút bấm Chạy dự đoán kích thước lớn, màu thẫm cực sang */
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

/* Thanh radio hoạt động nằm ngang gọn gàng */
div[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. HEADER
# =========================================================
st.markdown("""
<div class="header-box">
    <div class="header-title">🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ</div>
    <div class="header-sub">Ứng dụng Mô hình Cây quyết định trong Quản trị Rủi ro Ngân hàng BIDV</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 5. CHIA BỐ CỤC SONG SONG (Tránh bị đẩy kết quả xuống đáy web)
# =========================================================
left_col, right_col = st.columns(2, gap="large")

# ---------------------------------------------------------
# --- CỘT BÊN TRÁI: NHẬP LIỆU (Gom gọn các ô nhập) ---
# ---------------------------------------------------------
with left_col:
    st.markdown('<div class="section-title">1. NHẬP THÔNG TIN KHÁCH HÀNG</div>', unsafe_allow_html=True)
    
    # Chia nhỏ cột nội bộ để tối ưu không gian hiển thị
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        age = st.slider("🎂 Tuổi", 20, 80, 35)
        credit_sco = st.slider("💳 Điểm tín dụng", 495, 800, 650)
        balance = st.number_input("💰 Số dư tài khoản (VND)", min_value=0, value=50000000, step=1000000)
    
    with sub_col2:
        monthly_ir = st.number_input("💵 Thu nhập hàng tháng (VND)", min_value=0, value=15000000, step=1000000)
        nums_service = st.slider("🏦 Số lượng dịch vụ sử dụng", 1, 8, 3)
        engagement_score = st.slider("🤝 Điểm tương tác app", 0, 100, 50)
        
    active_text = st.radio("📱 Hoạt động gần đây", ["Có", "Không"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 CHẠY TEST CASE MÔ HÌNH")

# ---------------------------------------------------------
# --- CỘT BÊN PHẢI: HIỂN THỊ KẾT QUẢ ĐỒ HỌA SANG TRỌNG ---
# ---------------------------------------------------------
with right_col:
    st.markdown('<div class="section-title">2. KẾT QUẢ PHÂN TÍCH & DỰ ĐOÁN</div>', unsafe_allow_html=True)
    
    if predict_btn:
        # =====================================================
        # ENCODE & LOGIC GIỮ NGUYÊN 100% KHÔNG SAI LỆCH CỦA BẠN
        # =====================================================
        active_member = 1 if active_text == "Có" else 0
        
        features_order = [
            'monthly_ir', 'credit_sco', 'nums_service', 
            'engagement_score', 'balance', 'age', 'active_member'
        ]
        
        input_df = pd.DataFrame([{
            'monthly_ir': monthly_ir,
            'credit_sco': credit_sco,
            'nums_service': nums_service,
            'engagement_score': engagement_score,
            'balance': balance,
            'age': age,
            'active_member': active_member
        }])
        
        final_input = input_df[features_order]
        
        # Tính toán xác suất từ mô hình tuning của bạn
        risk_score = model.predict_proba(final_input)[0][1]
        risk_percent = round(risk_score * 100, 2)
        
        # Phân định màu sắc động và nội dung khuyến nghị dựa trên bộ khung cũ của bạn
        if risk_percent < 30:
            risk_level = "THẤP (Low Risk)"
            prediction_text = "✅ Khách hàng có khả năng tiếp tục sử dụng dịch vụ"
            recommendation = "Duy trì mối quan hệ tốt và tiếp tục chăm sóc định kỳ, giới thiệu các gói sản phẩm tài chính dài hạn."
            color = "#2E7D32"        # Xanh lá cây thẫm cao cấp
            bg_color = "#E8F5E9"     # Nền xanh lá nhạt
        elif risk_percent <= 70:
            risk_level = "TRUNG BÌNH (Medium Risk)"
            prediction_text = "⚠️ Khách hàng có nguy cơ rời bỏ"
            recommendation = "Nên chăm sóc chủ động: Điều phối quản lý quan hệ khách hàng (RM) gọi điện tư vấn, đề xuất ưu đãi giảm phí dịch vụ hoặc tặng voucher."
            color = "#F57C00"        # Màu cam tinh tế
            bg_color = "#FFF3E0"     # Nền cam nhạt
        else:
            risk_level = "CAO (High Risk)"
            prediction_text = "🚨 Khách hàng có nguy cơ rời bỏ"
            recommendation = "Cần kích hoạt quy trình ứng phó khẩn cấp: Chuyển dữ liệu sang trung tâm xử lý dữ liệu và liên hệ trực tiếp trong vòng 24h để thực hiện các đặc quyền giữ chân tối đa."
            color = "#C62828"        # Màu đỏ sẫm cảnh báo rủi ro
            bg_color = "#FFEBEE"     # Nền đỏ nhạt

        # Đầu ra đồ họa dạng Hộp Card đổ bóng, bo tròn viền chuẩn Apple UI
        st.markdown(f"""
        <div style="background-color: white; padding: 28px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.06); border: 1px solid #EAF2F8; margin-top: 10px;">
            
            <p style="font-size: 13px; color: #64748B; margin-bottom: 3px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;">RISK SCORE (Tỷ lệ rủi ro)</p>
            <h1 style="color: {color}; margin-top: 0; font-size: 56px; font-weight: 800; line-height: 1;">{risk_percent}%</h1>
            
            <div style="display: inline-block; background-color: {bg_color}; padding: 6px 16px; border-radius: 20px; margin-top: 10px; margin-bottom: 20px;">
                <span style="font-size: 13px; color: {color}; font-weight: 700;">● Mức độ: {risk_level}</span>
            </div>
            
            <h3 style="color: #1E293B; font-weight: 700; margin-top: 5px; font-size: 18px;">
                {prediction_text}
            </h3>
            
            <div style="background-color: {bg_color}25; border-left: 5px solid {color}; padding: 20px; border-radius: 8px; margin-top: 25px; border-top: 1px solid {color}10; border-right: 1px solid {color}10; border-bottom: 1px solid {color}10;">
                <div style="display: flex; align-items: center; gap: 8px; color: {color}; font-weight: 700; font-size: 15px;">
                    <span>🎯</span> Khuyến nghị hành động chủ động:
                </div>
                <p style="margin-top: 10px; color: #475569; font-size: 14px; line-height: 1.6; font-weight: 500; margin-bottom: 0;">
                    {recommendation}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Trạng thái tĩnh khi người dùng mở trang web lên và chưa nhấn nút dự đoán
        st.info("👋 Hệ thống đang sẵn sàng. Vui lòng nhập hoặc điều chỉnh các thông tin khách hàng ở bảng bên trái, sau đó nhấn nút 'CHẠY TEST CASE MÔ HÌNH' để kiểm tra kết quả phân tách rủi ro.")
