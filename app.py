import streamlit as st
import pandas as pd
import joblib

# 1. Cấu hình trang nền rộng
st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# 2. Tải mô hình thuật toán
@st.cache_resource
def load_my_model():
    return joblib.load("bidv_churn_modeltuning.pkl")

try:
    model = load_my_model()
except:
    st.error("🚨 Không tìm thấy file mô hình 'bidv_churn_modeltuning.pkl'!")
    st.stop()

# 3. CSS dọn dẹp giao diện phẳng và bo góc mềm mại
st.markdown("""
<style>
/* Xóa các đường viền thừa của Streamlit */
div[data-testid="stForm"] { border: none !important; }
.block-container { padding-top: 2.5rem !important; }

/* Header tối giản phong cách Dashboard hiện đại */
.main-header {
    background-color: #FFFFFF;
    padding: 20px 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 40px;
}
.main-title {
    font-size: 28px;
    font-weight: 700;
    color: #004B87;
    letter-spacing: -0.5px;
}
.main-sub {
    font-size: 14px;
    color: #64748B;
    margin-top: 4px;
}

/* Định hình form nhập liệu sạch sẽ */
.input-card {
    background-color: #F4F7FA;
    padding: 25px;
    border-radius: 14px;
    margin-bottom: 20px;
}

/* Nút bấm phẳng (Flat Design) */
div.stButton > button {
    background-color: #004B87 !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 14px 0 !important;
    border-radius: 10px !important;
    border: none !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(0, 75, 135, 0.15) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    background-color: #003560 !important;
    transform: translateY(-1px) !important;
}

/* Thanh chọn ngang */
div[data-testid="stRadio"] > div { flex-direction: row !important; gap: 25px; }
</style>
""", unsafe_allow_html=True)

# 4. HIỂN THỊ HEADER TỐI GIẢN
st.markdown("""
<div class="main-header">
    <div class="main-title">🏦 QUAN SÁT & DỰ BÁO RỦI RO KHÁCH HÀNG RỜI BỎ</div>
    <div class="main-sub">Hệ thống phân tích hành vi tài chính thời gian thực ứng dụng Thuật toán Cây quyết định</div>
</div>
""", unsafe_allow_html=True)

# 5. CHIA HAI CỘT SONG SONG
left_col, right_col = st.columns([1.1, 0.9], gap="large")

# --- CỘT TRÁI: KHU VỰC ĐIỀU CHỈNH BIẾN ---
with left_col:
    st.caption("⚙️ BẢNG ĐIỀU CHỈNH THÔNG SỐ")
    
    sub1, sub2 = st.columns(2, gap="medium")
    with sub1:
        age = st.slider("🎂 Tuổi khách hàng", 20, 80, 35)
        credit_sco = st.slider("💳 Điểm tín dụng", 495, 800, 650)
        balance = st.number_input("💰 Số dư tài khoản (VND)", min_value=0, value=50000000, step=5000000)
    
    with sub2:
        monthly_ir = st.number_input("💵 Thu nhập hàng tháng (VND)", min_value=0, value=15000000, step=5000000)
        nums_service = st.slider("🏦 Số lượng dịch vụ", 1, 8, 3)
        engagement_score = st.slider("🤝 Điểm tương tác ứng dụng", 0, 100, 50)
        
    active_text = st.radio("📱 Trạng thái tương tác gần đây", ["Có", "Không"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 CHẠY PHÂN TÍCH MÔ HÌNH")

# --- CỘT PHẢI: KHU VỰC HIỂN THỊ KẾT QUẢ ---
with right_col:
    st.caption("📊 KẾT QUẢ ĐÁNH GIÁ TỪ HỆ THỐNG")
    
    if predict_btn:
        # Giữ nguyên logic tính toán của bạn
        active_member = 1 if active_text == "Có" else 0
        features_order = ['monthly_ir', 'credit_sco', 'nums_service', 'engagement_score', 'balance', 'age', 'active_member']
        
        input_df = pd.DataFrame([{'monthly_ir': monthly_ir, 'credit_sco': credit_sco, 'nums_service': nums_service,
                                  'engagement_score': engagement_score, 'balance': balance, 'age': age, 'active_member': active_member}])
        
        risk_score = model.predict_proba(input_df[features_order])[0][1]
        risk_percent = round(risk_score * 100, 2)
        
        # Thiết lập bảng màu mảnh, không dùng mảng màu đậm gây loè loẹt
        if risk_percent < 30:
            risk_level = "Thấp (Low Risk)"
            prediction_text = "Tối ưu — Khách hàng có xu hướng Gắn bó lâu dài"
            recommendation = "Duy trì tần suất chăm sóc tự động qua app. Đề xuất các sản phẩm tích lũy chéo hoặc gói vay ưu đãi kỳ hạn dài để tối đa hóa giá trị vòng đời khách hàng."
            brand_color = "#0A84FF" 
            bg_badge = "#E5F2FF"
        elif risk_percent <= 70:
            risk_level = "Trung bình (Medium Risk)"
            prediction_text = "Chú ý — Khách hàng bắt đầu giảm tương tác"
            recommendation = "Hệ thống tự động điều phối dữ liệu về chi nhánh quản lý. Nhân viên RM cần chủ động liên hệ tặng gói miễn phí chuyển khoản trọn đời hoặc voucher quà tặng để làm mới mối liên kết."
            brand_color = "#FF9500" 
            bg_badge = "#FFF2E0"
        else:
            risk_level = "Cao (High Risk)"
            prediction_text = "Cảnh báo đỏ — Nguy cơ rời bỏ rất nghiêm trọng"
            recommendation = "🚨 Kích hoạt kịch bản giải cứu khẩn cấp. Lãnh đạo phòng dịch vụ khách hàng trực tiếp thẩm định, đưa ra chính sách đặc quyền miễn giảm toàn bộ phí dịch vụ hoặc áp dụng biên độ lãi suất đặc cách."
            brand_color = "#FF3B30" 
            bg_badge = "#FFEBEA"

        # Giải pháp an toàn: Gom HTML thành một biến chuỗi phẳng để tránh lỗi cú pháp thụt lề đầu dòng
        html_layout = f"""
        <div style="background-color: #FFFFFF; padding: 30px; border-radius: 16px; box-shadow: 0 12px 40px rgba(0,0,0,0.04); border: 1px solid #E2E8F0; margin-top: 5px;">
            <span style="font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">XÁC SUẤT RỦI RO CHI TIẾT</span>
            <h1 style="color: {brand_color}; margin: 5px 0 15px 0; font-size: 60px; font-weight: 700; letter-spacing: -1.5px; line-height: 1;">{risk_percent}%</h1>
            
            <div style="display: inline-block; background-color: {bg_badge}; padding: 6px 14px; border-radius: 8px; margin-bottom: 25px;">
                <span style="font-size: 13px; color: {brand_color}; font-weight: 600;">● Phân nhóm: {risk_level}</span>
            </div>
            
            <div style="font-size: 16px; font-weight: 700; color: #1E293B; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid #F1F5F9;">
                {prediction_text}
            </div>
            
            <div style="background-color: #F8FAFC; border-left: 4px solid {brand_color}; padding: 20px; border-radius: 0 12px 12px 0;">
                <div style="color: #0F172A; font-weight: 700; font-size: 14px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                    <span>🎯</span> PHƯƠNG ÁN XỬ LÝ ĐỀ XUẤT:
                </div>
                <p style="color: #475569; font-size: 13.5px; line-height: 1.6; font-weight: 500; margin: 0;">
                    {recommendation}
                </p>
            </div>
        </div>
        """
        # Đẩy biến chuỗi vào hàm xuất
        st.markdown(html_layout, unsafe_allow_html=True)
        
    else:
        st.info("👋 Vui lòng điều chỉnh các thông số tài chính của khách hàng ở cột bên trái và bấm nút 'CHẠY PHÂN TÍCH MÔ HÌNH' để xem kết quả đánh giá.")
