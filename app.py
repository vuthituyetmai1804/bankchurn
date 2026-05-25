import pandas as pd
import joblib
import time
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# LOAD MODEL (Đã bỏ hoàn toàn Scaler)
# =========================================================
model = joblib.load("bidv_churn_modeltuning.pkl")

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    /* ... (Giữ nguyên các style background, header, button của bạn) ... */

    /* Thêm các class bị thiếu để render đúng */
    .ai-result-card { 
        background: #072c2b; 
        border-radius: 28px; 
        padding: 40px; 
        color: white; 
    }
    .ai-circle { 
        width: 200px; height: 200px; border: 6px solid; border-radius: 50%; 
        display: flex; align-items: center; justify-content: center; margin: 20px auto; 
    }
    .ai-percent { font-size: 50px; font-weight: bold; }
    .ai-risk-title { text-align: center; font-size: 30px; font-weight: bold; margin: 20px 0; }
    .ai-sub { text-align: center; color: #aaa; margin-bottom: 20px; }
    .ai-mini-card { background: #1e1e1e; padding: 20px; border-radius: 15px; border-left: 5px solid #00ffae; }
    .ai-mini-title { color: #888; font-size: 16px; }
    .ai-mini-content { color: white; font-size: 18px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)
# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="header-box">
    <div class="header-title">
        🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ
    </div>
    <div class="header-sub">
        Ứng dụng Mô hình Cây quyết định trong Quản trị Rủi ro Ngân hàng BIDV
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("""
<div class="modern-card">
<h2>📋 Thông tin khách hàng</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# =========================================================
# LEFT COLUMN
# =========================================================
with col1:
    age = st.slider(
        "🎂 Tuổi",
        20, 80, 35
    )

    credit_sco = st.slider(
        "💳 Điểm tín dụng",
        495, 800, 650
    )

    balance = st.number_input(
        "💰 Số dư tài khoản (VND)",
        min_value=0,
        value=50000000,
        step=1000000
    )

# =========================================================
# RIGHT COLUMN
# =========================================================
with col2:
    monthly_ir = st.number_input(
        "💵 Thu nhập hàng tháng (VND)",
        min_value=0,
        value=15000000,
        step=1000000
    )

    nums_service = st.slider(
        "🏦 Số lượng dịch vụ sử dụng",
        1, 8, 3
    )

    engagement_score = st.slider(
        "🤝 Điểm tương tác app",
        0, 100, 50
    )

    active_text = st.radio(
        "📱 Hoạt động gần đây",
        ["Có", "Không"]
    )

# =========================================================
# ENCODE INPUT
# =========================================================
active_member = 1 if active_text == "Có" else 0

# =========================================================
# PREDICT BUTTON
# =========================================================
predict_btn = st.button("🔍 DỰ ĐOÁN NGAY")

# =========================================================
# PREDICTION LOGIC
# =========================================================
if predict_btn:
    # =====================================================
    # 1. TẠO DATAFRAME VỚI ĐÚNG 7 CỘT THEO ĐÚNG THỨ TỰ YÊU CẦU
    # =====================================================
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
    
    # Đảm bảo thứ tự cột gửi vào mô hình chuẩn xác 100%
    final_input = input_df[features_order]

    # =====================================================
    # 2. DỰ ĐOÁN TRỰC TIẾP KHÔNG QUA SCALER
    # =====================================================
    risk_score = model.predict_proba(final_input)[0][1]
    risk_percent = round(risk_score * 100, 2)

    # =====================================================
    # RISK LEVEL
    # =====================================================
    if risk_percent < 30:
        risk_level = "🟢 LOW RISK"
        prediction_text = "✅ Khách hàng có khả năng tiếp tục sử dụng dịch vụ"
        recommendation = "✅ Duy trì mối quan hệ tốt và tiếp tục chăm sóc định kỳ."
        color = "green"
    elif risk_percent <= 70:
        risk_level = "🟡 MEDIUM RISK"
        prediction_text = "⚠️ Khách hàng có nguy cơ rời bỏ"
        recommendation = "📞 Nên chăm sóc chủ động: Gọi điện tư vấn, tặng ưu đãi lãi suất, voucher."
        color = "orange"
    else:
        risk_level = "🔴 HIGH RISK"
        prediction_text = "⚠️ Khách hàng có nguy cơ rời bỏ"
        recommendation = "🚨 Cần liên hệ khẩn cấp trong 24h để giữ chân khách hàng."
        color = "red"

    # =====================================================
    # =====================================================
    # AI RESULT LAYOUT
    # =====================================================

    left_panel, right_panel = st.columns([1.15, 0.85])

    # =====================================================
    # LEFT PANEL
    # =====================================================
    
    with left_panel:
    
        st.markdown("""
        <div class="custom-card">
        <h2>📋 Thông tin khách hàng</h2>
        <p style="color:#5f6c7b;">
        AI Banking Analytics • BIDV Churn Prediction
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # =====================================================
    # RIGHT PANEL
    # =====================================================
    with right_panel:    
        html_code = f"""
        <div class="ai-result-card">
            <h2 style="color:white;text-align:center;">KẾT QUẢ DỰ ĐOÁN</h2>
            <div class="ai-circle" style="border-color:{glow}; box-shadow:0 0 20px {glow};">
                <div class="ai-percent">{risk_percent}%</div>
            </div>
            <div class="ai-risk-title">{risk_name}</div>
            <div class="ai-sub">Khách hàng có khả năng rời bỏ dịch vụ</div>
            <div class="ai-mini-card">
                <div class="ai-mini-title">🎯 Khuyến nghị hành động</div>
                <div class="ai-mini-content">{recommendation}</div>
            </div>
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)
