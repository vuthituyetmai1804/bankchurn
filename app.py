import streamlit as st
import pandas as pd
import joblib

# Sử dụng decorator để cache model
@st.cache_resource
def load_model():
    # Đảm bảo file .pkl nằm cùng thư mục với app.py
    return joblib.load("bidv_churn_modeltuning.pkl")

# Gọi hàm load model
try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Không thể tải model: {e}")
    st.stop()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BIDV Churn Prediction",
    page_icon="🏦",
    layout="wide"
)
# =========================================================
# CUSTOM CSS 
import streamlit as st
import streamlit.components.v1 as components

# Cấu hình trang rộng để hiển thị dashboard đẹp hơn
st.set_page_config(layout="wide")

# Khai báo UI bằng HTML và CSS hiện đại
html_code = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <link href="https://googleapis.com" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: #f4f7f6; color: #2d3748; padding: 30px; }
        
        /* Banner tối giản, sang trọng */
        .header-banner {
            background: linear-gradient(135deg, #006A4E 0%, #004D38 100%);
            padding: 40px;
            border-radius: 20px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 106, 78, 0.15);
            margin-bottom: 30px;
        }
        .header-banner h1 { font-size: 28px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 8px; }
        .header-banner p { font-size: 14px; color: #a3e2d1; font-weight: 300; }

        /* Khung Grid chia 2 cột rõ ràng, gọn gàng */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin-bottom: 30px;
        }

        /* Thẻ Card chứa form */
        .card {
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.03);
            border: 1px solid #e2e8f0;
        }
        .card-title {
            font-size: 18px;
            font-weight: 600;
            color: #1a202c;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .card-title::before {
            content: ''; display: inline-block; width: 4px; height: 18px; background: #006A4E; border-radius: 2px;
        }

        /* Form control tinh tế */
        .form-group { margin-bottom: 22px; }
        .form-group label { display: block; font-size: 13px; font-weight: 500; color: #718096; margin-bottom: 8px; }
        
        /* Custom input số */
        .input-control {
            width: 100%; padding: 12px 16px; border-radius: 10px; border: 1px solid #cbd5e0;
            font-size: 15px; transition: all 0.2s; background: #f8fafc;
        }
        .input-control:focus { outline: none; border-color: #006A4E; background: white; box-shadow: 0 0 0 3px rgba(0,106,78,0.1); }

        /* Thay thế slider thô bằng slider thanh mảnh công nghệ */
        .slider-container { display: flex; align-items: center; gap: 15px; }
        .slider-control {
            flex: 1; -webkit-appearance: none; height: 6px; background: #e2e8f0; border-radius: 3px; outline: none;
        }
        .slider-control::-webkit-slider-thumb {
            -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; background: #006A4E; cursor: pointer;
            border: 3px solid white; box-shadow: 0 2px 6px rgba(0,0,0,0.15); transition: 0.1s;
        }
        .slider-control::-webkit-slider-thumb:hover { transform: scale(1.1); }
        .slider-value { font-size: 14px; font-weight: 600; color: #006A4E; min-width: 40px; text-align: right; }

        /* Custom Radio Button dạng Switch cao cấp */
        .radio-group { display: flex; gap: 12px; }
        .radio-btn {
            flex: 1; position: relative; text-align: center;
        }
        .radio-btn input { position: absolute; opacity: 0; width: 0; height: 0; }
        .radio-label {
            display: block; padding: 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
            font-size: 14px; cursor: pointer; transition: all 0.2s; font-weight: 500;
        }
        .radio-btn input:checked + .radio-label {
            background: rgba(0, 106, 78, 0.08); border-color: #006A4E; color: #006A4E;
        }

        /* Nút bấm hiệu ứng Đẳng cấp */
        .btn-submit {
            width: 100%; padding: 16px; background: #006A4E; color: white; border: none; border-radius: 12px;
            font-size: 16px; font-weight: 600; cursor: pointer; transition: all 0.3s;
            box-shadow: 0 4px 12px rgba(0, 106, 78, 0.2); margin-top: 10px;
        }
        .btn-submit:hover { background: #004D38; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(0, 106, 78, 0.3); }
    </style>
</head>
<body>

    <div class="header-banner">
        <h1>HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ</h1>
        <p>Ứng dụng Mô hình Cây quyết định trong Quản trị Rủi ro Ngân hàng BIDV</p>
    </div>

    <div class="dashboard-grid">
        <!-- Cột 1 -->
        <div class="card">
            <div class="card-title">Thông tin khách hàng</div>
            
            <div class="form-group">
                <label>Tuổi</label>
                <div class="slider-container">
                    <input type="range" class="slider-control" min="18" max="100" value="35" oninput="this.nextElementSibling.innerText = this.value">
                    <span class="slider-value">35</span>
                </div>
            </div>

            <div class="form-group">
                <label>Điểm tín dụng</label>
                <div class="slider-container">
                    <input type="range" class="slider-control" min="300" max="850" value="650" oninput="this.nextElementSibling.innerText = this.value">
                    <span class="slider-value">650</span>
                </div>
            </div>

            <div class="form-group">
                <label>Số dư tài khoản (VND)</label>
                <input type="number" class="input-control" value="5000000">
            </div>
        </div>

        <!-- Cột 2 -->
        <div class="card">
            <div class="card-title">Chỉ số tương tác</div>

            <div class="form-group">
                <label>Thu nhập hàng tháng (VND)</label>
                <input type="number" class="input-control" value="15000000">
            </div>
            
            <div class="form-group">
                <label>Số lượng dịch vụ sử dụng</label>
                <div class="slider-container">
                    <input type="range" class="slider-control" min="1" max="10" value="3" oninput="this.nextElementSibling.innerText = this.value">
                    <span class="slider-value">3</span>
                </div>
            </div>

            <div class="form-group">
                <label>Điểm tương tác app</label>
                <div class="slider-container">
                    <input type="range" class="slider-control" min="1" max="100" value="74" oninput="this.nextElementSibling.innerText = this.value">
                    <span class="slider-value">74</span>
                </div>
            </div>

            <div class="form-group">
                <label>Hoạt động gần đây</label>
                <div class="radio-group">
                    <div class="radio-btn">
                        <input type="radio" id="active-yes" name="active" checked>
                        <label for="active-yes" class="radio-label">Có hoạt động</label>
                    </div>
                    <div class="radio-btn">
                        <input type="radio" id="active-no" name="active">
                        <label for="active-no" class="radio-label">Không</label>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <button class="btn-submit">BẮT ĐẦU DỰ ĐOÁN HÀNH VI</button>

</body>
</html>
"""

# Nhúng vào Streamlit với chiều cao phù hợp
components.html(html_code, height=750, scrolling=True)
# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="header-box">
    <div class="header-title">🏦 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ</div>
    <div class="header-sub">Ứng dụng Mô hình Cây quyết định trong Quản trị Rủi ro Ngân hàng BIDV</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("## 📋 Nhập thông tin khách hàng")

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
    # OUTPUT GRAPHICS
    # =====================================================
    st.markdown("---")
    st.markdown("# 📊 KẾT QUẢ PHÂN TÍCH")

    # =====================================================
    # METRICS
    # =====================================================
    colA, colB, colC = st.columns(3)

    with colA:
        st.metric(
            label="RISK SCORE",
            value=f"{risk_percent}%"
        )

    with colB:
        st.metric(
            label="RISK LEVEL",
            value=risk_level
        )

    with colC:
        st.metric(
            label="PREDICTION",
            value="CHURN" if risk_percent >= 50 else "STAY"
        )

    # =====================================================
    # PROGRESS BAR
    # =====================================================
    st.progress(int(risk_percent))

    # =====================================================
    # PREDICTION RESULT BOX
    # =====================================================
    st.markdown(f"""
    <div class="result-box">
        <h2 style="color:{color}; text-align: center; margin: 0;">
            {prediction_text}
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # RECOMMENDATION BOX
    # =====================================================
    st.markdown(f"""
    <div class="recommend-box"
    style="
        background-color:white;
        border-left:8px solid {color};
        margin-top:20px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.08);
    ">
    <h3 style="margin-top: 0;">🎯 Khuyến nghị hành động:</h3>
    <p style="margin-bottom: 0;">{recommendation}</p>
    </div>
    """, unsafe_allow_html=True)
