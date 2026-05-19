import streamlit as st

st.set_page_config(page_title="BIDV Churn Prediction", layout="wide")

# Logo BIDV (Giả sử bạn có file logo trong folder assets)
# st.sidebar.image("assets/bidv_logo.png", width=200)

st.title("🏦 HỆ THỐNG DỰ BÁO RỦI RO CHURN - BIDV")
st.markdown("### Đề tài: Ứng dụng AI trong quản trị khách hàng")

st.info("Hệ thống hỗ trợ RM phát hiện sớm khách hàng có nguy cơ rời bỏ dịch vụ.")

# Vẽ flow quy trình bằng cột
col1, col2, col3, col4 = st.columns(4)
col1.metric("Bước 1", "Dữ liệu khách hàng")
col2.metric("Bước 2", "Xử lý AI")
col3.metric("Bước 3", "Dự đoán Churn")
col4.metric("Bước 4", "Cảnh báo RM")

st.markdown("---")
st.write("Sử dụng Menu bên trái để truy cập các tính năng.")
