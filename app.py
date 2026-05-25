if st.button("🔍 DỰ ĐOÁN NGAY", type="primary"):
    # ==================== CHUẨN BỊ DỮ LIỆU ĐÚNG CÁCH ====================
    input_data = {
        'credit_sco': credit_score,
        'age': age,
        'balance': balance,
        'monthly_ir': monthly_income,
        'tenure_ye': tenure,
        'nums_card': nums_card,
        'nums_service': nums_service,
        'engagement_score': engagement_score,
        'active_member': 1 if active_member == "Có" else 0,
        'married': 1,                    # mặc định
        'last_transaction_month': 3,     # mặc định
        'risk_score': 0.2,               # tạm thời, sẽ tính sau nếu cần
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Danh sách cột scaler mong đợi (quan trọng!)
    cols_to_scale = [
        'credit_sco', 'age', 'balance', 'monthly_ir', 
        'nums_card', 'nums_service', 'engagement_score', 
        'tenure_ye', 'risk_score'
    ]
    
    # Scale chỉ các cột cần thiết
    input_scaled = input_df.copy()
    
    try:
        input_scaled[cols_to_scale] = scaler.transform(input_scaled[cols_to_scale])
        
        # Predict
        proba = model.predict_proba(input_scaled)[0][1]
        risk_score_percent = proba * 100
        
        # ==================== HIỂN THỊ KẾT QUẢ ====================
        st.success("**DỰ ĐOÁN HOÀN TẤT**")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric(label="**RISK SCORE**", value=f"{risk_score_percent:.1f}%")
        
        with col_res2:
            if risk_score_percent < 30:
                st.markdown("### 🟢 **LOW RISK**")
            elif risk_score_percent <= 70:
                st.markdown("### 🟡 **MEDIUM RISK**")
            else:
                st.markdown("### 🔴 **HIGH RISK**")
        
        if risk_score_percent >= 50:
            st.error("⚠️ **Khách hàng có nguy cơ rời bỏ**")
        else:
            st.success("✅ **Khách hàng có khả năng tiếp tục sử dụng dịch vụ**")
        
        # Khuyến nghị
        st.markdown("### 💡 Khuyến nghị")
        if risk_score_percent >= 70:
            st.error("🚨 Cần liên hệ khẩn cấp trong 24h - Ưu đãi đặc biệt, tư vấn cá nhân")
        elif risk_score_percent >= 40:
            st.warning("⚠️ Nên chăm sóc chủ động: Gọi điện, tặng voucher, nâng lãi suất")
        else:
            st.success("✅ Duy trì mối quan hệ tốt, theo dõi định kỳ")
            
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")
        st.info("💡 Kiểm tra xem model và scaler có khớp với các cột không?")
