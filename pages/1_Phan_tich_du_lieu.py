import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Phân tích dữ liệu (EDA)")
# Giả sử bạn có file data.csv, nếu chưa có hãy tạo dataframe mẫu
st.write("### Dataset Preview")
# df = pd.read_csv("data.csv") 
# st.dataframe(df.head())

st.write("### Correlation Heatmap")
fig, ax = plt.subplots()
# sns.heatmap(df.corr(), ax=ax)
st.pyplot(fig)
