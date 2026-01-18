import streamlit as st
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm

st.set_page_config(page_title="Panel Data Calc", layout="wide")

st.title("📊 Kalkulator Analisis Panel Data")
st.markdown("Aplikasi ini membantu anda menjalankan regresi Fixed Effects dan Random Effects.")

# Bahagian Upload
uploaded_file = st.file_uploader("Muat naik fail CSV anda", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Pratonton Data")
    st.dataframe(df.head())

    # Tetapan Kolum
    cols = df.columns.tolist()
    st.sidebar.header("Tetapan Model")
    id_col = st.sidebar.selectbox("Pilih Unit (Entity/ID):", cols)
    time_col = st.sidebar.selectbox("Pilih Masa (Time/Year):", cols)
    y_col = st.sidebar.selectbox("Pembolehubah Bersandar (Y):", cols)
    x_cols = st.sidebar.multiselect("Pembolehubah Bebas (X):", cols)

    if st.sidebar.button("Jalankan Analisis"):
        if not x_cols:
            st.error("Sila pilih sekurang-kurangnya satu pembolehubah X.")
        else:
            # Sediakan data
            df[time_col] = pd.to_datetime(df[time_col]).dt.year # Pastikan tahun dalam format betul
            data = df.set_index([id_col, time_col])
            
            # Tambah constant
            exog = sm.add_constant(data[x_cols])
            
            # Model Fixed Effects
            fe_model = PanelOLS(data[y_col], exog, entity_effects=True)
            fe_res = fe_model.fit()
            
            st.success("Analisis Berjaya!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Fixed Effects Results")
                st.text(fe_res.summary)
