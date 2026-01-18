import streamlit as st
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
from linearmodels.panel import compare
import statsmodels.api as sm
from scipy import stats

st.set_page_config(page_title="Panel Data Expert", layout="wide")
st.title("🔬 Kalkulator Panel Data Komprehensif")

# --- SIDEBAR & UPLOAD ---
uploaded_file = st.file_uploader("Muat naik CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()
    
    with st.sidebar:
        st.header("Konfigurasi Data")
        id_col = st.selectbox("Unit (Entity):", cols)
        time_col = st.selectbox("Masa (Time):", cols)
        y_col = st.selectbox("Y (Dependent):", cols)
        x_cols = st.multiselect("X (Independent):", cols)
        st.divider()
        run_btn = st.button("JALANKAN SEMUA ANALISIS")

    if run_btn:
        # Preparation
        data = df.set_index([id_col, time_col])
        exog = sm.add_constant(data[x_cols])
        
        # --- 1. ESTIMASI MODEL ---
        st.header("1. Estimasi Model")
        
        # Pooled OLS
        model_pooled = PooledOLS(data[y_col], exog).fit()
        
        # FEM (One-way: Entity)
        model_fe_1 = PanelOLS(data[y_col], exog, entity_effects=True).fit()
        
        # FEM (Two-way: Entity & Time)
        model_fe_2 = PanelOLS(data[y_col], exog, entity_effects=True, time_effects=True).fit()
        
        # REM (One-way)
        model_re_1 = RandomEffects(data[y_col], exog).fit()
        
        # Paparkan Perbandingan
        st.write(compare({"Pooled OLS": model_pooled, "FEM (1-Way)": model_fe_1, 
                          "FEM (2-Way)": model_fe_2, "REM": model_re_1}))

        # --- 2. MODEL SELECTION ---
        st.header("2. Pemilihan Model (Model Selection)")
        
        # Hausman Test (Simple Version: FE vs RE)
        # Formula: (b_fe - b_re)' * [Var(b_fe) - Var(b_re)]^-1 * (b_fe - b_re)
        b_fe = model_fe_1.params
        b_re = model_re_1.params
        v_fe = model_fe_1.cov
        v_re = model_re_1.cov
        
        diff = b_fe - b_re
        diff_cov = v_fe - v_re
        # Kita hanya ambil subset X (tanpa constant) untuk Hausman
        hausman_stat = diff.dot(np.linalg.inv(diff_cov)).dot(diff)
        dof = len(b_fe)
        p_val = 1 - stats.chi2.cdf(hausman_stat, dof)
        
        st.subheader("Hausman Test (FEM vs REM)")
        st.write(f"Chi-Sq Stat: {hausman_stat:.4f}, P-value: {p_val:.4f}")
        if p_val < 0.05:
            st.success("Keputusan: Gunakan Fixed Effects (FEM)")
        else:
            st.success("Keputusan: Gunakan Random Effects (REM)")

        # --- 3. DIAGNOSTIC & RAWATAN ---
        st.header("3. Diagnostic & Rawatan (Robustness)")
        
        st.info("Menjalankan Model dengan Robust Standard Errors (Rawatan untuk Heteroscedasticity & Autocorrelation)")
        
        # Rawatan: Menggunakan 'clustered' standard errors
        model_robust = PanelOLS(data[y_col], exog, entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
        
        st.subheader("Keputusan Akhir (Model Dirawat/Robust)")
        st.text(model_robust.summary)
