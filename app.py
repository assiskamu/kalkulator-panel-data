import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, compare
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.ardl import ARDL, ardl_select_order
import statsmodels.api as sm
from scipy import stats

st.set_page_config(page_title="Eco-Analyzer Pro", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚀 Navigation")
analysis_mode = st.sidebar.radio("Select Analysis Type:", ["Panel Data Analysis", "Time Series (ARDL & Causality)"])

# --- FILE UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()

    # ------------------------------------------------------------------
    # MODE 1: PANEL DATA (SAMA SEPERTI SEBELUM INI)
    # ------------------------------------------------------------------
    if analysis_mode == "Panel Data Analysis":
        st.header("🔬 Panel Data Analysis Mode")
        st.dataframe(df.head(5))
        
        with st.sidebar:
            id_col = st.selectbox("Entity (ID):", cols)
            time_col = st.selectbox("Time:", cols)
            y_col = st.selectbox("Y Variable:", cols)
            x_cols = st.multiselect("X Variables:", cols)
            run_panel = st.button("Run Panel Analysis")

        if run_panel:
            # (Kod Panel Data anda yang sebelum ini diletakkan di sini)
            # Dipendekkan untuk ruang, pastikan anda kekalkan logik Tab 1-4 yang lepas
            st.success("Panel Data Engine Ready. (Execute logic here)")

    # ------------------------------------------------------------------
    # MODE 2: TIME SERIES (ARDL & CAUSALITY)
    # ------------------------------------------------------------------
    elif analysis_mode == "Time Series (ARDL & Causality)":
        st.header("📈 Time Series Analysis: ARDL & Causality")
        st.info("Note: Ensure your data is for a single entity (e.g., one country) or filter your data first.")
        
        with st.sidebar:
            y_ts = st.selectbox("Dependent Variable (Y):", cols, key="y_ts")
            x_ts = st.selectbox("Independent Variable (X):", cols, key="x_ts")
            max_lag = st.slider("Select Max Lag:", 1, 10, 4)
            run_ts = st.button("Run Time Series Analysis")

        if run_ts:
            tab_ts1, tab_ts2, tab_ts3 = st.tabs(["📌 Stationarity", "🔗 ARDL Model", "🧬 Granger Causality"])

            with tab_ts1:
                st.subheader("1. Stationarity Test (ADF)")
                def check_stationarity(series, name):
                    res = adfuller(series.dropna())
                    st.write(f"**Variable: {name}**")
                    st.write(f"ADF Stat: {res[0]:.4f} | P-value: {res[1]:.4f}")
                    if res[1] < 0.05:
                        st.success(f"{name} is Stationary I(0)")
                    else:
                        st.warning(f"{name} is Non-Stationary. Check I(1) by differencing.")

                check_stationarity(df[y_ts], y_ts)
                check_stationarity(df[x_ts], x_ts)
                st.write("**Note:** ARDL is valid if variables are I(0), I(1), or a mix. It fails if any are I(2).")

            with tab_ts2:
                st.subheader("2. ARDL Model Estimation")
                # Auto lag selection
                sel_res = ardl_select_order(df[y_ts], max_lag, df[[x_ts]], max_lag, ic='aic')
                st.write(f"Optimal Lags Selected (AIC): {sel_res.model_order}")
                
                model_ardl = sel_res.model.fit()
                st.text(model_ardl.summary())
                
                # Bounding Test Concept
                st.write("**Bound Test Interpretation:** If F-Stat > Upper Bound, Long-run relationship exists.")

            with tab_ts3:
                st.subheader("3. Granger Causality Test")
                st.write(f"Testing if **{x_ts}** causes **{y_ts}**")
                
                # Granger test needs a 2D array
                gc_data = df[[y_ts, x_ts]].dropna()
                gc_res = grangercausalitytests(gc_data, maxlag=max_lag, verbose=False)
                
                # Displaying results for each lag
                for lag in range(1, max_lag + 1):
                    p_val = gc_res[lag][0]['ssr_chi2test'][1]
                    st.write(f"Lag {lag}: P-value = {p_val:.4f}")
                    if p_val < 0.05:
                        st.success(f"At lag {lag}, {x_ts} GRANGER CAUSES {y_ts}")
                    else:
                        st.write(f"At lag {lag}, no causality found.")

else:
    st.info("Please upload a CSV file to begin.")
