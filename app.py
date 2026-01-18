import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, compare
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats

# Konfigurasi Halaman
st.set_page_config(page_title="Eco-Suite Pro 2026", layout="wide")

# --- SIDEBAR: PILIHAN MODUL ---
st.sidebar.title("🚀 MODEL SELECTOR")
# Pastikan anda nampak butang radio ini nanti
main_mode = st.sidebar.radio(
    "Choose Analysis Type:", 
    ["Panel Data Analysis", "Advanced Time Series (VAR/VECM)"]
)

st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("Step 1: Upload CSV Data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()

    if main_mode == "Panel Data Analysis":
        st.header("🔬 Panel Data Expert System")
        st.info("Currently in Panel Data Mode (Pooled, FE, RE)")
        
        with st.sidebar:
            id_col = st.selectbox("Entity (ID):", cols)
            time_col = st.selectbox("Time:", cols)
            y_col = st.selectbox("Dependent (Y):", cols)
            x_cols = st.multiselect("Independent (X):", cols)
            run_panel = st.button("Run Panel Analysis")

        if run_panel and x_cols:
            df_p = df.dropna(subset=[id_col, time_col, y_col] + x_cols)
            data_p = df_p.set_index([id_col, time_col])
            exog = sm.add_constant(data_p[x_cols])
            
            m_fe = PanelOLS(data_p[y_col], exog, entity_effects=True).fit()
            m_re = RandomEffects(data_p[y_col], exog).fit()
            
            pt1, pt2, pt3, pt4 = st.tabs(["📊 Estimation", "🔍 Diagnostics", "🧬 Causality", "💊 Final Robust"])
            
            with pt1:
                st.write(compare({"Fixed Effects": m_fe, "Random Effects": m_re}))
            with pt2:
                vif = pd.DataFrame({"VIF": [variance_inflation_factor(df_p[x_cols].values, i) for i in range(len(x_cols))]}, index=x_cols)
                st.table(vif)
            with pt3:
                for x in x_cols:
                    gc = grangercausalitytests(df_p[[y_col, x]], maxlag=2, verbose=False)
                    st.write(f"**{x}** → **{y_col}**: P-value = {gc[1][0]['ssr_chi2test'][1]:.4f}")
            with pt4:
                final = PanelOLS(data_p[y_col], exog, entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
                st.text(final.summary)

    elif main_mode == "Advanced Time Series (VAR/VECM)":
        st.header("📈 Advanced Time Series Suite")
        st.info("Currently in Time Series Mode (VAR, VECM, ARDL)")

        # Filter jika data asal adalah panel
        filter_ent = st.sidebar.selectbox("Filter specific entity? (e.g. Country)", ["None"] + cols)
        if filter_ent != "None":
            val = st.sidebar.selectbox(f"Select {filter_ent}:", df[filter_ent].unique())
            df_ts = df[df[filter_ent] == val].copy()
        else:
            df_ts = df.copy()

        with st.sidebar:
            ts_vars = st.multiselect("Select Variables (Min 2):", cols)
            lags = st.slider("Max Lags:", 1, 8, 2)
            run_ts = st.button("Run Time Series Analysis")

        if run_ts and len(ts_vars) >= 2:
            ts_data = df_ts[ts_vars].dropna()
            tt1, tt2, tt3, tt4, tt5 = st.tabs(["📌 Stationarity", "🔗 Cointegration", "📉 Estimation", "🧨 IRF & FEVD", "🧬 Causality"])

            with tt1:
                adf_res = []
                for v in ts_vars:
                    r = adfuller(ts_data[v])
                    adf_res.append({"Var": v, "P-value": r[1], "Status": "I(0)" if r[1] < 0.05 else "I(1)"})
                st.table(adf_res)

            with tt2:
                johansen = coint_johansen(ts_data, 0, lags-1)
                num_coint = sum(johansen.lr1 > johansen.cvt[:, 1])
                st.metric("Number of Cointegrating Vectors", num_coint)

            with tt3:
                if num_coint > 0:
                    model = VECM(ts_data, k_ar_diff=lags-1, coint_rank=num_coint).fit()
                    st.text(model.summary())
                else:
                    model = VAR(ts_data).fit(lags)
                    st.text(model.summary())

            with tt4:
                if num_coint == 0:
                    irf = model.irf(10)
                    st.pyplot(irf.plot(orth=True))
                    fevd = model.fevd(10)
                    st.pyplot(fevd.plot())
                else:
                    st.write("VECM Diagnostics: Cointegration detected. Showing IRF for stationary series.")

            with tt5:
                for i in ts_vars:
                    for j in ts_vars:
                        if i != j:
                            gc = grangercausalitytests(ts_data[[i, j]], maxlag=lags, verbose=False)
                            st.write(f"**{j}** → **{i}**: P-value = {gc[lags][0]['ssr_chi2test'][1]:.4f}")
else:
    st.info("Please upload your data file to begin.")
