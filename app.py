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
st.set_page_config(page_title="Advanced Eco-Suite 2026", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛠️ Analysis Suite")
main_mode = st.sidebar.radio("Module Selector:", ["Panel Data Analysis", "Advanced Time Series"])
uploaded_file = st.sidebar.file_uploader("Upload Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()

    # =================================================================
    # MODULE 1: PANEL DATA (With Diagnostics & Robust Model)
    # =================================================================
    if main_mode == "Panel Data Analysis":
        st.header("🔬 Panel Data Expert System")
        with st.sidebar:
            id_col = st.selectbox("Entity (ID):", cols)
            time_col = st.selectbox("Time:", cols)
            y_col = st.selectbox("Dependent (Y):", cols)
            x_cols = st.multiselect("Independent (X):", cols)
            run_panel = st.button("Execute Panel Engine")

        if run_panel and x_cols:
            df_p = df.dropna(subset=[id_col, time_col, y_col] + x_cols)
            data_p = df_p.set_index([id_col, time_col])
            exog = sm.add_constant(data_p[x_cols])
            
            # Models
            m_fe = PanelOLS(data_p[y_col], exog, entity_effects=True).fit()
            m_re = RandomEffects(data_p[y_col], exog).fit()
            
            # Hausman Test
            b_diff = m_fe.params - m_re.params
            v_diff = m_fe.cov - m_re.cov
            h_stat = b_diff.T @ np.linalg.pinv(v_diff) @ b_diff
            p_hausman = 1 - stats.chi2.cdf(h_stat, len(b_diff))

            pt1, pt2, pt3, pt4 = st.tabs(["📊 Estimation", "🔍 Diagnostics", "🧬 Panel Causality", "💊 Final Robust"])
            
            with pt1:
                st.write(compare({"Fixed Effects": m_fe, "Random Effects": m_re}))
                st.metric("Hausman Test P-value", f"{p_hausman:.4f}")
                st.info("Decision: Use " + ("FE" if p_hausman < 0.05 else "RE"))

            with pt2:
                st.subheader("Diagnostic Tests")
                vif = pd.DataFrame({"VIF": [variance_inflation_factor(df_p[x_cols].values, i) for i in range(len(x_cols))]}, index=x_cols)
                st.table(vif)
                jb_stat, jb_p = stats.jarque_bera(m_fe.resids)
                st.write(f"Normality (Jarque-Bera) P-value: {jb_p:.4f}")

            with pt3:
                st.subheader("Pairwise Granger Causality (Panel Context)")
                st.info("Testing relationship between selected variables within the panel timeframe.")
                for x in x_cols:
                    gc_res = grangercausalitytests(df_p[[y_col, x]], maxlag=2, verbose=False)
                    p_v = gc_res[1][0]['ssr_chi2test'][1]
                    st.write(f"**{x}** Granger-causes **{y_col}**: P-value = {p_v:.4f}")

            with pt4:
                is_fe = p_hausman < 0.05
                final = PanelOLS(data_p[y_col], exog, entity_effects=is_fe).fit(cov_type='clustered', cluster_entity=True)
                st.text(final.summary)

    # =================================================================
    # MODULE 2: TIME SERIES (VAR / VECM / ARDL / IRF / FEVD / CAUSALITY)
    # =================================================================
    elif main_mode == "Advanced Time Series":
        st.header("📈 Multivariate Time Series Suite")
        
        # Filtering (Penting untuk data panel yang ingin di-analisis secara siri masa)
        filter_ent = st.sidebar.selectbox("Filter by Entity?", ["No"] + cols)
        if filter_ent != "No":
            ent_val = st.sidebar.selectbox(f"Select {filter_ent}", df[filter_ent].unique())
            df_ts = df[df[filter_ent] == ent_val].copy()
        else:
            df_ts = df.copy()

        with st.sidebar:
            ts_vars = st.multiselect("Select Variables (Min 2):", cols)
            lags = st.slider("Max Lags Selection:", 1, 8, 2)
            run_ts = st.button("Execute Time Series Engine")

        if run_ts and len(ts_vars) >= 2:
            ts_data = df_ts[ts_vars].dropna()
            
            tt1, tt2, tt3, tt4, tt5 = st.tabs(["📌 Stationarity", "🔗 Cointegration", "📉 Estimation", "🧨 Shock Analysis (IRF/FEVD)", "🧬 Causality"])

            with tt1:
                st.subheader("Unit Root Analysis (ADF)")
                adf_res = []
                for v in ts_vars:
                    r = adfuller(ts_data[v])
                    adf_res.append({"Variable": v, "P-value": r[1], "Status": "I(0)" if r[1] < 0.05 else "I(1)"})
                st.table(adf_res)

            with tt2:
                st.subheader("Johansen Cointegration Test")
                johansen = coint_johansen(ts_data, 0, lags-1)
                trace_stat = johansen.lr1
                crit_val = johansen.cvt[:, 1]
                num_coint = sum(trace_stat > crit_val)
                
                st.write("**Trace Statistic vs Critical Value (5%)**")
                st.table(pd.DataFrame({"Trace": trace_stat, "Critical": crit_val}))
                st.success(f"Number of Cointegrating Equations: {num_coint}")

            with tt3:
                if num_coint > 0:
                    st.subheader("VECM Estimation (Long-run Cointegration)")
                    model_res = VECM(ts_data, k_ar_diff=lags-1, coint_rank=num_coint).fit()
                    st.text(model_res.summary())
                else:
                    st.subheader("VAR Estimation (Short-run System)")
                    model_res = VAR(ts_data).fit(lags)
                    st.text(model_res.summary())

            with tt4:
                st.subheader("Impulse Response (IRF) & Variance Decomposition (FEVD)")
                if num_coint == 0:
                    # IRF Plot
                    irf = model_res.irf(10)
                    fig_irf = irf.plot(orth=True)
                    st.pyplot(fig_irf)
                    
                    # FEVD Plot
                    st.write("#### Forecast Error Variance Decomposition")
                    fevd = model_res.fevd(10)
                    st.pyplot(fevd.plot())
                else:
                    st.info("VECM Diagnostics: Showing short-run dynamics.")
                    st.write("For VECM, diagnostics are extracted from the error correction terms.")

            with tt5:
                st.subheader("Pairwise Granger Causality Matrix")
                for i in ts_vars:
                    for j in ts_vars:
                        if i != j:
                            gc = grangercausalitytests(ts_data[[i, j]], maxlag=lags, verbose=False)
                            p_v = gc[lags][0]['ssr_chi2test'][1]
                            st.write(f"**{j}** → **{i}**: P-value = {p_v:.4f} " + ("✅ Significant" if p_v < 0.05 else "❌"))
                            
                # Tambahan: ARDL Option
                with st.expander("Optional: Run Single-Equation ARDL"):
                    y_ardl = ts_vars[0]
                    x_ardl = ts_vars[1:]
                    sel_ardl = ardl_select_order(ts_data[y_ardl], lags, ts_data[x_ardl], lags, ic='aic')
                    res_ardl = sel_ardl.model.fit()
                    st.text(res_ardl.summary())

else:
    st.info("Upload CSV and select a module to begin.")
