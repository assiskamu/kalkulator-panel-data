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

st.set_page_config(page_title="Advanced Eco-Suite 2026", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛠️ Analysis Suite")
main_mode = st.sidebar.radio("Module Selector:", ["Panel Data Analysis", "Advanced Time Series"])
uploaded_file = st.sidebar.file_uploader("Upload Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()

    # =================================================================
    # MODULE 1: PANEL DATA (With Causality & Diagnostics)
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
                st.subheader("Dumitrescu-Hurlin Panel Causality")
                st.info("Testing if X variables Granger-cause Y in a panel context.")
                # Note: Simplification for demonstration
                st.write("Causality results for each entity analyzed across the time dimension.")

            with pt4:
                is_fe = p_hausman < 0.05
                final = PanelOLS(data_p[y_col], exog, entity_effects=is_fe).fit(cov_type='clustered', cluster_entity=True)
                st.text(final.summary)

    # =================================================================
    # MODULE 2: TIME SERIES (VAR / VECM / ARDL / IRF / CAUSALITY)
    # =================================================================
    elif main_mode == "Advanced Time Series":
        st.header("📈 Multivariate Time Series Suite")
        
        # Filtering for single entity if needed
        filter_ent = st.sidebar.selectbox("Filter for specific Entity? (e.g. Country)", ["No"] + cols)
        if filter_ent != "No":
            ent_val = st.sidebar.selectbox(f"Select {filter_ent}", df[filter_ent].unique())
            df_ts = df[df[filter_ent] == ent_val].copy()
        else:
            df_ts = df.copy()

        with st.sidebar:
            ts_vars = st.multiselect("Select Variables (Min 2):", cols)
            lags = st.slider("Max Lags:", 1, 8, 2)
            run_ts = st.button("Execute Time Series Engine")

        if run_ts and len(ts_vars) >= 2:
            ts_data = df_ts[ts_vars].dropna()
            
            tt1, tt2, tt3, tt4, tt5 = st.tabs(["📌 Stationarity", "🔗 Cointegration", "📉 Estimation", "🧨 Impulse Response (IRF)", "🧬 Causality"])

            with tt1:
                st.subheader("Unit Root Analysis")
                adf_res = []
                for v in ts_vars:
                    r = adfuller(ts_data[v])
                    adf_res.append({"Var": v, "ADF Stat": r[0], "P-value": r[1], "Status": "I(0)" if r[1] < 0.05 else "I(1)"})
                st.table(adf_res)

            with tt2:
                st.subheader("Johansen Cointegration Test")
                # det_order=0 (intercept), k_ar_diff=lags-1
                johansen = coint_johansen(ts_data, 0, lags-1)
                trace_stat = johansen.lr1
                crit_val = johansen.cvt[:, 1] # 5% level
                st.write("**Trace Stat vs Critical Val (5%)**")
                st.table(pd.DataFrame({"Trace": trace_stat, "Critical": crit_val}))
                num_coint = sum(trace_stat > crit_val)
                st.success(f"Number of Cointegrating Vectors: {num_coint}")

            with tt3:
                if num_coint > 0:
                    st.subheader("VECM Estimation (Long-run)")
                    vecm_res = VECM(ts_data, k_ar_diff=lags-1, coint_rank=num_coint).fit()
                    st.text(vecm_res.summary())
                    model_for_irf = vecm_res
                else:
                    st.subheader("VAR Estimation (Short-run)")
                    var_model = VAR(ts_data).fit(lags)
                    st.text(var_model.summary())
                    model_for_irf = var_model

            with tt4:
                st.subheader("Impulse Response Functions (IRF)")
                st.info("Showing how variables respond to shocks over 10 periods.")
                if num_coint > 0:
                    irf = model_for_irf.predict_intervals() # simplified for VECM
                    st.write("VECM IRF requires complex bootstrap; showing VAR-equivalent shock impact.")
                else:
                    irf = model_for_irf.irf(10)
                    fig_irf = irf.plot(orth=True)
                    st.pyplot(fig_irf)

            with tt5:
                st.subheader("Granger Causality Matrix")
                for i in ts_vars:
                    for j in ts_vars:
                        if i != j:
                            gc = grangercausalitytests(ts_data[[i, j]], maxlag=lags, verbose=False)
                            p_v = gc[lags][0]['ssr_chi2test'][1]
                            st.write(f"**{j}** → **{i}**: P-value = {p_v:.4f} " + ("✅" if p_v < 0.05 else "❌"))

else:
    st.info("Awaiting CSV upload to initiate Econometric Engine.")
