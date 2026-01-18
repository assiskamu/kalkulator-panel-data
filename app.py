import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, compare
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats

# Konfigurasi halaman
st.set_page_config(page_title="Advanced Panel Analyzer", layout="wide")

st.title("🔬 Professional Econometric Panel Data System")
st.markdown("A comprehensive tool for Pooled OLS, Fixed Effects, and Random Effects Analysis.")

# --- 1. MUAT NAIK FAIL ---
uploaded_file = st.file_uploader("Step 1: Upload CSV File (Long Panel Format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(5))
    cols = df.columns.tolist()
    
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        id_col = st.selectbox("Entity (ID) Column:", cols)
        time_col = st.selectbox("Time (Year/Period) Column:", cols)
        y_col = st.selectbox("Dependent Variable (Y):", cols)
        x_cols = st.multiselect("Independent Variables (X):", cols)
        st.divider()
        run_btn = st.button("RUN FULL ECONOMETRIC ANALYSIS")

    if run_btn:
        if not x_cols:
            st.error("Please select at least one X variable.")
        else:
            try:
                # Pembersihan Data
                df_clean = df.dropna(subset=[id_col, time_col, y_col] + x_cols)
                
                # Semakan Duplikasi
                duplicates = df_clean.duplicated(subset=[id_col, time_col]).sum()
                if duplicates > 0:
                    st.error(f"Duplicate Error: {duplicates} entries found for the same Entity-Time index.")
                else:
                    data = df_clean.set_index([id_col, time_col])
                    exog = sm.add_constant(data[x_cols])
                    
                    # ESTIMASI MODEL
                    m_pooled = PooledOLS(data[y_col], exog).fit()
                    m_fe = PanelOLS(data[y_col], exog, entity_effects=True).fit()
                    m_re = RandomEffects(data[y_col], exog).fit()

                    # TABS
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparison", "🔍 Selection Logic", "🔬 Diagnostics", "💊 Robust Results"])

                    # --- TAB 1: COMPARISON ---
                    with tab1:
                        st.subheader("Summary Table of Estimations")
                        st.write(compare({"Pooled OLS": m_pooled, "Fixed Effects": m_fe, "Random Effects": m_re}))
                        st.info("Check the consistency of coefficients across models. Large swings in values may indicate endogeneity issues.")

                    # --- TAB 2: SELECTION TESTS ---
                    with tab2:
                        st.header("Step 2: Model Selection Strategy")

                        # A. Chow Test (F-test)
                        st.subheader("1. Chow Test (Pooled vs Fixed Effects)")
                        st.write("**H₀:** Pooled OLS is appropriate. | **H₁:** Fixed Effects is required.")
                        f_pval = m_fe.f_pooled.pval
                        st.metric("F-test P-value", f"{f_pval:.4f}")
                        if f_pval < 0.05:
                            st.error("Decision: Reject H₀. Significant entity effects exist. Use Fixed Effects (FE).")
                        else:
                            st.success("Decision: Fail to Reject H₀. Pooled OLS is sufficient.")

                        st.divider()

                        # B. Hausman Test
                        st.subheader("2. Hausman Test (Fixed vs Random Effects)")
                        st.write("**H₀:** Random Effects is efficient. | **H₁:** Fixed Effects is consistent.")
                        b_fe, b_re = m_fe.params, m_re.params
                        v_fe, v_re = m_fe.cov, m_re.cov
                        h_stat = (b_fe - b_re).T @ np.linalg.pinv(v_fe - v_re) @ (b_fe - b_re)
                        p_hausman = 1 - stats.chi2.cdf(h_stat, len(b_fe))
                        st.metric("Hausman P-value", f"{p_hausman:.4f}")
                        if p_hausman < 0.05:
                            st.error("Decision: Reject H₀. Use Fixed Effects (FE).")
                        else:
                            st.success("Decision: Fail to Reject H₀. Use Random Effects (RE).")

                    # --- TAB 3: DIAGNOSTICS ---
                    with tab3:
                        st.header("Step 3: Classic Diagnostic Testing")

                        # 1. VIF
                        st.subheader("1. Multicollinearity (VIF)")
                        vif_data = pd.DataFrame()
                        vif_data["Variable"] = x_cols
                        vif_data["VIF"] = [variance_inflation_factor(df_clean[x_cols].values, i) for i in range(len(x_cols))]
                        st.table(vif_data)
                        st.write("VIF > 10 suggests serious multicollinearity issues.")

                        # 2. Normality (FIXED VERSION)
                        st.subheader("2. Normality of Residuals (Jarque-Bera)")
                        resids = m_fe.resids
                        jb_stat, jb_pval = stats.jarque_bera(resids) # Memulangkan 2 nilai sahaja
                        st.metric("Jarque-Bera P-value", f"{jb_pval:.4f}")
                        if jb_pval < 0.05:
                            st.warning("Result: Residuals are non-normal. This is common in panel data; OLS remains reliable for large samples.")
                        else:
                            st.success("Result: Residuals are normally distributed.")

                        # 3. Heteroscedasticity Visual
                        st.subheader("3. Heteroscedasticity Visual (Residuals vs Fitted)")
                        fig_res, ax_res = plt.subplots()
                        ax_res.scatter(m_fe.predict(), resids)
                        ax_res.axhline(0, color='red', linestyle='--')
                        ax_res.set_xlabel("Fitted Values")
                        ax_res.set_ylabel("Residuals")
                        st.pyplot(fig_res)

                    # --- TAB 4: ROBUST RESULTS ---
                    with tab4:
                        st.header("Step 4: Final Adjusted Model")
                        st.info("Treatment: Clustered Standard Errors applied to handle Heteroscedasticity & Autocorrelation.")
                        
                        is_fe = p_hausman < 0.05
                        res_robust = PanelOLS(data[y_col], exog, entity_effects=is_fe).fit(cov_type='clustered', cluster_entity=True)
                        
                        df_res = pd.DataFrame({
                            "Coefficient": res_robust.params,
                            "Std. Error": res_robust.std_errors,
                            "t-Stat": res_robust.tstats,
                            "P-value": res_robust.pvalues
                        })

                        st.write(f"### Final Selected Model: **{'Fixed Effects' if is_fe else 'Random Effects'}**")
                        
                        def color_sig(v): return 'color: green; font-weight: bold' if v < 0.05 else 'color: black'
                        st.dataframe(df_res.style.format("{:.4f}").applymap(color_sig, subset=['P-value']))

                        # Auto-Interpretation
                        st.subheader("📝 Results Interpretation Guide")
                        for var in x_cols:
                            coef = res_robust.params[var]
                            pval = res_robust.pvalues[var]
                            sig = "statistically significant" if pval < 0.05 else "not significant"
                            dir = "positive" if coef > 0 else "negative"
                            st.write(f"- For every 1 unit increase in **{var}**, **{y_col}** is expected to change by **{coef:.4f}** units in a **{dir}** direction. This result is **{sig}**.")

                        st.download_button("Download Full Report", str(res_robust.summary), "academic_report.txt")

            except Exception as e:
                st.error(f"Error: {e}")
