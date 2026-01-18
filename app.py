import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, compare
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats

st.set_page_config(page_title="Academic Panel Analyzer", layout="wide")
st.title("🔬 Professional Econometric Panel Data System")

uploaded_file = st.file_uploader("Step 1: Upload CSV File (Long Panel Format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(5))
    cols = df.columns.tolist()
    
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        id_col = st.selectbox("Entity (ID):", cols)
        time_col = st.selectbox("Time (Year):", cols)
        y_col = st.selectbox("Dependent Variable (Y):", cols)
        x_cols = st.multiselect("Independent Variables (X):", cols)
        st.divider()
        run_btn = st.button("RUN FULL ECONOMETRIC ANALYSIS")

    if run_btn:
        try:
            # 0. Data Preparation
            df_clean = df.dropna(subset=[id_col, time_col, y_col] + x_cols)
            data = df_clean.set_index([id_col, time_col])
            exog = sm.add_constant(data[x_cols])
            
            # RUN MODELS
            m_pooled = PooledOLS(data[y_col], exog).fit()
            m_fe = PanelOLS(data[y_col], exog, entity_effects=True).fit()
            m_re = RandomEffects(data[y_col], exog).fit()

            tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Comparisons", "🔍 Selection Logic", "🔬 Detailed Diagnostics", "💊 Final Robust Results"])

            # --- TAB 1: COMPARISON ---
            with tab1:
                st.subheader("Summary Table of Estimations")
                st.write(compare({"Pooled OLS": m_pooled, "Fixed Effects": m_fe, "Random Effects": m_re}))
                st.info("Compare the R-squared and Coefficients across different estimators. Significant changes in coefficients between FE and RE often hint at endogeneity.")

            # --- TAB 2: SELECTION TESTS ---
            with tab2:
                st.header("Step 2: Model Selection Strategy")

                # 1. Chow Test (F-test for Fixed Effects)
                st.subheader("1. Chow Test (Pooled vs Fixed Effects)")
                st.markdown("**Hypotheses:**")
                st.write("- **H₀ (Null):** All entity intercepts are zero (Pooled OLS is appropriate).")
                st.write("- **H₁ (Alternative):** At least one entity intercept is different (Fixed Effects is required).")
                f_pval = m_fe.f_pooled.pval
                st.metric("F-test P-value", f"{f_pval:.4f}")
                if f_pval < 0.05:
                    st.error("Decision: Reject H₀. Significant individual effects exist. Fixed Effects (FE) is more appropriate than Pooled OLS.")
                else:
                    st.success("Decision: Fail to Reject H₀. Individual effects are not significant. Pooled OLS is sufficient.")

                st.divider()

                # 2. Hausman Test
                st.subheader("2. Hausman Test (Fixed Effects vs Random Effects)")
                st.markdown("**Hypotheses:**")
                st.write("- **H₀ (Null):** Differences in coefficients are not systematic (Random Effects is efficient).")
                st.write("- **H₁ (Alternative):** Differences are systematic (Fixed Effects is consistent).")
                
                b_fe, b_re = m_fe.params, m_re.params
                v_fe, v_re = m_fe.cov, m_re.cov
                h_stat = (b_fe - b_re).T @ np.linalg.pinv(v_fe - v_re) @ (b_fe - b_re)
                p_hausman = 1 - stats.chi2.cdf(h_stat, len(b_fe))
                
                st.metric("Hausman P-value", f"{p_hausman:.4f}")
                if p_hausman < 0.05:
                    st.error("Decision: Reject H₀ (P < 0.05). Individual effects are correlated with predictors. Use the Fixed Effects (FE) Model.")
                else:
                    st.success("Decision: Fail to Reject H₀ (P > 0.05). Individual effects are random. Use the Random Effects (RE) Model for higher efficiency.")

            # --- TAB 3: DIAGNOSTICS ---
            with tab3:
                st.header("Step 3: Classic Diagnostic Testing")

                # 1. Multicollinearity (VIF)
                st.subheader("1. Multicollinearity Check")
                vif_data = pd.DataFrame()
                vif_data["Variable"] = x_cols
                vif_data["VIF"] = [variance_inflation_factor(df_clean[x_cols].values, i) for i in range(len(x_cols))]
                st.table(vif_data)
                st.write("**Interpretation:** A VIF > 10 indicates high multicollinearity. If VIF is high, your standard errors may be inflated, making variables appear insignificant.")

                # 2. Normality of Residuals
                st.subheader("2. Normality of Residuals (Jarque-Bera)")
                st.markdown("**Hypotheses:**")
                st.write("- **H₀:** Residuals are normally distributed.")
                st.write("- **H₁:** Residuals are not normally distributed.")
                resids = m_fe.resids
                jb_stat, jb_pval, _, _ = stats.jarque_bera(resids)
                st.metric("Jarque-Bera P-value", f"{jb_pval:.4f}")
                if jb_pval < 0.05:
                    st.warning("Result: Residuals are non-normal. However, for large samples, OLS estimators remain unbiased due to the Central Limit Theorem.")
                else:
                    st.success("Result: Residuals are normally distributed.")

                # 3. Heteroscedasticity Note
                st.subheader("3. Heteroscedasticity & Autocorrelation")
                st.info("In Panel Data, cross-sectional heteroscedasticity and serial correlation are almost always present. Instead of individual tests, modern econometrics applies 'Robust Clustered Standard Errors' to fix both simultaneously.")

            # --- TAB 4: FINAL MODEL ---
            with tab4:
                st.header("Step 4: Final Adjusted Model")
                st.markdown("This model applies **Driscoll-Kraay / Clustered Standard Errors** to provide valid P-values regardless of Heteroscedasticity or Autocorrelation.")
                
                is_fe = p_hausman < 0.05
                res_robust = PanelOLS(data[y_col], exog, entity_effects=is_fe).fit(cov_type='clustered', cluster_entity=True)
                
                # Formatted Table
                df_res = pd.DataFrame({
                    "Coefficient": res_robust.params,
                    "Std. Error": res_robust.std_errors,
                    "t-Statistic": res_robust.tstats,
                    "P-value": res_robust.pvalues
                })
                
                def highlight_sig(v): return 'color: green; font-weight: bold' if v < 0.05 else 'color: black'
                st.write(f"### Selected Model: **{'Fixed Effects' if is_fe else 'Random Effects'}**")
                st.dataframe(df_res.style.format("{:.4f}").applymap(highlight_sig, subset=['P-value']))

                # Auto-Interpretation of Variables
                st.subheader("📝 Automatic Results Interpretation")
                for var in x_cols:
                    coef = res_robust.params[var]
                    pval = res_robust.pvalues[var]
                    sig = "statistically significant" if pval < 0.05 else "not statistically significant"
                    direction = "positive" if coef > 0 else "negative"
                    
                    st.write(f"- The variable **{var}** has a **{direction}** effect on {y_col} ($\beta$ = {coef:.4f}). This relationship is **{sig}** at the 5% level.")

                st.download_button("📥 Download Full Academic Report", str(res_robust.summary), "academic_report.txt")

        except Exception as e:
            st.error(f"Econometric Engine Error: {e}")
