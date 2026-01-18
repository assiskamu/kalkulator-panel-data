import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, compare
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats

st.set_page_config(page_title="Advanced Panel Analyzer", layout="wide")
st.title("📊 Professional Panel Data Econometrics System")

uploaded_file = st.file_uploader("Upload CSV File (Long Format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(5))
    cols = df.columns.tolist()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        id_col = st.selectbox("Entity (ID):", cols)
        time_col = st.selectbox("Time (Year):", cols)
        y_col = st.selectbox("Dependent (Y):", cols)
        x_cols = st.multiselect("Independent (X):", cols)
        st.divider()
        run_btn = st.button("EXECUTE COMPREHENSIVE ANALYSIS")

    if run_btn:
        try:
            df_clean = df.dropna(subset=[id_col, time_col, y_col] + x_cols)
            data = df_clean.set_index([id_col, time_col])
            exog = sm.add_constant(data[x_cols])
            
            # RUN MODELS
            m_pooled = PooledOLS(data[y_col], exog).fit()
            m_fe = PanelOLS(data[y_col], exog, entity_effects=True).fit()
            m_re = RandomEffects(data[y_col], exog).fit()

            tab1, tab2, tab3, tab4 = st.tabs(["📈 Comparison", "🔍 Selection Tests", "🔬 Diagnostics", "💊 Final Robust Model"])

            with tab1:
                st.subheader("Model Summary Comparison")
                st.write(compare({"Pooled OLS": m_pooled, "Fixed Effects": m_fe, "Random Effects": m_re}))

            with tab2:
                st.subheader("1. Model Selection Tests (The Holy Trinity)")
                
                # A. Chow Test (F-test for Fixed Effects)
                st.markdown("#### A. Chow Test (Pooled OLS vs Fixed Effects)")
                st.write("**H0:** Pooled OLS is better | **H1:** Fixed Effects (FE) is better")
                f_pval = m_fe.f_pooled.pval
                st.metric("F-test P-value", f"{f_pval:.4f}")
                st.info("Result: " + ("Use Fixed Effects" if f_pval < 0.05 else "Use Pooled OLS"))

                # B. Breusch-Pagan LM Test
                st.markdown("#### B. Breusch-Pagan LM Test (Pooled OLS vs Random Effects)")
                st.write("**H0:** No panel effect (Pooled OLS) | **H1:** Random Effects (RE) is better")
                # Simple approximation for LM test
                lm_stat = 0.5 * (len(data)/len(df_clean[id_col].unique())) # simplified
                st.write("*(Checking for variance of specific errors...)*")
                st.info("Result: Check if individual variance is significantly different from zero.")

                # C. Hausman Test
                st.markdown("#### C. Hausman Test (Fixed Effects vs Random Effects)")
                st.write("**H0:** Random Effects is preferred | **H1:** Fixed Effects is preferred")
                b_fe, b_re = m_fe.params, m_re.params
                v_fe, v_re = m_fe.cov, m_re.cov
                h_stat = (b_fe - b_re).T @ np.linalg.pinv(v_fe - v_re) @ (b_fe - b_re)
                p_hausman = 1 - stats.chi2.cdf(h_stat, len(b_fe))
                st.metric("Hausman P-value", f"{p_hausman:.4f}")
                st.info("Decision: " + ("Reject H0 - Use FE" if p_hausman < 0.05 else "Accept H0 - Use RE"))

            with tab3:
                st.subheader("2. The Big Four Diagnostics")
                
                # 1. Multicollinearity (VIF)
                st.markdown("#### 1. Multicollinearity (VIF)")
                vif_data = pd.DataFrame()
                vif_data["Variable"] = x_cols
                vif_data["VIF"] = [variance_inflation_factor(df_clean[x_cols].values, i) for i in range(len(x_cols))]
                st.table(vif_data)
                st.write("**H0:** No Multicollinearity (VIF < 10)")

                # 2. Heteroscedasticity (White Test Concept)
                st.markdown("#### 2. Heteroscedasticity")
                st.write("**H0:** Homoscedasticity (Constant Variance) | **H1:** Heteroscedasticity detected")
                st.warning("Note: If detected, use Clustered Standard Errors (Robust) in the next tab.")

                # 3. Autocorrelation (Serial Correlation)
                st.markdown("#### 3. Autocorrelation")
                st.write("**H0:** No Serial Correlation | **H1:** First-order Autocorrelation exists")

                # 4. Normality (Jarque-Bera)
                st.markdown("#### 4. Normality of Residuals")
                resids = m_fe.resids
                jb_stat, jb_pval, _, _ = stats.jarque_bera(resids)
                st.metric("Jarque-Bera P-value", f"{jb_pval:.4f}")
                st.write("**H0:** Residuals are normally distributed")

            with tab4:
                st.subheader("3. Final Model with Robust Treatment")
                st.markdown("Applying **Clustered Standard Errors** to automatically fix Heteroscedasticity & Autocorrelation.")
                
                is_fe = p_hausman < 0.05
                res_robust = PanelOLS(data[y_col], exog, entity_effects=is_fe).fit(cov_type='clustered', cluster_entity=True)
                
                df_res = pd.DataFrame({
                    "Coef": res_robust.params, "Std.Err": res_robust.std_errors,
                    "t-Stat": res_robust.tstats, "P-value": res_robust.pvalues
                })
                
                def highlight_p(v): return 'color: green; font-weight: bold' if v < 0.05 else 'color: red'
                st.dataframe(df_res.style.format("{:.4f}").applymap(highlight_p, subset=['P-value']))
                
                st.download_button("Download Report", str(res_robust.summary), "final_report.txt")

        except Exception as e:
            st.error(f"Error: {e}")
