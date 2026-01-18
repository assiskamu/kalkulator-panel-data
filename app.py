import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, compare
import statsmodels.api as sm
from scipy import stats

# Page configuration
st.set_page_config(page_title="Panel Data Expert System", layout="wide")

st.title("🔬 Comprehensive Panel Data Analyzer")
st.markdown("Automated estimation for Pooled OLS, Fixed Effects (1-way/2-way), and Random Effects.")

# --- 1. FILE UPLOAD & AUTO-PREVIEW ---
uploaded_file = st.file_uploader("Upload your CSV file (Long Panel Format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10))
    
    cols = df.columns.tolist()
    
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        id_col = st.selectbox("Entity (ID) Column:", cols)
        time_col = st.selectbox("Time (Year/Period) Column:", cols)
        y_col = st.selectbox("Dependent Variable (Y):", cols)
        x_cols = st.multiselect("Independent Variables (X):", cols)
        st.divider()
        run_btn = st.button("RUN FULL ANALYSIS")

    # --- 2. VALIDATION & ANALYSIS ---
    if run_btn:
        if not x_cols:
            st.error("Please select at least one X variable.")
        else:
            try:
                # Data Cleaning (Removing missing values)
                df_clean = df.dropna(subset=[id_col, time_col, y_col] + x_cols)
                
                # Check for duplicates
                duplicates = df_clean.duplicated(subset=[id_col, time_col]).sum()
                if duplicates > 0:
                    st.error(f"Duplicate Error: {duplicates} entries found for the same Entity-Time index. Please clean your data.")
                else:
                    data = df_clean.set_index([id_col, time_col])
                    exog = sm.add_constant(data[x_cols])
                    
                    # TABS
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "📈 Model Comparison", "🔍 Selection Tests", "💊 Robust Model"])

                    with tab1:
                        st.subheader("Entity Trends")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        for label, grp in df_clean.groupby(id_col):
                            grp.sort_values(time_col).plot(x=time_col, y=y_col, ax=ax, label=label, marker='o')
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                        st.pyplot(fig)

                    with tab2:
                        st.subheader("Estimation Comparison")
                        m_pooled = PooledOLS(data[y_col], exog).fit()
                        m_fe1 = PanelOLS(data[y_col], exog, entity_effects=True).fit()
                        m_fe2 = PanelOLS(data[y_col], exog, entity_effects=True, time_effects=True).fit()
                        m_re = RandomEffects(data[y_col], exog).fit()
                        
                        st.write(compare({"Pooled": m_pooled, "FE (1-Way)": m_fe1, "FE (2-Way)": m_fe2, "RE": m_re}))

                    with tab3:
                        st.subheader("Diagnostic Decisions")
                        # Hausman Test
                        b_fe = m_fe1.params
                        b_re = m_re.params
                        v_fe = m_fe1.cov
                        v_re = m_re.cov
                        diff = b_fe - b_re
                        h_stat = diff.T @ np.linalg.pinv(v_fe - v_re) @ diff
                        p_hausman = 1 - stats.chi2.cdf(h_stat, len(b_fe))
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Hausman P-Value", f"{p_hausman:.4f}")
                            if p_hausman < 0.05:
                                st.error("Result: Use Fixed Effects (FE)")
                            else:
                                st.success("Result: Use Random Effects (RE)")
                        
                        with c2:
                            st.metric("F-Test (Pooled vs FE) P-Value", f"{m_fe1.f_pooled.pval:.4f}")
                            if m_fe1.f_pooled.pval < 0.05:
                                st.error("Result: OLS is Inadequate")
                            else:
                                st.success("Result: OLS is Sufficient")

                    with tab4:
                        st.subheader("Final Robust Model (Corrected)")
                        st.info("Treatment: Clustered Standard Errors applied to handle Heteroscedasticity and Autocorrelation.")
                        
                        is_fe = p_hausman < 0.05
                        res = PanelOLS(data[y_col], exog, entity_effects=is_fe).fit(cov_type='clustered', cluster_entity=True)
                        
                        # Clean Table Logic
                        df_res = pd.DataFrame({
                            "Coefficient": res.params,
                            "Std. Error": res.std_errors,
                            "t-Stat": res.tstats,
                            "P-value": res.pvalues
                        })

                        def color_p(val):
                            return 'color: green; font-weight: bold' if val < 0.05 else 'color: black'

                        st.write(f"### Selected Model: **{'Fixed Effects' if is_fe else 'Random Effects'}**")
                        st.dataframe(df_res.style.format("{:.4f}").applymap(color_p, subset=['P-value']))

                        m1, m2, m3 = st.columns(3)
                        m1.metric("R-Squared (Within)", f"{res.rsquared_within:.4f}")
                        m2.metric("Observations", int(res.nobs))
                        m3.metric("F-Statistic", f"{res.f_statistic.stat:.2f}")

                        with st.expander("Show Full Raw Summary"):
                            st.text(res.summary)
                        
                        st.download_button("Download Report", str(res.summary), file_name="panel_report.txt")

            except Exception as e:
                st.error(f"Critical Error: {e}")
else:
    st.info("👋 Welcome! Please upload your CSV file to begin the analysis.")
