import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, compare
import statsmodels.api as sm
from scipy import stats

st.set_page_config(page_title="Panel Data Expert System", layout="wide")

st.title("🔬 Comprehensive Panel Data Analyzer")
st.markdown("This application estimates Pooled OLS, Fixed Effects, and Random Effects models with automated diagnostic testing.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your CSV file (Long Panel Format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()
    
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        id_col = st.selectbox("Entity (ID) Column:", cols)
        time_col = st.selectbox("Time (Year/Period) Column:", cols)
        y_col = st.selectbox("Dependent Variable (Y):", cols)
        x_cols = st.multiselect("Independent Variables (X):", cols)
        st.divider()
        run_btn = st.button("RUN FULL ANALYSIS")

    if run_btn:
        if not x_cols:
            st.error("Please select at least one X variable.")
        else:
            # Data Preparation
            df_clean = df.dropna(subset=[y_col] + x_cols)
            data = df_clean.set_index([id_col, time_col])
            exog = sm.add_constant(data[x_cols])
            
            # Create Tabs for better UI
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Visualization", "📈 Model Estimation", "🔍 Selection & Diagnostics", "💊 Final Robust Model"])

            # --- TAB 1: VISUALIZATION ---
            with tab1:
                st.subheader("Entity Specific Trends")
                fig, ax = plt.subplots(figsize=(10, 5))
                for label, grp in df_clean.groupby(id_col):
                    grp.sort_values(time_col).plot(x=time_col, y=y_col, ax=ax, label=label, marker='o')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                st.pyplot(fig)
                st.info("Visual check: If the lines have different levels/intercepts, Fixed Effects (FE) is likely necessary.")

            # --- TAB 2: ESTIMATION ---
            with tab2:
                st.subheader("Comparison of All Models")
                m_pooled = PooledOLS(data[y_col], exog).fit()
                m_fe1 = PanelOLS(data[y_col], exog, entity_effects=True).fit()
                m_fe2 = PanelOLS(data[y_col], exog, entity_effects=True, time_effects=True).fit()
                m_re = RandomEffects(data[y_col], exog).fit()
                
                comparison = compare({
                    "Pooled OLS": m_pooled, 
                    "FE (One-way)": m_fe1, 
                    "FE (Two-way)": m_fe2, 
                    "Random Effects": m_re
                })
                st.write(comparison)

            # --- TAB 3: DIAGNOSTICS ---
            with tab3:
                st.subheader("Model Selection Tests")
                
                # Hausman Test Implementation
                b_fe = m_fe1.params
                b_re = m_re.params
                v_fe = m_fe1.cov
                v_re = m_re.cov
                diff = b_fe - b_re
                precision = np.linalg.pinv(v_fe - v_re)
                stat = diff.T @ precision @ diff
                p_hausman = 1 - stats.chi2.cdf(stat, len(b_fe))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**1. Hausman Test (FE vs RE)**")
                    st.metric("P-Value", f"{p_hausman:.4f}")
                    if p_hausman < 0.05:
                        st.error("Decision: Use Fixed Effects (FE)")
                        st.write("Reason: Null hypothesis rejected. Individual effects are correlated with regressors.")
                    else:
                        st.success("Decision: Use Random Effects (RE)")
                        st.write("Reason: Failed to reject Null. RE is more efficient.")

                with col2:
                    st.markdown("**2. Pooled vs FE (F-test)**")
                    st.write(f"F-stat P-value: {m_fe1.f_pooled.pval:.4f}")
                    if m_fe1.f_pooled.pval < 0.05:
                        st.error("Decision: Pooled OLS is Inadequate")
                    else:
                        st.success("Decision: Pooled OLS is Sufficient")

            # --- TAB 4: ROBUST TREATMENT ---
            with tab4:
                st.subheader("Final Model with Robust Standard Errors")
                st.write("This model addresses **Heteroscedasticity** and **Serial Correlation** using Clustered Covariance.")
                
                # Automatically choose model based on Hausman
                is_fe = p_hausman < 0.05
                final_res = PanelOLS(data[y_col], exog, entity_effects=is_fe).fit(cov_type='clustered', cluster_entity=True)
                
                st.write(f"Showing results for: **{'Fixed Effects' if is_fe else 'Random Effects'} Model (Robust)**")
                st.text(final_res.summary)
                
                # Download Results
                st.download_button("Download Summary as TXT", str(final_res.summary), file_name="regression_results.txt")
