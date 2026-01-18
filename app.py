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

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your CSV file (Long Panel Format)", type=["csv"])

if uploaded_file:
    # AUTO-SHOW DATA IMMEDIATELY
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview (Auto-loaded)")
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
                # Data Preparation & Duplicate Check
                df_clean = df.dropna(subset=[id_col, time_col, y_col] + x_cols)
                
                # Check for duplicates in Entity-Time index
                duplicates = df_clean.duplicated(subset=[id_col, time_col]).sum()
                if duplicates > 0:
                    st.error(f"Error: Found {duplicates} duplicate entries for the same Entity and Time. Please clean your data.")
                else:
                    data = df_clean.set_index([id_col, time_col])
                    exog = sm.add_constant(data[x_cols])
                    
                    # Create Tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "📈 Models", "🔍 Selection Tests", "💊 Robust Model"])

                    with tab1:
                        st.subheader("Entity Trends Over Time")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        for label, grp in df_clean.groupby(id_col):
                            grp.sort_values(time_col).plot(x=time_col, y=y_col, ax=ax, label=label, marker='o')
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                        st.pyplot(fig)

                    with tab2:
                        st.subheader("Model Comparison")
                        # Estimations
                        m_pooled = PooledOLS(data[y_col], exog).fit()
                        m_fe1 = PanelOLS(data[y_col], exog, entity_effects=True).fit()
                        m_fe2 = PanelOLS(data[y_col], exog, entity_effects=True, time_effects=True).fit()
                        m_re = RandomEffects(data[y_col], exog).fit()
                        
                        st.write(compare({"Pooled": m_pooled, "FE (1-Way)": m_fe1, "FE (2-Way)": m_fe2, "RE": m_re}))

                    with tab3:
                        st.subheader("Diagnostics & Selection")
                        # Hausman Test
                        b_fe = m_fe1.params
                        b_re = m_re.params
                        v_fe = m_fe1.cov
                        v_re = m_re.cov
                        diff = b_fe - b_re
                        h_stat = diff.T @ np.linalg.pinv(v_fe - v_re) @ diff
                        p_hausman = 1 - stats.chi2.cdf(h_stat, len(b_fe))
                        
                        st.markdown(f"**Hausman Test P-Value:** `{p_hausman:.4f}`")
                        if p_hausman < 0.05:
                            st.error("Decision: Use Fixed Effects (FE)")
                        else:
                            st.success("Decision: Use Random Effects (RE)")
                            
                        # F-test for Fixed Effects
                        st.markdown(f"**F-test (Pooled vs FE) P-Value:** `{m_fe1.f_pooled.pval:.4f}`")

                    with tab4:
                        st.subheader("Final Robust Results")
                        st.info("Treatment: Clustered Standard Errors (Corrects for Heteroscedasticity & Serial Correlation)")
                        
                        is_fe = p_hausman < 0.05
                        res_robust = PanelOLS(data[y_col], exog, entity_effects=is_fe).fit(cov_type='clustered', cluster_entity=True)
                        
                        st.text(res_robust.summary)
                        st.download_button("Export Results", str(res_robust.summary), file_name="panel_analysis.txt")

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin.")
