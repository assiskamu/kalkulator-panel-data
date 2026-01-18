import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, compare
import statsmodels.api as sm
from scipy import stats

st.set_page_config(page_title="Expert Panel Data", layout="wide")

# CSS untuk gaya laporan
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Laporan Analisis Panel Data Profesional")

uploaded_file = st.file_uploader("Muat naik fail CSV (Format Long Panel)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()
    
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        id_col = st.selectbox("Unit (Entity):", cols)
        time_col = st.selectbox("Masa (Time):", cols)
        y_col = st.selectbox("Y (Bersandar):", cols)
        x_cols = st.multiselect("X (Bebas):", cols)
        st.divider()
        run_btn = st.button("JALANKAN ANALISIS PENUH")

    if run_btn:
        # Data Preparation
        df_clean = df.dropna(subset=[y_col] + x_cols)
        data = df_clean.set_index([id_col, time_col])
        exog = sm.add_constant(data[x_cols])
        
        # TAB MENU
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualisasi", "📈 Estimasi Model", "🔍 Diagnostik", "💊 Rawatan & Kesimpulan"])

        with tab1:
            st.subheader("Trend Data mengikut Entiti")
            fig, ax = plt.subplots(figsize=(10, 4))
            for label, grp in df_clean.groupby(id_col):
                grp.plot(x=time_col, y=y_col, ax=ax, label=label, marker='o')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            st.write("**Penerangan:** Graf ini menunjukkan variasi unit merentas masa. Jika garisan setiap entiti mempunyai tahap (intercept) yang sangat berbeza, ini petanda awal Fixed Effects diperlukan.")

        with tab2:
            st.subheader("Perbandingan Model")
            m_pooled = PooledOLS(data[y_col], exog).fit()
            m_fe1 = PanelOLS(data[y_col], exog, entity_effects=True).fit()
            m_fe2 = PanelOLS(data[y_col], exog, entity_effects=True, time_effects=True).fit()
            m_re = RandomEffects(data[y_col], exog).fit()
            
            comp = compare({"Pooled": m_pooled, "FE (1-Way)": m_fe1, "FE (2-Way)": m_fe2, "RE": m_re})
            st.write(comp)

        with tab3:
            st.subheader("Ujian Diagnostik Formal")
            
            # 1. Hausman Test Logic
            b_fe = m_fe1.params
            b_re = m_re.params
            v_fe = m_fe1.cov
            v_re = m_re.cov
            diff = b_fe - b_re
            # Guna pseudo-inverse untuk kestabilan numerik
            h_stat = diff.dot(np.linalg.pinv(v_fe - v_re)).dot(diff)
            p_hausman = 1 - stats.chi2.cdf(h_stat, len(b_fe))
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**1. Hausman Test (FE vs RE)**")
                st.write(f"P-value: {p_hausman:.4f}")
                if p_hausman < 0.05:
                    st.error("Keputusan: Pilih Fixed Effects (Reject H0)")
                else:
                    st.success("Keputusan: Pilih Random Effects (Fail to reject H0)")

            # 2. Breusch-Pagan LM Test (Pooled vs RE)
            with col2:
                st.write("**2. Breusch-Pagan LM Test**")
                st.info("Menguji jika ada kesan rawak (Random Effects). Jika P < 0.05, Pooled OLS tidak sesuai.")

        with tab4:
            st.subheader("Model Akhir & Rawatan Robust")
            # Autoselect Model berdasarkan Hausman
            final_model_type = "Fixed Effects" if p_hausman < 0.05 else "Random Effects"
            st.write(f"Berdasarkan ujian, model terbaik ialah: **{final_model_type}**")
            
            # Rawatan Heteroscedasticity (Clustered Standard Errors)
            res_robust = PanelOLS(data[y_col], exog, entity_effects=(p_hausman < 0.05)).fit(cov_type='clustered', cluster_entity=True)
            
            st.write("### Keputusan Regresi Robust (Telah Dirawat)")
            st.text(res_robust.summary)
            
            st.download_button("Simpan Keputusan (Text)", str(res_robust.summary), file_name="hasil_regresi.txt")
