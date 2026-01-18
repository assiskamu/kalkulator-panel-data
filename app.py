"""Academic Econometrics Suite (Streamlit)

MODULES
- Panel Data: Pooled OLS, Fixed Effects (entity/time/two-way), Random Effects (entity RE)
  + model selection (FE vs pooled, RE vs pooled LM for balanced panel, Hausman FE vs RE)
  + diagnostics (VIF, Breusch–Pagan heteroskedasticity on pooled residuals, Pesaran CD on residuals for balanced panel)
  + robust covariance options (robust, clustered, Driscoll–Kraay via kernel covariance)

- Time Series (separate file; annual; time column = year): VAR, VECM, ARDL
  + data screening + transformation suggestions
  + stationarity tests (ADF + KPSS)
  + lag selection
  + model-consistent causality tests:
      VAR: VARResults.test_causality
      VECM: VECMResults.test_granger_causality
      ARDL: Wald tests on lagged X terms (Granger-style)
  + diagnostics
  + export Excel report (reproducible)

NOTES
- This app intentionally avoids "panel Granger" implemented naively on stacked data.
- LM test for RE vs pooled is implemented ONLY for balanced panels using the standard formula
  described in Stata's xttest0 documentation (Breusch–Pagan LM with Baltagi-Li note). See:
  Stata xtreg postestimation PDF, Methods and Formulas section.

Author: rewritten for academic defensibility.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from scipy import stats

# Panel
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

# OLS diagnostics
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Time series
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ardl import ardl_select_order


# -----------------------------
# Streamlit global setup
# -----------------------------

st.set_page_config(page_title="Academic Econometrics Suite", layout="wide")


# -----------------------------
# Helpers: IO
# -----------------------------

def _read_uploaded_file(uploaded) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Read CSV/XLSX from Streamlit uploader.

    Returns (df, meta) where meta contains information for reporting.
    """
    meta = {"file_name": getattr(uploaded, "name", "uploaded")}
    if uploaded is None:
        raise ValueError("No file uploaded.")

    name = meta["file_name"].lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
        meta["file_type"] = "csv"
        return df, meta

    if name.endswith(".xlsx") or name.endswith(".xls"):
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox("Pilih sheet", xls.sheet_names, key=f"sheet_{meta['file_name']}")
        df = pd.read_excel(xls, sheet_name=sheet)
        meta["file_type"] = "excel"
        meta["sheet"] = sheet
        return df, meta

    raise ValueError("Supported formats: .csv, .xlsx")


def _to_numeric_safe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _excel_bytes(sheets: Dict[str, pd.DataFrame], notes: Optional[Dict[str, str]] = None) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        if notes:
            note_df = pd.DataFrame({"item": list(notes.keys()), "value": list(notes.values())})
            note_df.to_excel(writer, sheet_name="meta", index=False)
        for sheet, df in sheets.items():
            # Excel sheet name limit
            safe = sheet[:31]
            df.to_excel(writer, sheet_name=safe, index=False)
    bio.seek(0)
    return bio.read()


# -----------------------------
# Helpers: Screening & transformations
# -----------------------------

def _screen_transform_suggestions(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.dropna()
        if s.empty:
            continue

        minv = float(s.min())
        maxv = float(s.max())
        zeros = int((s == 0).sum())
        neg = int((s < 0).sum())
        skew = float(stats.skew(s, bias=False)) if len(s) >= 3 else np.nan
        kurt = float(stats.kurtosis(s, bias=False)) if len(s) >= 4 else np.nan

        # Outliers via IQR
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            outlier_share = 0.0
        else:
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            outlier_share = float(((s < lo) | (s > hi)).mean())

        suggestion = "none"
        reason = ""

        # Simple, explainable rules; do not auto-apply.
        if neg > 0:
            suggestion = "none"
            reason = "Ada nilai negatif; log tidak sesuai."
        else:
            if (skew is not np.nan) and (abs(skew) >= 1.2):
                if zeros > 0:
                    suggestion = "log1p"
                    reason = "Skew tinggi dan ada sifar; log1p boleh kurangkan skew."
                else:
                    suggestion = "log"
                    reason = "Skew tinggi dan semua positif; log boleh stabilkan varians."
            else:
                suggestion = "none"
                reason = "Tiada isu skew yang jelas (berdasarkan rule-of-thumb)."

        rows.append(
            {
                "variable": c,
                "min": minv,
                "max": maxv,
                "zeros": zeros,
                "negatives": neg,
                "skewness": skew,
                "kurtosis": kurt,
                "outlier_share": outlier_share,
                "suggestion": suggestion,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _apply_transform(df: pd.DataFrame, transform_map: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Apply per-column transforms: none, log, log1p, standardize.

    Returns (df_transformed, notes)
    """
    out = df.copy()
    notes = {}

    for col, tr in transform_map.items():
        if tr == "none" or col not in out.columns:
            continue

        s = pd.to_numeric(out[col], errors="coerce")
        if tr == "log":
            if (s <= 0).any():
                notes[col] = "SKIPPED log: ada nilai <= 0"
                continue
            out[col] = np.log(s)
            notes[col] = "Applied log"

        elif tr == "log1p":
            if (s < 0).any():
                notes[col] = "SKIPPED log1p: ada nilai < 0"
                continue
            out[col] = np.log1p(s)
            notes[col] = "Applied log1p"

        elif tr == "standardize":
            mu = float(s.mean())
            sd = float(s.std(ddof=0))
            if sd == 0:
                notes[col] = "SKIPPED standardize: sd=0"
                continue
            out[col] = (s - mu) / sd
            notes[col] = "Applied standardization (z-score)"

        else:
            notes[col] = f"Unknown transform '{tr}' ignored"

    return out, notes


# -----------------------------
# Helpers: Panel
# -----------------------------

def _panel_prepare(
    df_raw: pd.DataFrame,
    id_col: str,
    time_col: str,
    y_col: str,
    x_cols: List[str],
    add_time_dummies: bool,
    transform_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, str]]:
    df = df_raw.copy()

    needed = [id_col, time_col, y_col] + x_cols
    df = df[needed].copy()

    # Apply transforms on numeric columns only (safe)
    df, tr_notes = _apply_transform(df, transform_map)

    # Convert y and numeric X to numeric
    df = _to_numeric_safe(df, [y_col] + x_cols)

    # Drop rows with missing y
    df = df.dropna(subset=[y_col])

    # Set MultiIndex for linearmodels
    df[id_col] = df[id_col].astype(str)
    # year/time can be int; keep as is but ensure sortable
    df = df.sort_values([id_col, time_col])

    # Duplicate panel index check
    if df.duplicated(subset=[id_col, time_col]).any():
        dup = df[df.duplicated(subset=[id_col, time_col], keep=False)][[id_col, time_col]].drop_duplicates()
        raise ValueError(
            "Duplicate (entity,time) rows detected. Fix your data. Examples:\n"
            + dup.head(10).to_string(index=False)
        )

    df = df.set_index([id_col, time_col])

    y = df[y_col]

    X = df[x_cols].copy()

    # Handle categorical X via dummies
    cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("category")]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Optional time dummies (for RE or pooled when user wants time FE)
    if add_time_dummies:
        # time index is level 1
        t = df.index.get_level_values(1)
        td = pd.get_dummies(t, prefix="time", drop_first=True)
        td.index = df.index
        X = pd.concat([X, td], axis=1)

    X = sm.add_constant(X, has_constant="add")

    # Drop any row with missing in X
    keep = (~y.isna()) & (~X.isna().any(axis=1))
    y = y.loc[keep]
    X = X.loc[keep]

    meta = {
        "n_obs": str(len(y)),
        "n_entities": str(y.index.get_level_values(0).nunique()),
        "time_min": str(y.index.get_level_values(1).min()),
        "time_max": str(y.index.get_level_values(1).max()),
    }

    return df, y, X, {**meta, **{f"transform_{k}": v for k, v in tr_notes.items()}}


def _is_balanced_panel(index: pd.MultiIndex) -> Tuple[bool, int, int]:
    entities = index.get_level_values(0)
    times = index.get_level_values(1)
    n = entities.nunique()
    # count Ti per entity
    counts = pd.Series(1, index=index).groupby(level=0).sum()
    balanced = counts.nunique() == 1
    T = int(counts.iloc[0]) if balanced else int(counts.median())
    return balanced, int(n), int(T)


def _bp_lm_test_re_vs_pooled_balanced(pooled_resid: pd.Series) -> Tuple[float, float]:
    """Breusch–Pagan LM test for random effects (balanced panel only).

    Uses standard balanced-panel formula shown in Stata xttest0 documentation.

    LM = (nT)/(2(T-1)) * ( [sum_i (sum_t v_it)^2 / sum_it v_it^2] - 1 )^2
    ~ Chi-square(1)

    See: Stata xtreg postestimation (xttest0) Methods and Formulas.
    """
    idx = pooled_resid.index
    balanced, n, T = _is_balanced_panel(idx)
    if not balanced:
        raise ValueError("LM test implemented only for balanced panels in this app.")
    if T <= 2:
        raise ValueError("LM test requires T>2.")

    v = pooled_resid.copy()
    # Sum over time per entity
    sum_i = v.groupby(level=0).sum()
    S1 = float((sum_i ** 2).sum())
    S2 = float((v ** 2).sum())
    if S2 == 0:
        raise ValueError("Residual variance is zero; LM undefined.")

    lm = (n * T) / (2 * (T - 1)) * ((S1 / S2) - 1) ** 2
    p = 1 - stats.chi2.cdf(lm, df=1)
    return float(lm), float(p)


def _hausman_test_aligned(fe_params: pd.Series, fe_cov: pd.DataFrame, re_params: pd.Series, re_cov: pd.DataFrame) -> Tuple[float, float, int]:
    """Hausman test FE vs RE with coefficient alignment.

    H = (b_FE - b_RE)' [Var(b_FE) - Var(b_RE)]^{-1} (b_FE - b_RE)
    ~ Chi-square(k)

    Returns (stat, pvalue, dof)
    """
    common = fe_params.index.intersection(re_params.index)
    # Drop constant if present (often included in RE/pooled but not meaningful in FE)
    common = pd.Index([c for c in common if c.lower() not in {"const", "intercept"}])
    if len(common) == 0:
        raise ValueError("No common coefficients between FE and RE for Hausman test.")

    b = (fe_params.loc[common] - re_params.loc[common]).values.reshape(-1, 1)
    V = fe_cov.loc[common, common] - re_cov.loc[common, common]

    # Numerical stabilization
    V = V.values
    # If V not PD, use pseudo-inverse (common in practice). Be explicit in UI.
    try:
        Vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        Vinv = np.linalg.pinv(V)

    stat = float((b.T @ Vinv @ b).ravel()[0])
    dof = int(len(common))
    p = 1 - stats.chi2.cdf(stat, df=dof)
    return stat, float(p), dof


def _vif_table(X: pd.DataFrame) -> pd.DataFrame:
    Xv = X.copy()
    Xv = Xv.drop(columns=[c for c in Xv.columns if c.lower() in {"const", "intercept"}], errors="ignore")
    # Need numeric only
    Xv = Xv.apply(pd.to_numeric, errors="coerce")
    Xv = Xv.dropna(axis=1, how="all")

    if Xv.shape[1] == 0:
        return pd.DataFrame({"variable": [], "VIF": []})

    # Drop rows with NA (VIF requires full matrix)
    Xv = Xv.dropna()
    if len(Xv) < 5:
        return pd.DataFrame({"variable": list(Xv.columns), "VIF": [np.nan] * Xv.shape[1]})

    vals = Xv.values
    vifs = []
    for i in range(Xv.shape[1]):
        vifs.append(float(variance_inflation_factor(vals, i)))

    return pd.DataFrame({"variable": list(Xv.columns), "VIF": vifs}).sort_values("VIF", ascending=False)


def _pesaran_cd_test_balanced(resid: pd.Series) -> Tuple[float, float]:
    """Pesaran CD test for cross-sectional dependence (balanced panel only).

    CD = sqrt(2T/(N(N-1))) * sum_{i<j} rho_ij
    where rho_ij is correlation of residuals across time.
    """
    idx = resid.index
    balanced, N, T = _is_balanced_panel(idx)
    if not balanced:
        raise ValueError("Pesaran CD implemented only for balanced panels in this app.")
    if N < 3:
        raise ValueError("Pesaran CD requires N>=3.")

    # reshape to T x N
    df = resid.rename("e").reset_index()
    df.columns = ["entity", "time", "e"]
    pivot = df.pivot(index="time", columns="entity", values="e")

    # correlation matrix
    corr = pivot.corr()
    # sum upper triangle excluding diagonal
    iu = np.triu_indices_from(corr.values, k=1)
    rhos = corr.values[iu]
    S = np.nansum(rhos)

    cd = math.sqrt(2 * T / (N * (N - 1))) * S
    p = 2 * (1 - stats.norm.cdf(abs(cd)))
    return float(cd), float(p)


# -----------------------------
# Helpers: Time series
# -----------------------------

@dataclass
class StationarityResult:
    variable: str
    adf_p: float
    kpss_p: float
    classification: str


def _adf_pvalue(s: pd.Series, regression: str = "c") -> float:
    s = s.dropna().astype(float)
    if len(s) < 10:
        return np.nan
    try:
        return float(adfuller(s, regression=regression, autolag="AIC")[1])
    except Exception:
        return np.nan


def _kpss_pvalue(s: pd.Series, regression: str = "c") -> float:
    s = s.dropna().astype(float)
    if len(s) < 10:
        return np.nan
    try:
        # nlags="auto" can fail in rare cases; fallback to a small number
        return float(kpss(s, regression=regression, nlags="auto")[1])
    except Exception:
        try:
            return float(kpss(s, regression=regression, nlags=min(8, max(1, len(s)//10)))[1])
        except Exception:
            return np.nan


def _stationarity_suite(df: pd.DataFrame, cols: List[str], adf_reg: str, kpss_reg: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for c in cols:
        s = df[c].dropna()
        adf_p = _adf_pvalue(s, regression=adf_reg)
        kpss_p = _kpss_pvalue(s, regression=kpss_reg)

        # Simple, explicit classification rule (not "truth", but consistent):
        # - stationary if ADF rejects (p<0.05) AND KPSS does not reject (p>0.05)
        # - non-stationary if ADF fails (p>=0.05) AND KPSS rejects (p<=0.05)
        # - otherwise "mixed/unclear"
        if (not np.isnan(adf_p)) and (not np.isnan(kpss_p)):
            if adf_p < 0.05 and kpss_p > 0.05:
                cl = "I(0) (stationary)"
            elif adf_p >= 0.05 and kpss_p <= 0.05:
                cl = "I(1) likely (non-stationary in level)"
            else:
                cl = "Mixed/unclear"
        else:
            cl = "Insufficient data"

        rows.append({"variable": c, "ADF_p": adf_p, "KPSS_p": kpss_p, "classification": cl})

    return pd.DataFrame(rows)


def _ts_prepare(df_raw: pd.DataFrame, year_col: str, vars_: List[str], transform_map: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df_raw.copy()

    needed = [year_col] + vars_
    df = df[needed].copy()

    # Apply transforms
    df, tr_notes = _apply_transform(df, transform_map)

    # Numeric conversion
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = _to_numeric_safe(df, vars_)

    df = df.dropna(subset=[year_col])
    df = df.sort_values(year_col)

    if df[year_col].duplicated().any():
        dup = df[df[year_col].duplicated(keep=False)][[year_col]].drop_duplicates()
        raise ValueError("Duplicate year values detected. Fix your time index. Examples:\n" + dup.head(10).to_string(index=False))

    df = df.set_index(year_col)
    df.index.name = "year"

    # Drop missing rows in selected vars
    df = df.dropna(subset=vars_)

    meta = {
        "T": str(len(df)),
        "year_min": str(int(df.index.min())) if len(df) else "",
        "year_max": str(int(df.index.max())) if len(df) else "",
        **{f"transform_{k}": v for k, v in tr_notes.items()},
    }
    return df, meta


def _matrix_from_var_causality(var_res, vars_: List[str]) -> pd.DataFrame:
    rows = []
    for caused in vars_:
        row = {"caused": caused}
        for causing in vars_:
            if causing == caused:
                row[causing] = np.nan
                continue
            try:
                test = var_res.test_causality(caused=caused, causing=[causing], kind="wald")
                row[causing] = float(test.pvalue)
            except Exception:
                row[causing] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _matrix_from_vecm_causality(vecm_res, vars_: List[str]) -> pd.DataFrame:
    rows = []
    for caused in vars_:
        row = {"caused": caused}
        for causing in vars_:
            if causing == caused:
                row[causing] = np.nan
                continue
            try:
                test = vecm_res.test_granger_causality(caused=caused, causing=[causing])
                row[causing] = float(test.pvalue)
            except Exception:
                row[causing] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _ardl_granger_style_tests(ardl_res, y: str, x_list: List[str]) -> pd.DataFrame:
    """Granger-style causality: H0: all lagged terms of X are jointly zero.

    This is a pragmatic diagnostic, not a philosophical statement about true causality.
    """
    params = ardl_res.params
    names = list(params.index)

    out_rows = []
    for x in x_list:
        # Identify parameter names involving x (e.g., L0.x, L1.x, ...)
        terms = [nm for nm in names if (nm.endswith(f".{x}") or f"({x})" in nm or f"{x}" in nm) and ("L" in nm)]
        # Filter out y terms
        terms = [t for t in terms if x in t and y not in t]
        terms = list(dict.fromkeys(terms))

        if len(terms) == 0:
            out_rows.append({"causing": x, "F_stat": np.nan, "p_value": np.nan, "df_num": np.nan, "df_denom": np.nan, "terms": ""})
            continue

        # Build restriction matrix R*beta = 0 for selected terms
        k = len(params)
        R = np.zeros((len(terms), k))
        for i, t in enumerate(terms):
            j = names.index(t)
            R[i, j] = 1.0

        try:
            w = ardl_res.wald_test(R)
            # statsmodels returns a ContrastResults-like object
            p = float(w.pvalue)
            stat = float(w.statistic)
            # for OLS-type models, statistic is chi2 by default; use F version if available
            # We'll report both if possible
            out_rows.append({
                "causing": x,
                "stat": stat,
                "p_value": p,
                "terms": ", ".join(terms)[:200],
            })
        except Exception:
            out_rows.append({"causing": x, "stat": np.nan, "p_value": np.nan, "terms": ", ".join(terms)[:200]})

    return pd.DataFrame(out_rows)


# -----------------------------
# UI components
# -----------------------------

st.title("Academic Econometrics Suite")
st.caption("Panel models (Pooled/FE/RE) + Time series models (VAR/VECM/ARDL) with validation + export. No fake panel causality.")

mode = st.sidebar.radio("Module", ["Panel data", "Time series (separate file)"], index=0)


# -----------------------------
# PANEL MODULE
# -----------------------------

if mode == "Panel data":
    st.header("Panel Data Module")

    uploaded = st.file_uploader("Upload panel dataset (.csv / .xlsx)", type=["csv", "xlsx", "xls"], key="panel_uploader")
    if uploaded is None:
        st.info("Upload dataset panel untuk mula.")
        st.stop()

    try:
        df_raw, meta_file = _read_uploaded_file(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("Data preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    cols = df_raw.columns.tolist()
    if len(cols) < 3:
        st.error("Dataset terlalu kecil. Perlukan sekurang-kurangnya id, time, y, dan X.")
        st.stop()

    st.subheader("Specification")
    c1, c2, c3 = st.columns(3)
    with c1:
        id_col = st.selectbox("Entity (id) column", cols)
    with c2:
        time_col = st.selectbox("Time column (year/period)", cols)
    with c3:
        y_col = st.selectbox("Dependent variable (Y)", [c for c in cols if c not in {id_col, time_col}])

    x_cols = st.multiselect(
        "Regressors (X)",
        [c for c in cols if c not in {id_col, time_col, y_col}],
        default=[c for c in cols if c not in {id_col, time_col, y_col}][:3],
    )

    st.markdown("#### Data screening: transform suggestions")
    numeric_candidates = [y_col] + x_cols
    screen_df = _screen_transform_suggestions(df_raw, numeric_candidates)
    st.dataframe(screen_df, use_container_width=True)

    st.markdown("#### Transform controls (optional)")
    st.write("Ini bukan auto. Kalau anda apply, anda bertanggungjawab justify dalam penulisan.")

    transform_map = {c: "none" for c in numeric_candidates}
    tcols = st.columns(3)
    for i, c in enumerate(numeric_candidates):
        with tcols[i % 3]:
            default_sug = "none"
            if not screen_df.empty and (screen_df["variable"] == c).any():
                default_sug = str(screen_df.loc[screen_df["variable"] == c, "suggestion"].iloc[0])
            transform_map[c] = st.selectbox(
                f"Transform: {c}",
                ["none", "log", "log1p", "standardize"],
                index=["none", "log", "log1p", "standardize"].index(default_sug if default_sug in ["none", "log", "log1p", "standardize"] else "none"),
                key=f"panel_tr_{c}",
            )

    st.markdown("#### Effects & covariance")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        entity_fe = st.checkbox("Entity fixed effects", value=True)
    with e2:
        time_fe = st.checkbox("Time fixed effects", value=False)
    with e3:
        twoway = st.checkbox("Two-way FE (entity + time)", value=False)
    with e4:
        cov_choice = st.selectbox(
            "Covariance / SE",
            ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both", "driscoll_kraay"],
            index=2,
        )

    if twoway:
        entity_fe = True
        time_fe = True

    st.markdown("#### Run")
    run = st.button("Estimate panel models", type="primary")

    if not run:
        st.stop()

    # Prepare data
    try:
        df_panel, y, X, meta_prep = _panel_prepare(
            df_raw,
            id_col=id_col,
            time_col=time_col,
            y_col=y_col,
            x_cols=x_cols,
            add_time_dummies=(time_fe and not entity_fe),  # for pooled; FE handles time via time_effects
            transform_map=transform_map,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Build covariance kwargs
    cov_kwargs = {}
    cov_type = "unadjusted"
    if cov_choice == "unadjusted":
        cov_type = "unadjusted"
    elif cov_choice == "robust":
        cov_type = "robust"
    elif cov_choice == "cluster_entity":
        cov_type = "clustered"
        cov_kwargs = {"cluster_entity": True}
    elif cov_choice == "cluster_time":
        cov_type = "clustered"
        cov_kwargs = {"cluster_time": True}
    elif cov_choice == "cluster_both":
        cov_type = "clustered"
        cov_kwargs = {"cluster_entity": True, "cluster_time": True}
    elif cov_choice == "driscoll_kraay":
        cov_type = "kernel"
        cov_kwargs = {"kernel": "bartlett"}

    st.subheader("Estimation results")

    results = {}

    # Pooled
    try:
        pooled_mod = PooledOLS(y, X)
        pooled_res = pooled_mod.fit(cov_type=cov_type, **cov_kwargs)
        results["PooledOLS"] = pooled_res
        st.markdown("##### Pooled OLS")
        st.text(pooled_res.summary)
    except Exception as e:
        st.error(f"Pooled OLS failed: {e}")
        pooled_res = None

    # FE
    fe_res = None
    if entity_fe or time_fe:
        try:
            fe_mod = PanelOLS(
                y,
                X,
                entity_effects=bool(entity_fe),
                time_effects=bool(time_fe),
                drop_absorbed=True,
            )
            fe_res = fe_mod.fit(cov_type=cov_type, **cov_kwargs)
            results["FixedEffects"] = fe_res
            st.markdown("##### Fixed Effects")
            st.text(fe_res.summary)
        except Exception as e:
            st.error(f"Fixed Effects failed: {e}")

    # RE (entity RE). If time FE requested, include time dummies in X already? We added only for pooled; so handle here.
    re_res = None
    try:
        X_re = X
        if time_fe:
            # Add time dummies for RE explicitly
            time_index = y.index.get_level_values(1)
            td = pd.get_dummies(time_index, prefix="time", drop_first=True)
            td.index = y.index
            X_re = pd.concat([X, td], axis=1)
            # Avoid duplicate columns
            X_re = X_re.loc[:, ~X_re.columns.duplicated()]

        re_mod = RandomEffects(y, X_re)
        re_res = re_mod.fit(cov_type=cov_type, **cov_kwargs)
        results["RandomEffects"] = re_res
        st.markdown("##### Random Effects (entity RE)")
        st.text(re_res.summary)
    except Exception as e:
        st.error(f"Random Effects failed: {e}")

    st.subheader("Model selection")

    sel_rows = []

    # FE vs pooled
    if fe_res is not None:
        try:
            fp = fe_res.f_pooled
            sel_rows.append({"test": "F-test (FE vs pooled)", "stat": float(fp.stat), "p_value": float(fp.pval), "df": str(fp.df)})
        except Exception as e:
            sel_rows.append({"test": "F-test (FE vs pooled)", "stat": np.nan, "p_value": np.nan, "df": f"error: {e}"})

    # LM test RE vs pooled (balanced only)
    if pooled_res is not None:
        try:
            lm_stat, lm_p = _bp_lm_test_re_vs_pooled_balanced(pooled_res.resids)
            sel_rows.append({"test": "Breusch–Pagan LM (RE vs pooled; balanced only)", "stat": lm_stat, "p_value": lm_p, "df": "chi2(1)"})
        except Exception as e:
            sel_rows.append({"test": "Breusch–Pagan LM (RE vs pooled)", "stat": np.nan, "p_value": np.nan, "df": f"skipped: {e}"})

    # Hausman FE vs RE
    if fe_res is not None and re_res is not None:
        try:
            stat, p, dof = _hausman_test_aligned(fe_res.params, fe_res.cov, re_res.params, re_res.cov)
            sel_rows.append({"test": "Hausman (FE vs RE)", "stat": stat, "p_value": p, "df": f"chi2({dof})"})
        except Exception as e:
            sel_rows.append({"test": "Hausman (FE vs RE)", "stat": np.nan, "p_value": np.nan, "df": f"error: {e}"})

    sel_df = pd.DataFrame(sel_rows)
    st.dataframe(sel_df, use_container_width=True)

    # Recommend model (simple, explicit rules)
    st.subheader("Recommended model (rule-based)")
    rec = "PooledOLS"
    rationale = []

    # If FE vs pooled significant => FE preferred
    fe_p = None
    re_p = None
    haus_p = None

    for _, r in sel_df.iterrows():
        if r["test"].startswith("F-test"):
            fe_p = r["p_value"]
        if r["test"].startswith("Breusch"):
            re_p = r["p_value"]
        if r["test"].startswith("Hausman"):
            haus_p = r["p_value"]

    if (fe_p is not None) and pd.notna(fe_p) and fe_p < 0.05:
        rec = "FixedEffects"
        rationale.append("FE vs pooled: significant → pooled rejected.")

    # If pooled not rejected but RE LM significant -> RE may be preferred
    if rec == "PooledOLS" and (re_p is not None) and pd.notna(re_p) and re_p < 0.05:
        rec = "RandomEffects"
        rationale.append("RE vs pooled: LM significant (balanced only) → pooled rejected.")

    # If both FE and RE plausible, use Hausman
    if (fe_res is not None) and (re_res is not None) and (haus_p is not None) and pd.notna(haus_p):
        if haus_p < 0.05:
            rec = "FixedEffects"
            rationale.append("Hausman significant → RE inconsistent → choose FE.")
        else:
            # Only switch to RE if it exists
            if re_res is not None:
                rec = "RandomEffects"
                rationale.append("Hausman not significant → RE acceptable and efficient.")

    st.write(f"**Recommended:** {rec}")
    if rationale:
        st.write("Rationale:")
        for x in rationale:
            st.write(f"- {x}")

    st.subheader("Diagnostics")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("##### Multicollinearity (VIF)")
        vif_df = _vif_table(X)
        st.dataframe(vif_df, use_container_width=True)

    with d2:
        st.markdown("##### Heteroskedasticity (Breusch–Pagan on pooled residuals)")
        if pooled_res is not None:
            try:
                # BP requires exog without multiindex alignment issues
                y_sm = y.values
                X_sm = X.values
                lm, lm_p, f, f_p = het_breuschpagan(pooled_res.resids.values, X_sm)
                bp_df = pd.DataFrame([
                    {"stat": lm, "p_value": lm_p, "test": "LM"},
                    {"stat": f, "p_value": f_p, "test": "F"},
                ])
                st.dataframe(bp_df, use_container_width=True)
            except Exception as e:
                st.write(f"Skipped: {e}")
        else:
            st.write("Need pooled model to compute this test.")

    with d3:
        st.markdown("##### Cross-sectional dependence (Pesaran CD; balanced only)")
        base_resid = None
        if fe_res is not None:
            base_resid = fe_res.resids
        elif pooled_res is not None:
            base_resid = pooled_res.resids

        if base_resid is not None:
            try:
                cd, cd_p = _pesaran_cd_test_balanced(base_resid)
                st.dataframe(pd.DataFrame([{"CD_stat": cd, "p_value": cd_p}]), use_container_width=True)
            except Exception as e:
                st.write(f"Skipped: {e}")
        else:
            st.write("No residuals available.")

    st.subheader("Export report")

    export_tables: Dict[str, pd.DataFrame] = {
        "panel_selection": sel_df,
        "panel_vif": vif_df,
        "panel_screening": screen_df,
    }

    # Coeff tables
    def _coef_table(res) -> pd.DataFrame:
        if res is None:
            return pd.DataFrame()
        out = pd.DataFrame({
            "coef": res.params,
            "std_err": res.std_errors,
            "t": res.tstats,
            "p": res.pvalues,
        }).reset_index().rename(columns={"index": "term"})
        return out

    export_tables["coef_pooled"] = _coef_table(pooled_res)
    export_tables["coef_fe"] = _coef_table(fe_res)
    export_tables["coef_re"] = _coef_table(re_res)

    meta = {**meta_file, **meta_prep, "recommended_model": rec, "covariance": cov_choice}
    xbytes = _excel_bytes(export_tables, notes=meta)

    st.download_button(
        "Download panel report (Excel)",
        data=xbytes,
        file_name="panel_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# -----------------------------
# TIME SERIES MODULE (separate file, year)
# -----------------------------

else:
    st.header("Time Series Module (Separate File, Annual)")

    uploaded = st.file_uploader("Upload time series dataset (.csv / .xlsx)", type=["csv", "xlsx", "xls"], key="ts_uploader")
    if uploaded is None:
        st.info("Upload dataset siri masa untuk mula.")
        st.stop()

    try:
        df_raw, meta_file = _read_uploaded_file(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("Data preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    cols = df_raw.columns.tolist()
    if "year" in cols:
        default_year = "year"
    else:
        default_year = cols[0]

    c1, c2 = st.columns([1, 3])
    with c1:
        year_col = st.selectbox("Year column", cols, index=cols.index(default_year) if default_year in cols else 0)

    with c2:
        vars_ = st.multiselect(
            "Variables (numeric time series)",
            [c for c in cols if c != year_col],
            default=[c for c in cols if c != year_col][:3],
        )

    if len(vars_) < 2:
        st.error("Pilih sekurang-kurangnya 2 variables untuk VAR/VECM. (ARDL boleh 1+X, tetapi app ini buat screening dulu).")
        st.stop()

    st.subheader("Data screening: transform suggestions")
    screen_df = _screen_transform_suggestions(df_raw, vars_)
    st.dataframe(screen_df, use_container_width=True)

    st.markdown("#### Transform controls (optional)")
    transform_map = {c: "none" for c in vars_}
    tcols = st.columns(3)
    for i, c in enumerate(vars_):
        with tcols[i % 3]:
            default_sug = "none"
            if not screen_df.empty and (screen_df["variable"] == c).any():
                default_sug = str(screen_df.loc[screen_df["variable"] == c, "suggestion"].iloc[0])
            transform_map[c] = st.selectbox(
                f"Transform: {c}",
                ["none", "log", "log1p", "standardize"],
                index=["none", "log", "log1p", "standardize"].index(default_sug if default_sug in ["none", "log", "log1p", "standardize"] else "none"),
                key=f"ts_tr_{c}",
            )

    # Prepare TS data
    try:
        ts_df, meta_prep = _ts_prepare(df_raw, year_col=year_col, vars_=vars_, transform_map=transform_map)
    except Exception as e:
        st.error(str(e))
        st.stop()

    T = len(ts_df)
    st.write(f"**Sample length (T)** = {T} (annual)")

    st.subheader("Stationarity screening (ADF + KPSS)")
    adf_reg = st.selectbox("ADF deterministic term", ["c", "ct"], index=0, help="c=constant, ct=constant+trend")
    kpss_reg = st.selectbox("KPSS deterministic term", ["c", "ct"], index=0)

    stat_df = _stationarity_suite(ts_df, vars_, adf_reg=adf_reg, kpss_reg=kpss_reg)
    st.dataframe(stat_df, use_container_width=True)

    st.markdown("**Interpretation rule used:** ADF p<0.05 AND KPSS p>0.05 → I(0); ADF p>=0.05 AND KPSS p<=0.05 → I(1) likely; else mixed/unclear.")

    st.subheader("Model")
    model = st.radio("Choose time series model", ["VAR", "VECM", "ARDL"], horizontal=True)

    if model == "VAR":
        st.markdown("### VAR")
        endog = st.multiselect("Endogenous variables", vars_, default=vars_)
        if len(endog) < 2:
            st.error("VAR requires at least 2 endogenous variables.")
            st.stop()

        # Guardrail for annual T=40: keep maxlag reasonable by default
        default_maxlag = min(4, max(1, T // 10))
        maxlags = st.slider("Max lag to consider (annual guardrail)", 1, min(8, max(2, T // 5)), default_maxlag)
        ic = st.selectbox("Lag selection criterion", ["bic", "aic", "hqic"], index=0)
        trend = st.selectbox("Deterministic term", ["c", "ct", "n"], index=0)

        run = st.button("Fit VAR", type="primary")
        if not run:
            st.stop()

        data = ts_df[endog].copy()

        try:
            var_mod = VAR(data)
            sel = var_mod.select_order(maxlags=maxlags, trend=trend)
            lag = int(getattr(sel, ic))
            if lag < 1:
                lag = 1

            var_res = var_mod.fit(lag, trend=trend)

            st.markdown(f"**Selected lag ({ic})** = {lag}")
            st.text(var_res.summary())

            # Diagnostics
            st.subheader("VAR diagnostics")
            diag_rows = []
            try:
                diag_rows.append({"test": "stability", "result": str(var_res.is_stable(verbose=False))})
            except Exception:
                pass
            try:
                wn = var_res.test_whiteness(nlags=lag)
                diag_rows.append({"test": "whiteness", "stat": float(wn.statistic), "p_value": float(wn.pvalue)})
            except Exception:
                pass
            try:
                nm = var_res.test_normality()
                diag_rows.append({"test": "normality", "stat": float(nm.statistic), "p_value": float(nm.pvalue)})
            except Exception:
                pass

            st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

            # Causality
            st.subheader("Causality (model-consistent)")
            st.write("Matrix below shows p-values for H0: 'causing' does NOT Granger-cause 'caused' (Wald test).")
            caus_df = _matrix_from_var_causality(var_res, endog)
            st.dataframe(caus_df, use_container_width=True)

            # Export
            export = {
                "ts_screening": screen_df,
                "ts_stationarity": stat_df,
                "var_lag_selection": pd.DataFrame(sel.summary().as_html(), index=[0]),
                "var_causality_p": caus_df,
            }

            # Coefs
            coef_tables = []
            for eqn in endog:
                params = var_res.params[eqn].reset_index().rename(columns={"index": "term", eqn: "coef"})
                params["equation"] = eqn
                coef_tables.append(params)
            export["var_params"] = pd.concat(coef_tables, ignore_index=True)

            meta = {**meta_file, **meta_prep, "model": "VAR", "endog": ",".join(endog), "selected_lag": str(lag), "trend": trend, "ic": ic}
            xbytes = _excel_bytes(export, notes=meta)

            st.download_button(
                "Download TS report (Excel)",
                data=xbytes,
                file_name="ts_report_VAR.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"VAR failed: {e}")


    elif model == "VECM":
        st.markdown("### VECM")
        endog = st.multiselect("Endogenous variables", vars_, default=vars_)
        if len(endog) < 2:
            st.error("VECM requires at least 2 endogenous variables.")
            st.stop()

        # Gatekeeping: all must be I(1) likely
        stat_subset = stat_df[stat_df["variable"].isin(endog)].copy()
        not_i1 = stat_subset[~stat_subset["classification"].str.contains("I\(1\)", regex=True)]
        if len(not_i1) > 0:
            st.error("VECM gatekeeping failed: semua endog perlu I(1) (at least 'I(1) likely' under ADF+KPSS rule).")
            st.dataframe(not_i1, use_container_width=True)
            st.stop()

        # Annual guardrail
        default_k = 1
        k_ar_diff = st.slider("Lag in differences (k_ar_diff)", 1, min(4, max(1, T // 10)), default_k)
        det = st.selectbox("Deterministic term", ["nc", "co", "ct"], index=1, help="nc: none, co: constant in cointegration relation, ct: constant+trend")

        run = st.button("Fit VECM", type="primary")
        if not run:
            st.stop()

        data = ts_df[endog].copy()

        try:
            # Johansen test for rank decision
            # det_order mapping for coint_johansen: -1 none, 0 constant, 1 trend
            det_order = {"nc": -1, "co": 0, "ct": 1}[det]
            joh = coint_johansen(data, det_order, k_ar_diff)

            # Build trace table
            trace = joh.lr1
            cv_5 = joh.cvt[:, 1]
            rank = int(np.sum(trace > cv_5))

            jtab = pd.DataFrame({
                "rank": list(range(len(trace))),
                "trace_stat": trace,
                "crit_5pct": cv_5,
                "reject_H0_rank_leq_r": trace > cv_5,
            })

            st.subheader("Johansen cointegration test (trace)")
            st.dataframe(jtab, use_container_width=True)
            st.write(f"**Selected rank (5%)** = {rank}")

            if rank == 0:
                st.error("No cointegration (rank=0). VECM not appropriate. Use VAR on differenced data or ARDL.")
                st.stop()

            vecm_mod = VECM(data, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic=det)
            vecm_res = vecm_mod.fit()

            st.text(vecm_res.summary())

            st.subheader("Causality (model-consistent)")
            st.write("p-values for H0: 'causing' does NOT Granger-cause 'caused' (VECM test).")
            caus_df = _matrix_from_vecm_causality(vecm_res, endog)
            st.dataframe(caus_df, use_container_width=True)

            # Diagnostics (available in statsmodels)
            st.subheader("VECM diagnostics")
            diag_rows = []
            try:
                wn = vecm_res.test_whiteness(nlags=k_ar_diff)
                diag_rows.append({"test": "whiteness", "stat": float(wn.statistic), "p_value": float(wn.pvalue)})
            except Exception:
                pass
            try:
                nm = vecm_res.test_normality()
                diag_rows.append({"test": "normality", "stat": float(nm.statistic), "p_value": float(nm.pvalue)})
            except Exception:
                pass
            st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

            export = {
                "ts_screening": screen_df,
                "ts_stationarity": stat_df,
                "johansen_trace": jtab,
                "vecm_causality_p": caus_df,
            }

            meta = {**meta_file, **meta_prep, "model": "VECM", "endog": ",".join(endog), "k_ar_diff": str(k_ar_diff), "deterministic": det, "rank": str(rank)}
            xbytes = _excel_bytes(export, notes=meta)

            st.download_button(
                "Download TS report (Excel)",
                data=xbytes,
                file_name="ts_report_VECM.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"VECM failed: {e}")


    else:
        st.markdown("### ARDL")

        y = st.selectbox("Dependent variable (Y)", vars_)
        x_list = st.multiselect("Regressors (X)", [v for v in vars_ if v != y], default=[v for v in vars_ if v != y][:2])
        if len(x_list) == 0:
            st.error("ARDL requires at least one regressor X.")
            st.stop()

        # Guardrail for annual T=40
        max_p = st.slider("Max lag for Y (p)", 1, min(4, max(2, T // 10)), 2)
        max_q = st.slider("Max lag for X (q)", 0, min(4, max(1, T // 10)), 2)
        ic = st.selectbox("Model selection IC", ["bic", "aic"], index=0)
        trend = st.selectbox("Deterministic term", ["c", "ct", "n"], index=0)

        run = st.button("Fit ARDL", type="primary")
        if not run:
            st.stop()

        data_y = ts_df[y]
        data_X = ts_df[x_list]

        # Gatekeeping: block if any I(2) suspicion (we only have I(0)/I(1)/mixed classification here)
        # We warn if any variable is mixed/unclear
        unclear = stat_df[stat_df["variable"].isin([y] + x_list) & stat_df["classification"].str.contains("Mixed|Insufficient", regex=True)]
        if len(unclear) > 0:
            st.warning("Stationarity classification is mixed/unclear for some variables. ARDL bounds inference may be unreliable. Consider transformations or differencing.")
            st.dataframe(unclear, use_container_width=True)

        try:
            sel = ardl_select_order(data_y, data_X, maxlag=max_p, maxorder=max_q, ic=ic, trend=trend)
            ardl_res = sel.model.fit()

            st.markdown("**Selected ARDL order**")
            st.write(sel)
            st.text(ardl_res.summary())

            st.subheader("ARDL diagnostics")
            diag_rows = []
            try:
                sc = ardl_res.test_serial_correlation()
                diag_rows.append({"test": "serial_correlation", "stat": float(sc.statistic), "p_value": float(sc.pvalue)})
            except Exception:
                pass
            try:
                ht = ardl_res.test_heteroskedasticity()
                diag_rows.append({"test": "heteroskedasticity", "stat": float(ht.statistic), "p_value": float(ht.pvalue)})
            except Exception:
                pass
            try:
                nm = ardl_res.test_normality()
                diag_rows.append({"test": "normality", "stat": float(nm.statistic), "p_value": float(nm.pvalue)})
            except Exception:
                pass

            st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

            st.subheader("Causality (model-consistent)")
            st.write("Granger-style: H0 all lagged terms of X are jointly zero (Wald test on ARDL regression terms).")
            caus_df = _ardl_granger_style_tests(ardl_res, y=y, x_list=x_list)
            st.dataframe(caus_df, use_container_width=True)

            export = {
                "ts_screening": screen_df,
                "ts_stationarity": stat_df,
                "ardl_causality": caus_df,
            }

            # Coef table
            coef = pd.DataFrame({
                "term": ardl_res.params.index,
                "coef": ardl_res.params.values,
                "std_err": ardl_res.bse.values,
                "t": ardl_res.tvalues.values,
                "p": ardl_res.pvalues.values,
            })
            export["ardl_params"] = coef
            export["ardl_diagnostics"] = pd.DataFrame(diag_rows)

            meta = {**meta_file, **meta_prep, "model": "ARDL", "Y": y, "X": ",".join(x_list), "max_p": str(max_p), "max_q": str(max_q), "ic": ic, "trend": trend}
            xbytes = _excel_bytes(export, notes=meta)

            st.download_button(
                "Download TS report (Excel)",
                data=xbytes,
                file_name="ts_report_ARDL.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"ARDL failed: {e}")

