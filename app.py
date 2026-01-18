"""Academic Econometrics Suite (Streamlit) — Panel + Time Series (separate file)

GOAL
- Provide defensible, reviewer-friendly workflows for:
  (A) Panel data: Pooled OLS, FE (entity/time/two-way), RE, plus CRE (Mundlak)
  (B) Time series (annual; year index): VAR, VECM, ARDL

KEY DESIGN PRINCIPLES
- Validation first: refuse to run invalid setups (duplicate time index, object dtypes in regressors, etc.).
- No fake results: remove naive 'panel Granger' on stacked data.
- Reproducibility: export audit-ready Excel (or ZIP of CSV if Excel engine missing).
- Academic guidance: explicit hypotheses and decision rules shown in-app.

DEPENDENCIES (recommended in requirements.txt)
- streamlit
- pandas
- numpy
- scipy
- statsmodels
- linearmodels
- matplotlib
- openpyxl OR xlsxwriter (for Excel export)

"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from scipy import stats

# plotting
import matplotlib.pyplot as plt

# Panel models
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

# General OLS / helpers
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Time-series
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ardl import ARDL, ardl_select_order


# ============================================================
# UI / STYLE
# ============================================================

st.set_page_config(page_title="Academic Econometrics Suite", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
<style>
:root{
  --bg: #0b1220;
  --card: #101a2e;
  --card2:#0f1a33;
  --text: #e9eef8;
  --muted:#a8b3cf;
  --accent:#4fd1c5;
  --accent2:#7aa2ff;
  --warn:#ffcc66;
  --bad:#ff6b6b;
  --good:#8be28b;
}

html, body, [class*="css"]  {
  color: var(--text) !important;
}

.block-container{padding-top: 1.3rem;}

/* Header */
.app-title{
  font-size: 1.55rem;
  font-weight: 800;
  letter-spacing: .2px;
  margin: 0 0 0.25rem 0;
}
.app-sub{
  color: var(--muted);
  margin-top: 0;
}

/* Cards */
.card{
  background: linear-gradient(180deg, rgba(16,26,46,0.95) 0%, rgba(10,18,35,0.95) 100%);
  border: 1px solid rgba(122,162,255,0.18);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
.card h3{margin:0 0 .35rem 0; font-size:1.05rem;}
.card p{margin:.15rem 0; color: var(--muted);}

.badge{
  display:inline-block;
  padding:2px 10px;
  border-radius:999px;
  font-size:.78rem;
  border:1px solid rgba(79,209,197,0.35);
  color: var(--text);
  background: rgba(79,209,197,0.08);
  margin-right:.35rem;
}

.badge-warn{border-color: rgba(255,204,102,0.45); background: rgba(255,204,102,0.10);}
.badge-bad{border-color: rgba(255,107,107,0.55); background: rgba(255,107,107,0.10);}

.hr{
  height:1px;
  background: rgba(168,179,207,0.18);
  margin: 10px 0 14px 0;
}

/* Tables (HTML) */
.table-wrap{margin: 0.35rem 0 1.2rem 0;}
.table-title{font-weight:700; margin: 0 0 0.35rem 0; color: var(--text);}

.table-wrap table{
  width: 100%;
  border-collapse: collapse;
  border: 1px solid rgba(168,179,207,0.35);
  border-radius: 12px;
  overflow:hidden;
}
.table-wrap th, .table-wrap td{
  border: 1px solid rgba(168,179,207,0.25);
  padding: 8px 10px;
  font-size: 0.92rem;
}
.table-wrap th{
  background: rgba(122,162,255,0.14);
  color: var(--text);
  text-align: left;
}
.table-wrap tr:nth-child(even) td{
  background: rgba(255,255,255,0.02);
}

/* Small helper text */
.small{color: var(--muted); font-size: 0.9rem;}

</style>
""",
        unsafe_allow_html=True,
    )


inject_css()

st.markdown(
    """
<div class="app-title">Academic Econometrics Suite</div>
<p class="app-sub">Panel Data + Time Series (separate file, annual). Built for defensible workflows: validation → tests → model choice → diagnostics → export.</p>
<div class="hr"></div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UTILITIES
# ============================================================


def html_table(df: pd.DataFrame, title: str = "", max_rows: int = 300) -> None:
    """Render a DataFrame as an HTML table with vertical & horizontal gridlines."""
    if df is None:
        return
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)
        title = f"{title} (showing first {max_rows} rows)"

    # avoid very wide floats
    for c in df2.columns:
        if pd.api.types.is_float_dtype(df2[c]):
            df2[c] = df2[c].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")

    html = df2.to_html(index=False, escape=False)
    st.markdown(
        f"""
<div class="table-wrap">
  <div class="table-title">{title}</div>
  {html}
</div>
""",
        unsafe_allow_html=True,
    )


def warn_box(msg: str) -> None:
    st.markdown(f"<span class='badge badge-warn'>Warning</span> {msg}", unsafe_allow_html=True)


def bad_box(msg: str) -> None:
    st.markdown(f"<span class='badge badge-bad'>Critical</span> {msg}", unsafe_allow_html=True)


def good_box(msg: str) -> None:
    st.markdown(f"<span class='badge'>OK</span> {msg}", unsafe_allow_html=True)


def _read_uploaded(uploaded) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Read CSV/XLSX from Streamlit uploader."""
    if uploaded is None:
        raise ValueError("No file uploaded")

    meta = {"file_name": getattr(uploaded, "name", "uploaded")}
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


def _engine_for_excel() -> Optional[str]:
    """Return an available Excel writer engine, or None if none available."""
    try:
        import openpyxl  # noqa: F401

        return "openpyxl"
    except Exception:
        pass
    try:
        import xlsxwriter  # noqa: F401

        return "xlsxwriter"
    except Exception:
        return None


def export_report_bytes(sheets: Dict[str, pd.DataFrame], meta: Dict[str, str]) -> Tuple[bytes, str]:
    """Export tables as Excel if possible; otherwise ZIP of CSVs."""
    engine = _engine_for_excel()

    # Clean sheet names and ensure df
    safe_sheets: Dict[str, pd.DataFrame] = {}
    for k, v in sheets.items():
        if v is None:
            continue
        if not isinstance(v, pd.DataFrame):
            v = pd.DataFrame(v)
        safe_sheets[k[:31]] = v

    meta_df = pd.DataFrame({"key": list(meta.keys()), "value": list(meta.values())})

    if engine is not None:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine=engine) as writer:
            meta_df.to_excel(writer, sheet_name="meta", index=False)
            for name, df in safe_sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        bio.seek(0)
        return bio.read(), "xlsx"

    # ZIP fallback
    zbio = io.BytesIO()
    with zipfile.ZipFile(zbio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.csv", meta_df.to_csv(index=False))
        for name, df in safe_sheets.items():
            zf.writestr(f"{name[:31]}.csv", df.to_csv(index=False))
    zbio.seek(0)
    return zbio.read(), "zip"


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def screen_transform_suggestions(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Simple, explainable screening for suggesting log/log1p transforms."""
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        zeros = int((s == 0).sum())
        neg = int((s < 0).sum())
        skew = float(stats.skew(s, bias=False)) if len(s) >= 3 else np.nan

        # outlier share via IQR
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            out_share = 0.0
        else:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out_share = float(((s < lo) | (s > hi)).mean())

        sugg = "none"
        reason = ""
        if neg > 0:
            sugg = "none"
            reason = "Negative values detected; log transforms not valid."
        else:
            if (not np.isnan(skew)) and abs(skew) >= 1.2:
                if zeros > 0:
                    sugg = "log1p"
                    reason = "High skewness with zeros; log1p can reduce skew."
                else:
                    sugg = "log"
                    reason = "High skewness (all positive); log can stabilize variance."
            else:
                sugg = "none"
                reason = "No strong skewness signal (rule-of-thumb)."

        rows.append(
            {
                "variable": c,
                "min": float(s.min()),
                "max": float(s.max()),
                "zeros": zeros,
                "negatives": neg,
                "skewness": skew,
                "outlier_share": out_share,
                "suggestion": sugg,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def apply_transforms(df: pd.DataFrame, transform_map: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = df.copy()
    notes: Dict[str, str] = {}
    for col, tr in transform_map.items():
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        if tr == "none":
            continue
        if tr == "log":
            if (s <= 0).any():
                notes[col] = "SKIP log: non-positive values"
                continue
            out[col] = np.log(s)
            notes[col] = "Applied log"
        elif tr == "log1p":
            if (s < 0).any():
                notes[col] = "SKIP log1p: negative values"
                continue
            out[col] = np.log1p(s)
            notes[col] = "Applied log1p"
        elif tr == "zscore":
            out[col] = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0)
            notes[col] = "Applied z-score"
        else:
            notes[col] = f"Unknown transform '{tr}' ignored"
    return out, notes


# ============================================================
# TIME SERIES: STATIONARITY + TRANSFORMS
# ============================================================


def adf_test(series: pd.Series, regression: str = "c") -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 8:
        return {"adf_stat": np.nan, "p_value": np.nan, "nobs": len(s)}
    stat, p, _, _, _, _ = adfuller(s, regression=regression, autolag="AIC")
    return {"adf_stat": float(stat), "p_value": float(p), "nobs": float(len(s))}


def kpss_test(series: pd.Series, regression: str = "c") -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 8:
        return {"kpss_stat": np.nan, "p_value": np.nan, "nobs": len(s)}
    stat, p, _, _ = kpss(s, regression=regression, nlags="auto")
    return {"kpss_stat": float(stat), "p_value": float(p), "nobs": float(len(s))}


def integration_order(series: pd.Series, regression: str = "c") -> Tuple[int, pd.DataFrame]:
    """Classify I(0)/I(1)/I(2) using ADF+KPSS on level and diffs.

    Decision rule (simple and explicit):
    - Stationary if ADF p<0.05 and KPSS p>0.05
    - Otherwise, difference and test again

    Returns (d, table) where d is 0/1/2 and table shows results.
    """
    s0 = pd.to_numeric(series, errors="coerce")

    def _is_stationary(s: pd.Series) -> Tuple[bool, Dict[str, float], Dict[str, float]]:
        a = adf_test(s, regression=regression)
        k = kpss_test(s, regression=regression)
        ok = (not np.isnan(a["p_value"])) and (a["p_value"] < 0.05) and (not np.isnan(k["p_value"])) and (k["p_value"] > 0.05)
        return ok, a, k

    rows = []
    ok0, a0, k0 = _is_stationary(s0)
    rows.append({"order": "level", **a0, **{f"{k}": v for k, v in k0.items() if k != "nobs"}, "kpss_nobs": k0.get("nobs")})
    if ok0:
        return 0, pd.DataFrame(rows)

    s1 = s0.diff().dropna()
    ok1, a1, k1 = _is_stationary(s1)
    rows.append({"order": "diff1", **a1, **{f"{k}": v for k, v in k1.items() if k != "nobs"}, "kpss_nobs": k1.get("nobs")})
    if ok1:
        return 1, pd.DataFrame(rows)

    s2 = s1.diff().dropna()
    ok2, a2, k2 = _is_stationary(s2)
    rows.append({"order": "diff2", **a2, **{f"{k}": v for k, v in k2.items() if k != "nobs"}, "kpss_nobs": k2.get("nobs")})
    if ok2:
        return 2, pd.DataFrame(rows)

    # fallback: treat as I(2+) if still not stationary
    return 2, pd.DataFrame(rows)


def plot_series(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(df.index, df.values)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)


def plot_multi_series(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots()
    for c in df.columns:
        ax.plot(df.index, df[c].values, label=c)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)


def plot_residual_diagnostics(resid: pd.Series, fitted: Optional[pd.Series] = None, title_prefix: str = "") -> None:
    r = pd.to_numeric(resid, errors="coerce").dropna()
    if len(r) < 5:
        return

    # Residual vs fitted (if available)
    if fitted is not None:
        f = pd.to_numeric(fitted, errors="coerce").reindex(r.index).dropna()
        common = r.index.intersection(f.index)
        if len(common) >= 5:
            fig, ax = plt.subplots()
            ax.scatter(f.loc[common], r.loc[common])
            ax.axhline(0, linewidth=1)
            ax.set_title(f"{title_prefix} Residual vs Fitted")
            ax.set_xlabel("Fitted")
            ax.set_ylabel("Residual")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, clear_figure=True)

    # QQ plot
    fig, ax = plt.subplots()
    sm.qqplot(r, line="45", ax=ax)
    ax.set_title(f"{title_prefix} Residual QQ plot")
    st.pyplot(fig, clear_figure=True)


# ============================================================
# PANEL: PREPARE + TESTS
# ============================================================


def panel_validate(df: pd.DataFrame, id_col: str, time_col: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}

    if id_col not in df.columns or time_col not in df.columns:
        raise ValueError("id_col or time_col not found")

    d = df.copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[id_col] = d[id_col].astype(str)

    if d[time_col].isna().any():
        raise ValueError("Time column contains NA after numeric conversion")

    dup = d.duplicated([id_col, time_col]).sum()
    if dup > 0:
        raise ValueError(f"Duplicate (entity,time) rows detected: {dup}")

    # Balanced/unbalanced
    counts = d.groupby(id_col)[time_col].nunique()
    meta["N_entities"] = str(int(counts.shape[0]))
    meta["T_min"] = str(int(counts.min()))
    meta["T_median"] = str(float(counts.median()))
    meta["T_max"] = str(int(counts.max()))
    meta["balanced"] = str(bool(counts.min() == counts.max()))

    return meta


def panel_design_matrix(df: pd.DataFrame, y: str, x_cols: List[str], add_const: bool = True) -> Tuple[pd.Series, pd.DataFrame, Dict[str, str]]:
    """Build y and X. Auto-dummy categorical columns (drop_first)."""
    notes: Dict[str, str] = {}

    yv = pd.to_numeric(df[y], errors="coerce")
    Xraw = df[x_cols].copy()

    # Identify categorical/object columns
    cat_cols = [c for c in Xraw.columns if (Xraw[c].dtype == "object" or str(Xraw[c].dtype).startswith("category"))]
    if len(cat_cols) > 0:
        notes["dummies"] = f"Created dummies for: {', '.join(cat_cols)} (drop_first=True)"
        Xraw = pd.get_dummies(Xraw, columns=cat_cols, drop_first=True)

    # numeric conversion
    for c in Xraw.columns:
        Xraw[c] = pd.to_numeric(Xraw[c], errors="coerce")

    # drop rows with NA in y or X
    data = pd.concat([yv.rename("__y__"), Xraw], axis=1)
    before = len(data)
    data = data.dropna()
    after = len(data)
    if after < before:
        notes["listwise_delete"] = f"Dropped {before-after} rows due to missing values in y/X"

    yv2 = data["__y__"]
    X2 = data.drop(columns=["__y__"])

    if add_const:
        X2 = sm.add_constant(X2, has_constant="add")

    return yv2, X2, notes


def panel_add_time_dummies(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    dummies = pd.get_dummies(df[time_col].astype(int), prefix="T", drop_first=True)
    return dummies


def panel_vif(X: pd.DataFrame) -> pd.DataFrame:
    Xv = X.copy()
    if "const" in Xv.columns:
        Xv = Xv.drop(columns=["const"])
    if Xv.shape[1] <= 1:
        return pd.DataFrame({"variable": Xv.columns, "VIF": [np.nan] * Xv.shape[1]})

    vals = []
    for i, col in enumerate(Xv.columns):
        try:
            v = variance_inflation_factor(Xv.values, i)
        except Exception:
            v = np.nan
        vals.append({"variable": col, "VIF": float(v) if v is not None else np.nan})
    return pd.DataFrame(vals).sort_values("VIF", ascending=False)


def panel_bp_hetero(pool_res) -> Optional[pd.DataFrame]:
    try:
        e = np.asarray(pool_res.resids)
        X = np.asarray(pool_res.model.exog)
        lm, pval, fval, fp = het_breuschpagan(e, X)
        return pd.DataFrame(
            {
                "test": ["Breusch–Pagan"],
                "lm_stat": [float(lm)],
                "lm_p": [float(pval)],
                "f_stat": [float(fval)],
                "f_p": [float(fp)],
            }
        )
    except Exception:
        return None


def panel_pesaran_cd(resids: pd.Series, id_col: str, time_col: str, df_raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Pesaran CD test (balanced only)."""
    try:
        tmp = df_raw[[id_col, time_col]].copy()
        tmp["resid"] = resids.values
        piv = tmp.pivot(index=time_col, columns=id_col, values="resid")
        if piv.isna().any().any():
            return None
        T = piv.shape[0]
        N = piv.shape[1]
        if N < 3 or T < 5:
            return None
        C = piv.corr().values
        # average off-diagonal correlation
        off = C[np.triu_indices(N, 1)]
        rho_bar = float(np.mean(off))
        cd = math.sqrt((2 * T) / (N * (N - 1))) * np.sum(off)
        p = 2 * (1 - stats.norm.cdf(abs(cd)))
        return pd.DataFrame(
            {
                "test": ["Pesaran CD"],
                "N": [N],
                "T": [T],
                "CD_stat": [float(cd)],
                "p_value": [float(p)],
                "avg_corr": [rho_bar],
            }
        )
    except Exception:
        return None


def panel_resid_ar1_test(resids: pd.Series, id_col: str, time_col: str, df_raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Indicative residual AR(1) test: e_it = rho e_i,t-1 + u_it. Not Wooldridge; but flags serial correlation."""
    try:
        tmp = df_raw[[id_col, time_col]].copy()
        tmp["resid"] = resids.values
        tmp = tmp.sort_values([id_col, time_col])
        tmp["resid_l1"] = tmp.groupby(id_col)["resid"].shift(1)
        tmp = tmp.dropna()
        if len(tmp) < 10:
            return None
        X = sm.add_constant(tmp["resid_l1"].values)
        y = tmp["resid"].values
        fit = sm.OLS(y, X).fit()
        rho = float(fit.params[1])
        p = float(fit.pvalues[1])
        return pd.DataFrame({"test": ["Residual AR(1)"], "rho": [rho], "p_value": [p], "nobs": [len(tmp)]})
    except Exception:
        return None


def hausman_aligned(fe_res, re_res) -> pd.DataFrame:
    """Hausman test with coefficient alignment and pseudoinverse; warns if not reliable."""
    b_fe = fe_res.params
    b_re = re_res.params

    # drop intercept if present
    b_fe = b_fe.drop(labels=["const"], errors="ignore")
    b_re = b_re.drop(labels=["const"], errors="ignore")

    common = b_fe.index.intersection(b_re.index)
    if len(common) == 0:
        return pd.DataFrame({"hausman_stat": [np.nan], "df": [0], "p_value": [np.nan], "note": ["No common coefficients"]})

    bdiff = (b_fe[common] - b_re[common]).values.reshape(-1, 1)

    V_fe = fe_res.cov
    V_re = re_res.cov
    V_fe = V_fe.loc[common, common]
    V_re = V_re.loc[common, common]

    Vdiff = (V_fe - V_re).values
    # pseudo-inverse to handle singular
    Vinv = np.linalg.pinv(Vdiff)
    statv = float((bdiff.T @ Vinv @ bdiff).ravel()[0])
    df = int(len(common))
    p = float(1 - stats.chi2.cdf(statv, df))

    note = "OK"
    if not np.isfinite(statv) or statv < 0:
        note = "Not reliable (non-positive or non-finite statistic)."

    return pd.DataFrame({"hausman_stat": [statv], "df": [df], "p_value": [p], "note": [note]})


def re_vs_pooled_lr_test(df_raw: pd.DataFrame, id_col: str, time_col: str, y: str, X: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Likelihood ratio test: random intercept (MixedLM) vs pooled OLS.

    This is an academically defensible alternative to BP LM, and works for unbalanced panels.
    """
    try:
        # Build a dataframe for statsmodels
        tmp = df_raw[[id_col, time_col]].copy()
        tmp["y"] = pd.to_numeric(df_raw.loc[X.index, y].values, errors="coerce")
        for c in X.columns:
            tmp[c] = X[c].values

        tmp = tmp.dropna()
        # Pooled OLS
        ols = sm.OLS(tmp["y"].values, tmp[X.columns].values).fit()

        # MixedLM random intercept
        md = sm.MixedLM(tmp["y"].values, tmp[X.columns].values, groups=tmp[id_col].values)
        re = md.fit(reml=False, method="lbfgs", disp=False)

        lr = 2 * (re.llf - ols.llf)
        p = 1 - stats.chi2.cdf(lr, 1)

        return pd.DataFrame({"test": ["Random intercept LR"], "lr_stat": [float(lr)], "df": [1], "p_value": [float(p)]})
    except Exception:
        return None


# ============================================================
# TIME SERIES: COINTEGRATION / CAUSALITY HELPERS
# ============================================================


def johansen_table(df: pd.DataFrame, det_order: int, k_ar_diff: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    j = coint_johansen(df.values, det_order, k_ar_diff)

    trace = pd.DataFrame(
        {
            "rank": list(range(len(j.lr1))),
            "trace_stat": j.lr1,
            "cv_90": j.cvt[:, 0],
            "cv_95": j.cvt[:, 1],
            "cv_99": j.cvt[:, 2],
        }
    )
    maxeig = pd.DataFrame(
        {
            "rank": list(range(len(j.lr2))),
            "maxeig_stat": j.lr2,
            "cv_90": j.cvm[:, 0],
            "cv_95": j.cvm[:, 1],
            "cv_99": j.cvm[:, 2],
        }
    )
    return trace, maxeig


def engle_granger_cointegration(y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """Residual-based cointegration check (Engle–Granger):

    1) regress y on X (levels)
    2) ADF on residuals (no constant) using statsmodels' approximate p-value

    Note: Critical values differ from standard ADF; treat as indicative.
    """
    try:
        data = pd.concat([y.rename("y"), X], axis=1).dropna()
        y2 = data["y"]
        X2 = sm.add_constant(data.drop(columns=["y"]))
        fit = sm.OLS(y2.values, X2.values).fit()
        resid = pd.Series(fit.resid, index=data.index)
        stat, p, *_ = adfuller(resid.dropna().values, regression="n", autolag="AIC")
        return pd.DataFrame({"test": ["Engle–Granger (ADF on residuals)"], "adf_stat": [float(stat)], "p_value": [float(p)]})
    except Exception:
        return pd.DataFrame({"test": ["Engle–Granger"], "adf_stat": [np.nan], "p_value": [np.nan]})


def ardl_long_run(ardl_res) -> pd.DataFrame:
    """Compute long-run coefficients for ARDL: sum(beta_x_lags) / (1 - sum(phi_y_lags))."""
    params = ardl_res.params
    # Identify y-lag params
    ylag = [k for k in params.index if k.startswith("y.L") or ".L" in k and k.startswith("y")]
    # statsmodels ARDL names vary; we'll infer by model's endog name
    endog_name = ardl_res.model.endog_names
    ylag = [k for k in params.index if k.startswith(f"{endog_name}.") and ".L" in k]

    denom = 1.0
    for k in ylag:
        denom -= float(params[k])

    # Collect regressor long-run
    rows = []
    for nm in params.index:
        if nm == "const" or nm == "trend":
            continue
        if nm.startswith(f"{endog_name}."):
            continue
        # group by base regressor before .L
        base = nm.split(".")[0]
        # sum all lags for that base
    bases = {}
    for nm in params.index:
        if nm == "const" or nm == "trend":
            continue
        if nm.startswith(f"{endog_name}."):
            continue
        base = nm.split(".")[0]
        bases.setdefault(base, 0.0)
        bases[base] += float(params[nm])

    for base, num in bases.items():
        rows.append({"variable": base, "long_run_coef": num / denom if denom != 0 else np.nan, "denom": denom})

    return pd.DataFrame(rows)


def ardl_ecm(y: pd.Series, X: pd.DataFrame, ardl_res) -> Tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    """Estimate a simple ECM implied by ARDL order.

    Steps:
    - Compute long-run coefficients (theta)
    - ECT_{t-1} = y_{t-1} - (c + theta' X_{t-1})
    - Regress Δy_t on Δy lags, ΔX lags, and ECT_{t-1}

    This provides the key academic quantity: speed of adjustment (ECT coefficient).
    """
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    yv = data["y"]
    Xv = data.drop(columns=["y"])

    # Long-run
    lr = ardl_long_run(ardl_res)
    theta = {r["variable"]: r["long_run_coef"] for _, r in lr.iterrows()}

    # Constant
    const = float(ardl_res.params.get("const", 0.0))

    # ECT
    ect = yv.shift(1) - (const + sum(theta.get(col, 0.0) * Xv[col].shift(1) for col in Xv.columns))

    # Build short-run regressors based on ARDL selected lags
    # We parse model lags from ardl_res.model
    p = int(getattr(ardl_res.model, "_max_lag", 1))  # rough

    dy = yv.diff()
    df_ecm = pd.DataFrame({"dy": dy, "ect_l1": ect})

    # Add Δy lags up to p-1
    for L in range(1, max(1, p)):
        df_ecm[f"dy_l{L}"] = dy.shift(L)

    # Add ΔX lags using observed lag structure in params (best-effort)
    for col in Xv.columns:
        dx = Xv[col].diff()
        df_ecm[f"d{col}"] = dx
        # add one lag by default
        df_ecm[f"d{col}_l1"] = dx.shift(1)

    df_ecm = df_ecm.dropna()

    Y = df_ecm["dy"].values
    Xmat = sm.add_constant(df_ecm.drop(columns=["dy"]).values)
    fit = sm.OLS(Y, Xmat).fit()

    # output table
    out = pd.DataFrame(
        {
            "term": ["const"] + [c for c in df_ecm.drop(columns=["dy"]).columns],
            "coef": fit.params,
            "std_err": fit.bse,
            "t": fit.tvalues,
            "p_value": fit.pvalues,
        }
    )

    return out, fit


# ============================================================
# SIDEBAR NAV
# ============================================================

with st.sidebar:
    st.markdown("<div class='card'><h3>Navigation</h3><p>Choose module and upload the correct dataset type.</p></div>", unsafe_allow_html=True)
    module = st.radio("Module", ["Panel Data", "Time Series (separate file)"])
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# ============================================================
# MODULE: PANEL DATA
# ============================================================


def run_panel_module() -> None:
    st.markdown("<div class='card'><h3>Panel Data — Pooled / FE / RE / CRE</h3><p>Workflow: validate → screen → estimate → select → diagnose → export.</p></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload panel dataset (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="panel_upl")
    if uploaded is None:
        st.info("Upload a panel dataset to start.")
        return

    df_raw, meta_file = _read_uploaded(uploaded)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    cols = list(df_raw.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        id_col = st.selectbox("Entity column (ID)", cols)
    with c2:
        time_col = st.selectbox("Time column", cols)
    with c3:
        y_col = st.selectbox("Dependent variable (Y)", [c for c in cols if c not in [id_col, time_col]])

    x_cols = st.multiselect("Regressors (X)", [c for c in cols if c not in [id_col, time_col, y_col]])
    if len(x_cols) == 0:
        st.warning("Select at least one regressor.")
        return

    # Validate panel
    try:
        panel_meta = panel_validate(df_raw, id_col, time_col)
    except Exception as e:
        bad_box(str(e))
        return

    # Panel meta cards
    st.markdown(
        f"""
<div class='card'>
  <h3>Panel structure</h3>
  <p><span class='badge'>N={panel_meta['N_entities']}</span>
     <span class='badge'>Tmin={panel_meta['T_min']}</span>
     <span class='badge'>Tmax={panel_meta['T_max']}</span>
     <span class='badge'>Balanced={panel_meta['balanced']}</span></p>
  <p class='small'>If the panel is unbalanced, selection tests are still provided; some dependence tests may be skipped.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Screening
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    with st.expander("Data screening: transform suggestions (optional)", expanded=False):
        sug = screen_transform_suggestions(df_raw, [y_col] + x_cols)
        html_table(sug, "Screening summary")
        st.caption("These are suggestions only. Apply transformations only when substantively justified.")

    # Optional transforms
    with st.expander("Apply transformations (optional)", expanded=False):
        tr_options = ["none", "log", "log1p", "zscore"]
        transform_map = {}
        for c in [y_col] + x_cols:
            transform_map[c] = st.selectbox(f"Transform {c}", tr_options, key=f"tr_{c}")
        df_work, tr_notes = apply_transforms(df_raw, transform_map)
        if tr_notes:
            html_table(pd.DataFrame({"variable": list(tr_notes.keys()), "note": list(tr_notes.values())}), "Transform notes")
    if 'df_work' not in locals():
        df_work = df_raw.copy()
        tr_notes = {}

    # Effects + covariance
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        entity_fe = st.checkbox("Entity FE", value=True)
    with ec2:
        time_fe = st.checkbox("Time FE", value=False)
    with ec3:
        cov_choice = st.selectbox(
            "Covariance / SE",
            ["robust", "cluster_entity", "cluster_time", "driscoll_kraay"],
        )
    with ec4:
        use_cre = st.checkbox("CRE (Mundlak) for RE", value=False)

    # Prepare data for linearmodels: MultiIndex
    dfp = df_work.copy()
    dfp[id_col] = dfp[id_col].astype(str)
    dfp[time_col] = pd.to_numeric(dfp[time_col], errors="coerce").astype(int)
    dfp = dfp.sort_values([id_col, time_col])

    # Build y and X (with dummies) aligned with dfp index
    yv, X, notes = panel_design_matrix(dfp, y_col, x_cols, add_const=True)

    # Align dfp to yv/X rows
    dfp_aligned = dfp.loc[yv.index]

    # Add time dummies to pooled and RE when time_fe only
    time_dum = None
    if time_fe and not entity_fe:
        time_dum = panel_add_time_dummies(dfp_aligned, time_col)
        X = pd.concat([X, time_dum.set_index(X.index)], axis=1)

    # Set MultiIndex
    dfp_aligned = dfp_aligned.set_index([id_col, time_col])
    y_panel = yv.set_axis(dfp_aligned.index)
    X_panel = X.set_axis(dfp_aligned.index)

    # Cov settings
    cov_kw = {}
    if cov_choice == "robust":
        cov_type = "robust"
    elif cov_choice == "cluster_entity":
        cov_type = "clustered"
        cov_kw = {"cluster_entity": True}
    elif cov_choice == "cluster_time":
        cov_type = "clustered"
        cov_kw = {"cluster_time": True}
    else:
        cov_type = "driscoll-kraay"
        cov_kw = {"kernel": "bartlett"}

    # Fit models
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("Estimation")

    with st.spinner("Fitting models..."):
        # pooled
        m_pool = PooledOLS(y_panel, X_panel).fit(cov_type=cov_type, **cov_kw)

        # FE
        m_fe = PanelOLS(y_panel, X_panel, entity_effects=entity_fe, time_effects=time_fe).fit(cov_type=cov_type, **cov_kw)

        # RE (optionally CRE)
        X_re = X_panel.copy()
        cre_notes = ""
        if use_cre:
            # Mundlak: add entity means of each regressor (excluding const and time dummies)
            base_cols = [c for c in X_re.columns if c != "const" and not str(c).startswith("T_")]
            means = X_re[base_cols].groupby(level=0).transform("mean")
            means = means.add_prefix("mean_")
            X_re = pd.concat([X_re, means], axis=1)
            cre_notes = f"CRE added means for: {', '.join(base_cols)}"

        m_re = RandomEffects(y_panel, X_re).fit(cov_type=cov_type, **cov_kw)

    # Tables: coefficients
    def coef_table(res) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "term": res.params.index,
                "coef": res.params.values,
                "std_err": res.std_errors.values,
                "t": res.tstats.values,
                "p_value": res.pvalues.values,
            }
        )

    tab1, tab2, tab3 = st.tabs(["Pooled OLS", "Fixed Effects", "Random Effects (RE/CRE)"])
    with tab1:
        html_table(coef_table(m_pool), "Pooled OLS coefficients")
        st.code(str(m_pool.summary), language="text")
    with tab2:
        html_table(coef_table(m_fe), "FE coefficients")
        st.code(str(m_fe.summary), language="text")
    with tab3:
        if cre_notes:
            warn_box(cre_notes)
        html_table(coef_table(m_re), "RE/CRE coefficients")
        st.code(str(m_re.summary), language="text")

    # Model selection
    st.subheader("Model selection")

    sel_rows = []
    # FE vs pooled: linearmodels provides f_pooled
    try:
        f = m_fe.f_pooled
        sel_rows.append({"test": "FE vs pooled (F)", "stat": float(f.stat), "df": str(f.df), "p_value": float(f.pval)})
    except Exception:
        sel_rows.append({"test": "FE vs pooled (F)", "stat": np.nan, "df": "", "p_value": np.nan})

    # RE vs pooled: LR test (robust alternative)
    lr = re_vs_pooled_lr_test(dfp.reset_index(drop=True).loc[yv.index], id_col, time_col, y_col, X)
    if lr is not None:
        sel_rows.append({"test": "RE vs pooled (LR random intercept)", "stat": float(lr.loc[0, "lr_stat"]), "df": "1", "p_value": float(lr.loc[0, "p_value"])})
    else:
        sel_rows.append({"test": "RE vs pooled (LR)", "stat": np.nan, "df": "1", "p_value": np.nan})

    # Hausman
    haus = hausman_aligned(m_fe, m_re)
    sel_rows.append({"test": "Hausman (FE vs RE)", "stat": float(haus.loc[0, "hausman_stat"]), "df": str(int(haus.loc[0, "df"])), "p_value": float(haus.loc[0, "p_value"]), "note": haus.loc[0, "note"]})

    sel_df = pd.DataFrame(sel_rows)
    html_table(sel_df, "Selection tests")

    with st.expander("Hypotheses (H0/H1) — panel selection", expanded=False):
        st.markdown(
            """
- **FE vs pooled (F-test)**
  - H0: No entity/time effects are needed (pooled is adequate).
  - H1: Effects are needed (FE is preferred).

- **RE vs pooled (LR random intercept)**
  - H0: Random intercept variance = 0 (pooled adequate).
  - H1: Random intercept variance > 0 (RE preferred).

- **Hausman (FE vs RE)**
  - H0: RE is consistent (prefer RE for efficiency).
  - H1: RE is inconsistent (prefer FE).
"""
        )

    # Diagnostics
    st.subheader("Diagnostics")

    diag_tables: Dict[str, pd.DataFrame] = {}

    vif_df = panel_vif(X)
    html_table(vif_df, "VIF (based on design matrix)")
    diag_tables["panel_vif"] = vif_df

    bp_df = panel_bp_hetero(m_pool)
    if bp_df is not None:
        html_table(bp_df, "Breusch–Pagan (pooled residuals) — heteroskedasticity")
        diag_tables["panel_bp"] = bp_df

    # Residual dependence tests
    # Pesaran CD (only if balanced and pivotable)
    if panel_meta["balanced"] == "True":
        cd_df = panel_pesaran_cd(m_fe.resids, id_col, time_col, dfp_aligned.reset_index())
        if cd_df is not None:
            html_table(cd_df, "Pesaran CD (FE residuals)")
            diag_tables["panel_cd"] = cd_df
        else:
            warn_box("Pesaran CD skipped (missing values after pivot or insufficient N/T).")
    else:
        warn_box("Pesaran CD not computed for unbalanced panels (by design).")

    ar1_df = panel_resid_ar1_test(m_fe.resids, id_col, time_col, dfp_aligned.reset_index())
    if ar1_df is not None:
        html_table(ar1_df, "Residual AR(1) test (indicative)")
        diag_tables["panel_ar1"] = ar1_df

    # Warnings
    red_flags = []
    if bp_df is not None and float(bp_df.loc[0, "lm_p"]) < 0.05:
        red_flags.append("Heteroskedasticity detected (BP p<0.05). Use robust/cluster/DK SE.")
    if ar1_df is not None and float(ar1_df.loc[0, "p_value"]) < 0.05:
        red_flags.append("Residual serial correlation indicated (AR(1) p<0.05). Prefer cluster/DK SE.")
    if len(red_flags) > 0:
        bad_box(" ".join(red_flags))

    # Export
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("Export report")

    export_tables: Dict[str, pd.DataFrame] = {
        "panel_selection": sel_df,
        "pooled_coef": coef_table(m_pool),
        "fe_coef": coef_table(m_fe),
        "re_coef": coef_table(m_re),
        **diag_tables,
    }

    meta = {
        **meta_file,
        "module": "panel",
        "id_col": id_col,
        "time_col": time_col,
        "y": y_col,
        "x": ",".join(x_cols),
        "entity_fe": str(entity_fe),
        "time_fe": str(time_fe),
        "covariance": cov_choice,
        "cre_mundlak": str(use_cre),
        **panel_meta,
        **tr_notes,
        **notes,
    }

    b, ext = export_report_bytes(export_tables, meta)
    st.download_button(
        f"Download report ({ext.upper()})",
        data=b,
        file_name=f"panel_report.{ext}",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if ext == "xlsx" else "application/zip",
    )


# ============================================================
# MODULE: TIME SERIES (SEPARATE FILE)
# ============================================================


def run_ts_module() -> None:
    st.markdown("<div class='card'><h3>Time Series — VAR / VECM / ARDL (Annual, year index)</h3><p>Workflow: validate → screen → stationarity → cointegration gate → estimate → diagnostics → causality → export.</p></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload time series dataset (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="ts_upl")
    if uploaded is None:
        st.info("Upload a time series dataset to start.")
        return

    df_raw, meta_file = _read_uploaded(uploaded)

    cols = list(df_raw.columns)
    if len(cols) < 3:
        bad_box("Dataset seems too small. Need a year column + at least 2 variables.")
        return

    time_col = st.selectbox("Time column (year)", cols)

    # validate time index
    dft = df_raw.copy()
    dft[time_col] = pd.to_numeric(dft[time_col], errors="coerce")
    if dft[time_col].isna().any():
        bad_box("Year column has NA after numeric conversion.")
        return

    dft[time_col] = dft[time_col].astype(int)
    if dft[time_col].duplicated().any():
        bad_box("Duplicate years detected. Time series requires unique time index.")
        return

    dft = dft.sort_values(time_col)
    dft = dft.set_index(time_col)

    # variable selection
    var_candidates = [c for c in cols if c != time_col]
    ts_vars = st.multiselect("Select variables (numeric)", var_candidates, default=var_candidates[: min(3, len(var_candidates))])
    if len(ts_vars) < 2:
        st.warning("Select at least 2 variables.")
        return

    # numeric conversion
    dft2 = dft.copy()
    dft2 = safe_numeric(dft2, ts_vars)

    # drop missing
    before = len(dft2)
    dft2 = dft2.dropna(subset=ts_vars)
    after = len(dft2)

    T = len(dft2)
    st.markdown(
        f"""
<div class='card'>
  <h3>Time series integrity</h3>
  <p><span class='badge'>T={T}</span> <span class='badge'>Annual</span>
     <span class='badge'>Dropped rows={before-after}</span></p>
  <p class='small'>For annual TS, keep lags small unless T is very large. With T≈40, VAR can work with small k and modest lags; ARDL is usually the most robust choice.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Screening suggestions
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    with st.expander("Data screening: transform suggestions (optional)", expanded=False):
        sug = screen_transform_suggestions(dft2.reset_index(), ts_vars)
        html_table(sug, "Screening summary")

    # Apply transforms
    with st.expander("Apply transformations (optional)", expanded=False):
        transform_map = {}
        for c in ts_vars:
            transform_map[c] = st.selectbox(f"Transform {c}", ["none", "log", "log1p", "zscore"], key=f"ts_tr_{c}")
        dft3, tr_notes = apply_transforms(dft2, transform_map)
        if tr_notes:
            html_table(pd.DataFrame({"variable": list(tr_notes.keys()), "note": list(tr_notes.values())}), "Transform notes")
    if 'dft3' not in locals():
        dft3 = dft2
        tr_notes = {}

    # Plot raw series
    st.subheader("Visual screening")
    plot_multi_series(dft3[ts_vars], "Selected series (possibly transformed)")

    # Stationarity
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("Stationarity (ADF + KPSS)")

    reg = st.selectbox("Deterministic terms for tests", ["c", "ct"], index=0, help="c=constant, ct=constant+trend")

    integ_rows = []
    station_tables = {}
    max_d = 0
    for v in ts_vars:
        d, tbl = integration_order(dft3[v], regression=reg)
        max_d = max(max_d, d)
        station_tables[f"station_{v}"] = tbl
        integ_rows.append({"variable": v, "integration_order": d})

    integ_df = pd.DataFrame(integ_rows)
    html_table(integ_df, "Integration order summary")

    with st.expander("Details: ADF/KPSS tables per variable", expanded=False):
        for v in ts_vars:
            html_table(station_tables[f"station_{v}"], f"{v}: tests")

    if max_d >= 2:
        bad_box("At least one series appears I(2) (or non-stationary after 2 differences). VECM and ARDL assumptions are violated. Consider differencing or alternative data.")

    # Differencing workflow
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("Transformation for modeling")

    mode = st.radio(
        "Modeling data form",
        ["Levels", "First differences", "Log-differences (Δlog)"],
        index=0,
        help="Use Levels for VECM when cointegrated; differences for VAR when no cointegration. Δlog is common for growth rates when data are positive.",
    )

    data_model = dft3[ts_vars].copy()
    trans_note = ""

    if mode == "First differences":
        data_model = data_model.diff().dropna()
        trans_note = "Using first differences."
    elif mode == "Log-differences (Δlog)":
        if (data_model <= 0).any().any():
            bad_box("Δlog requires strictly positive series. Apply log/log1p first or choose another mode.")
            return
        data_model = np.log(data_model).diff().dropna()
        trans_note = "Using log-differences."

    if trans_note:
        warn_box(trans_note)

    # Model selection: VAR / VECM / ARDL
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    model_type = st.selectbox("Time series model", ["VAR", "VECM", "ARDL"], index=0)

    # Hard gate: ARDL and VECM require no I(2) series.
    if max_d >= 2 and model_type in ["VECM", "ARDL"]:
        st.error("Blocked: At least one series appears I(2). ARDL and VECM are not valid under I(2). Use differencing/log-differencing, or revisit data.")
        return

    report_tables: Dict[str, pd.DataFrame] = {}

    # Common: lag cap
    maxlag_default = min(3, max(1, int(math.floor(len(data_model) / 10))))
    maxlag = st.slider("Max lag (cap)", 1, min(8, max(1, len(data_model) // 3)), value=maxlag_default)

    # ========= VAR =========
    if model_type == "VAR":
        st.subheader("VAR")

        # lag selection
        with st.spinner("Selecting lag order..."):
            sel = VAR(data_model).select_order(maxlags=maxlag)
        sel_tbl = pd.DataFrame({"criterion": ["AIC", "BIC", "HQIC", "FPE"], "selected_lag": [sel.aic, sel.bic, sel.hqic, sel.fpe]})
        html_table(sel_tbl, "Lag selection")
        report_tables["var_lag_selection"] = sel_tbl

        p = st.selectbox("VAR lag p", sorted(set([int(x) for x in [sel.aic, sel.bic, sel.hqic, sel.fpe] if x is not None and not np.isnan(x)] + [1])), index=0)

        with st.spinner("Fitting VAR..."):
            var_res = VAR(data_model).fit(p)

        # Coefs
        # Flatten to a table
        params = var_res.params
        coef_long = []
        for eq in params.columns:
            for term in params.index:
                coef_long.append({"equation": eq, "term": term, "coef": float(params.loc[term, eq])})
        coef_df = pd.DataFrame(coef_long)
        html_table(coef_df, "VAR coefficients (long format)")
        report_tables["var_coef"] = coef_df

        # Diagnostics
        diag_rows = []
        try:
            roots = np.abs(var_res.roots)
            stable = bool(np.all(roots < 1))
            diag_rows.append({"diagnostic": "Stability (all roots < 1)", "value": stable})
        except Exception:
            diag_rows.append({"diagnostic": "Stability", "value": "NA"})

        try:
            sc = var_res.test_serial_correlation()
            diag_rows.append({"diagnostic": "Serial corr (Portmanteau) p", "value": float(sc.pvalue)})
        except Exception:
            pass

        try:
            norm = var_res.test_normality()
            diag_rows.append({"diagnostic": "Normality (JB) p", "value": float(norm.pvalue)})
        except Exception:
            pass

        diag_df = pd.DataFrame(diag_rows)
        html_table(diag_df, "VAR diagnostics")
        report_tables["var_diagnostics"] = diag_df

        # Residual diagnostics plots
        st.subheader("Residual diagnostics")
        resid = pd.DataFrame(var_res.resid, index=data_model.index, columns=data_model.columns)
        plot_multi_series(resid, "VAR residuals")
        plot_residual_diagnostics(resid.iloc[:, 0], title_prefix="VAR")

        # IRF & FEVD
        st.subheader("Impulse response & FEVD")
        steps = st.slider("Horizon (steps)", 5, 20, 10)

        try:
            irf = var_res.irf(steps)
            fig = irf.plot(orth=False)
            st.pyplot(fig)
        except Exception:
            warn_box("IRF plot failed in this environment.")

        try:
            fevd = var_res.fevd(steps)
            fig = fevd.plot()
            st.pyplot(fig)
        except Exception:
            warn_box("FEVD plot failed in this environment.")

        # Causality matrix (Granger-style) using fitted VAR
        st.subheader("Causality (Granger-style, model-consistent)")
        pvals = []
        for caused in data_model.columns:
            for causing in data_model.columns:
                if caused == causing:
                    continue
                try:
                    test = var_res.test_causality(caused, [causing], kind="wald")
                    pvals.append({"causing": causing, "caused": caused, "p_value": float(test.pvalue)})
                except Exception:
                    pvals.append({"causing": causing, "caused": caused, "p_value": np.nan})
        caus_df = pd.DataFrame(pvals)
        html_table(caus_df, "VAR causality p-values")
        report_tables["var_causality"] = caus_df

        with st.expander("Hypotheses (H0/H1) — VAR causality", expanded=False):
            st.markdown(
                """
- For each pair (X → Y):
  - **H0**: X does not Granger-cause Y (all lagged coefficients of X in Y-equation are jointly zero).
  - **H1**: X Granger-causes Y.

Interpretation: predictive precedence, not causal identification.
"""
            )

    # ========= VECM =========
    elif model_type == "VECM":
        st.subheader("VECM")

        if max_d >= 2:
            bad_box("VECM requires I(1) variables. Your stationarity screening indicates I(2).")
            return

        # For Johansen/VECM, cointegration is tested on levels.
        levels = dft3[ts_vars].dropna()
        if len(levels) < 20:
            warn_box("T is small for Johansen; cointegration inference may be weak.")

        det_order = st.selectbox("Johansen deterministic term (det_order)", [0, 1], index=0, help="0: constant in cointegration space; 1: trend.")
        k_ar_diff = st.slider("k_ar_diff (lags in Δ terms)", 1, 5, 1)

        with st.spinner("Running Johansen cointegration test..."):
            trace, maxeig = johansen_table(levels, det_order=det_order, k_ar_diff=k_ar_diff)

        html_table(trace, "Johansen trace test")
        html_table(maxeig, "Johansen max-eigen test")
        report_tables["vecm_johansen_trace"] = trace
        report_tables["vecm_johansen_maxeig"] = maxeig

        # Rank decision (simple): number of trace stats above 95% critical
        rank_95 = int((trace["trace_stat"] > trace["cv_95"]).sum())
        warn_box(f"Suggested cointegration rank (trace @95%): {rank_95}. Adjust based on theory/diagnostics.")

        rank = st.slider("Choose cointegration rank", 0, min(rank_95 + 1, len(ts_vars) - 1), value=min(rank_95, len(ts_vars) - 1))
        if rank == 0:
            bad_box("Selected rank=0 implies no cointegration. Prefer VAR on differences instead of VECM.")
            return

        with st.spinner("Fitting VECM..."):
            vecm = VECM(levels, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic="co")
            vecm_res = vecm.fit()

        # Summary
        st.code(str(vecm_res.summary()), language="text")

        # Extract alpha (adjustment) and beta (cointegration) where available
        try:
            alpha = pd.DataFrame(vecm_res.alpha, index=ts_vars, columns=[f"CI{j+1}" for j in range(rank)])
            beta = pd.DataFrame(vecm_res.beta, index=ts_vars, columns=[f"CI{j+1}" for j in range(rank)])
            html_table(alpha.reset_index().rename(columns={"index": "variable"}), "Adjustment coefficients (alpha)")
            html_table(beta.reset_index().rename(columns={"index": "variable"}), "Cointegration vectors (beta)")
            report_tables["vecm_alpha"] = alpha.reset_index().rename(columns={"index": "variable"})
            report_tables["vecm_beta"] = beta.reset_index().rename(columns={"index": "variable"})
        except Exception:
            warn_box("Could not extract alpha/beta tables in this environment.")

        # Causality (Granger) matrix
        st.subheader("Causality (Granger-style, VECM)")
        pvals = []
        for caused in ts_vars:
            for causing in ts_vars:
                if causing == caused:
                    continue
                try:
                    test = vecm_res.test_granger_causality(caused=caused, causing=[causing])
                    pvals.append({"causing": causing, "caused": caused, "p_value": float(test.pvalue)})
                except Exception:
                    pvals.append({"causing": causing, "caused": caused, "p_value": np.nan})
        caus_df = pd.DataFrame(pvals)
        html_table(caus_df, "VECM causality p-values")
        report_tables["vecm_causality"] = caus_df

        with st.expander("Hypotheses (H0/H1) — Johansen & VECM causality", expanded=False):
            st.markdown(
                """
- **Johansen cointegration**
  - H0: cointegration rank ≤ r
  - H1: cointegration rank > r

- **VECM Granger-style causality**
  - H0: lagged causing-variable terms do not jointly predict the caused equation.
  - H1: they do.

Interpretation remains predictive (not structural causality).
"""
            )

        # IRF & FEVD (best effort by converting to VAR)
        st.subheader("IRF / FEVD (best-effort)")
        steps = st.slider("Horizon (steps)", 5, 20, 10)
        try:
            var_from_vecm = vecm_res.vecm_to_var()
            irf = var_from_vecm.irf(steps)
            fig = irf.plot(orth=False)
            st.pyplot(fig)
            fevd = var_from_vecm.fevd(steps)
            fig = fevd.plot()
            st.pyplot(fig)
        except Exception:
            warn_box("IRF/FEVD conversion not available here. You can still report alpha/beta and short-run dynamics.")

    # ========= ARDL =========
    else:
        st.subheader("ARDL")

        if max_d >= 2:
            bad_box("ARDL assumes variables are I(0)/I(1), not I(2). Your screening indicates I(2).")
            return

        y = st.selectbox("Dependent variable (Y)", ts_vars)
        x = st.multiselect("Regressors (X)", [v for v in ts_vars if v != y], default=[v for v in ts_vars if v != y][: min(2, len(ts_vars) - 1)])
        if len(x) == 0:
            st.warning("Select at least one regressor.")
            return

        ic = st.selectbox("Information criterion", ["bic", "aic"], index=0)
        trend = st.selectbox("Trend term", ["c", "ct", "n"], index=0, help="c=const, ct=const+trend, n=no constant")
        max_p = st.slider("Max lag for Y", 1, min(6, max(1, T // 5)), 2)
        max_q = st.slider("Max lag for X", 0, min(6, max(1, T // 5)), 2)

        # Fit selection
        levels = dft3[[y] + x].dropna()
        data_y = levels[y]
        data_X = levels[x]

        with st.spinner("Selecting ARDL order..."):
            sel = ardl_select_order(endog=data_y, maxlag=max_p, exog=data_X, maxorder=max_q, ic=ic, trend=trend)

        # Summarize selected order
        # sel.model is ARDL model
        try:
            chosen = str(sel.model)
        except Exception:
            chosen = "Selected ARDL model"

        st.markdown(f"<div class='card'><h3>Selected order</h3><p class='small'>{chosen}</p></div>", unsafe_allow_html=True)

        with st.spinner("Fitting ARDL..."):
            ardl_res = sel.model.fit()

        st.code(str(ardl_res.summary()), language="text")

        # Cointegration check (Engle–Granger; indicative)
        st.subheader("Cointegration check (ARDL context)")
        eg = engle_granger_cointegration(data_y, data_X)
        html_table(eg, "Engle–Granger residual-based cointegration (indicative)")
        report_tables["ardl_cointegration"] = eg

        # Long-run + ECM
        st.subheader("Long-run & ECM")
        lr = ardl_long_run(ardl_res)
        html_table(lr, "Long-run coefficients (computed)")
        report_tables["ardl_long_run"] = lr

        ecm_tbl, ecm_fit = ardl_ecm(data_y, data_X, ardl_res)
        html_table(ecm_tbl, "ECM regression (Δy on Δ terms + ECT_{t-1})")
        report_tables["ardl_ecm"] = ecm_tbl

        # Diagnostics
        st.subheader("ARDL diagnostics")
        resid = pd.Series(ardl_res.resid, index=levels.index)
        fitted = pd.Series(ardl_res.fittedvalues, index=levels.index)
        plot_residual_diagnostics(resid, fitted, title_prefix="ARDL")

        # Causality (Granger-style) via Wald tests: H0 all lagged X terms in Δy eq are zero.
        st.subheader("Causality (Granger-style, ARDL via Wald on lagged X terms)")
        # We approximate using ECM regression terms dX and dX_l1
        wald_rows = []
        try:
            # Build restriction indices for each X based on term names
            terms = list(ecm_tbl["term"])  # includes const
            for xvar in x:
                idxs = [i for i, t in enumerate(terms) if t in [f"d{xvar}", f"d{xvar}_l1"]]
                if not idxs:
                    wald_rows.append({"causing": xvar, "caused": y, "p_value": np.nan, "note": "No Δ terms in ECM"})
                    continue
                R = np.zeros((len(idxs), len(terms)))
                for r_i, p_i in enumerate(idxs):
                    R[r_i, p_i] = 1.0
                w = ecm_fit.wald_test(R)
                wald_rows.append({"causing": xvar, "caused": y, "p_value": float(w.pvalue), "stat": float(w.statistic)})
        except Exception:
            wald_rows.append({"causing": "NA", "caused": y, "p_value": np.nan})

        wald_df = pd.DataFrame(wald_rows)
        html_table(wald_df, "ARDL/ECM causality p-values")
        report_tables["ardl_causality"] = wald_df

        with st.expander("Hypotheses (H0/H1) — ARDL/ECM", expanded=False):
            st.markdown(
                """
- **Engle–Granger residual cointegration (indicative)**
  - H0: residual has unit root (no cointegration)
  - H1: residual is stationary (cointegration)

- **ECM speed of adjustment (ECT)**
  - Expect negative and significant coefficient on ECT_{t-1} for convergence.

- **Granger-style causality (Wald)**
  - H0: lagged ΔX terms jointly do not predict ΔY
  - H1: they do
"""
            )

    # Export
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("Export report")

    meta = {
        **meta_file,
        "module": "time_series",
        "time_col": time_col,
        "variables": ",".join(ts_vars),
        "T": str(T),
        "mode": mode,
        "model": model_type,
        "test_regression": reg,
        **tr_notes,
    }

    # Add stationarity summary to report
    report_tables["stationarity_summary"] = integ_df
    for k, v in station_tables.items():
        report_tables[k] = v

    b, ext = export_report_bytes(report_tables, meta)
    st.download_button(
        f"Download report ({ext.upper()})",
        data=b,
        file_name=f"ts_report.{ext}",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if ext == "xlsx" else "application/zip",
    )


# ============================================================
# RUN
# ============================================================

if module == "Panel Data":
    run_panel_module()
else:
    run_ts_module()
