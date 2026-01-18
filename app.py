"""Academic Econometrics Suite (Streamlit)

This app is designed for reviewer-facing (audit-friendly) econometric workflows.

Key principles:
- Validation first (block invalid setups).
- No fake outputs.
- Explicit hypotheses (H0/H1) and decision rules.
- Reproducible exports (configuration + tables).

Recommended requirements.txt
- streamlit
- pandas
- numpy
- scipy
- statsmodels
- matplotlib
- linearmodels
- openpyxl OR xlsxwriter

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

import matplotlib.pyplot as plt

# --- statsmodels
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ardl import ARDL, ardl_select_order

# Optional (bounds test exists only in some statsmodels versions)
try:
    from statsmodels.tsa.ardl import ardl_bonds_test  # type: ignore
except Exception:
    ardl_bonds_test = None  # type: ignore

# --- linearmodels (panel)
try:
    from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
    _HAS_LINEARMODELS = True
except Exception:
    PanelOLS = PooledOLS = RandomEffects = None  # type: ignore
    _HAS_LINEARMODELS = False


# ============================================================
# UI / STYLE
# ============================================================

st.set_page_config(page_title="Academic Econometrics Suite", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
<style>
:root{
  --bg0:#070B14;
  --bg1:#0B1220;
  --panel:#0F1A33;
  --card:#101F3D;
  --card2:#0D1833;
  --border: rgba(140, 180, 255, 0.22);
  --text:#EAF1FF;
  --muted:#A9B5D6;
  --accent:#00D1B2;
  --accent2:#7AA2FF;
  --accent3:#FFB703;
  --good:#44E58B;
  --bad:#FF5C77;
}

html, body, [class*="css"]{
  color: var(--text) !important;
}

/* Background */
.stApp{
  background: radial-gradient(1200px 700px at 20% 0%, rgba(122,162,255,0.18), transparent 60%),
              radial-gradient(900px 600px at 90% 10%, rgba(0,209,178,0.14), transparent 55%),
              linear-gradient(180deg, var(--bg0), var(--bg1));
}

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(15,26,51,0.95), rgba(9,15,31,0.95));
  border-right: 1px solid rgba(140,180,255,0.18);
}

.block-container{padding-top: 1.2rem; padding-bottom: 2.2rem;}

/* Headings */
.h-title{font-size:1.7rem; font-weight:900; margin:0 0 .25rem 0; letter-spacing:.2px;}
.h-sub{color:var(--muted); margin:.1rem 0 0 0;}

/* Cards */
.card{
  background: linear-gradient(180deg, rgba(16,31,61,0.96), rgba(13,24,51,0.96));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.42);
}
.card h3{margin:0 0 .4rem 0; font-size:1.08rem;}
.card p{margin:.25rem 0; color:var(--muted)}

.badge{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem;
  border:1px solid rgba(0,209,178,0.35); background: rgba(0,209,178,0.10); margin-right:.35rem;}

/* Table styling with vertical + horizontal lines */
.table-wrap{margin: .4rem 0 1rem 0;}
.table-wrap table{border-collapse:collapse; width:100%; font-size:0.92rem;}
.table-wrap th, .table-wrap td{
  border:1px solid rgba(140,180,255,0.26);
  padding:7px 10px;
}
.table-wrap th{
  background: rgba(122,162,255,0.14);
  color: var(--text);
  font-weight: 800;
}
.table-wrap tr:nth-child(even) td{background: rgba(255,255,255,0.02);}

/* Reduce whitespace for code blocks */
code{color: var(--text) !important;}

/* Make Streamlit metrics nicer */
[data-testid="stMetricValue"]{color: var(--text) !important;}
[data-testid="stMetricDelta"]{color: var(--muted) !important;}

</style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ============================================================
# COMMON HELPERS
# ============================================================


def _to_numeric_safe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _read_uploaded_file(uploaded) -> Tuple[pd.DataFrame, Dict[str, str]]:
    name = getattr(uploaded, "name", "uploaded")
    if name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        raise ValueError("Unsupported file type. Upload .csv or .xlsx")

    meta = {
        "file_name": name,
        "n_rows_raw": str(len(df)),
        "n_cols_raw": str(df.shape[1]),
    }
    return df, meta


def _df_to_html(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None:
        return "<div class='table-wrap'><em>None</em></div>"
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    # Avoid scientific notation noise
    with pd.option_context("display.float_format", lambda x: f"{x:,.6g}"):
        html = d.to_html(index=False, escape=False)
    return f"<div class='table-wrap'>{html}</div>"


def st_table(df: pd.DataFrame, caption: Optional[str] = None, max_rows: int = 200) -> None:
    if caption:
        st.markdown(f"**{caption}**")
    st.markdown(_df_to_html(df, max_rows=max_rows), unsafe_allow_html=True)


def _fig_show(fig) -> None:
    """Render matplotlib figures with anti-overlap defaults.

    - Applies tight_layout/constrained_layout if possible.
    - Rotates x tick labels when there are many ticks.
    """
    try:
        # Prefer constrained layout when available (helps multi-axes figures like IRF/FEVD)
        if hasattr(fig, "set_constrained_layout"):
            fig.set_constrained_layout(True)
    except Exception:
        pass

    # Rotate dense x-ticks to avoid overlap
    try:
        for ax in getattr(fig, "axes", []):
            xt = ax.get_xticklabels()
            if len(xt) >= 8:
                for lab in xt:
                    lab.set_rotation(45)
                    lab.set_ha("right")
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    st.pyplot(fig, clear_figure=True)


def _line_plot(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=130)
    for c in df.columns:
        ax.plot(df.index, df[c].values, label=str(c), linewidth=1.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(loc="upper left", ncol=3, fontsize=9, frameon=False)
    _fig_show(fig)


# ============================================================
# TRANSFORMATION SCREENING
# ============================================================


def _screen_transform_suggestions(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        zeros = int((s == 0).sum())
        neg = int((s < 0).sum())
        skew = float(stats.skew(s, bias=False)) if len(s) >= 3 else np.nan
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            out_share = 0.0
        else:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out_share = float(((s < lo) | (s > hi)).mean())

        suggestion, reason = "none", ""
        if neg > 0:
            suggestion = "none"
            reason = "Ada nilai negatif; log tidak sesuai."
        else:
            if (not np.isnan(skew)) and abs(skew) >= 1.2:
                if zeros > 0:
                    suggestion = "log1p"
                    reason = "Skew tinggi + ada sifar; log1p boleh kurangkan skew."
                else:
                    suggestion = "log"
                    reason = "Skew tinggi + semua positif; log boleh stabilkan varians."
            else:
                suggestion = "none"
                reason = "Tiada isu skew yang jelas (rule-of-thumb)."

        rows.append(
            {
                "variable": c,
                "min": float(s.min()),
                "max": float(s.max()),
                "zeros": zeros,
                "negatives": neg,
                "skewness": skew,
                "outlier_share": out_share,
                "suggestion": suggestion,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _apply_transform(df: pd.DataFrame, transform_map: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = df.copy()
    notes: Dict[str, str] = {}
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
            mu, sd = float(s.mean()), float(s.std(ddof=0))
            if sd == 0:
                notes[col] = "SKIPPED standardize: sd=0"
                continue
            out[col] = (s - mu) / sd
            notes[col] = "Applied z-score"
        else:
            notes[col] = f"Unknown transform '{tr}' ignored"
    return out, notes


# ============================================================
# PANEL HELPERS
# ============================================================


def _panel_prepare(
    df_raw: pd.DataFrame,
    id_col: str,
    time_col: str,
    y_col: str,
    x_cols: List[str],
    add_time_dummies: bool,
    transform_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, str]]:
    df = df_raw[[id_col, time_col, y_col] + x_cols].copy()
    df, tr_notes = _apply_transform(df, transform_map)
    df = _to_numeric_safe(df, [y_col] + x_cols)

    df = df.dropna(subset=[y_col])
    df[id_col] = df[id_col].astype(str)
    df = df.sort_values([id_col, time_col])

    if df.duplicated(subset=[id_col, time_col]).any():
        dup = df[df.duplicated(subset=[id_col, time_col], keep=False)][[id_col, time_col]].drop_duplicates()
        raise ValueError(
            "Duplicate (entity,time) rows detected. Fix your data. Examples:\n" + dup.head(10).to_string(index=False)
        )

    df = df.set_index([id_col, time_col])

    y = df[y_col]
    X = df[x_cols].copy()

    cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("category")]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    if add_time_dummies:
        t = df.index.get_level_values(1)
        td = pd.get_dummies(t, prefix="time", drop_first=True)
        td.index = df.index
        X = pd.concat([X, td], axis=1)

    X = sm.add_constant(X, has_constant="add")

    keep = (~y.isna()) & (~X.isna().any(axis=1))
    y = y.loc[keep]
    X = X.loc[keep]

    meta = {
        "n_obs": str(len(y)),
        "n_entities": str(y.index.get_level_values(0).nunique()),
        "time_min": str(y.index.get_level_values(1).min()),
        "time_max": str(y.index.get_level_values(1).max()),
    }
    meta.update({f"transform_{k}": v for k, v in tr_notes.items()})
    return df, y, X, meta


def _is_balanced_panel(index: pd.MultiIndex) -> Tuple[bool, int, int]:
    entities = index.get_level_values(0)
    n = entities.nunique()
    counts = pd.Series(1, index=index).groupby(level=0).sum()
    balanced = counts.nunique() == 1
    T = int(counts.iloc[0]) if balanced else int(counts.median())
    return balanced, int(n), int(T)


def _bp_lm_test_re_vs_pooled_balanced(pooled_resid: pd.Series) -> Tuple[float, float]:
    idx = pooled_resid.index
    balanced, n, T = _is_balanced_panel(idx)
    if not balanced:
        raise ValueError("LM test implemented only for balanced panels in this app.")
    if T <= 2:
        raise ValueError("LM test requires T>2.")

    v = pooled_resid.copy()
    sum_i = v.groupby(level=0).sum()
    S1 = float((sum_i**2).sum())
    S2 = float((v**2).sum())
    if S2 == 0:
        raise ValueError("Residual variance is zero; LM undefined.")

    lm = (n * T) / (2 * (T - 1)) * ((S1 / S2) - 1) ** 2
    p = 1 - stats.chi2.cdf(lm, df=1)
    return float(lm), float(p)


def _hausman_test_aligned(fe_params: pd.Series, fe_cov: pd.DataFrame, re_params: pd.Series, re_cov: pd.DataFrame) -> Tuple[float, float, int]:
    common = fe_params.index.intersection(re_params.index)
    common = pd.Index([c for c in common if c.lower() not in {"const", "intercept"}])
    if len(common) == 0:
        raise ValueError("No common coefficients between FE and RE for Hausman test.")

    b = (fe_params.loc[common] - re_params.loc[common]).values.reshape(-1, 1)
    V = (fe_cov.loc[common, common] - re_cov.loc[common, common]).values
    try:
        Vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        Vinv = np.linalg.pinv(V)
    stat = float((b.T @ Vinv @ b).ravel()[0])
    dof = int(len(common))
    p = 1 - stats.chi2.cdf(stat, df=dof)
    return stat, float(p), dof


def _vif_table(X: pd.DataFrame) -> pd.DataFrame:
    Xv = X.copy().drop(columns=[c for c in X.columns if c.lower() in {"const", "intercept"}], errors="ignore")
    Xv = Xv.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    Xv = Xv.dropna()
    if Xv.shape[1] == 0:
        return pd.DataFrame({"variable": [], "VIF": []})
    if len(Xv) < 5:
        return pd.DataFrame({"variable": list(Xv.columns), "VIF": [np.nan] * Xv.shape[1]})

    vals = Xv.values
    vifs = [float(variance_inflation_factor(vals, i)) for i in range(Xv.shape[1])]
    return pd.DataFrame({"variable": list(Xv.columns), "VIF": vifs}).sort_values("VIF", ascending=False)


def _pesaran_cd_test_balanced(resid: pd.Series) -> Tuple[float, float]:
    idx = resid.index
    balanced, N, T = _is_balanced_panel(idx)
    if not balanced:
        raise ValueError("Pesaran CD implemented only for balanced panels in this app.")
    if N < 3:
        raise ValueError("Pesaran CD requires N>=3.")

    df = resid.rename("e").reset_index()
    df.columns = ["entity", "time", "e"]
    pivot = df.pivot(index="time", columns="entity", values="e")
    corr = pivot.corr()
    iu = np.triu_indices_from(corr.values, k=1)
    rhos = corr.values[iu]
    S = np.nansum(rhos)
    cd = math.sqrt(2 * T / (N * (N - 1))) * S
    p = 2 * (1 - stats.norm.cdf(abs(cd)))
    return float(cd), float(p)


def _wooldridge_serial_corr_test(y: pd.Series, X: pd.DataFrame) -> Tuple[float, float, float]:
    """Wooldridge/Drukker (2003) AR(1) serial correlation test (implementation-style).

    Steps (conceptually):
      1) Estimate first-differenced model without constant, cluster by entity
      2) Obtain residuals u_it
      3) Regress u_it on u_{i,t-1} without constant, cluster by entity
      4) Test H0: coef = -0.5

    Returns (coef, t_stat, p_value).
    """
    idx = y.index
    ent = idx.get_level_values(0)

    df = pd.concat([y.rename("y"), X.drop(columns=[c for c in X.columns if c.lower() in {"const", "intercept"}], errors="ignore")], axis=1)
    df = df.sort_index()

    # First differences within entity
    d = df.groupby(level=0).diff()
    d = d.dropna()

    dy = d["y"]
    dX = d.drop(columns=["y"], errors="ignore")

    # Step 1: diff regression, no constant
    mod1 = sm.OLS(dy.values, dX.values)
    try:
        res1 = mod1.fit(cov_type="cluster", cov_kwds={"groups": ent.loc[d.index].values})
    except Exception:
        res1 = mod1.fit()

    u = pd.Series(res1.resid, index=d.index, name="u")

    # Step 2: u on lag u, no constant
    u_lag = u.groupby(level=0).shift(1)
    reg = pd.concat([u, u_lag.rename("u_lag")], axis=1).dropna()

    mod2 = sm.OLS(reg["u"].values, reg[["u_lag"]].values)
    try:
        res2 = mod2.fit(cov_type="cluster", cov_kwds={"groups": ent.loc[reg.index].values})
    except Exception:
        res2 = mod2.fit()

    beta = float(res2.params[0])
    se = float(res2.bse[0]) if float(res2.bse[0]) != 0 else np.nan
    t_stat = (beta + 0.5) / se if (se is not np.nan) else np.nan
    # normal approximation for p
    p = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
    return beta, float(t_stat), float(p)


# ============================================================
# TIME SERIES HELPERS
# ============================================================


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
        return float(kpss(s, regression=regression, nlags="auto")[1])
    except Exception:
        try:
            return float(kpss(s, regression=regression, nlags=min(8, max(1, len(s) // 10)))[1])
        except Exception:
            return np.nan


def _stationarity_suite(df: pd.DataFrame, cols: List[str], adf_reg: str, kpss_reg: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for c in cols:
        s = df[c].dropna()
        adf_p = _adf_pvalue(s, regression=adf_reg)
        kpss_p = _kpss_pvalue(s, regression=kpss_reg)

        if (not np.isnan(adf_p)) and (not np.isnan(kpss_p)):
            if adf_p < 0.05 and kpss_p > 0.05:
                cl = "I(0) (stationary)"
                integ = 0
            elif adf_p >= 0.05 and kpss_p <= 0.05:
                cl = "I(1) likely (non-stationary in level)"
                integ = 1
            else:
                cl = "Mixed/unclear"
                integ = np.nan
        else:
            cl = "Insufficient data"
            integ = np.nan

        rows.append({"variable": c, "ADF_p": adf_p, "KPSS_p": kpss_p, "classification": cl, "integration_order": integ})

    return pd.DataFrame(rows)


def _integration_order_by_diff(s: pd.Series, adf_reg: str, kpss_reg: str, max_diff: int = 2) -> Tuple[int, List[Tuple[int, float, float, str]]]:
    """Determine integration order up to I(2) using ADF+KPSS rule per differencing level."""
    history = []
    cur = s.copy()
    for d in range(0, max_diff + 1):
        adf_p = _adf_pvalue(cur, regression=adf_reg)
        kpss_p = _kpss_pvalue(cur, regression=kpss_reg)
        if (not np.isnan(adf_p)) and (not np.isnan(kpss_p)):
            if adf_p < 0.05 and kpss_p > 0.05:
                history.append((d, adf_p, kpss_p, "stationary"))
                return d, history
            elif adf_p >= 0.05 and kpss_p <= 0.05:
                history.append((d, adf_p, kpss_p, "non-stationary"))
            else:
                history.append((d, adf_p, kpss_p, "mixed"))
        else:
            history.append((d, adf_p, kpss_p, "insufficient"))

        cur = cur.diff().dropna()

    return max_diff + 1, history


def _ts_prepare(df_raw: pd.DataFrame, year_col: str, vars_: List[str], transform_map: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df_raw[[year_col] + vars_].copy()
    df, tr_notes = _apply_transform(df, transform_map)
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = _to_numeric_safe(df, vars_)
    df = df.dropna(subset=[year_col]).sort_values(year_col)

    if df[year_col].duplicated().any():
        dup = df[df[year_col].duplicated(keep=False)][[year_col]].drop_duplicates()
        raise ValueError("Duplicate year values detected. Fix your time index. Examples:\n" + dup.head(10).to_string(index=False))

    df = df.set_index(year_col)
    df.index.name = "year"
    df = df.dropna(subset=vars_)

    meta = {
        "T": str(len(df)),
        "year_min": str(int(df.index.min())) if len(df) else "",
        "year_max": str(int(df.index.max())) if len(df) else "",
    }
    meta.update({f"transform_{k}": v for k, v in tr_notes.items()})
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


def _parse_ardl_terms(index: List[str]) -> Dict[str, Dict[int, str]]:
    """Map var -> lag -> param_name using robust patterns for statsmodels ARDL."""
    out: Dict[str, Dict[int, str]] = {}
    for nm in index:
        # Pattern: var.Lk
        if ".L" in nm:
            try:
                var, lag_s = nm.split(".L", 1)
                lag = int(lag_s)
                out.setdefault(var, {})[lag] = nm
                continue
            except Exception:
                pass
        # Pattern: Lk.var
        if nm.startswith("L") and "." in nm:
            try:
                lag_s, var = nm.split(".", 1)
                lag = int(lag_s.replace("L", ""))
                out.setdefault(var, {})[lag] = nm
                continue
            except Exception:
                pass
    return out


def _ardl_long_run(ardl_res, y: str, x_list: List[str]) -> Tuple[pd.DataFrame, float, float]:
    params = ardl_res.params
    names = list(params.index)
    m = _parse_ardl_terms(names)

    # Endog lag sum
    phi_lags = m.get(y, {})
    phi = 0.0
    p = 0
    for lag, nm in phi_lags.items():
        if lag >= 1:
            phi += float(params[nm])
            p = max(p, lag)

    denom = 1.0 - phi

    # Constant
    const = float(params.get("const", params.get("intercept", 0.0)))
    const_lr = const / denom if denom != 0 else np.nan

    rows = []
    for x in x_list:
        betas = 0.0
        q = 0
        for lag, nm in m.get(x, {}).items():
            if lag >= 0:
                betas += float(params[nm])
                q = max(q, lag)
        lr = betas / denom if denom != 0 else np.nan
        rows.append({"variable": x, "sum_beta_lags": betas, "long_run_coef": lr, "max_lag_included": q})

    return pd.DataFrame(rows), const_lr, float(p)


def _ardl_ecm(ardl_res, ts_df: pd.DataFrame, y: str, x_list: List[str], adf_reg: str, kpss_reg: str) -> Tuple[Optional[sm.regression.linear_model.RegressionResultsWrapper], pd.DataFrame, pd.Series]:
    """Build ECM representation consistent with the estimated ARDL lag structure (as far as params naming allows)."""
    params = ardl_res.params
    names = list(params.index)
    m = _parse_ardl_terms(names)

    # Determine p and q_i from estimated model (based on detected terms)
    p = max([lag for lag in m.get(y, {}).keys() if lag >= 1], default=0)
    q_map = {x: max([lag for lag in m.get(x, {}).keys() if lag >= 0], default=0) for x in x_list}

    lr_table, const_lr, _p = _ardl_long_run(ardl_res, y=y, x_list=x_list)
    lr_map = {r["variable"]: float(r["long_run_coef"]) for _, r in lr_table.iterrows()}

    # Build ECT_{t-1}
    df = ts_df[[y] + x_list].copy().astype(float)
    ect = df[y].shift(1) - const_lr
    for x in x_list:
        ect = ect - lr_map.get(x, np.nan) * df[x].shift(1)

    # Build regression: Δy_t on ECT_{t-1} + Σ Δy_{t-i} (i=1..p-1) + Σ Σ Δx_{t-j}
    dy = df[y].diff()
    reg = pd.DataFrame({"dy": dy, "ect_l1": ect})

    for i in range(1, max(p, 1)):
        reg[f"d{y}_L{i}"] = dy.shift(i)

    for x in x_list:
        dx = df[x].diff()
        # include lag 0..q
        for j in range(0, q_map.get(x, 0) + 1):
            reg[f"d{x}_L{j}"] = dx.shift(j)

    reg = reg.dropna()
    if len(reg) < 10:
        return None, lr_table, ect

    Y = reg["dy"]
    X = reg.drop(columns=["dy"])
    X = sm.add_constant(X, has_constant="add")

    mod = sm.OLS(Y.values, X.values)
    # HAC with lag=1 is a sensible default for annual
    try:
        res = mod.fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    except Exception:
        res = mod.fit()

    # Build nice table
    tab = pd.DataFrame(
        {
            "term": ["const"] + list(X.columns[1:]),
            "coef": res.params,
            "std_err": res.bse,
            "t": res.tvalues,
            "p": res.pvalues,
        }
    )

    # Attach as attribute
    res.ecm_table_ = tab  # type: ignore
    return res, lr_table, ect


def _engle_granger_resid_coint(df: pd.DataFrame, y: str, x_list: List[str], trend: str = "c") -> Dict[str, float]:
    Y = df[y].astype(float)
    X = df[x_list].astype(float)
    X = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(Y.values, X.values).fit()
    resid = pd.Series(ols.resid, index=df.index)
    try:
        p = float(adfuller(resid.dropna().values, regression="n", autolag="AIC")[1])
    except Exception:
        p = np.nan
    return {"eg_adf_p_resid": p}


def _var_lag_table(sel) -> pd.DataFrame:
    """Create a defensible lag selection table from VAR select_order result."""
    rows = []
    for crit in ["aic", "bic", "hqic", "fpe"]:
        v = getattr(sel, crit, None)
        if v is None:
            continue
        try:
            if isinstance(v, (pd.Series, pd.DataFrame)):
                s = pd.Series(v)
            else:
                s = pd.Series(v)
            for lag, val in s.items():
                rows.append({"criterion": crit, "lag": int(lag), "value": float(val)})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame({"note": ["Lag selection table unavailable in this statsmodels version."]})
    return pd.DataFrame(rows).sort_values(["criterion", "lag"])


def _plot_residual_diagnostics(residuals: pd.Series, fitted: Optional[pd.Series], title_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4), dpi=130)
    if fitted is not None:
        ax.scatter(fitted, residuals, s=16)
        ax.set_xlabel("Fitted")
    else:
        ax.plot(residuals.index, residuals.values, linewidth=1.2)
        ax.set_xlabel("Index")
    ax.set_ylabel("Residual")
    ax.set_title(f"{title_prefix}: Residuals", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    _fig_show(fig)

    # QQ
    fig2, ax2 = plt.subplots(figsize=(10.5, 4), dpi=130)
    sm.qqplot(residuals.values, line="45", ax=ax2)
    ax2.set_title(f"{title_prefix}: QQ plot", fontsize=12, fontweight="bold")
    _fig_show(fig2)


def _plot_var_irf_fevd(var_res, steps: int = 10) -> None:
    """Plot IRF and FEVD with layout safeguards.

    Notes:
    - statsmodels may return a Figure or a list/array depending on version.
    - We always funnel figures through _fig_show().
    """

    # IRF
    try:
        irf = var_res.irf(steps)
        fig = irf.plot(orth=False)
        # Some versions return ndarray of axes; extract figure safely
        if hasattr(fig, "get_figure"):
            fig = fig.get_figure()
        if hasattr(fig, "set_size_inches"):
            try:
                fig.set_size_inches(11.5, 7.5, forward=True)
            except Exception:
                pass
        if hasattr(fig, "axes"):
            _fig_show(fig)
        else:
            st.info("IRF plot rendered as multiple figures in this environment.")
    except Exception as e:
        st.warning(f"IRF not available: {e}")

    # FEVD
    try:
        fevd = var_res.fevd(steps)
        fig = fevd.plot()
        if hasattr(fig, "get_figure"):
            fig = fig.get_figure()
        if hasattr(fig, "set_size_inches"):
            try:
                fig.set_size_inches(11.5, 7.5, forward=True)
            except Exception:
                pass
        if hasattr(fig, "axes"):
            _fig_show(fig)
        else:
            st.info("FEVD plot rendered as multiple figures in this environment.")
    except Exception as e:
        st.warning(f"FEVD not available: {e}")




# ============================================================
# EXPORT HELPERS
# ============================================================


def _excel_bytes(tables: Dict[str, pd.DataFrame], notes: Optional[Dict[str, str]] = None) -> bytes:
    """Write Excel to bytes. Try openpyxl then xlsxwriter; fallback to ZIP of CSV."""
    # Ensure config is always present
    if notes is not None and "config" not in tables:
        tables = dict(tables)
        tables["config"] = pd.DataFrame([{"key": k, "value": v} for k, v in notes.items()])

    bio = io.BytesIO()

    engines = ["openpyxl", "xlsxwriter"]
    last_err: Optional[Exception] = None

    for eng in engines:
        try:
            with pd.ExcelWriter(bio, engine=eng) as writer:
                for name, df in tables.items():
                    if df is None:
                        continue
                    sheet = str(name)[:31]
                    d = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
                    d.to_excel(writer, sheet_name=sheet, index=False)
            return bio.getvalue()
        except Exception as e:
            last_err = e
            bio = io.BytesIO()

    # Fallback to ZIP of CSV
    zbio = io.BytesIO()
    with zipfile.ZipFile(zbio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            if df is None:
                continue
            d = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            zf.writestr(f"{name}.csv", d.to_csv(index=False))
    if notes is not None:
        zf_note = pd.DataFrame([{"key": k, "value": v} for k, v in notes.items()])
        with zipfile.ZipFile(zbio, mode="a", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("config.csv", zf_note.to_csv(index=False))

    # If you reached here, Excel not possible
    return zbio.getvalue()


# ============================================================
# APP LAYOUT
# ============================================================

st.markdown("<div class='h-title'>Academic Econometrics Suite</div>", unsafe_allow_html=True)
st.markdown("<div class='h-sub'>Panel (Pooled/FE/RE/CRE) + Annual Time Series (VAR/VECM/ARDL) — with validation, diagnostics, and audit-ready export.</div>", unsafe_allow_html=True)

mode = st.sidebar.radio("Module", ["Panel data", "Time series (separate file)"], index=0)


# ============================================================
# PANEL MODULE
# ============================================================

if mode == "Panel data":
    if not _HAS_LINEARMODELS:
        st.error("Package 'linearmodels' not available in this environment. Install it to use the Panel module.")
        st.stop()

    st.header("Panel Data Module")

    uploaded = st.file_uploader("Upload panel dataset (.csv / .xlsx)", type=["csv", "xlsx", "xls"], key="panel_uploader")
    if uploaded is None:
        st.info("Upload dataset panel untuk mula.")
        st.stop()

    df_raw, meta_file = _read_uploaded_file(uploaded)

    with st.expander("Data preview", expanded=True):
        st.dataframe(df_raw.head(25), use_container_width=True)

    cols = df_raw.columns.tolist()
    if len(cols) < 4:
        st.error("Dataset terlalu kecil. Perlukan sekurang-kurangnya id, time, Y, dan sekurang-kurangnya 1 X.")
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
    if len(x_cols) == 0:
        st.error("Pilih sekurang-kurangnya 1 regressor X.")
        st.stop()

    st.subheader("Data screening (transform suggestions)")
    numeric_candidates = [y_col] + x_cols
    screen_df = _screen_transform_suggestions(df_raw, numeric_candidates)
    st_table(screen_df, max_rows=200)

    st.subheader("Transform controls (optional)")
    st.caption("Ini bukan auto. Kalau anda apply, anda bertanggungjawab justify dalam penulisan.")

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

    st.subheader("Effects & covariance")
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
        entity_fe, time_fe = True, True

    st.subheader("Run")
    run = st.button("Estimate panel models", type="primary")
    if not run:
        st.stop()

    df_panel, y, X, meta_prep = _panel_prepare(
        df_raw,
        id_col=id_col,
        time_col=time_col,
        y_col=y_col,
        x_cols=x_cols,
        add_time_dummies=(time_fe and not entity_fe),
        transform_map=transform_map,
    )

    # Covariance mapping
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

    # Metrics header
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Observations", meta_prep.get("n_obs", ""))
    with m2:
        st.metric("Entities", meta_prep.get("n_entities", ""))
    with m3:
        st.metric("Time min", meta_prep.get("time_min", ""))
    with m4:
        st.metric("Time max", meta_prep.get("time_max", ""))

    st.subheader("Estimation results")

    results: Dict[str, object] = {}

    # Pooled
    pooled_res = None
    try:
        pooled_mod = PooledOLS(y, X)
        pooled_res = pooled_mod.fit(cov_type=cov_type, **cov_kwargs)
        results["PooledOLS"] = pooled_res
        with st.expander("Pooled OLS summary", expanded=False):
            st.text(pooled_res.summary)
    except Exception as e:
        st.error(f"Pooled OLS failed: {e}")

    # FE
    fe_res = None
    if entity_fe or time_fe:
        try:
            fe_mod = PanelOLS(y, X, entity_effects=bool(entity_fe), time_effects=bool(time_fe), drop_absorbed=True)
            fe_res = fe_mod.fit(cov_type=cov_type, **cov_kwargs)
            results["FixedEffects"] = fe_res
            with st.expander("Fixed Effects summary", expanded=False):
                st.text(fe_res.summary)
        except Exception as e:
            st.error(f"Fixed Effects failed: {e}")

    # RE
    re_res = None
    try:
        X_re = X
        if time_fe:
            time_index = y.index.get_level_values(1)
            td = pd.get_dummies(time_index, prefix="time", drop_first=True)
            td.index = y.index
            X_re = pd.concat([X, td], axis=1)
            X_re = X_re.loc[:, ~X_re.columns.duplicated()]
        re_mod = RandomEffects(y, X_re)
        re_res = re_mod.fit(cov_type=cov_type, **cov_kwargs)
        results["RandomEffects"] = re_res
        with st.expander("Random Effects summary", expanded=False):
            st.text(re_res.summary)
    except Exception as e:
        st.error(f"Random Effects failed: {e}")

    # CRE (Mundlak)
    cre_res = None
    try:
        # add entity means of X (excluding const)
        X0 = X.drop(columns=[c for c in X.columns if c.lower() in {"const", "intercept"}], errors="ignore")
        means = X0.groupby(level=0).transform("mean")
        means.columns = [f"mean_{c}" for c in means.columns]
        X_cre = pd.concat([X, means], axis=1)

        if time_fe:
            time_index = y.index.get_level_values(1)
            td = pd.get_dummies(time_index, prefix="time", drop_first=True)
            td.index = y.index
            X_cre = pd.concat([X_cre, td], axis=1)
            X_cre = X_cre.loc[:, ~X_cre.columns.duplicated()]

        cre_mod = RandomEffects(y, X_cre)
        cre_res = cre_mod.fit(cov_type=cov_type, **cov_kwargs)
        results["CRE_Mundlak"] = cre_res
        with st.expander("CRE (Mundlak) summary", expanded=False):
            st.text(cre_res.summary)
    except Exception:
        pass

    st.subheader("Model selection")
    st.caption(
        "Hypotheses: (1) FE vs pooled F-test: H0 pooled sufficient; (2) RE vs pooled LM: H0 var(u_i)=0; (3) Hausman: H0 RE consistent."
    )

    sel_rows = []
    if fe_res is not None:
        try:
            fp = fe_res.f_pooled
            sel_rows.append({"test": "F-test (FE vs pooled)", "stat": float(fp.stat), "p_value": float(fp.pval), "df": str(fp.df)})
        except Exception as e:
            sel_rows.append({"test": "F-test (FE vs pooled)", "stat": np.nan, "p_value": np.nan, "df": f"error: {e}"})

    if pooled_res is not None:
        try:
            lm_stat, lm_p = _bp_lm_test_re_vs_pooled_balanced(pooled_res.resids)
            sel_rows.append({"test": "Breusch–Pagan LM (RE vs pooled; balanced only)", "stat": lm_stat, "p_value": lm_p, "df": "chi2(1)"})
        except Exception as e:
            sel_rows.append({"test": "Breusch–Pagan LM (RE vs pooled)", "stat": np.nan, "p_value": np.nan, "df": f"skipped: {e}"})

    if fe_res is not None and re_res is not None:
        try:
            stat, p, dof = _hausman_test_aligned(fe_res.params, fe_res.cov, re_res.params, re_res.cov)
            sel_rows.append({"test": "Hausman (FE vs RE)", "stat": stat, "p_value": p, "df": f"chi2({dof})"})
        except Exception as e:
            sel_rows.append({"test": "Hausman (FE vs RE)", "stat": np.nan, "p_value": np.nan, "df": f"error: {e}"})

    sel_df = pd.DataFrame(sel_rows)
    st_table(sel_df)

    st.subheader("Recommended model (rule-based)")
    rec = "PooledOLS"
    rationale: List[str] = []

    fe_p = None
    re_p = None
    haus_p = None
    for _, r in sel_df.iterrows():
        if str(r["test"]).startswith("F-test"):
            fe_p = r["p_value"]
        if str(r["test"]).startswith("Breusch"):
            re_p = r["p_value"]
        if str(r["test"]).startswith("Hausman"):
            haus_p = r["p_value"]

    if fe_p is not None and pd.notna(fe_p) and fe_p < 0.05:
        rec = "FixedEffects"
        rationale.append("FE vs pooled: significant → pooled rejected.")

    if rec == "PooledOLS" and re_p is not None and pd.notna(re_p) and re_p < 0.05:
        rec = "RandomEffects"
        rationale.append("RE vs pooled: LM significant (balanced only) → pooled rejected.")

    if (fe_res is not None) and (re_res is not None) and (haus_p is not None) and pd.notna(haus_p):
        if haus_p < 0.05:
            rec = "FixedEffects"
            rationale.append("Hausman significant → RE inconsistent → choose FE.")
        else:
            rec = "RandomEffects"
            rationale.append("Hausman not significant → RE acceptable and efficient.")

    st.markdown(f"<div class='card'><h3>Recommended: {rec}</h3>" + "".join([f"<p>• {x}</p>" for x in rationale]) + "</div>", unsafe_allow_html=True)

    st.subheader("Diagnostics")
    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.markdown("##### Multicollinearity (VIF)")
        vif_df = _vif_table(X)
        st_table(vif_df)

    with d2:
        st.markdown("##### Heteroskedasticity (Breusch–Pagan; pooled residuals)")
        if pooled_res is not None:
            try:
                lm, lm_p, f, f_p = het_breuschpagan(pooled_res.resids.values, X.values)
                bp_df = pd.DataFrame([
                    {"test": "LM", "stat": float(lm), "p_value": float(lm_p)},
                    {"test": "F", "stat": float(f), "p_value": float(f_p)},
                ])
                st_table(bp_df)
            except Exception as e:
                st.write(f"Skipped: {e}")
        else:
            st.write("Need pooled model.")

    with d3:
        st.markdown("##### Cross-sectional dependence (Pesaran CD; balanced only)")
        base_resid = fe_res.resids if fe_res is not None else (pooled_res.resids if pooled_res is not None else None)
        if base_resid is not None:
            try:
                cd, cd_p = _pesaran_cd_test_balanced(base_resid)
                st_table(pd.DataFrame([{"CD_stat": cd, "p_value": cd_p}]))
            except Exception as e:
                st.write(f"Skipped: {e}")
        else:
            st.write("No residuals available.")

    with d4:
        st.markdown("##### Serial correlation (Wooldridge/Drukker AR(1))")
        st.caption("H0: no AR(1). Test implemented via first-difference residual regression; H0 tested as coef = -0.5.")
        base_for_test = fe_res if fe_res is not None else pooled_res
        if base_for_test is not None:
            try:
                beta, t_stat, p_val = _wooldridge_serial_corr_test(y, X)
                st_table(pd.DataFrame([{"coef(u_lag)": beta, "t_stat(H0: coef=-0.5)": t_stat, "p_value": p_val}]))
            except Exception as e:
                st.write(f"Skipped: {e}")
        else:
            st.write("Need at least one fitted model.")

    st.subheader("Export report")

    def _coef_table(res) -> pd.DataFrame:
        if res is None:
            return pd.DataFrame()
        out = pd.DataFrame({
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.std_errors.values,
            "t": res.tstats.values,
            "p": res.pvalues.values,
        })
        return out

    export_tables: Dict[str, pd.DataFrame] = {
        "panel_selection": sel_df,
        "panel_screening": screen_df,
        "panel_vif": vif_df,
        "coef_pooled": _coef_table(pooled_res),
        "coef_fe": _coef_table(fe_res),
        "coef_re": _coef_table(re_res),
        "coef_cre_mundlak": _coef_table(cre_res),
    }

    config = {
        **meta_file,
        **meta_prep,
        "id_col": id_col,
        "time_col": time_col,
        "y_col": y_col,
        "x_cols": ",".join(x_cols),
        "entity_fe": str(entity_fe),
        "time_fe": str(time_fe),
        "twoway": str(twoway),
        "covariance": cov_choice,
        "recommended_model": rec,
    }

    xbytes = _excel_bytes(export_tables, notes=config)

    # Decide mime based on whether it looks like zip
    is_zip = xbytes[:2] == b"PK"

    st.download_button(
        "Download panel report",
        data=xbytes,
        file_name=("panel_report.zip" if is_zip else "panel_report.xlsx"),
        mime=("application/zip" if is_zip else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    )


# ============================================================
# TIME SERIES MODULE
# ============================================================

else:
    st.header("Time Series Module (Separate File, Annual)")

    uploaded = st.file_uploader("Upload time series dataset (.csv / .xlsx)", type=["csv", "xlsx", "xls"], key="ts_uploader")
    if uploaded is None:
        st.info("Upload dataset siri masa untuk mula.")
        st.stop()

    df_raw, meta_file = _read_uploaded_file(uploaded)

    with st.expander("Data preview", expanded=True):
        st.dataframe(df_raw.head(25), use_container_width=True)

    cols = df_raw.columns.tolist()
    default_year = "year" if "year" in cols else cols[0]

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
        st.error("Pilih sekurang-kurangnya 2 variables untuk VAR/VECM. (ARDL perlukan 1 Y + sekurang-kurangnya 1 X).")
        st.stop()

    st.subheader("Data screening (transform suggestions)")
    screen_df = _screen_transform_suggestions(df_raw, vars_)
    st_table(screen_df)

    st.subheader("Transform controls (optional)")
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

    ts_df, meta_prep = _ts_prepare(df_raw, year_col=year_col, vars_=vars_, transform_map=transform_map)
    T = len(ts_df)

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("T (annual)", T)
    with m2:
        st.metric("Year min", meta_prep.get("year_min", ""))
    with m3:
        st.metric("Year max", meta_prep.get("year_max", ""))

    st.subheader("Plots: levels")
    _line_plot(ts_df[vars_], title="Time series (levels)")

    st.subheader("Stationarity screening")
    adf_reg = st.selectbox("ADF deterministic term", ["c", "ct"], index=0)
    kpss_reg = st.selectbox("KPSS deterministic term", ["c", "ct"], index=0)

    stat_df = _stationarity_suite(ts_df, vars_, adf_reg=adf_reg, kpss_reg=kpss_reg)
    st_table(stat_df)

    st.caption("Rule used: ADF p<0.05 AND KPSS p>0.05 → I(0); ADF p>=0.05 AND KPSS p<=0.05 → I(1) likely; else mixed/unclear.")

    # I(2) screening
    integ_rows = []
    max_d = 0
    for v in vars_:
        d, hist = _integration_order_by_diff(ts_df[v], adf_reg=adf_reg, kpss_reg=kpss_reg, max_diff=2)
        max_d = max(max_d, d)
        integ_rows.append({"variable": v, "integration_order_est": d})
    integ_df = pd.DataFrame(integ_rows)

    with st.expander("Integration order check (up to I(2))", expanded=False):
        st_table(integ_df)
        if max_d >= 2:
            st.warning("At least one variable looks I(2) or worse under ADF+KPSS-by-differencing. VECM/ARDL cointegration inference is not appropriate without further work.")

    st.subheader("Model")
    model = st.radio("Choose time series model", ["VAR", "VECM", "ARDL"], horizontal=True)

    # Common config export dict
    base_config = {**meta_file, **meta_prep, "year_col": year_col, "vars": ",".join(vars_), "adf_reg": adf_reg, "kpss_reg": kpss_reg}

    # ---------------- VAR ----------------
    if model == "VAR":
        st.markdown("### VAR")
        endog = st.multiselect("Endogenous variables", vars_, default=vars_)
        if len(endog) < 2:
            st.error("VAR requires at least 2 endogenous variables.")
            st.stop()

        # Spurious guardrail
        has_i1 = (stat_df[stat_df["variable"].isin(endog)]["classification"].str.contains("I\\(1\\)", regex=True)).any()
        data_mode = st.radio("VAR data mode", ["First differences", "Levels"], index=0 if has_i1 else 1, horizontal=True)
        if has_i1 and data_mode == "Levels":
            confirm = st.checkbox("I understand VAR in levels may be spurious if variables are I(1) without cointegration.", value=False)
            if not confirm:
                st.warning("Switch to 'First differences' or confirm the risk.")
                st.stop()

        default_maxlag = min(4, max(1, T // 10))
        maxlags = st.slider("Max lag to consider (annual guardrail)", 1, min(8, max(2, T // 5)), default_maxlag)
        ic = st.selectbox("Lag selection criterion", ["bic", "aic", "hqic"], index=0)
        trend = st.selectbox("Deterministic term", ["c", "ct", "n"], index=0)
        steps = st.slider("IRF/FEVD steps", 5, 20, 10)

        run = st.button("Fit VAR", type="primary")
        if not run:
            st.stop()

        data = ts_df[endog].copy()
        if data_mode == "First differences":
            data = data.diff().dropna()

        try:
            var_mod = VAR(data)
            sel = var_mod.select_order(maxlags=maxlags, trend=trend)
            lag = int(getattr(sel, ic))
            if lag < 1:
                lag = 1
            var_res = var_mod.fit(lag, trend=trend)

            st.markdown(f"**Selected lag ({ic})** = {lag}")
            with st.expander("VAR summary", expanded=False):
                st.text(var_res.summary())

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

            st_table(pd.DataFrame(diag_rows))

            st.subheader("Causality (model-consistent)")
            st.caption("Matrix shows p-values for H0: 'causing' does NOT Granger-cause 'caused' (Wald test).")
            caus_df = _matrix_from_var_causality(var_res, endog)
            st_table(caus_df)

            st.subheader("IRF and FEVD")
            _plot_var_irf_fevd(var_res, steps=steps)

            # Export tables
            lag_tbl = _var_lag_table(sel)
            coef_tables = []
            for eqn in endog:
                params = var_res.params[eqn].reset_index().rename(columns={"index": "term", eqn: "coef"})
                params["equation"] = eqn
                coef_tables.append(params)

            export = {
                "ts_screening": screen_df,
                "ts_stationarity": stat_df,
                "ts_integration": integ_df,
                "var_lag_selection": lag_tbl,
                "var_causality_p": caus_df,
                "var_params": pd.concat(coef_tables, ignore_index=True),
                "var_diagnostics": pd.DataFrame(diag_rows),
            }

            config = {**base_config, "model": "VAR", "endog": ",".join(endog), "selected_lag": str(lag), "trend": trend, "ic": ic, "data_mode": data_mode}
            xbytes = _excel_bytes(export, notes=config)
            is_zip = xbytes[:2] == b"PK"

            st.download_button(
                "Download TS report",
                data=xbytes,
                file_name=("ts_report_VAR.zip" if is_zip else "ts_report_VAR.xlsx"),
                mime=("application/zip" if is_zip else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            )

        except Exception as e:
            st.error(f"VAR failed: {e}")

    # ---------------- VECM ----------------
    elif model == "VECM":
        st.markdown("### VECM")
        endog = st.multiselect("Endogenous variables", vars_, default=vars_)
        if len(endog) < 2:
            st.error("VECM requires at least 2 endogenous variables.")
            st.stop()

        # Gatekeeping: require all I(1) and no I(2)
        if max_d >= 2:
            st.error("VECM blocked: at least one series looks I(2)+ under differencing-based checks.")
            st.stop()

        stat_subset = stat_df[stat_df["variable"].isin(endog)].copy()
        not_i1 = stat_subset[~stat_subset["classification"].str.contains("I\\(1\\)", regex=True)]
        if len(not_i1) > 0:
            st.error("VECM blocked: all endogenous variables must be I(1) under the ADF+KPSS rule.")
            st_table(not_i1)
            st.stop()

        default_k = 1
        k_ar_diff = st.slider("Lag in differences (k_ar_diff)", 1, min(6, max(1, T // 8)), default_k)
        det = st.selectbox(
            "Deterministic term",
            ["nc", "co", "ct"],
            index=1,
            help="nc: none, co: constant in cointegration relation, ct: constant + trend in cointegration relation",
        )

        run = st.button("Fit VECM", type="primary")
        if not run:
            st.stop()

        data = ts_df[endog].copy()

        try:
            det_order = {"nc": -1, "co": 0, "ct": 1}[det]
            joh = coint_johansen(data, det_order, k_ar_diff)

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
            st_table(jtab)
            st.markdown(f"**Selected rank (5%)** = {rank}")

            if rank == 0:
                st.error("No cointegration (rank=0). VECM not appropriate. Use VAR on differenced data or ARDL.")
                st.stop()

            vecm_mod = VECM(data, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic=det)
            vecm_res = vecm_mod.fit()

            with st.expander("VECM summary", expanded=False):
                st.text(vecm_res.summary())

            # Key VECM objects
            try:
                alpha = pd.DataFrame(vecm_res.alpha, index=endog)
                alpha.columns = [f"coint_{i+1}" for i in range(alpha.shape[1])]
            except Exception:
                alpha = pd.DataFrame()

            try:
                beta = pd.DataFrame(vecm_res.beta, index=endog)
                beta.columns = [f"coint_{i+1}" for i in range(beta.shape[1])]
            except Exception:
                beta = pd.DataFrame()

            if not alpha.empty:
                st.subheader("Adjustment coefficients (alpha)")
                st_table(alpha.reset_index().rename(columns={"index": "variable"}))

            if not beta.empty:
                st.subheader("Cointegration vectors (beta)")
                st_table(beta.reset_index().rename(columns={"index": "variable"}))

            st.subheader("Causality (model-consistent)")
            st.caption("Matrix shows p-values for H0: 'causing' does NOT Granger-cause 'caused' (VECM test).")
            caus_df = _matrix_from_vecm_causality(vecm_res, endog)
            st_table(caus_df)

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
            st_table(pd.DataFrame(diag_rows))

            export = {
                "ts_screening": screen_df,
                "ts_stationarity": stat_df,
                "ts_integration": integ_df,
                "johansen_trace": jtab,
                "alpha": alpha.reset_index().rename(columns={"index": "variable"}) if not alpha.empty else pd.DataFrame(),
                "beta": beta.reset_index().rename(columns={"index": "variable"}) if not beta.empty else pd.DataFrame(),
                "vecm_causality_p": caus_df,
                "vecm_diagnostics": pd.DataFrame(diag_rows),
            }

            config = {**base_config, "model": "VECM", "endog": ",".join(endog), "k_ar_diff": str(k_ar_diff), "deterministic": det, "rank": str(rank)}
            xbytes = _excel_bytes(export, notes=config)
            is_zip = xbytes[:2] == b"PK"

            st.download_button(
                "Download TS report",
                data=xbytes,
                file_name=("ts_report_VECM.zip" if is_zip else "ts_report_VECM.xlsx"),
                mime=("application/zip" if is_zip else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            )

        except Exception as e:
            st.error(f"VECM failed: {e}")

    # ---------------- ARDL ----------------
    else:
        st.markdown("### ARDL")

        y_name = st.selectbox("Dependent variable (Y)", vars_)
        x_list = st.multiselect("Regressors (X)", [v for v in vars_ if v != y_name], default=[v for v in vars_ if v != y_name][:2])
        if len(x_list) == 0:
            st.error("ARDL requires at least one regressor X.")
            st.stop()

        # Gate: ARDL not for I(2)
        if max_d >= 2:
            st.error("ARDL blocked: at least one series looks I(2)+. Use different transformations/specification.")
            st.stop()

        max_p = st.slider("Max lag for Y (p)", 1, min(6, max(2, T // 8)), 2)
        max_q = st.slider("Max lag for X (q)", 0, min(6, max(1, T // 8)), 2)
        ic = st.selectbox("Model selection IC", ["bic", "aic"], index=0)
        trend = st.selectbox("Deterministic term", ["c", "ct", "n"], index=0)

        run = st.button("Fit ARDL", type="primary")
        if not run:
            st.stop()

        data_y = ts_df[y_name].astype(float)
        data_X = ts_df[x_list].astype(float)

        try:
            sel = ardl_select_order(data_y, max_p, data_X, max_q, ic=ic, trend=trend)
            ardl_res = sel.model.fit()

            st.markdown("**Selected ARDL order**")
            with st.expander("ARDL selected order object", expanded=False):
                st.write(sel)

            with st.expander("ARDL summary", expanded=False):
                st.text(ardl_res.summary())

            # Cointegration (Bounds test if available; else honest fallback)
            st.subheader("Cointegration test")
            coint_rows = []
            bounds_available = ardl_bonds_test is not None
            if bounds_available:
                try:
                    bt = ardl_bonds_test(ardl_res)
                    # bt may be result object; attempt to extract
                    coint_rows.append({"test": "ARDL bounds test", "stat": getattr(bt, "statistic", np.nan), "p_value": getattr(bt, "pvalue", np.nan)})
                except Exception as e:
                    bounds_available = False
                    st.warning(f"Bounds test not usable in this environment: {e}")

            if not bounds_available:
                st.info("Bounds test is not available in this statsmodels environment. Showing Engle–Granger residual-based test as indicative only.")
                eg = _engle_granger_resid_coint(ts_df[[y_name] + x_list], y=y_name, x_list=x_list, trend=trend)
                coint_rows.append({"test": "Engle–Granger (indicative)", "stat": np.nan, "p_value": eg.get("eg_adf_p_resid", np.nan)})

            st_table(pd.DataFrame(coint_rows))

            st.subheader("Long-run & ECM")
            lr_tbl, const_lr, p_est = _ardl_long_run(ardl_res, y=y_name, x_list=x_list)
            st_table(lr_tbl)

            ecm_res, lr_tbl2, ect_series = _ardl_ecm(ardl_res, ts_df, y=y_name, x_list=x_list, adf_reg=adf_reg, kpss_reg=kpss_reg)
            if ecm_res is None:
                st.warning("ECM could not be estimated reliably (insufficient aligned observations).")
            else:
                # highlight ect term
                ecm_tab = getattr(ecm_res, "ecm_table_", pd.DataFrame())
                st_table(ecm_tab)
                try:
                    ect_row = ecm_tab[ecm_tab["term"] == "ect_l1"].iloc[0]
                    st.markdown(
                        f"<div class='card'><h3>Speed of adjustment (ECT)</h3><p>ECT coefficient (expected negative if stable): <b>{ect_row['coef']:.4g}</b>, p-value: <b>{ect_row['p']:.4g}</b></p></div>",
                        unsafe_allow_html=True,
                    )
                except Exception:
                    pass

            st.subheader("ARDL diagnostics")
            diag_rows = []
            # statsmodels ARDLResults exposes some tests in newer versions
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

            # ARCH LM
            try:
                from statsmodels.stats.diagnostic import het_arch
                arch_stat, arch_p, _, _ = het_arch(ardl_res.resid)
                diag_rows.append({"test": "ARCH LM", "stat": float(arch_stat), "p_value": float(arch_p)})
            except Exception:
                pass

            # CUSUM (if available)
            try:
                from statsmodels.stats.diagnostic import breaks_cusumolsresid
                cus_stat, cus_p, _ = breaks_cusumolsresid(ardl_res)
                diag_rows.append({"test": "CUSUM", "stat": float(cus_stat), "p_value": float(cus_p)})
            except Exception:
                pass

            st_table(pd.DataFrame(diag_rows))

            # Plots (avoid overlap)
            st.subheader("Residual diagnostics plots")
            resid = pd.Series(ardl_res.resid, index=ts_df.index[-len(ardl_res.resid):])
            try:
                fitted = pd.Series(ardl_res.fittedvalues, index=resid.index)
            except Exception:
                fitted = None
            _plot_residual_diagnostics(resid.dropna(), fitted.dropna() if fitted is not None else None, title_prefix="ARDL")

            # Granger-style Wald tests
            st.subheader("Causality (model-consistent; Wald on ARDL terms)")
            st.caption("H0: all lagged terms of X are jointly zero (in the ARDL regression).")
            # A light-weight implementation: test all params containing each x
            caus_rows = []
            names = list(ardl_res.params.index)
            for x in x_list:
                terms = [nm for nm in names if (x in nm and (".L" in nm or nm.startswith("L"))) and (y_name not in nm)]
                terms = list(dict.fromkeys(terms))
                if not terms:
                    caus_rows.append({"causing": x, "stat": np.nan, "p_value": np.nan, "terms": ""})
                    continue
                k = len(names)
                R = np.zeros((len(terms), k))
                for i, tnm in enumerate(terms):
                    R[i, names.index(tnm)] = 1.0
                try:
                    w = ardl_res.wald_test(R)
                    caus_rows.append({"causing": x, "stat": float(w.statistic), "p_value": float(w.pvalue), "terms": ", ".join(terms)[:250]})
                except Exception:
                    caus_rows.append({"causing": x, "stat": np.nan, "p_value": np.nan, "terms": ", ".join(terms)[:250]})

            caus_df = pd.DataFrame(caus_rows)
            st_table(caus_df)

            # Export
            coef = pd.DataFrame({
                "term": ardl_res.params.index,
                "coef": ardl_res.params.values,
                "std_err": ardl_res.bse.values,
                "t": ardl_res.tvalues.values,
                "p": ardl_res.pvalues.values,
            })

            export = {
                "ts_screening": screen_df,
                "ts_stationarity": stat_df,
                "ts_integration": integ_df,
                "cointegration": pd.DataFrame(coint_rows),
                "ardl_params": coef,
                "ardl_long_run": lr_tbl,
                "ecm_table": getattr(ecm_res, "ecm_table_", pd.DataFrame()) if ecm_res is not None else pd.DataFrame(),
                "ardl_diagnostics": pd.DataFrame(diag_rows),
                "ardl_causality": caus_df,
            }

            config = {**base_config, "model": "ARDL", "Y": y_name, "X": ",".join(x_list), "max_p": str(max_p), "max_q": str(max_q), "ic": ic, "trend": trend}
            xbytes = _excel_bytes(export, notes=config)
            is_zip = xbytes[:2] == b"PK"

            st.download_button(
                "Download TS report",
                data=xbytes,
                file_name=("ts_report_ARDL.zip" if is_zip else "ts_report_ARDL.xlsx"),
                mime=("application/zip" if is_zip else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            )

        except Exception as e:
            st.error(f"ARDL failed: {e}")
