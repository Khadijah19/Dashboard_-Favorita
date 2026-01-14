# pages/0_Admin.py
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# Imports projet
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.hf_artifacts import read_latest, download_artifacts_from_latest
from utils.data_loader import load_train_from_hf

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Admin", page_icon="🛠️", layout="wide")

HF_REPO_ID = os.getenv("HF_REPO_ID", "khadidia-77/favorita")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")
PARQUET_NAME = os.getenv("PARQUET_NAME", "train_last10w.parquet")
MAX_WEEKS = 10

# ============================================================
# CSS (adopté depuis app.py)
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 1600px;
}

/* ===== HERO SECTION ===== */
.dashboard-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    border-radius: 24px;
    padding: 3.5rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 25px 70px rgba(15, 23, 42, 0.4);
    position: relative;
    overflow: hidden;
}

.dashboard-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 700px;
    height: 700px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, transparent 60%);
    border-radius: 50%;
    animation: float 8s ease-in-out infinite;
}

.dashboard-hero::after {
    content: '';
    position: absolute;
    bottom: -40%;
    left: -15%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%);
    border-radius: 50%;
    animation: float 10s ease-in-out infinite reverse;
}

@keyframes float {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(30px, -30px); }
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero-title {
    color: white;
    font-size: 3rem;
    font-weight: 900;
    margin: 0 0 0.8rem 0;
    letter-spacing: -0.04em;
    text-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    line-height: 1.1;
}

.hero-subtitle {
    color: rgba(255, 255, 255, 0.95);
    font-size: 1.2rem;
    margin: 0;
    font-weight: 400;
    line-height: 1.6;
}

/* ===== CHART SECTIONS ===== */
.chart-section {
    background: white;
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0 8px 35px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(0, 0, 0, 0.04);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.chart-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
}

.chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid #f1f5f9;
}

.chart-title {
    font-size: 1.4rem;
    font-weight: 900;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.chart-icon {
    width: 10px;
    height: 10px;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    border-radius: 50%;
    display: inline-block;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fafaf9 0%, #ffffff 100%);
    border-right: 2px solid rgba(0, 0, 0, 0.06);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

section[data-testid="stSidebar"] h2 {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.5rem;
    font-weight: 900;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #f1f5f9;
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 8px;
    font-weight: 600;
    color: #1e293b;
    border: 1px solid #e2e8f0;
}

/* ===== INFO BOXES ===== */
.stInfo {
    background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
}

/* ===== DATA TABLE ===== */
.dataframe {
    border: none !important;
    border-radius: 8px;
    overflow: hidden;
}

.dataframe thead tr {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
}

.dataframe thead th {
    color: #1e293b;
    font-weight: 700;
    padding: 0.8rem;
    font-size: 0.85rem;
}

.dataframe tbody tr:hover {
    background: #f8fafc;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    transform: translateY(-2px);
}

/* ===== FOOTER ===== */
.dashboard-footer {
    text-align: center;
    color: #64748b;
    padding: 2rem 0;
    margin-top: 3rem;
    border-top: 2px solid #f1f5f9;
}

.footer-title {
    font-size: 0.95rem;
    margin: 0;
    font-weight: 700;
    color: #1e293b;
}

.footer-text {
    font-size: 0.8rem;
    margin: 0.5rem 0 0 0;
    opacity: 0.8;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Helpers
# ============================================================
def _safe_expm1(x):
    x = np.asarray(x, dtype="float64")
    x = np.clip(x, -50, 50)
    return np.expm1(x)

def _to_bool_onpromotion(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)

    ss = s.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "t", "yes", "y"}
    falsy = {"false", "0", "f", "no", "n", "nan", "none", ""}
    return ss.apply(lambda v: True if v in truthy else (False if v in falsy else False))

@st.cache_resource(show_spinner=False)
def load_artifacts_latest(hf_token):
    return download_artifacts_from_latest(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        hf_token=hf_token,
        artifacts_dir="artifacts",
        cache_dir=".cache/favorita_artifacts",
    )

@st.cache_data(show_spinner=False)
def load_data_weeks(weeks: int):
    df_ = load_train_from_hf(
        weeks=int(weeks),
        filename=PARQUET_NAME,
    )
    df_["date"] = pd.to_datetime(df_["date"], errors="coerce").dt.normalize()
    df_ = df_.dropna(subset=["date"])

    keep = ["date", "store_nbr", "item_nbr", "onpromotion", "unit_sales"]
    keep = [c for c in keep if c in df_.columns]
    return df_[keep].copy()

# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
<div class="dashboard-hero">
  <div class="hero-content">
    <div class="hero-title">Administration</div>
    <div class="hero-subtitle">
      Monitoring du modèle et évaluation sur une fenêtre récente<br>
      Source: HuggingFace · Dataset: {HF_REPO_ID} · Fichier: {PARQUET_NAME}
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## Configuration")

    eval_weeks = st.selectbox(
        "Fenêtre de données pour l'évaluation (semaines)",
        [10, 8, 4, 3, 2, 1],
        index=0,
    )
    eval_days = st.slider(
        "Taille du jeu de validation (derniers jours)",
        min_value=7, max_value=28, value=14, step=1
    )
    max_rows = st.number_input(
        "Cap lignes (échantillonnage si trop gros)",
        min_value=50_000, max_value=1_000_000, value=300_000, step=50_000
    )

    st.divider()
    run_eval = st.button("Calculer les performances", width="stretch")

# ============================================================
# Statut du modèle (latest)
# ============================================================
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.markdown(
    """
<div class="chart-header">
  <div class="chart-title">
    <span class="chart-icon"></span>
    Statut du modèle
  </div>
</div>
""",
    unsafe_allow_html=True,
)

try:
    latest = read_latest(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
    st.success(f"Latest run: {latest.get('run_id')} | Updated at: {latest.get('updated_at')}")
    with st.expander("Afficher latest.json", expanded=False):
        st.json(latest)
except Exception as e:
    st.warning("latest.json est introuvable pour le moment.")
    st.caption(str(e))

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Evaluation
# ============================================================
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.markdown(
    """
<div class="chart-header">
  <div class="chart-title">
    <span class="chart-icon"></span>
    Evaluation du modèle
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("L'évaluation utilise les derniers jours de la fenêtre sélectionnée comme jeu de validation.")

if run_eval:
    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        hf_token = None

    # 1) Load artifacts latest
    try:
        model, pipe, feature_cols, meta = load_artifacts_latest(hf_token)
    except Exception as e:
        st.error("Impossible de charger les artefacts latest depuis HuggingFace.")
        st.exception(e)
        st.stop()

    st.caption(
        f"Artifacts loaded | run={meta.get('run_id')} | trained_at={meta.get('trained_at', meta.get('updated_at'))}"
    )

    # 2) Load data
    with st.spinner("Chargement des données..."):
        df = load_data_weeks(int(eval_weeks))

    needed = {"date", "store_nbr", "item_nbr", "onpromotion", "unit_sales"}
    missing = needed - set(df.columns)
    if missing:
        st.error(f"Colonnes manquantes pour l'évaluation: {sorted(list(missing))}")
        st.stop()

    df = df.copy()
    df["onpromotion"] = _to_bool_onpromotion(df["onpromotion"])
    df["unit_sales"] = pd.to_numeric(df["unit_sales"], errors="coerce")
    df = df.dropna(subset=["unit_sales"])
    df["unit_sales"] = df["unit_sales"].clip(lower=0)

    # 3) Split last eval_days
    max_d = df["date"].max()
    cut_d = max_d - pd.Timedelta(days=int(eval_days) - 1)
    valid = df.loc[df["date"] >= cut_d].copy()

    if len(valid) == 0:
        st.warning("Jeu de validation vide. Vérifier les dates.")
        st.stop()

    # 4) Sample
    if len(valid) > int(max_rows):
        valid = valid.sample(int(max_rows), random_state=42)
        st.info(f"Validation échantillonnée: {len(valid):,} lignes")

    st.caption(
        f"Validation window: {valid['date'].min().date()} to {valid['date'].max().date()} | n={len(valid):,}"
    )

    # 5) Build X/y
    y_true_log = np.log1p(valid["unit_sales"].values.astype("float64"))

    X_input = valid[["date", "store_nbr", "item_nbr", "onpromotion"]].copy()
    X_input["date"] = pd.to_datetime(X_input["date"]).dt.normalize()
    X_input["store_nbr"] = X_input["store_nbr"].astype(int)
    X_input["item_nbr"] = X_input["item_nbr"].astype(int)
    X_input["onpromotion"] = X_input["onpromotion"].astype(bool)

    with st.spinner("Transformation et prédiction..."):
        X_enriched = pipe.transform(X_input)
        X = (
            X_enriched.reindex(columns=feature_cols, fill_value=0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        y_pred_log = model.predict(X)

    # 6) Metrics log + units
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
    r2_log = float(r2_score(y_true_log, y_pred_log))

    y_true = valid["unit_sales"].values.astype("float64")
    y_pred = _safe_expm1(y_pred_log)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (units)", f"{rmse:,.3f}")
    c2.metric("MAE (units)", f"{mae:,.3f}")
    c3.metric("R2 (units)", f"{r2:,.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("RMSE (log1p)", f"{rmse_log:,.4f}")
    c5.metric("MAE (log1p)", f"{mae_log:,.4f}")
    c6.metric("R2 (log1p)", f"{r2_log:,.4f}")

    preview = valid[["date", "store_nbr", "item_nbr", "onpromotion"]].copy()
    preview["y_true_unit_sales"] = y_true.astype("float32")
    preview["y_pred_unit_sales"] = y_pred.astype("float32")
    preview["abs_err"] = np.abs(preview["y_true_unit_sales"] - preview["y_pred_unit_sales"]).astype("float32")

    with st.expander("Aperçu des erreurs (top 200)", expanded=False):
        st.dataframe(preview.sort_values("abs_err", ascending=False).head(200), width="stretch")

    st.success("Evaluation terminée.")

else:
    st.info("Cliquer sur le bouton pour lancer l'évaluation avec les paramètres de la barre latérale.")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.divider()
st.markdown(
    """
<div class="dashboard-footer">
  <p class="footer-title">Favorita Forecast Dashboard</p>
  <p class="footer-text">© 2026 · Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
