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
from utils.training import train_and_publish
from utils.data_loader import load_train_from_hf


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Admin", page_icon="⚙️", layout="wide")

HF_REPO_ID = os.getenv("HF_REPO_ID", "khadidia-77/favorita")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")
PARQUET_NAME = os.getenv("PARQUET_NAME", "train_last10w.parquet")


# ============================================================
# CSS PREMIUM
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding: 1.5rem 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}

/* ===== HERO SECTION ===== */
.admin-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    border-radius: 24px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(15, 23, 42, 0.4);
}

.admin-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.admin-hero::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -5%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero-title {
    color: white;
    font-size: 2.5rem;
    font-weight: 900;
    margin: 0 0 0.8rem 0;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.hero-subtitle {
    color: rgba(255, 255, 255, 0.85);
    font-size: 1.1rem;
    margin: 0;
    font-weight: 400;
}

.hero-badge {
    display: inline-block;
    background: rgba(59, 130, 246, 0.2);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    color: #93c5fd;
    font-weight: 600;
    margin-top: 1rem;
    font-size: 0.85rem;
}

/* ===== SECTION CARDS ===== */
.section-card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.section-card:hover {
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

.section-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 1.5rem;
    padding-bottom: 0.8rem;
    border-bottom: 2px solid #f1f5f9;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-icon {
    width: 8px;
    height: 8px;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    border-radius: 50%;
    display: inline-block;
}

/* ===== METRICS GRID ===== */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.2rem;
    margin: 1.5rem 0;
}

.metric-card {
    background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid;
    transition: all 0.3s ease;
}

.metric-card:nth-child(1) { border-left-color: #3b82f6; }
.metric-card:nth-child(2) { border-left-color: #8b5cf6; }
.metric-card:nth-child(3) { border-left-color: #10b981; }
.metric-card:nth-child(4) { border-left-color: #f59e0b; }

.metric-card:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
}

.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 900;
    color: #0f172a;
    line-height: 1;
}

.metric-detail {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    transform: translateY(-2px);
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    background: #f8fafc;
    border-radius: 8px;
    font-weight: 600;
    color: #1e293b;
    border: 1px solid #e2e8f0;
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

/* ===== ALERTS ===== */
.stAlert {
    border-radius: 10px;
    border: none;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="admin-hero">
    <div class="hero-content">
        <div class="hero-title">Administration</div>
        <div class="hero-subtitle">Entraînement & Gestion des Artefacts HuggingFace</div>
        <div class="hero-badge">HF Repository: khadidia-77/favorita</div>
    </div>
</div>
""", unsafe_allow_html=True)


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
    falsy  = {"false", "0", "f", "no", "n", "nan", "none", ""}
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
        columns=["date", "store_nbr", "item_nbr", "onpromotion", "unit_sales"],
    )
    df_["date"] = pd.to_datetime(df_["date"], errors="coerce").dt.normalize()
    df_ = df_.dropna(subset=["date"])
    return df_


# ============================================================
# Latest info
# ============================================================
try:
    latest = read_latest()
    st.success(f"Latest run: {latest.get('run_id')} (maj: {latest.get('updated_at')})")
except Exception as e:
    st.warning("Pas de latest.json trouvé pour l'instant.")
    st.caption(str(e))


# ============================================================
# Train + Publish
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><span class="section-icon"></span>Entraînement du Modèle</div>', unsafe_allow_html=True)

weeks_window = st.selectbox("Fenêtre d'entraînement (semaines)", [10, 8, 4, 3, 2], index=0)

if st.button("Retrain + Publish sur HF", use_container_width=True):
    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        hf_token = None

    with st.spinner("Entraînement + publication en cours..."):
        res = train_and_publish(
            weeks_window=int(weeks_window),
            hf_repo_id=HF_REPO_ID,
            hf_token=hf_token,
        )

    st.success("✅ Terminé ! Nouveau modèle publié.")
    st.json(res.get("published", {}))
    st.json(res.get("train_metrics", {}))

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Eval performances du modèle actuel
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><span class="section-icon"></span>Performances du Modèle Actuel</div>', unsafe_allow_html=True)

with st.expander("Configurer l'évaluation", expanded=True):
    eval_weeks = st.selectbox("Fenêtre de données pour l'évaluation (semaines)", [10, 8, 4, 3, 2], index=0, key="eval_weeks")
    eval_days = st.slider("Taille du jeu de validation (derniers jours)", min_value=7, max_value=28, value=14, step=1)
    max_rows = st.number_input("Cap lignes (échantillonnage si trop gros)", min_value=50_000, max_value=1_000_000, value=300_000, step=50_000)
    run_eval = st.button("Calculer les performances", use_container_width=True)

if run_eval:
    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        hf_token = None

    # Load artifacts latest
    try:
        model, pipe, feature_cols, meta = load_artifacts_latest(hf_token)
    except Exception as e:
        st.error("❌ Impossible de charger les artefacts latest depuis HF.")
        st.exception(e)
        st.stop()

    st.caption(
        f"✅ Artifacts: run={meta.get('run_id')} | trained_at={meta.get('trained_at', meta.get('updated_at'))}"
    )

    # Load data window
    with st.spinner("Chargement des données..."):
        df = load_data_weeks(int(eval_weeks))

    needed = {"date", "store_nbr", "item_nbr", "onpromotion", "unit_sales"}
    missing = needed - set(df.columns)
    if missing:
        st.error(f"❌ Colonnes manquantes dans la base pour évaluer: {sorted(list(missing))}")
        st.stop()

    df = df.copy()
    df["onpromotion"] = _to_bool_onpromotion(df["onpromotion"])
    df["unit_sales"] = pd.to_numeric(df["unit_sales"], errors="coerce")
    df = df.dropna(subset=["unit_sales"])
    df["unit_sales"] = df["unit_sales"].clip(lower=0)

    # Split: last eval_days as validation
    max_d = df["date"].max()
    cut_d = max_d - pd.Timedelta(days=int(eval_days) - 1)
    valid = df.loc[df["date"] >= cut_d].copy()

    if len(valid) == 0:
        st.warning("⚠️ Jeu de validation vide (check dates).")
        st.stop()

    # Sample if too big
    if len(valid) > int(max_rows):
        valid = valid.sample(int(max_rows), random_state=42)
        st.info(f"📌 Validation échantillonnée à {len(valid):,} lignes")

    st.caption(f"Validation: {valid['date'].min().date()} → {valid['date'].max().date()} | n={len(valid):,}")

    # Build X / y (log)
    y_true_log = np.log1p(valid["unit_sales"].values.astype("float64"))

    X_input = valid[["date", "store_nbr", "item_nbr", "onpromotion"]].copy()
    X_input["date"] = pd.to_datetime(X_input["date"]).dt.normalize()
    X_input["store_nbr"] = X_input["store_nbr"].astype(int)
    X_input["item_nbr"] = X_input["item_nbr"].astype(int)
    X_input["onpromotion"] = X_input["onpromotion"].astype(bool)

    with st.spinner("Transformation + prédiction..."):
        X_enriched = pipe.transform(X_input)
        X = (X_enriched
             .reindex(columns=feature_cols, fill_value=0)
             .replace([np.inf, -np.inf], np.nan)
             .fillna(0))

        y_pred_log = model.predict(X)

    # Metrics in log + in units
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae_log  = float(mean_absolute_error(y_true_log, y_pred_log))
    r2_log   = float(r2_score(y_true_log, y_pred_log))

    y_true = valid["unit_sales"].values.astype("float64")
    y_pred = _safe_expm1(y_pred_log)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">RMSE (unités)</div>
            <div class="metric-value">{rmse:,.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">MAE (unités)</div>
            <div class="metric-value">{mae:,.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">R² (unités)</div>
            <div class="metric-value">{r2:,.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">RMSE (log)</div>
            <div class="metric-value">{rmse_log:,.4f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Aperçu
    preview = valid[["date", "store_nbr", "item_nbr", "onpromotion"]].copy()
    preview["y_true_unit_sales"] = y_true.astype("float32")
    preview["y_pred_unit_sales"] = y_pred.astype("float32")
    preview["abs_err"] = np.abs(preview["y_true_unit_sales"] - preview["y_pred_unit_sales"]).astype("float32")

    with st.expander("Aperçu des erreurs (top 200)", expanded=False):
        st.dataframe(preview.sort_values("abs_err", ascending=False).head(200), use_container_width=True)

    st.success("✅ Évaluation terminée.")

st.markdown('</div>', unsafe_allow_html=True)
