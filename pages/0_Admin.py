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
st.title("🛠️ Admin — Monitoring & Évaluation (HF-only)")

HF_REPO_ID = os.getenv("HF_REPO_ID", "khadidia-77/favorita")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")
PARQUET_NAME = os.getenv("PARQUET_NAME", "train_last10w.parquet")

# 👉 Mets ça dans tes variables d’environnement Streamlit Cloud (Settings > Secrets / Advanced)
# Exemple:
# GITHUB_ACTIONS_URL="https://github.com/<user>/<repo>/actions/workflows/train.yml"
GITHUB_ACTIONS_URL = os.getenv("GITHUB_ACTIONS_URL", "").strip()

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
    df_ = load_train_from_hf(weeks=int(weeks), filename=PARQUET_NAME)
    df_["date"] = pd.to_datetime(df_["date"], errors="coerce").dt.normalize()
    df_ = df_.dropna(subset=["date"])
    return df_

# ============================================================
# SECTION: Trigger training (GitHub Actions) — SAFE
# ============================================================
st.subheader("🚀 Entraînement automatique (recommandé)")

st.info(
    "✅ Ici on **n’entraîne pas** dans Streamlit (ça fait crasher l’app). "
    "Le retrain doit se faire via **GitHub Actions**, puis les artefacts sont publiés sur HuggingFace."
)

cA, cB = st.columns([0.55, 0.45])
with cA:
    if GITHUB_ACTIONS_URL:
        # Streamlit récent : link_button existe, sinon markdown suffit
        try:
            st.link_button("🧠 Ouvrir GitHub Actions (Run workflow)", GITHUB_ACTIONS_URL, use_container_width=True)
        except Exception:
            st.markdown(f"👉 **GitHub Actions**: {GITHUB_ACTIONS_URL}")
    else:
        st.warning(
            "⚠️ Variable `GITHUB_ACTIONS_URL` non définie. "
            "Ajoute-la dans Streamlit Cloud pour avoir le bouton direct."
        )
with cB:
    st.caption(
        "Astuce: tu peux configurer un **schedule** (ex: chaque nuit) + un entraînement manuel (workflow_dispatch)."
    )

st.divider()

# ============================================================
# Latest info (HF latest.json)
# ============================================================
st.subheader("🧾 Dernier run publié (HuggingFace)")

try:
    latest = read_latest(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, hf_token=None)
    st.success(f"Latest run: {latest.get('run_id')} — maj: {latest.get('updated_at') or latest.get('trained_at')}")
    with st.expander("Voir latest.json"):
        st.json(latest)
except Exception as e:
    st.warning("Pas de latest.json trouvé pour l’instant.")
    st.caption(str(e))

# ============================================================
# ✅ Eval performances du modèle actuel (latest)
# ============================================================
st.divider()
st.subheader("📊 Performances du modèle actuel (latest)")

with st.expander("Configurer l'évaluation", expanded=True):
    eval_weeks = st.selectbox("Fenêtre de données pour l'évaluation (semaines)", [10, 8, 6, 4, 3, 2], index=0)
    eval_days = st.slider("Taille du jeu de validation (derniers jours)", min_value=7, max_value=28, value=14, step=1)
    max_rows = st.number_input(
        "Cap lignes (échantillonnage si trop gros)",
        min_value=50_000, max_value=1_000_000, value=300_000, step=50_000
    )
    run_eval = st.button("📈 Calculer les performances", use_container_width=True)

if run_eval:
    # Token HF (si repo privé)
    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        hf_token = os.getenv("HF_TOKEN", None)

    # 1) Load artifacts latest
    try:
        model, pipe, feature_cols, meta = load_artifacts_latest(hf_token)
    except Exception as e:
        st.error("❌ Impossible de charger les artefacts latest depuis HF.")
        st.exception(e)
        st.stop()

    st.caption(f"✅ Artifacts: run={meta.get('run_id')} | trained_at={meta.get('trained_at', meta.get('updated_at'))}")

    # 2) Load data window
    with st.spinner("📥 Chargement des données..."):
        df = load_data_weeks(int(eval_weeks))

    needed = {"date", "store_nbr", "item_nbr", "onpromotion", "unit_sales"}
    missing = needed - set(df.columns)
    if missing:
        st.error(f"❌ Colonnes manquantes pour évaluer: {sorted(list(missing))}")
        st.stop()

    df = df.copy()
    df["onpromotion"] = _to_bool_onpromotion(df["onpromotion"])
    df["unit_sales"] = pd.to_numeric(df["unit_sales"], errors="coerce")
    df = df.dropna(subset=["unit_sales"])
    df["unit_sales"] = df["unit_sales"].clip(lower=0)

    # 3) Split: last eval_days as validation
    max_d = df["date"].max()
    cut_d = max_d - pd.Timedelta(days=int(eval_days) - 1)
    valid = df.loc[df["date"] >= cut_d].copy()

    if len(valid) == 0:
        st.warning("⚠️ Jeu de validation vide (check dates).")
        st.stop()

    # 4) Sample if too big
    if len(valid) > int(max_rows):
        valid = valid.sample(int(max_rows), random_state=42)
        st.info(f"📌 Validation échantillonnée à {len(valid):,} lignes")

    st.caption(f"🗓️ Validation: {valid['date'].min().date()} → {valid['date'].max().date()} | n={len(valid):,}")

    # 5) Build X / y (log)
    y_true_log = np.log1p(valid["unit_sales"].values.astype("float64"))

    X_input = valid[["date", "store_nbr", "item_nbr", "onpromotion"]].copy()
    X_input["date"] = pd.to_datetime(X_input["date"]).dt.normalize()
    X_input["store_nbr"] = X_input["store_nbr"].astype(int)
    X_input["item_nbr"] = X_input["item_nbr"].astype(int)
    X_input["onpromotion"] = X_input["onpromotion"].astype(bool)

    with st.spinner("🧠 Transformation + prédiction..."):
        X_enriched = pipe.transform(X_input)
        X = (
            X_enriched.reindex(columns=feature_cols, fill_value=0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        y_pred_log = model.predict(X)

    # 6) Metrics in log + in units
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae_log  = float(mean_absolute_error(y_true_log, y_pred_log))
    r2_log   = float(r2_score(y_true_log, y_pred_log))

    y_true = valid["unit_sales"].values.astype("float64")
    y_pred = _safe_expm1(y_pred_log)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (unités)", f"{rmse:,.3f}")
    c2.metric("MAE (unités)", f"{mae:,.3f}")
    c3.metric("R² (unités)", f"{r2:,.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("RMSE (log1p)", f"{rmse_log:,.4f}")
    c5.metric("MAE (log1p)", f"{mae_log:,.4f}")
    c6.metric("R² (log1p)", f"{r2_log:,.4f}")

    preview = valid[["date", "store_nbr", "item_nbr", "onpromotion"]].copy()
    preview["y_true_unit_sales"] = y_true.astype("float32")
    preview["y_pred_unit_sales"] = y_pred.astype("float32")
    preview["abs_err"] = np.abs(preview["y_true_unit_sales"] - preview["y_pred_unit_sales"]).astype("float32")

    with st.expander("🔎 Aperçu des erreurs (top 200)", expanded=False):
        st.dataframe(preview.sort_values("abs_err", ascending=False).head(200), use_container_width=True)

    st.success("✅ Évaluation terminée.")
