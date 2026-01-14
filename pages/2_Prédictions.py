# pages/2_Predictions.py
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np

# Fix import utils sur Streamlit Cloud
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_train_from_hf
from utils.hf_artifacts import download_artifacts_from_latest

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Prédictions - Favorita",
    page_icon="📦",
    layout="wide",
)

HF_REPO_ID = os.getenv("HF_REPO_ID", "khadidia-77/favorita")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

PARQUET_NAME = "train_last10w.parquet"
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

/* ===== KPI CARDS ===== */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    max-width: 1100px;
    margin: 0 auto 2rem auto;
}

.kpi-card {
    background: white;
    border-radius: 18px;
    padding: 2rem 1.5rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    border-left: 5px solid;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 100px;
    height: 100px;
    background: radial-gradient(circle, rgba(0, 0, 0, 0.03) 0%, transparent 70%);
    border-radius: 50%;
}

.kpi-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 16px 45px rgba(0, 0, 0, 0.14);
}

.kpi-label {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
}

.kpi-value {
    font-size: 2.2rem;
    font-weight: 900;
    color: #0f172a;
    line-height: 1;
    position: relative;
    z-index: 1;
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

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    padding: 0.8rem 1.5rem;
    border: 1px solid #e2e8f0;
    border-bottom: none;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    color: white !important;
    border: none;
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
# Load HF artifacts
# ============================================================
@st.cache_resource(show_spinner=True)
def load_artifacts_hf():
    return download_artifacts_from_latest(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        hf_token=HF_TOKEN,
        artifacts_dir="artifacts",
        cache_dir=".cache/favorita_artifacts",
    )

try:
    model, pipe, feature_cols, meta = load_artifacts_hf()
except Exception as e:
    st.error("Impossible de charger les artefacts depuis HuggingFace.")
    st.exception(e)
    st.stop()

# ============================================================
# Load recent data
# ============================================================
@st.cache_data(show_spinner=True)
def load_recent_data(weeks: int):
    # NOTE: si ton loader ne supporte pas columns=, supprime ce paramètre
    df_ = load_train_from_hf(
        weeks=int(weeks),
        filename=PARQUET_NAME,
        columns=["date", "store_nbr", "item_nbr", "onpromotion"],
    )
    df_["date"] = pd.to_datetime(df_["date"], errors="coerce").dt.normalize()
    df_ = df_.dropna(subset=["date"])
    return df_

WEEKS = int(st.session_state.get("weeks_window", MAX_WEEKS))
WEEKS = min(WEEKS, MAX_WEEKS)

df = load_recent_data(WEEKS)

store_list = np.sort(df["store_nbr"].dropna().unique()).tolist()
item_list = np.sort(df["item_nbr"].dropna().unique()).tolist()

min_d = df["date"].min().date()
max_d = df["date"].max().date()

# ============================================================
# HEADER (pro)
# ============================================================
st.markdown(
    f"""
<div class="dashboard-hero">
  <div class="hero-content">
    <div class="hero-title">Prévisions de ventes</div>
    <div class="hero-subtitle">
      Prédictions générées à partir des artefacts du modèle publié sur HuggingFace.<br>
      Fenêtre de données: <b>{WEEKS} semaines</b> · Période: {min_d} → {max_d} · Source: {PARQUET_NAME}
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption(
    f"Model loaded | run={meta.get('run_id')} | trained_at={meta.get('trained_at', meta.get('updated_at'))}"
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## Paramètres")

    with st.expander("Prévision ponctuelle", expanded=True):
        future_max = max_d + timedelta(days=60)
        date_in = st.date_input("Date", value=max_d, min_value=min_d, max_value=future_max)
        store_nbr = st.selectbox("Store", options=store_list, index=0)

        q = st.text_input("Recherche item", value="", placeholder="Saisir un identifiant d'item...")
        if q.strip():
            item_opts = [x for x in item_list if q.strip() in str(x)][:5000]
        else:
            item_opts = item_list[:5000]

        item_nbr = st.selectbox("Item", options=item_opts, index=0)
        onpromotion = st.checkbox("Promotion", value=False)

    with st.expander("Prévision sur période", expanded=False):
        date_range = st.date_input(
            "Période",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
            key="pred_date_range",
        )
        store_sel = st.multiselect("Stores", options=store_list, default=[])
        item_sel = st.multiselect("Items", options=item_opts, default=[])
        group_mode = st.selectbox(
            "Agrégation du graphique",
            ["Par couple (store, item)", "Par item", "Par store"],
            index=0,
        )
        top_n = st.slider("Nombre de séries maximum (Top N)", 1, 30, 10, 1)
        run_period = st.button("Lancer la prédiction", width="stretch")

# ============================================================
# Tabs
# ============================================================
tab1, tab2 = st.tabs(["Prévision ponctuelle", "Prévision sur période"])

# ============================================================
# TAB 1 — Single prediction
# ============================================================
with tab1:
    new_df = pd.DataFrame(
        {
            "date": [pd.to_datetime(date_in)],
            "store_nbr": [int(store_nbr)],
            "item_nbr": [int(item_nbr)],
            "onpromotion": [bool(onpromotion)],
        }
    )

    X_enriched = pipe.transform(new_df)
    X = (
        X_enriched.reindex(columns=feature_cols, fill_value=0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    pred_log = float(model.predict(X)[0])
    pred_sales = float(np.expm1(pred_log))

    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chart-header">
          <div class="chart-title">
            <span class="chart-icon"></span>
            Résultat de la prévision
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="kpi-container" style="grid-template-columns: repeat(3, 1fr);">
  <div class="kpi-card" style="border-left-color:#3b82f6;">
    <div class="kpi-label">Prévision (unités)</div>
    <div class="kpi-value">{pred_sales:.2f}</div>
  </div>
  <div class="kpi-card" style="border-left-color:#8b5cf6;">
    <div class="kpi-label">Valeur log1p</div>
    <div class="kpi-value">{pred_log:.4f}</div>
  </div>
  <div class="kpi-card" style="border-left-color:#ec4899;">
    <div class="kpi-label">Indicateur promotion</div>
    <div class="kpi-value">{'Oui' if onpromotion else 'Non'}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Date: {pd.to_datetime(date_in).strftime('%d/%m/%Y')}")
        st.write(f"Store: {store_nbr}")
    with c2:
        st.write(f"Item: {item_nbr}")
        st.write("Source des features: pipeline + features_columns du dernier run")

    with st.expander("Observation envoyée au pipeline", expanded=False):
        st.dataframe(new_df, width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2 — Period predictions + chart by couple store-item
# ============================================================
with tab2:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chart-header">
          <div class="chart-title">
            <span class="chart-icon"></span>
            Prévision sur période
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("Sélectionnez une période, filtrez si besoin, puis lancez la prédiction. Le graphique affiche les séries les plus importantes sur la période (Top N).")

    if run_period:
        # ---------- Dates ----------
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_d = pd.to_datetime(date_range[0])
            end_d = pd.to_datetime(date_range[1])
        else:
            start_d = pd.to_datetime(date_range)
            end_d = pd.to_datetime(date_range)

        if start_d > end_d:
            start_d, end_d = end_d, start_d

        # ---------- Filtrage ----------
        f = df.loc[(df["date"] >= start_d) & (df["date"] <= end_d)].copy()

        if store_sel:
            f = f.loc[f["store_nbr"].isin(store_sel)]
        if item_sel:
            f = f.loc[f["item_nbr"].isin(item_sel)]

        if len(f) == 0:
            st.warning("Aucune donnée disponible après application des filtres.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        # ---------- Cap volume ----------
        nmax = 300_000
        if len(f) > nmax:
            f = f.sample(nmax, random_state=42)
            st.info(f"Échantillonnage appliqué : {nmax:,} lignes.")

        # ---------- Prédiction ----------
        with st.spinner("Calcul des prédictions..."):
            # IMPORTANT : si ton pipeline a besoin exactement des 4 colonnes,
            # assure-toi que f contient bien "date, store_nbr, item_nbr, onpromotion"
            Xf_enriched = pipe.transform(f)

            Xf = (
                Xf_enriched.reindex(columns=feature_cols, fill_value=0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

            pred_log_arr = model.predict(Xf)
            pred = np.expm1(pred_log_arr)

        out = f[["date", "store_nbr", "item_nbr"]].copy()
        out["pred_unit_sales"] = pd.to_numeric(pred, errors="coerce").astype("float32")
        out = out.dropna(subset=["pred_unit_sales"])

        # ---------- Choix d'agrégation ----------
        if group_mode == "Par couple (store, item)":
            out["series"] = (
                out["store_nbr"].astype(int).astype(str)
                + " · "
                + out["item_nbr"].astype(int).astype(str)
            )
        elif group_mode == "Par item":
            out["series"] = out["item_nbr"].astype(int).astype(str)
        else:
            out["series"] = out["store_nbr"].astype(int).astype(str)

        # Agrégation journalière par série
        g = (
            out.groupby(["date", "series"], as_index=False)["pred_unit_sales"]
            .sum()
            .sort_values(["date", "series"])
        )

        # ---------- Top N séries ----------
        # (les plus importantes sur la période, par somme totale)
        rank = (
            g.groupby("series", as_index=False)["pred_unit_sales"]
            .sum()
            .sort_values("pred_unit_sales", ascending=False)
        )
        top_series = rank["series"].head(int(top_n)).tolist()
        g = g.loc[g["series"].isin(top_series)]

        if len(g) == 0:
            st.warning("Aucune série à afficher (Top N vide). Essayez d'augmenter Top N ou de réduire les filtres.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        # ---------- Plot pro (Plotly) ----------
        import plotly.express as px

        fig = px.line(
            g,
            x="date",
            y="pred_unit_sales",
            color="series",
            markers=False,
            labels={
                "date": "Date",
                "pred_unit_sales": "Ventes prédites (unités)",
                "series": "Série",
            },
        )
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            legend_title_text="",
        )

        st.plotly_chart(fig, width="stretch")

        # ---------- Table ----------
        with st.expander("Table de prédictions (aperçu)", expanded=False):
            st.dataframe(out.sort_values(["date", "store_nbr", "item_nbr"]).head(500), width="stretch")

        # ---------- Export ----------
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger le fichier CSV",
            data=csv,
            file_name="predictions_favorita.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("Définissez la période et les filtres dans la barre latérale, puis lancez la prédiction.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
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
