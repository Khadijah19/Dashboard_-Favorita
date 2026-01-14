# pages/2_Prédictions.py
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# ✅ Fix import utils sur Streamlit Cloud
ROOT = Path(__file__).resolve().parents[1]  # repo root (où se trouve utils/)
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_train_from_hf
from utils.hf_artifacts import download_artifacts_from_latest

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Prédictions - Favorita",
    page_icon="🔮",
    layout="wide",
)

# --- HF settings (dataset) ---
HF_REPO_ID = os.getenv("HF_REPO_ID", "khadidia-77/favorita")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))  # peut être None si repo public

PARQUET_NAME = "train_last10w.parquet"
MAX_WEEKS = 10

# ============================================================
# CSS (✅ adopté depuis app.py)
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
    grid-template-columns: repeat(2, 1fr);
    gap: 1.5rem;
    max-width: 900px;
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

.kpi-card:nth-child(1) { border-left-color: #3b82f6; }
.kpi-card:nth-child(2) { border-left-color: #8b5cf6; }

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
# LOAD HF ARTIFACTS (model + pipeline + features)
# ============================================================
@st.cache_resource(show_spinner=True)
def load_artifacts_hf():
    return download_artifacts_from_latest(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        hf_token=HF_TOKEN,  # ok même si None (repo public)
        artifacts_dir="artifacts",
        cache_dir=".cache/favorita_artifacts",
    )

try:
    model, pipe, feature_cols, meta = load_artifacts_hf()
    st.caption(
        f"✅ Model HF chargé | run={meta.get('run_id')} | trained_at={meta.get('trained_at', meta.get('updated_at'))}"
    )
except Exception as e:
    st.error("❌ Impossible de charger les artefacts depuis HuggingFace.")
    st.exception(e)
    st.stop()

# ============================================================
# LOAD DATA (HF ONLY)
# ============================================================
@st.cache_data(show_spinner=True)
def load_recent_data(weeks: int):
    df_ = load_train_from_hf(
        weeks=int(weeks),  # ✅ fix cache/param
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
# HEADER (✅ style app.py)
# ============================================================
st.markdown(
    f"""
<div class="dashboard-hero">
  <div class="hero-content">
    <div class="hero-title">🔮 Prédictions IA</div>
    <div class="hero-subtitle">
      Moteur de prévision des ventes (HF artifacts)<br>
      Fenêtre: <b>{WEEKS} semaines</b> · 📅 {min_d} → {max_d} · 📦 {PARQUET_NAME}
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## Configuration")

    with st.expander("⚡ Prédiction Instantanée", expanded=True):
        from datetime import timedelta

        future_max = max_d + timedelta(days=60)  # ou 365
        date_in = st.date_input("📅 Date", value=max_d, min_value=min_d, max_value=future_max)
        store_nbr = st.selectbox("🏪 Store", options=store_list, index=0)

        q = st.text_input("🔍 Rechercher un item", value="", placeholder="ID de l'item...")
        if q.strip():
            item_opts = [x for x in item_list if q.strip() in str(x)][:5000]
        else:
            item_opts = item_list[:5000]

        item_nbr = st.selectbox("📦 Item", options=item_opts, index=0)
        onpromotion = st.checkbox("🏷️ En promotion", value=False)

    with st.expander("📊 Prédiction sur Période", expanded=False):
        date_range = st.date_input(
            "Période",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
            key="pred_date_range",
        )
        store_sel = st.multiselect("Stores", options=store_list, default=[])
        item_sel = st.multiselect("Items", options=item_opts, default=[])
        run_period = st.button("🚀 Lancer Prédiction", width="stretch")

# ============================================================
# TABS
# ============================================================
tab1, tab2 = st.tabs(["⚡ Instantané", "📈 Période"])

# ============================================================
# TAB 1 — SINGLE PRED
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

    # ✅ Style "chart-section" (comme app.py)
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chart-header">
          <div class="chart-title">
            <span class="chart-icon"></span>
            Prévision instantanée
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI cards version prediction (réutilise kpi-card)
    st.markdown(
        f"""
<div class="kpi-container" style="max-width: 1100px;">
  <div class="kpi-card">
    <div class="kpi-label">Prévision estimée</div>
    <div class="kpi-value">{pred_sales:.2f}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Log(pred)</div>
    <div class="kpi-value">{pred_log:.4f}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # détails
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"📅 **Date** : {pd.to_datetime(date_in).strftime('%d/%m/%Y')}")
        st.write(f"🏪 **Store** : {store_nbr}")
    with c2:
        st.write(f"📦 **Item** : {item_nbr}")
        st.write(f"🏷️ **Promotion** : {'✅ Oui' if onpromotion else '❌ Non'}")

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔍 Détails de l'observation"):
        st.dataframe(new_df, width="stretch")

# ============================================================
# TAB 2 — PERIOD PRED
# ============================================================
with tab2:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chart-header">
          <div class="chart-title">
            <span class="chart-icon"></span>
            Prédictions sur période avec filtres
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if run_period:
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_d = pd.to_datetime(date_range[0])
            end_d = pd.to_datetime(date_range[1])
        else:
            start_d = pd.to_datetime(date_range)
            end_d = pd.to_datetime(date_range)

        if start_d > end_d:
            start_d, end_d = end_d, start_d

        f = df.loc[(df["date"] >= start_d) & (df["date"] <= end_d)].copy()

        if store_sel:
            f = f.loc[f["store_nbr"].isin(store_sel)]
        if item_sel:
            f = f.loc[f["item_nbr"].isin(item_sel)]

        if len(f) == 0:
            st.warning("⚠️ Aucune ligne après filtres.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        nmax = 300_000
        if len(f) > nmax:
            f = f.sample(nmax, random_state=42)
            st.info(f"📊 Dataset échantillonné : {nmax:,} lignes")

        with st.spinner("⚙️ Prédiction en cours..."):
            Xf_enriched = pipe.transform(f)
            Xf = (
                Xf_enriched.reindex(columns=feature_cols, fill_value=0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

            pred_log_arr = model.predict(Xf)
            pred = np.expm1(pred_log_arr)

        out = f[["date", "store_nbr", "item_nbr"]].copy()
        out["pred_unit_sales"] = pred.astype("float32")

        total = float(out["pred_unit_sales"].sum())
        avg = float(out["pred_unit_sales"].mean())
        nrows = int(len(out))

        # KPIs (style kpi-card)
        st.markdown(
            f"""
<div class="kpi-container" style="grid-template-columns: repeat(3, 1fr); max-width: 1100px;">
  <div class="kpi-card" style="border-left-color:#3b82f6;">
    <div class="kpi-label">Total prédit</div>
    <div class="kpi-value">{total:,.0f}</div>
  </div>
  <div class="kpi-card" style="border-left-color:#8b5cf6;">
    <div class="kpi-label">Moyenne / ligne</div>
    <div class="kpi-value">{avg:.2f}</div>
  </div>
  <div class="kpi-card" style="border-left-color:#ec4899;">
    <div class="kpi-label">Lignes</div>
    <div class="kpi-value">{nrows:,}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="chart-header" style="margin-top:1.5rem;">
  <div class="chart-title">
    <span class="chart-icon"></span>
    Évolution temporelle (total prédit)
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        g1 = out.groupby("date", as_index=False)["pred_unit_sales"].sum()
        st.line_chart(g1.set_index("date"))

        with st.expander("📄 Table de prédictions (aperçu)"):
            st.dataframe(out.head(200), width="stretch")

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger les prédictions (CSV)",
            data=csv,
            file_name="predictions_favorita.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("ℹ️ Configure la période + filtres dans la sidebar, puis clique sur **Lancer Prédiction**.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    """
<div class="dashboard-footer">
  <p class="footer-title">Favorita Forecast Dashboard</p>
  <p class="footer-text">© 2026 · Propulsé par Streamlit · Made with ❤️</p>
</div>
""",
    unsafe_allow_html=True,
)
