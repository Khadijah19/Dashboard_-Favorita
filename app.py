# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np

from utils.viz import bar_top_families_sum
from utils.data_loader import load_train_from_hf, load_items_hf, load_stores_hf

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Favorita Forecast Dashboard",
    page_icon="📦",
    layout="wide",
)

PARQUET_NAME = "train_last10w.parquet"
WEEKS_WINDOW = 10  # fenêtre fixée (stabilité)

# ============================================================
# CSS PREMIUM (hero centré + animations)
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1600px; }

/* HERO */
.dashboard-hero{
    background: linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);
    border-radius: 26px; padding: 4.2rem 2.8rem; margin-bottom: 2rem;
    box-shadow: 0 25px 70px rgba(15,23,42,0.42);
    position: relative; overflow: hidden; text-align: center;
}
.dashboard-hero::before{
    content:''; position:absolute; top:-55%; right:-22%;
    width:760px; height:760px;
    background: radial-gradient(circle, rgba(59,130,246,0.22) 0%, transparent 60%);
    border-radius:50%; animation: float 9s ease-in-out infinite;
}
.dashboard-hero::after{
    content:''; position:absolute; bottom:-48%; left:-18%;
    width:560px; height:560px;
    background: radial-gradient(circle, rgba(139,92,246,0.16) 0%, transparent 70%);
    border-radius:50%; animation: float 11s ease-in-out infinite reverse;
}
@keyframes float{ 0%,100%{transform:translate(0,0);} 50%{transform:translate(34px,-34px);} }

.hero-shine{
    position:absolute; inset:0;
    background: linear-gradient(120deg, rgba(255,255,255,0.0) 0%, rgba(255,255,255,0.06) 22%,
                                rgba(255,255,255,0.0) 45%, rgba(255,255,255,0.0) 100%);
    transform: translateX(-60%);
    animation: shine 5.5s ease-in-out infinite;
    pointer-events:none;
}
@keyframes shine{
    0%{transform:translateX(-65%);opacity:0.0;}
    15%{opacity:0.9;}
    45%{opacity:0.0;}
    100%{transform:translateX(85%);opacity:0.0;}
}
.hero-content{ position:relative; z-index:1; }
.hero-kicker{
    display:inline-block; padding:0.45rem 0.9rem; border-radius:999px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    color: rgba(255,255,255,0.92);
    font-weight:700; letter-spacing:0.06em;
    text-transform:uppercase; font-size:0.72rem;
    margin-bottom:0.9rem; backdrop-filter: blur(10px);
}
.hero-title{
    color:white; font-size:3.05rem; font-weight:950; margin:0 0 0.9rem 0;
    letter-spacing:-0.04em; line-height:1.08;
    text-shadow: 0 10px 24px rgba(0,0,0,0.24);
    animation: fadeUp 0.9s ease-out both;
}
.hero-subtitle{
    color: rgba(255,255,255,0.92);
    font-size:1.18rem; margin:0 auto; font-weight:420; line-height:1.65;
    max-width: 920px;
    animation: fadeUp 1.05s ease-out both; animation-delay:0.08s;
}
.hero-actions{
    display:flex; gap:0.85rem; justify-content:center; flex-wrap:wrap;
    margin-top:1.8rem;
    animation: fadeUp 1.15s ease-out both; animation-delay:0.14s;
}
.hero-chip{
    display:inline-flex; align-items:center; gap:0.55rem;
    padding:0.7rem 0.95rem; border-radius:14px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    color: rgba(255,255,255,0.92);
    font-weight:650; font-size:0.9rem;
    backdrop-filter: blur(10px);
}
.dot{
    width:8px; height:8px; border-radius:50%;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    box-shadow: 0 0 0 4px rgba(59,130,246,0.18);
}
@keyframes fadeUp{ from{opacity:0;transform:translateY(10px);} to{opacity:1;transform:translateY(0);} }

/* KPI */
.kpi-container{
    display:grid; grid-template-columns: repeat(2, 1fr);
    gap:1.5rem; max-width:700px; margin: 0 auto 2rem auto;
}
.kpi-card{
    background:white; border-radius:18px; padding:2rem 1.5rem;
    box-shadow:0 8px 30px rgba(0,0,0,0.08);
    border-left:5px solid; transition: all 0.25s ease;
    position:relative; overflow:hidden;
}
.kpi-card::before{
    content:''; position:absolute; top:0; right:0; width:100px; height:100px;
    background: radial-gradient(circle, rgba(0,0,0,0.03) 0%, transparent 70%);
    border-radius:50%;
}
.kpi-card:nth-child(1){ border-left-color:#3b82f6; }
.kpi-card:nth-child(2){ border-left-color:#8b5cf6; }
.kpi-card:hover{ transform: translateY(-6px); box-shadow: 0 16px 45px rgba(0,0,0,0.14); }
.kpi-label{
    font-size:0.75rem; color:#64748b; font-weight:800;
    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;
}
.kpi-value{
    font-size:2.2rem; font-weight:900; color:#0f172a; line-height:1;
}

/* Sections */
.chart-section{
    background:white; border-radius:20px; padding:2.5rem;
    box-shadow:0 8px 35px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.04);
    margin-bottom:2rem; position:relative; overflow:hidden;
}
.chart-section::before{
    content:''; position:absolute; top:0; left:0; right:0; height:4px;
    background: linear-gradient(90deg,#3b82f6 0%,#8b5cf6 50%,#ec4899 100%);
}
.chart-header{
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:2rem; padding-bottom:1.5rem;
    border-bottom:2px solid #f1f5f9;
}
.chart-title{
    font-size:1.4rem; font-weight:900; color:#0f172a;
    display:flex; align-items:center; gap:0.8rem;
}
.chart-icon{
    width:10px; height:10px; border-radius:50%;
    background: linear-gradient(135deg,#3b82f6 0%,#8b5cf6 100%);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#fafaf9 0%,#ffffff 100%);
    border-right: 2px solid rgba(0,0,0,0.06);
}
section[data-testid="stSidebar"] .block-container{ padding-top:2rem; }
section[data-testid="stSidebar"] h2{
    background: linear-gradient(135deg,#3b82f6 0%,#8b5cf6 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
    font-size:1.5rem; font-weight:900;
    margin-bottom:2rem; padding-bottom:1rem;
    border-bottom:2px solid #f1f5f9;
}

/* Info box */
.stInfo{
    background: linear-gradient(135deg,#dbeafe 0%,#e0e7ff 100%);
    border-left:4px solid #3b82f6; border-radius:8px;
}

/* Footer */
.dashboard-footer{
    text-align:center; color:#64748b; padding:2rem 0; margin-top:3rem;
    border-top:2px solid #f1f5f9;
}
.footer-title{ font-size:0.95rem; margin:0; font-weight:750; color:#1e293b; }
.footer-text{ font-size:0.85rem; margin:0.6rem 0 0 0; opacity:0.9; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div class="dashboard-hero">
  <div class="hero-shine"></div>
  <div class="hero-content">
    <div class="hero-kicker">Sales forecasting • Analytics</div>
    <div class="hero-title">Bienvenue sur Favorita Forecast</div>
    <div class="hero-subtitle">
      Vous pouvez explorer les ventes récentes et surtout faire des prédictions sur vos ventes futures.
    </div>
    <div class="hero-actions">
      <div class="hero-chip"><span class="dot"></span> Données récentes (10 semaines)</div>
      <div class="hero-chip"><span class="dot"></span> Filtres store / item / période</div>
      <div class="hero-chip"><span class="dot"></span> Courbes multi-séries</div>
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
    st.caption("Fenêtre fixée à 10 semaines pour assurer la stabilité.")
    st.divider()
    st.markdown("## Filtres")
    st.caption("La courbe s'affiche uniquement si au moins un store et un item sont sélectionnés.")

# ============================================================
# LOAD DATA (HF ONLY)
# ============================================================
@st.cache_data(show_spinner=True)
def load_all_fixed():
    train = load_train_from_hf(
        weeks=int(WEEKS_WINDOW),
        filename=PARQUET_NAME,
        columns=["date", "store_nbr", "item_nbr", "onpromotion", "unit_sales"],
    )
    items = load_items_hf("items.csv")
    stores = load_stores_hf("stores.csv")
    return train, items, stores


train, items, stores = load_all_fixed()

train["date"] = pd.to_datetime(train["date"], errors="coerce").dt.normalize()
train = train.dropna(subset=["date"])

min_d, max_d = train["date"].min(), train["date"].max()

store_list = np.sort(train["store_nbr"].dropna().unique()).tolist()
item_list = np.sort(train["item_nbr"].dropna().unique()).tolist()

# ============================================================
# SIDEBAR FILTERS (avec contrôle "période" obligatoire)
# ============================================================
with st.sidebar:
    # IMPORTANT: Streamlit peut renvoyer un seul jour si l'utilisateur clique "date unique"
    date_range = st.date_input(
        "Période",
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date(),
        help="Sélectionnez une date de début et une date de fin.",
        key="date_range",
    )

    # ✅ Validation : si ce n'est pas une paire (start, end), on affiche un message clair
    if not (isinstance(date_range, (tuple, list)) and len(date_range) == 2):
        st.info("Veuillez sélectionner une période (date de début et date de fin).")
        st.stop()

    store_sel = st.multiselect("Stores", store_list, default=[])

    q_item = st.text_input("Recherche item (id)", value="")
    if q_item.strip():
        item_opts = [x for x in item_list if q_item.strip() in str(x)][:5000]
    else:
        item_opts = item_list[:5000]

    item_sel = st.multiselect("Items", options=item_opts, default=[])

    chart_mode = st.selectbox(
        "Mode du graphique",
        ["Par couple (store, item)", "Par item", "Par store"],
        index=0,
    )

# ============================================================
# APPLY FILTERS
# ============================================================
start_d = pd.to_datetime(date_range[0])
end_d = pd.to_datetime(date_range[1])
if start_d > end_d:
    start_d, end_d = end_d, start_d

df = train.loc[(train["date"] >= start_d) & (train["date"] <= end_d)].copy()

if store_sel:
    df = df.loc[df["store_nbr"].isin(store_sel)]
if item_sel:
    df = df.loc[df["item_nbr"].isin(item_sel)]

df["unit_sales_pos"] = pd.to_numeric(df["unit_sales"], errors="coerce").fillna(0).clip(lower=0)

# ============================================================
# TOP FAMILIES DATASET (indépendant item filter)
# ============================================================
df_base = train.loc[(train["date"] >= start_d) & (train["date"] <= end_d)].copy()
if store_sel:
    df_base = df_base.loc[df_base["store_nbr"].isin(store_sel)]
df_base["unit_sales_pos"] = pd.to_numeric(df_base["unit_sales"], errors="coerce").fillna(0).clip(lower=0)

items_min = items[["item_nbr", "family"]].copy()
items_min["item_nbr"] = pd.to_numeric(items_min["item_nbr"], errors="coerce")
items_min["family"] = items_min["family"].fillna("UNKNOWN").astype(str)

# ============================================================
# KPIs
# ============================================================
n_stores = int(df["store_nbr"].nunique()) if len(df) else 0
n_items = int(df["item_nbr"].nunique()) if len(df) else 0

st.markdown(
    f"""
<div class="kpi-container">
  <div class="kpi-card">
    <div class="kpi-label">Stores actifs</div>
    <div class="kpi-value">{n_stores:,}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Items actifs</div>
    <div class="kpi-value">{n_items:,}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# CHARTS
# ============================================================
left, right = st.columns([2.1, 1])

with left:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chart-header">
          <div class="chart-title">
            <span class="chart-icon"></span>
            Ventes journalières (somme des ventes positives)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if (not store_sel) or (not item_sel):
        st.info("Sélectionnez au moins un store et un item pour afficher la courbe.")
    else:
        dplot = df[["date", "store_nbr", "item_nbr", "unit_sales_pos"]].copy()

        if chart_mode == "Par couple (store, item)":
            dplot["series"] = dplot["store_nbr"].astype(str) + " · " + dplot["item_nbr"].astype(str)
        elif chart_mode == "Par item":
            dplot["series"] = dplot["item_nbr"].astype(str)
        else:
            dplot["series"] = dplot["store_nbr"].astype(str)

        g = (
            dplot.groupby(["date", "series"], as_index=False)["unit_sales_pos"]
            .sum()
            .rename(columns={"unit_sales_pos": "value"})
        )

        pivot = g.pivot(index="date", columns="series", values="value").sort_index()
        st.line_chart(pivot)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chart-header">
          <div class="chart-title">
            <span class="chart-icon"></span>
            Top familles (somme des ventes positives)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_fam = df_base[["item_nbr", "unit_sales_pos"]].merge(items_min, on="item_nbr", how="left", copy=False)
    fig2 = bar_top_families_sum(df_fam, y_col="unit_sales_pos", top=10)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Aperçu des données filtrées"):
        st.dataframe(df.head(50), use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    """
<div class="dashboard-footer">
  <p class="footer-title">Favorita Forecast Dashboard</p>
  <p class="footer-text">
    Un tableau de bord pensé pour aller vite : filtrer, comparer, comprendre, puis passer à la prédiction.
    Utilisez la barre latérale pour ajuster la période et analyser les couples store–item.
  </p>
</div>
""",
    unsafe_allow_html=True,
)
