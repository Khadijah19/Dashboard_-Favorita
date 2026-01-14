# utils/data_loader.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from pathlib import Path
import pandas as pd

from huggingface_hub import hf_hub_download

# ============================================================
# HF config
# ============================================================
HF_DATASET_REPO = "khadidia-77/favorita"
HF_REPO_TYPE = "dataset"
DEFAULT_CACHE_DIR = ".cache/favorita_data"

# ============================================================
# Small retry helper (réseau HF parfois instable)
# ============================================================
def _retry(fn, tries: int = 3, base_sleep: float = 1.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2 ** i))
    raise last


def _hf_download(
    filename: str,
    repo_id: str = HF_DATASET_REPO,
    repo_type: str = HF_REPO_TYPE,
    cache_dir: str = DEFAULT_CACHE_DIR,
    hf_token: str | None = None,
) -> str:
    def _dl():
        return hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            cache_dir=cache_dir,
            token=hf_token,
        )

    return _retry(_dl, tries=3, base_sleep=1.0)


# ============================================================
# TRAIN loader (source of truth)
# IMPORTANT: on suppose que le parquet est DÉJÀ "last10w"
# ============================================================
def load_train_from_hf(
    repo_id: str = HF_DATASET_REPO,
    hf_token: str | None = None,
    filename: str = "train_last10w.parquet",
    cache_dir: str = DEFAULT_CACHE_DIR,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Charge un parquet depuis HuggingFace (cache local).
    Version stable: on ne refiltre pas en weeks ici (car le fichier est déjà last10w).
    """

    local = _hf_download(
        filename=filename,
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        cache_dir=cache_dir,
        hf_token=hf_token,
    )

    # Lecture parquet (colonnes si besoin)
    df = pd.read_parquet(local, columns=columns)

    # Sécuriser présence de 'date'
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"])
    else:
        # si l'appelant a oublié "date" dans columns=
        raise ValueError("La colonne 'date' doit être chargée (ajoute-la dans columns=...).")

    # Dtypes compacts
    if "store_nbr" in df.columns:
        df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").fillna(0).astype("int16")

    if "item_nbr" in df.columns:
        df["item_nbr"] = pd.to_numeric(df["item_nbr"], errors="coerce").fillna(0).astype("int32")

    if "unit_sales" in df.columns:
        df["unit_sales"] = pd.to_numeric(df["unit_sales"], errors="coerce").fillna(0).astype("float32")

    if "onpromotion" in df.columns:
        # robuste si jamais ça arrive en string/object
        s = df["onpromotion"]
        if s.dtype == "O":
            s = s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
        else:
            s = s.fillna(False).astype(bool)
        df["onpromotion"] = s

    return df


# ============================================================
# ITEMS / STORES loaders (on passe aussi par hf_hub_download)
# -> plus stable que pd.read_csv(url http)
# ============================================================
def load_items_hf(
    filename: str = "items.csv",
    repo_id: str = HF_DATASET_REPO,
    hf_token: str | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    local = _hf_download(filename, repo_id=repo_id, repo_type=HF_REPO_TYPE, cache_dir=cache_dir, hf_token=hf_token)
    df = pd.read_csv(local)

    df["item_nbr"] = pd.to_numeric(df["item_nbr"], errors="coerce").fillna(0).astype("int32")
    df["family"] = df["family"].fillna("UNKNOWN").astype(str).str.strip()
    if "class" in df.columns:
        df["class"] = pd.to_numeric(df["class"], errors="coerce").fillna(-1).astype("int16")
    if "perishable" in df.columns:
        df["perishable"] = pd.to_numeric(df["perishable"], errors="coerce").fillna(0).astype("int8")
    return df


def load_stores_hf(
    filename: str = "stores.csv",
    repo_id: str = HF_DATASET_REPO,
    hf_token: str | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    local = _hf_download(filename, repo_id=repo_id, repo_type=HF_REPO_TYPE, cache_dir=cache_dir, hf_token=hf_token)
    df = pd.read_csv(local)

    df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").fillna(0).astype("int16")
    for c in ["city", "state", "type"]:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str).str.strip()
    if "cluster" in df.columns:
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype("int16")
    return df


# ============================================================
# Autres CSV (même principe)
# ============================================================
def load_oil_hf(
    filename: str = "oil.csv",
    repo_id: str = HF_DATASET_REPO,
    hf_token: str | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    local = _hf_download(filename, repo_id=repo_id, repo_type=HF_REPO_TYPE, cache_dir=cache_dir, hf_token=hf_token)
    df = pd.read_csv(local, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["dcoilwtico"] = pd.to_numeric(df["dcoilwtico"], errors="coerce")
    return df


def load_transactions_hf(
    filename: str = "transactions.csv",
    repo_id: str = HF_DATASET_REPO,
    hf_token: str | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    local = _hf_download(filename, repo_id=repo_id, repo_type=HF_REPO_TYPE, cache_dir=cache_dir, hf_token=hf_token)
    df = pd.read_csv(local, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").fillna(0).astype("int16")
    df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0).astype("float32")
    return df


def load_holidays_hf(
    filename: str = "holidays_events.csv",
    repo_id: str = HF_DATASET_REPO,
    hf_token: str | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    local = _hf_download(filename, repo_id=repo_id, repo_type=HF_REPO_TYPE, cache_dir=cache_dir, hf_token=hf_token)
    df = pd.read_csv(local, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for c in ["type", "locale", "locale_name", "description"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()
    if "transferred" in df.columns:
        df["transferred"] = df["transferred"].fillna(False).astype(bool)
    return df


# ============================================================
# Compat layer (tes pages existantes)
# ============================================================
def load_train_recent(data_dir=None, weeks: int = 10, parquet_name: str = "train_last10w.parquet") -> pd.DataFrame:
    # weeks ignoré volontairement: le fichier est déjà last10w
    return load_train_from_hf(filename=parquet_name)

def load_items(data_dir=None) -> pd.DataFrame:
    return load_items_hf()

def load_stores(data_dir=None) -> pd.DataFrame:
    return load_stores_hf()

def load_oil(data_dir=None) -> pd.DataFrame:
    return load_oil_hf()

def load_transactions(data_dir=None) -> pd.DataFrame:
    return load_transactions_hf()

def load_holidays(data_dir=None) -> pd.DataFrame:
    return load_holidays_hf()


# ============================================================
# Admin helpers (local artifacts only)
# ============================================================
import json

def load_metadata(models_dir: str | Path = "models") -> dict:
    p = Path(models_dir) / "metadata.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_metrics(models_dir: str | Path = "models") -> dict:
    return load_metadata(models_dir=models_dir)
