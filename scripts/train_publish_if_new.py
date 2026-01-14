# scripts/train_publish_if_new.py
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.file_download import get_hf_file_metadata

from utils.data_loader import load_train_from_hf
from utils.training import ensure_hf_cache, resolve_data_dir, train_reference_model
from utils.hf_artifacts import publish_run_to_hf


def env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    try:
        return int(v)
    except Exception:
        return default


def get_parquet_fingerprint(repo_id: str, filename: str, token: str | None, repo_type: str = "dataset") -> str:
    """
    Retourne une "empreinte" stable du fichier sur HF.
    On privilégie commit_hash, sinon etag, sinon last_modified.
    """
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)
    meta = get_hf_file_metadata(url=url, token=token)

    commit_hash = getattr(meta, "commit_hash", None)
    etag = getattr(meta, "etag", None)
    last_modified = getattr(meta, "last_modified", None)

    if commit_hash:
        return f"commit:{commit_hash}"
    if etag:
        return f"etag:{etag}"
    if last_modified:
        return f"modified:{last_modified}"
    return "unknown"


def read_latest_run_metadata(repo_id: str, token: str | None, repo_type: str = "dataset") -> dict:
    """
    Lit artifacts/latest.json puis charge artifacts/runs/<run_id>/metadata.json.
    Si pas dispo -> {}
    """
    try:
        latest_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename="artifacts/latest.json",
            token=token,
        )
        with open(latest_path, "r", encoding="utf-8") as f:
            latest = json.load(f)

        run_dir = latest.get("run_dir")
        if not run_dir:
            return {}

        meta_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=f"{run_dir}/metadata.json",
            token=token,
        )
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return {}


def main():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN manquant. Ajoute-le dans GitHub Secrets (Settings > Secrets).")

    repo_id = os.getenv("HF_REPO_ID", "khadidia-77/favorita")
    repo_type = os.getenv("HF_REPO_TYPE", "dataset")
    parquet_name = os.getenv("PARQUET_NAME", "train_last10w.parquet")

    weeks_window = env_int("WEEKS_WINDOW", 10)

    # ⚠️ IMPORTANT: ton parquet last10w = ~70 jours => total_days doit être <= 70
    total_days = env_int("TOTAL_DAYS", 70)
    test_days = env_int("TEST_DAYS", 14)
    gap_days = env_int("GAP_DAYS", 3)
    feature_gap_days = env_int("FEATURE_GAP_DAYS", 3)
    sales_history_days = env_int("SALES_HISTORY_DAYS", 120)

    # 1) Fingerprint actuel du parquet sur HF
    current_fp = get_parquet_fingerprint(
        repo_id=repo_id,
        filename=parquet_name,
        token=hf_token,
        repo_type=repo_type,
    )
    print(f"📌 Current parquet fingerprint: {current_fp}")

    # 2) Fingerprint du dernier run (si existe)
    last_meta = read_latest_run_metadata(repo_id=repo_id, token=hf_token, repo_type=repo_type)
    last_fp = (
        last_meta.get("data_signature", {})
        .get("parquet_fingerprint", None)
    )
    print(f"📌 Last trained parquet fingerprint: {last_fp}")

    # 3) Si identique -> SKIP
    if last_fp and last_fp == current_fp:
        print("✅ Aucun changement détecté sur le parquet -> entraînement SKIPPÉ.")
        return

    print("🚀 Nouveau parquet (ou pas d'historique) -> entraînement + publish.")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 4) Snapshot HF (CSV nécessaires au pipeline)
    snapshot = ensure_hf_cache(
        repo_id=repo_id,
        cache_dir="data/hf_cache",
        force=False,
        hf_token=hf_token,
        repo_type="dataset",
    )
    data_dir = resolve_data_dir(snapshot)

    # 5) Charger le train parquet depuis HF
    df = load_train_from_hf(
        repo_id=repo_id,
        hf_token=hf_token,
        weeks=weeks_window,
        filename=parquet_name,
    )

    # 6) Entraîner (models/)
    metrics = train_reference_model(
        df_last10w=df,
        data_dir=str(data_dir),
        models_dir=str(models_dir),
        weeks_window=weeks_window,
        total_days=total_days,
        test_days=test_days,
        gap_days=gap_days,
        feature_gap_days=feature_gap_days,
        sales_history_days=sales_history_days,
        data_signature={
            "source": "github_actions",
            "repo_id": repo_id,
            "parquet": parquet_name,
            "weeks_window": weeks_window,
            "total_days": total_days,
            "test_days": test_days,
            "gap_days": gap_days,
            "parquet_fingerprint": current_fp,  # ✅ clé pour comparaison
        },
    )

    # 7) Publish HF + update latest.json
    latest = publish_run_to_hf(
        local_models_dir=str(models_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        hf_token=hf_token,
    )

    print("✅ TRAIN METRICS:", metrics)
    print("✅ PUBLISHED:", latest)


if __name__ == "__main__":
    main()
