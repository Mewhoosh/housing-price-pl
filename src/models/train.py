from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import dagshub
import joblib
import mlflow
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.features.build_features import build_features

load_dotenv()  # loads .env — MLFLOW_TRACKING_URI, USERNAME, PASSWORD

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).parents[2]
ARTEFACTS_DIR = ROOT / "model_artefacts"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"
DATA_PATH     = ROOT / "data" / "raw" / "otodom_all.csv"

# ---------------------------------------------------------------------------
# XGBoost config
# ---------------------------------------------------------------------------

XGB_PARAMS: dict = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=17,
    n_jobs=-1,
    verbosity=0,
)

EXPERIMENT_NAME = "housing-price-pl"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- DagsHub MLflow tracking ---
    dagshub.init(
        repo_owner="Mewhoosh",
        repo_name="housing-price-pl",
        mlflow=True,
    )

    # --- snapshot raw data before training ---
    _snapshot_data()

    # --- features ---
    feats = build_features(DATA_PATH)

    # --- MLflow run ---
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        log.info("MLflow run: %s", run.info.run_id)

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(
            feats.X_train, feats.y_train,
            eval_set=[(feats.X_test, feats.y_test)],
            verbose=False,
        )

        # metrics on test set — original price scale (matches notebook)
        y_pred_log = model.predict(feats.X_test)
        y_pred     = np.expm1(y_pred_log)
        y_true     = np.expm1(feats.y_test)

        mae  = float(mean_absolute_error(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        r2   = float(r2_score(y_true, y_pred))

        log.info("MAE: %s PLN | MAPE: %.1f%% | R²: %.3f", f"{mae:,.0f}", mape, r2)

        mlflow.log_params(XGB_PARAMS)
        mlflow.log_metrics({"mae": mae, "mape": mape, "r2": r2})
        mlflow.log_metric("train_rows", len(feats.X_train))
        mlflow.log_metric("test_rows",  len(feats.X_test))

        # --- save artefacts ---
        ARTEFACTS_DIR.mkdir(exist_ok=True)

        model_path = ARTEFACTS_DIR / "xgb_model_candidate.joblib"
        joblib.dump(model, model_path)

        enc_city_path = ARTEFACTS_DIR / "target_enc_city.json"
        enc_city_path.write_text(
            json.dumps(feats.enc_city, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        enc_neigh_path = ARTEFACTS_DIR / "target_enc_neighborhood.json"
        enc_neigh_path.write_text(
            json.dumps(feats.enc_neighborhood, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(enc_city_path))
        mlflow.log_artifact(str(enc_neigh_path))

        # city_neighborhoods.json — needed by app.py dropdowns
        (ARTEFACTS_DIR / "city_neighborhoods.json").write_text(
            json.dumps(feats.city_neighborhoods, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # meta.json — feature names + input ranges for app.py validation
        meta = {
            "features":    list(feats.X_train.columns),
            "rooms_range": [int(feats.X_train["rooms"].min()), int(feats.X_train["rooms"].max())],
            "floor_range": [int(feats.X_train["floor"].min()), int(feats.X_train["floor"].max())],
            "area_range":  [float(feats.X_train["area_m2"].min()), float(feats.X_train["area_m2"].max())],
        }
        (ARTEFACTS_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # store run id so evaluate.py can pick up this candidate
        candidate_meta = {
            "run_id": run.info.run_id,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        candidate_meta_path = ARTEFACTS_DIR / "candidate_meta.json"
        candidate_meta_path.write_text(
            json.dumps(candidate_meta, indent=2),
            encoding="utf-8",
        )

        log.info("Artefacts saved to %s", ARTEFACTS_DIR)


def _snapshot_data() -> None:
    """Copy current otodom_all.csv to data/snapshots/YYYY-MM.csv."""
    if not DATA_PATH.exists():
        log.warning("No data file found at %s — skipping snapshot", DATA_PATH)
        return

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    label = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dest  = SNAPSHOTS_DIR / f"{label}.csv"

    if dest.exists():
        log.info("Snapshot %s already exists — skipping", dest.name)
        return

    shutil.copy2(DATA_PATH, dest)
    log.info("Snapshot saved → %s", dest)


if __name__ == "__main__":
    train()
