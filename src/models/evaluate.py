from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

from src.features.build_features import build_features

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).parents[2]
ARTEFACTS_DIR = ROOT / "model_artefacts"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    candidate_meta_path = ARTEFACTS_DIR / "candidate_meta.json"
    production_path     = ARTEFACTS_DIR / "xgb_model.joblib"
    candidate_path      = ARTEFACTS_DIR / "xgb_model_candidate.joblib"

    if not candidate_path.exists():
        log.error("No candidate model found. Run train.py first.")
        return

    # Build the SAME test set for both models
    feats = build_features()

    y_true = np.expm1(feats.y_test.values)

    # --- evaluate candidate ---
    candidate = joblib.load(candidate_path)
    mae_new   = _mae(candidate, feats.X_test, y_true)
    log.info("Candidate MAE: %s PLN", f"{mae_new:,.0f}")

    # --- evaluate production (if exists) ---
    if production_path.exists():
        production = joblib.load(production_path)
        try:
            mae_old = _mae(production, feats.X_test, y_true)
            log.info("Production MAE on new test set: %s PLN", f"{mae_old:,.0f}")
        except ValueError:
            mae_old = float("inf")
            log.warning("Feature set mismatch — production model incompatible, auto-promoting candidate")
    else:
        mae_old = float("inf")
        log.info("No production model yet — candidate will be promoted automatically")

    # --- promote? ---
    if mae_new < mae_old:
        log.info(
            "Candidate is BETTER (%s < %s PLN) — promoting to production",
            f"{mae_new:,.0f}", f"{mae_old:,.0f}",
        )
        _promote(candidate_path, production_path, mae_new, mae_old)
    else:
        log.warning(
            "Candidate is NOT better (%s >= %s PLN) — keeping current production model",
            f"{mae_new:,.0f}", f"{mae_old:,.0f}",
        )


def _mae(model, X_test, y_true: np.ndarray) -> float:
    y_pred = np.expm1(model.predict(X_test))
    return float(mean_absolute_error(y_true, y_pred))


def _promote(
    candidate_path: Path,
    production_path: Path,
    mae_new: float,
    mae_old: float,
) -> None:
    shutil.copy2(candidate_path, production_path)
    log.info("Copied %s → %s", candidate_path.name, production_path.name)

    # promote encoding artefacts (candidate_meta already written by train.py)
    for name in ("target_enc_city.json", "target_enc_neighborhood.json"):
        src = ARTEFACTS_DIR / name
        if src.exists():
            log.info("Artefact promoted: %s", name)

    # update production meta
    meta_path = ARTEFACTS_DIR / "production_meta.json"
    candidate_meta = json.loads(
        (ARTEFACTS_DIR / "candidate_meta.json").read_text(encoding="utf-8")
    )
    production_meta = {
        **candidate_meta,
        "mae_old": mae_old,
        "mae_new": mae_new,
        "promoted": True,
    }
    meta_path.write_text(json.dumps(production_meta, indent=2), encoding="utf-8")
    log.info("Production meta updated → %s", meta_path)


if __name__ == "__main__":
    evaluate()
