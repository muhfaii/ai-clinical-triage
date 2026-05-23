"""
Main training script.

Usage:
    python -m ards_ml.training.train --data data/ARDS_Dataset.csv --out artefacts/
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier as _XGBClassifier


class XGBClassifier(_XGBClassifier):
    """XGBClassifier with sklearn 1.8+ tags compatibility (get_tags-based is_classifier)."""

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags
        tags = super().__sklearn_tags__() if hasattr(super(), "__sklearn_tags__") else Tags()
        tags.estimator_type = "classifier"
        return tags

from ..pipeline.pipeline import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET = "Mortality"
DROP_COLS = ["ICU_Length_of_Stay", TARGET]


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df[TARGET]
    return X, y


def split_data(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.15 / 0.85,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )
    log.info(
        f"Split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def calibrated_proba(pipeline, X):
    """Return Platt-calibrated probabilities if calibrator is attached."""
    raw = pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "_platt_calibrator"):
        return pipeline._platt_calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    return raw


def youden_threshold(pipeline, X_val, y_val):
    from sklearn.metrics import roc_curve
    proba = calibrated_proba(pipeline, X_val)
    fpr, tpr, thresholds = roc_curve(y_val, proba)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": 2.29,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    clf = XGBClassifier(**params)
    pipeline = build_pipeline(clf)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring="average_precision", n_jobs=-1,
    )
    return scores.mean()


def train_baseline(X_train, y_train) -> Pipeline:
    lr = LogisticRegression(
        solver="lbfgs", class_weight="balanced",
        max_iter=1000, random_state=RANDOM_STATE,
    )
    pipeline = build_pipeline(lr)
    pipeline.fit(X_train, y_train)
    log.info("Baseline logistic regression trained.")
    return pipeline


def train_xgb(X_train, y_train, X_val, y_val, n_trials: int = 50) -> Pipeline:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params.update(
        {
            "scale_pos_weight": 2.29,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "use_label_encoder": False,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
    )
    log.info(f"Best Optuna params: {best_params}")

    clf = XGBClassifier(**best_params)
    pipeline = build_pipeline(clf)
    pipeline.fit(X_train, y_train)

    # Platt scaling: fit sigmoid calibrator on validation set probabilities
    raw_val_proba = pipeline.predict_proba(X_val)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(raw_val_proba, y_val)
    pipeline._platt_calibrator = platt
    log.info("XGBoost trained and calibrated (Platt scaling on val set).")
    return pipeline, best_params, study


def main(data_path: str, out_dir: str, n_trials: int = 50):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Save test set for locked evaluation
    pd.concat([X_test, y_test], axis=1).to_csv(out / "test_set.csv", index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(out / "val_set.csv", index=False)

    # Baseline
    baseline = train_baseline(X_train, y_train)
    joblib.dump(baseline, out / "baseline_lr.pkl")

    # Primary XGBoost
    xgb_pipeline, best_params, study = train_xgb(
        X_train, y_train, X_val, y_val, n_trials=n_trials
    )
    joblib.dump(xgb_pipeline, out / "xgb_pipeline.pkl")

    # Tune threshold on validation set
    threshold = youden_threshold(xgb_pipeline, X_val, y_val)
    log.info(f"Operating threshold (Youden's J): {threshold:.4f}")

    # Persist threshold config
    config_path = Path(__file__).parent.parent / "config" / "threshold_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["operating_threshold"] = round(threshold, 4)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save best params
    with open(out / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    log.info(f"Artefacts saved to {out}")
    return xgb_pipeline, baseline, X_test, y_test, threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ARDS_Dataset.csv")
    parser.add_argument("--out", default="artefacts")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    main(args.data, args.out, args.trials)
