"""
Training script — logistic regression baseline + XGBoost primary model.

Usage:
    python -m diabetes_ml.training.train --data diabetes_ml/data/diabetes_prediction_dataset.csv
"""
import logging

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier as _XGBClassifier

from ..pipeline.pipeline import build_pipeline
from .data import load_data, split_data  # noqa: F401 — re-exported for run.py

log = logging.getLogger(__name__)

RANDOM_STATE = 42


class XGBClassifier(_XGBClassifier):
    """XGBClassifier with sklearn 1.8+ tags compatibility."""

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags
        tags = super().__sklearn_tags__() if hasattr(super(), "__sklearn_tags__") else Tags()
        tags.estimator_type = "classifier"
        return tags


def calibrated_proba(pipeline, X):
    """Return Platt-calibrated probabilities when calibrator is attached."""
    raw = pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "_platt_calibrator"):
        return pipeline._platt_calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    return raw


def f2_threshold(pipeline, X_val, y_val) -> float:
    """Find the probability threshold that maximises F2-score on the validation set."""
    from sklearn.metrics import fbeta_score
    proba = calibrated_proba(pipeline, X_val)
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [
        fbeta_score(y_val, (proba >= t).astype(int), beta=2, zero_division=0)
        for t in thresholds
    ]
    return float(thresholds[np.argmax(scores)])


def _objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 15.0),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    clf = XGBClassifier(**params)
    pipeline = build_pipeline(clf)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
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


def train_xgb(X_train, y_train, X_val, y_val, n_trials: int = 50):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = {
        **study.best_params,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    log.info(f"Best Optuna params: {best_params}")

    clf = XGBClassifier(**best_params)
    pipeline = build_pipeline(clf)
    pipeline.fit(X_train, y_train)

    # Platt calibration on validation set probabilities
    raw_val = pipeline.predict_proba(X_val)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(raw_val, y_val)
    pipeline._platt_calibrator = platt
    log.info("XGBoost trained and Platt-calibrated on val set.")

    return pipeline, best_params, study
