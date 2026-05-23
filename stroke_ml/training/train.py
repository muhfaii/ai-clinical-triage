import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier as _XGBClassifier

from ..pipeline.pipeline import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET = "stroke"
SCALE_POS_WEIGHT = 19.52  # 4861 / 249


class XGBClassifier(_XGBClassifier):
    """XGBClassifier with sklearn 1.8+ tags compatibility."""

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags
        tags = super().__sklearn_tags__() if hasattr(super(), "__sklearn_tags__") else Tags()
        tags.estimator_type = "classifier"
        return tags


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # bmi "N/A" strings are handled by FeatureEngineer, but coerce here too for safety
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    X = df.drop(columns=[TARGET], errors="ignore")
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
    log.info(f"Split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def calibrated_proba(pipeline, X):
    raw = pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "_platt_calibrator"):
        return pipeline._platt_calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    return raw


def f2_threshold(pipeline, X_val, y_val) -> float:
    """Select threshold that maximises F2-score (beta=2) on the validation set."""
    proba = calibrated_proba(pipeline, X_val)
    best_thresh, best_f2 = 0.3, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = (proba >= t).astype(int)
        f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_thresh = f2, float(t)
    log.info(f"F2-optimal threshold: {best_thresh:.2f}  (F2={best_f2:.4f})")
    return best_thresh


def train_baseline(X_train, y_train) -> Pipeline:
    lr = LogisticRegression(
        solver="lbfgs", class_weight="balanced",
        max_iter=1000, random_state=RANDOM_STATE,
    )
    # Baseline uses class_weight instead of SMOTE — keeps it as a clean reference
    pipeline = build_pipeline(lr, use_smote=False)
    pipeline.fit(X_train, y_train)
    log.info("Baseline logistic regression trained.")
    return pipeline


def _objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    clf = XGBClassifier(**params)
    pipeline = build_pipeline(clf, use_smote=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


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

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    })
    log.info(f"Best Optuna params: {best_params}")

    clf = XGBClassifier(**best_params)
    pipeline = build_pipeline(clf, use_smote=False)
    pipeline.fit(X_train, y_train)

    # Platt scaling on validation set
    raw_val_proba = pipeline.predict_proba(X_val)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(raw_val_proba, y_val)
    pipeline._platt_calibrator = platt
    log.info("XGBoost trained and calibrated (Platt scaling on val set).")
    return pipeline, best_params, study
