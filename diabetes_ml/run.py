"""
End-to-end training, evaluation, and SHAP persistence.

Usage:
    cd /Users/loop/Documents/ai-agent-hospital
    python -m diabetes_ml.run
    python -m diabetes_ml.run --data path/to/diabetes_prediction_dataset.csv --trials 100
"""
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml

from diabetes_ml.training.train import (
    load_data, split_data,
    train_baseline, train_xgb,
    calibrated_proba, f2_threshold,
)
from diabetes_ml.training.evaluate import (
    compute_primary_metrics,
    compute_secondary_metrics,
    subgroup_analysis,
    check_subgroup_gaps,
    print_report,
)
from diabetes_ml.explainability.shap_explainer import build_explainer, save_explainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_PATH = "diabetes_ml/data/diabetes_prediction_dataset.csv"
OUT_DIR = Path("diabetes_ml/artefacts")
CONFIG_PATH = Path("diabetes_ml/config/threshold_config.yaml")
N_TRIALS = 50


def main(data_path: str = DATA_PATH, n_trials: int = N_TRIALS):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    targets = config.get("performance_targets", {})

    # Load & split
    X, y = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Baseline
    log.info("Training baseline logistic regression...")
    baseline = train_baseline(X_train, y_train)
    joblib.dump(baseline, OUT_DIR / "baseline_lr.pkl")
    log.info("Baseline saved.")

    # Primary XGBoost with Optuna
    log.info(f"Tuning XGBoost ({n_trials} Optuna trials)...")
    xgb_pipeline, best_params, _ = train_xgb(X_train, y_train, X_val, y_val, n_trials=n_trials)
    joblib.dump(xgb_pipeline, OUT_DIR / "xgb_pipeline.pkl")

    with open(OUT_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Threshold: maximise F2 on validation set
    threshold = f2_threshold(xgb_pipeline, X_val, y_val)
    log.info(f"Operating threshold (max F2 on val): {threshold:.4f}")

    config["operating_threshold"] = round(threshold, 4)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Locked test-set evaluation
    log.info("Evaluating on locked test set...")
    y_prob_test = calibrated_proba(xgb_pipeline, X_test)

    primary = compute_primary_metrics(y_test, y_prob_test, threshold)
    secondary = compute_secondary_metrics(y_test, y_prob_test, threshold)

    subgroup_df = subgroup_analysis(
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
        y_prob_test,
        threshold,
    )

    print_report(primary, secondary, subgroup_df, targets)

    flagged = check_subgroup_gaps(subgroup_df, primary["auroc"])
    if flagged:
        log.warning(f"Subgroups with >5% AUROC gap vs overall: {flagged}")

    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump({**primary, **secondary}, f, indent=2)

    if not subgroup_df.empty:
        subgroup_df.to_csv(OUT_DIR / "subgroup_metrics.csv")

    # SHAP explainer
    log.info("Building SHAP explainer...")
    try:
        explainer = build_explainer(xgb_pipeline)
        save_explainer(explainer, str(OUT_DIR / "shap_explainer.pkl"))
        log.info("SHAP explainer saved.")
    except Exception as e:
        log.warning(f"SHAP explainer build failed: {e}")

    log.info(f"All artefacts saved to {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    main(args.data, args.trials)
