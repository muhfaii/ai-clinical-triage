"""
End-to-end training, evaluation, and SHAP persistence.

Usage:
    cd /Users/loop/Documents/ai-agent-hospital
    python -m stroke_ml.run
"""
import json
import logging
from pathlib import Path

import joblib
import yaml

from stroke_ml.training.train import (
    calibrated_proba,
    f2_threshold,
    load_data,
    split_data,
    train_baseline,
    train_xgb,
)
from stroke_ml.training.evaluate import (
    check_subgroup_recall_gaps,
    compute_ece,
    compute_primary_metrics,
    compute_secondary_metrics,
    print_report,
    subgroup_analysis,
)
from stroke_ml.explainability.shap_explainer import build_explainer, save_explainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_PATH = "stroke_ml/data/healthcare-dataset-stroke-data.csv"
OUT_DIR = Path("stroke_ml/artefacts")
N_TRIALS = 50


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    log.info("Training baseline logistic regression...")
    baseline = train_baseline(X_train, y_train)
    joblib.dump(baseline, OUT_DIR / "baseline_lr.pkl")

    log.info(f"Tuning XGBoost ({N_TRIALS} Optuna trials)...")
    xgb_pipeline, best_params, study = train_xgb(
        X_train, y_train, X_val, y_val, n_trials=N_TRIALS
    )
    joblib.dump(xgb_pipeline, OUT_DIR / "xgb_pipeline.pkl")

    with open(OUT_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    threshold = f2_threshold(xgb_pipeline, X_val, y_val)
    log.info(f"Operating threshold (F2-optimal on val): {threshold:.4f}")

    config_path = Path("stroke_ml/config/threshold_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["operating_threshold"] = round(threshold, 4)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    log.info("Evaluating on locked test set...")
    y_prob_test = calibrated_proba(xgb_pipeline, X_test)

    primary = compute_primary_metrics(y_test, y_prob_test, threshold)
    secondary = compute_secondary_metrics(y_test, y_prob_test, threshold)
    ece = compute_ece(y_test, y_prob_test)

    # Apply engineer+imputer to X_test so subgroup_analysis can find encoded columns
    eng = xgb_pipeline.named_steps["engineer"]
    imp = xgb_pipeline.named_steps["imputer"]
    X_test_eng = imp.transform(eng.transform(X_test)).reset_index(drop=True)

    subgroup_df = subgroup_analysis(
        X_test_eng,
        y_test.reset_index(drop=True),
        y_prob_test,
        threshold,
    )

    targets = config.get("performance_targets", {})
    print_report(primary, secondary, ece, subgroup_df, targets)

    flagged = check_subgroup_recall_gaps(subgroup_df, primary["recall"])
    if flagged:
        log.warning(f"Subgroups with >10pp recall gap vs overall: {flagged}")

    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump({**primary, **secondary, "ece": ece}, f, indent=2)

    if not subgroup_df.empty:
        subgroup_df.to_csv(OUT_DIR / "subgroup_metrics.csv")

    log.info("Building SHAP explainer...")
    try:
        explainer = build_explainer(xgb_pipeline)
        save_explainer(explainer, str(OUT_DIR / "shap_explainer.pkl"))
        log.info("SHAP explainer saved.")
    except Exception as e:
        log.warning(f"SHAP explainer build failed: {e}")

    log.info(f"All artefacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
