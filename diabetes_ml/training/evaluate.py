"""
Evaluation module.
Accuracy is intentionally excluded — misleading under ~10.8:1 class imbalance.
Primary metric: ROC-AUC. Clinical priority metric: F2-score (recall-weighted).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_primary_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "recall": recall,
        "precision": precision,
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold_used": threshold,
        "n": len(y_true),
        "n_positive": int(y_true.sum()),
    }


def compute_secondary_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "specificity": specificity,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "false_positives_per_100_neg": round((fp / (tn + fp)) * 100, 2) if (tn + fp) > 0 else 0.0,
    }


def clinical_risk_band(prob: float, blood_glucose: float = None, hba1c: float = None) -> dict:
    """Map a model probability to a clinical risk band, applying hard rule overrides."""
    hard_flag = bool(
        (blood_glucose is not None and blood_glucose >= 200) or
        (hba1c is not None and hba1c >= 6.5)
    )
    if prob >= 0.50:
        band, action = "High", "Recommend HbA1c or OGTT confirmatory test"
    elif prob >= 0.25:
        band, action = "Moderate", "Recommend lifestyle review and follow-up in 6 months"
    else:
        band, action = "Low", "Routine monitoring per standard care"

    return {
        "probability": round(prob, 4),
        "risk_band": band,
        "action": action,
        "hard_flag": hard_flag,
        "label": "Clinical Decision-Support Score — not a diagnosis",
    }


def subgroup_analysis(X: pd.DataFrame, y_true, y_prob, threshold: float) -> pd.DataFrame:
    results = []

    def _eval(mask, label):
        if mask.sum() < 10 or y_true[mask].nunique() < 2:
            return
        m = compute_primary_metrics(y_true[mask], y_prob[mask], threshold)
        m["subgroup"] = label
        results.append(m)

    # Age tiers (matches requirements bias audit)
    if "age" in X.columns:
        _eval(X["age"] < 40, "Age < 40")
        _eval((X["age"] >= 40) & (X["age"] <= 60), "Age 40–60")
        _eval(X["age"] > 60, "Age > 60")

    # Gender
    if "gender" in X.columns:
        for g in X["gender"].unique():
            _eval(X["gender"] == g, f"Gender = {g}")

    # Smoking history
    if "smoking_history" in X.columns:
        for s in X["smoking_history"].unique():
            _eval(X["smoking_history"] == s, f"Smoking = {s}")

    return pd.DataFrame(results).set_index("subgroup") if results else pd.DataFrame()


def check_performance_targets(metrics: dict, targets: dict) -> dict:
    return {
        "auroc":     metrics.get("auroc", 0)     >= targets.get("auroc", 0.95),
        "auprc":     metrics.get("auprc", 0)     >= targets.get("auprc", 0.80),
        "recall":    metrics.get("recall", 0)    >= targets.get("recall", 0.88),
        "precision": metrics.get("precision", 0) >= targets.get("precision", 0.75),
        "f2":        metrics.get("f2", 0)        >= targets.get("f2", 0.85),
    }


def check_subgroup_gaps(subgroup_df: pd.DataFrame, overall_auroc: float, max_gap: float = 0.05) -> list:
    if subgroup_df.empty or "auroc" not in subgroup_df.columns:
        return []
    gaps = subgroup_df["auroc"].apply(lambda x: abs(x - overall_auroc))
    return gaps[gaps > max_gap].index.tolist()


def print_report(primary: dict, secondary: dict, subgroup_df: pd.DataFrame, targets: dict):
    checks = check_performance_targets(primary, targets)

    print("\n" + "=" * 60)
    print("PRIMARY METRICS (test set)")
    print("=" * 60)
    for k in ("auroc", "auprc", "recall", "precision", "f2", "f1"):
        v = primary.get(k, 0)
        status = f"[{'PASS' if checks.get(k, True) else 'FAIL'}]" if k in checks else ""
        print(f"  {k:<20} {v:.4f}   {status}")

    cm_row = f"  TP={primary['tp']}  FP={primary['fp']}  TN={primary['tn']}  FN={primary['fn']}"
    print(f"\n  Confusion Matrix\n{cm_row}")
    print(f"  Threshold used: {primary['threshold_used']:.4f}")

    print("\nSECONDARY METRICS")
    print("-" * 40)
    for k, v in secondary.items():
        print(f"  {k:<35} {v:.4f}" if isinstance(v, float) else f"  {k:<35} {v}")

    if not subgroup_df.empty and "auroc" in subgroup_df.columns:
        print("\nSUBGROUP AUROC (bias audit)")
        print("-" * 40)
        overall = primary["auroc"]
        for sg, row in subgroup_df.iterrows():
            gap = abs(row["auroc"] - overall)
            flag = " *** GAP > 5%" if gap > 0.05 else ""
            print(f"  {sg:<40} {row['auroc']:.4f}  (n={row['n']}){flag}")

    all_pass = all(checks.values())
    print("\n" + (
        "ALL TARGETS MET — ready for deployment review" if all_pass
        else "ONE OR MORE TARGETS MISSED — review before deployment"
    ))
    print("=" * 60)
