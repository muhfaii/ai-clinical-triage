"""
Evaluation module.
NOTE: Near-random performance (~AUC 0.50) is expected with the current dataset
due to weak predictive signal (all feature correlations < 0.02 with target).
See Section 7 of requirements for enrichment roadmap.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

DATA_QUALITY_AUC_THRESHOLD = 0.60


def compute_primary_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": brier_score_loss(y_true, y_prob),
        "threshold_used": threshold,
        "n": len(y_true),
        "n_positive": int(y_true.sum()),
    }


def compute_secondary_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n_negative = tn + fp
    false_alarm_rate = round((fp / n_negative) * 100, 2) if n_negative > 0 else 0.0
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "false_alarm_rate_per_100": false_alarm_rate,
    }


def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_sizes = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    weights = bin_sizes / len(y_true)
    ece = np.sum(weights[:len(prob_true)] * np.abs(prob_true - prob_pred))
    return float(ece)


def subgroup_analysis(X: pd.DataFrame, y_true, y_prob, threshold: float) -> pd.DataFrame:
    results = []

    def _eval(mask, label):
        if mask.sum() < 10:
            return
        sub_y = y_true[mask]
        sub_p = y_prob[mask]
        if sub_y.nunique() < 2:
            return
        m = compute_primary_metrics(sub_y, sub_p, threshold)
        m["subgroup"] = label
        results.append(m)

    if "Age" in X.columns:
        _eval(X["Age"] < 40, "Age < 40")
        _eval((X["Age"] >= 40) & (X["Age"] <= 60), "Age 40–60")
        _eval(X["Age"] > 60, "Age > 60")

    if "Sex" in X.columns:
        _eval(X["Sex"] == 0, "Sex = Female")
        _eval(X["Sex"] == 1, "Sex = Male")

    if "Diabetes" in X.columns:
        _eval(X["Diabetes"] == 0, "Diabetes = No")
        _eval(X["Diabetes"] == 1, "Diabetes = Yes")

    if "Smoking" in X.columns:
        _eval(X["Smoking"] == 0, "Smoking = No")
        _eval(X["Smoking"] == 1, "Smoking = Yes")

    if "Previous_Heart_Problems" in X.columns:
        _eval(X["Previous_Heart_Problems"] == 0, "Prev Heart Problems = No")
        _eval(X["Previous_Heart_Problems"] == 1, "Prev Heart Problems = Yes")

    return pd.DataFrame(results).set_index("subgroup") if results else pd.DataFrame()


def check_performance_targets(metrics: dict, targets: dict) -> dict:
    return {
        "auroc": metrics.get("auroc", 0) >= targets.get("auroc", 0.75),
        "auprc": metrics.get("auprc", 0) >= targets.get("auprc", 0.50),
        "sensitivity": metrics.get("sensitivity", 0) >= targets.get("sensitivity", 0.80),
        "specificity": metrics.get("specificity", 0) >= targets.get("specificity", 0.65),
        "brier_score": metrics.get("brier_score", 1) <= targets.get("brier_score", 0.25),
    }


def check_subgroup_gaps(subgroup_df: pd.DataFrame, overall_auroc: float, max_gap: float = 0.05) -> list:
    if subgroup_df.empty or "auroc" not in subgroup_df.columns:
        return []
    gaps = subgroup_df["auroc"].apply(lambda x: abs(x - overall_auroc))
    return gaps[gaps > max_gap].index.tolist()


def print_report(primary: dict, secondary: dict, ece: float, subgroup_df: pd.DataFrame, targets: dict):
    print("\n" + "=" * 60)
    print("PRIMARY METRICS (test set)")
    print("=" * 60)

    if primary.get("auroc", 0) < DATA_QUALITY_AUC_THRESHOLD:
        print("\n  ⚠  WARNING: AUC below 0.60 — dataset has weak predictive signal.")
        print("     See Section 7 of requirements for enrichment actions.\n")

    checks = check_performance_targets(primary, targets)
    for k, v in primary.items():
        if k in checks:
            status = "PASS" if checks[k] else "FAIL"
            print(f"  {k:<20} {v:.4f}   [{status}]")
        else:
            print(f"  {k:<20} {v}")

    print("\nSECONDARY METRICS")
    print("-" * 40)
    for k, v in secondary.items():
        print(f"  {k:<30} {v}")
    print(f"  {'ece':<30} {ece:.4f}")

    if not subgroup_df.empty:
        print("\nSUBGROUP AUROC")
        print("-" * 40)
        overall_auroc = primary.get("auroc", 0)
        for sg, row in subgroup_df.iterrows():
            gap = abs(row["auroc"] - overall_auroc)
            flag = " *** GAP > 5%" if gap > 0.05 else ""
            print(f"  {sg:<40} {row['auroc']:.4f}  (n={row['n']}){flag}")

    all_pass = all(checks.values())
    print("\n" + ("ALL TARGETS MET" if all_pass else "ONE OR MORE TARGETS MISSED — review before deployment"))
    print("=" * 60)
