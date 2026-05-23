"""
Evaluation module — primary metrics, secondary metrics, and subgroup analysis.
Accuracy is intentionally excluded (misleading for 30/70 imbalanced target).
"""
import warnings

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
    recall_score,
    roc_auc_score,
)


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
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "false_alarm_rate_per_100": _false_alarm_rate(y_true, y_pred),
    }


def _false_alarm_rate(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n_alive = tn + fp
    return round((fp / n_alive) * 100, 2) if n_alive > 0 else 0.0


def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_sizes = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    weights = bin_sizes / len(y_true)
    ece = np.sum(weights[:len(prob_true)] * np.abs(prob_true - prob_pred))
    return float(ece)


def subgroup_analysis(X: pd.DataFrame, y_true, y_prob, threshold: float) -> pd.DataFrame:
    """
    Mandatory subgroup analysis per Section 6.3.
    Returns a DataFrame with AUROC and primary metrics per subgroup.
    """
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
        m["n"] = int(mask.sum())
        results.append(m)

    # Age tiers
    if "Age" in X.columns:
        _eval(X["Age"] < 50, "Age < 50")
        _eval((X["Age"] >= 50) & (X["Age"] <= 70), "Age 50–70")
        _eval(X["Age"] > 70, "Age > 70")

    # Sex
    if "Sex" in X.columns:
        _eval(X["Sex"] == 0, "Sex = Female")
        _eval(X["Sex"] == 1, "Sex = Male")

    # CCI burden — requires CCI column (added by FeatureEngineer)
    cci_col = "CCI" if "CCI" in X.columns else None
    if cci_col:
        _eval(X[cci_col] == 0, "CCI = 0")
        _eval((X[cci_col] >= 1) & (X[cci_col] <= 2), "CCI 1–2")
        _eval(X[cci_col] >= 3, "CCI ≥ 3")

    # PaO2/FiO2 tier
    if "PaO2_FiO2_Ratio" in X.columns:
        pf = X["PaO2_FiO2_Ratio"]
        _eval(pf >= 200, "P/F Mild (≥200)")
        _eval((pf >= 100) & (pf < 200), "P/F Moderate (100–200)")

    # Ventilation type
    if "Ventilation_Type" in X.columns:
        _eval(X["Ventilation_Type"] == 0, "Vent = None")
        _eval(X["Ventilation_Type"] == 1, "Vent = Non-invasive")
        _eval(X["Ventilation_Type"] == 2, "Vent = Invasive")

    # Smoking status
    if "Smoking_Status" in X.columns:
        _eval(X["Smoking_Status"] == 0, "Smoking = Never")
        _eval(X["Smoking_Status"] == 1, "Smoking = Ex-smoker")
        _eval(X["Smoking_Status"] == 2, "Smoking = Current")

    df = pd.DataFrame(results).set_index("subgroup") if results else pd.DataFrame()
    return df


def check_performance_targets(metrics: dict, targets: dict) -> dict:
    """Returns pass/fail for each minimum performance target."""
    checks = {
        "auroc": metrics.get("auroc", 0) >= targets.get("auroc", 0.85),
        "auprc": metrics.get("auprc", 0) >= targets.get("auprc", 0.70),
        "sensitivity": metrics.get("sensitivity", 0) >= targets.get("sensitivity", 0.80),
        "specificity": metrics.get("specificity", 0) >= targets.get("specificity", 0.75),
        "brier_score": metrics.get("brier_score", 1) <= targets.get("brier_score", 0.20),
    }
    return checks


def check_subgroup_gaps(subgroup_df: pd.DataFrame, overall_auroc: float, max_gap: float = 0.05) -> list:
    """Flags subgroups with AUROC gap > max_gap vs overall model."""
    if subgroup_df.empty or "auroc" not in subgroup_df.columns:
        return []
    gaps = subgroup_df["auroc"].apply(lambda x: abs(x - overall_auroc))
    flagged = gaps[gaps > max_gap].index.tolist()
    return flagged


def print_report(primary: dict, secondary: dict, ece: float, subgroup_df: pd.DataFrame, targets: dict):
    print("\n" + "=" * 60)
    print("PRIMARY METRICS (test set)")
    print("=" * 60)
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

    print(f"\n  ece                            {ece:.4f}")

    if not subgroup_df.empty:
        print("\nSUBGROUP AUROC")
        print("-" * 40)
        overall_auroc = primary.get("auroc", 0)
        for sg, row in subgroup_df.iterrows():
            gap = abs(row["auroc"] - overall_auroc)
            flag = " *** GAP > 5%" if gap > 0.05 else ""
            print(f"  {sg:<35} {row['auroc']:.4f}  (n={row['n']}){flag}")

    all_pass = all(checks.values())
    print("\n" + ("ALL TARGETS MET — ready for deployment review" if all_pass else
                  "ONE OR MORE TARGETS MISSED — review before deployment"))
    print("=" * 60)
