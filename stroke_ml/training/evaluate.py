import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    roc_auc_score,
)


def compute_primary_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    return {
        "recall": recall,
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f2": f2,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold_used": threshold,
        "n": len(y_true),
        "n_positive": int(y_true.sum()),
    }


def compute_secondary_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "accuracy": accuracy,
        "brier_score": brier_score_loss(y_true, y_prob),
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
        y_pred = (sub_p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(sub_y, y_pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results.append({
            "subgroup": label,
            "recall": recall,
            "n": int(mask.sum()),
            "n_positive": int(sub_y.sum()),
        })

    if "age" in X.columns:
        age = pd.to_numeric(X["age"], errors="coerce")
        _eval(age < 40, "age < 40")
        _eval((age >= 40) & (age <= 60), "age 40–60")
        _eval((age > 60) & (age <= 80), "age 60–80")
        _eval(age > 80, "age > 80")

    if "gender_Male" in X.columns:
        _eval(X["gender_Male"] == 1, "gender = Male")
        _eval(X["gender_Male"] == 0, "gender = Female")

    if "Residence_type" in X.columns:
        _eval(X["Residence_type"] == 1, "Residence = Urban")
        _eval(X["Residence_type"] == 0, "Residence = Rural")

    return pd.DataFrame(results).set_index("subgroup") if results else pd.DataFrame()


def check_subgroup_recall_gaps(subgroup_df: pd.DataFrame, overall_recall: float, max_gap: float = 0.10) -> list:
    if subgroup_df.empty or "recall" not in subgroup_df.columns:
        return []
    gaps = subgroup_df["recall"].apply(lambda x: overall_recall - x)
    return gaps[gaps > max_gap].index.tolist()


def print_report(primary: dict, secondary: dict, ece: float, subgroup_df: pd.DataFrame, targets: dict):
    print("\n" + "=" * 60)
    print("PRIMARY METRICS (test set)")
    print("=" * 60)

    gate_keys = {"recall", "auroc", "auprc", "f2"}
    for k, v in primary.items():
        if k in gate_keys:
            target = targets.get(k, 0)
            status = "PASS" if v >= target else "FAIL"
            print(f"  {k:<20} {v:.4f}   [target >= {target}  {status}]")
        elif k not in ("tp", "fp", "tn", "fn"):
            print(f"  {k:<20} {v}")

    cm_keys = ["tp", "fp", "tn", "fn"]
    if all(k in primary for k in cm_keys):
        print(f"\n  Confusion matrix  TP={primary['tp']}  FP={primary['fp']}  TN={primary['tn']}  FN={primary['fn']}")

    print("\nSECONDARY METRICS")
    print("-" * 40)
    for k, v in secondary.items():
        print(f"  {k:<30} {v:.4f}")
    print(f"  {'ece':<30} {ece:.4f}")

    if not subgroup_df.empty:
        print("\nSUBGROUP RECALL")
        print("-" * 40)
        overall_recall = primary.get("recall", 0)
        for sg, row in subgroup_df.iterrows():
            gap = overall_recall - row["recall"]
            flag = f"  *** GAP > 10pp (gap={gap:.2f})" if gap > 0.10 else ""
            print(f"  {sg:<40} {row['recall']:.4f}  (n={row['n']}, pos={row['n_positive']}){flag}")

    all_pass = all(primary.get(k, 0) >= targets.get(k, 0) for k in gate_keys)
    print("\n" + ("ALL TARGETS MET" if all_pass else "ONE OR MORE TARGETS MISSED — review before deployment"))
    print("=" * 60)
