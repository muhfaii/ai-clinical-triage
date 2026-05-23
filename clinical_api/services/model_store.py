"""
Loads all model artefacts once at startup and exposes them via get().
Missing artefacts are warned, not crashed — so the API starts even if
a model hasn't been trained yet.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml

from ..config import ARTEFACT_PATHS

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    pipeline: object
    explainer: Optional[object]
    threshold: float
    model_version: str
    clinical_rules: dict = field(default_factory=dict)


_store: dict[str, ModelBundle] = {}


def load_all() -> None:
    for name, paths in ARTEFACT_PATHS.items():
        if not paths["pipeline"].exists():
            log.warning(f"[{name}] pipeline artefact not found — model skipped")
            continue

        try:
            pipeline = joblib.load(paths["pipeline"])
        except Exception as e:
            log.warning(f"[{name}] pipeline failed to load ({e}) — model skipped")
            continue

        explainer = None
        if paths["explainer"].exists():
            try:
                explainer = joblib.load(paths["explainer"])
            except Exception as e:
                log.warning(f"[{name}] explainer load failed: {e}")

        with open(paths["config"]) as f:
            cfg = yaml.safe_load(f)

        _store[name] = ModelBundle(
            pipeline=pipeline,
            explainer=explainer,
            threshold=cfg.get("operating_threshold", 0.5),
            model_version=cfg.get("model_version", "1.0.0"),
            clinical_rules=cfg.get("clinical_rules", {}),
        )
        log.info(f"[{name}] loaded  threshold={_store[name].threshold}  version={_store[name].model_version}")


def get(name: str) -> Optional[ModelBundle]:
    return _store.get(name)


def loaded_models() -> list[str]:
    return list(_store.keys())


def get_top_shap_factors(bundle: ModelBundle, X: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Generic SHAP top-N extractor that works across all pipeline types."""
    if bundle.explainer is None:
        return []
    try:
        # Run everything before the classifier to get the processed feature matrix
        X_proc = bundle.pipeline[:-1].transform(X)
        sv = bundle.explainer.shap_values(X_proc)
        # Binary classification: shap_values returns list[neg, pos] or single array
        if isinstance(sv, list):
            sv = sv[1]
        sv_row = sv[0] if sv.ndim == 2 else sv

        try:
            raw_names = bundle.pipeline[:-1].get_feature_names_out()
            # Strip transformer prefixes added by ColumnTransformer (e.g. "scale__age" → "age")
            names = [n.split("__")[-1] for n in raw_names]
        except Exception:
            names = [f"feature_{i}" for i in range(len(sv_row))]

        top_idx = np.argsort(np.abs(sv_row))[::-1][:top_n]
        return [
            {
                "feature": names[i],
                "shap_value": round(float(sv_row[i]), 4),
                "direction": "increases" if sv_row[i] > 0 else "decreases",
            }
            for i in top_idx
        ]
    except Exception as e:
        log.warning(f"SHAP extraction failed: {e}")
        return []
