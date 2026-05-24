from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import APIRouter

from ..services import model_store

router = APIRouter()

_BASE = Path(__file__).parent.parent.parent

def _sit2stand_info() -> dict:
    cfg_path = _BASE / "sit2stand_ml/config/threshold_config.yaml"
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return {"version": cfg.get("model_version", "1.0.0"), "type": "signal_processing"}
    except Exception:
        return {"version": "unknown", "type": "signal_processing"}


@router.get("/health")
def health():
    models = {
        name: {
            "version": model_store.get(name).model_version,
            "threshold": model_store.get(name).threshold,
        }
        for name in model_store.loaded_models()
    }
    models["sit2stand"] = _sit2stand_info()
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": models,
    }
