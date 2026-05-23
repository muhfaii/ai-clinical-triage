from datetime import datetime, timezone

from fastapi import APIRouter

from ..services import model_store

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {
            name: {
                "version": model_store.get(name).model_version,
                "threshold": model_store.get(name).threshold,
            }
            for name in model_store.loaded_models()
        },
    }
