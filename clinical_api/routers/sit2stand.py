import uuid

import yaml
from fastapi import APIRouter, HTTPException

from ..config import DISCLAIMER
from ..models.sit2stand import RepMetrics, Sit2StandInput, Sit2StandResult
from sit2stand_ml.pipeline.pipeline import analyze

router = APIRouter()

from pathlib import Path as _Path
_cfg_path = _Path(__file__).parent.parent.parent / "sit2stand_ml/config/threshold_config.yaml"
with open(_cfg_path) as _f:
    _thresholds = yaml.safe_load(_f)["thresholds"]


@router.post("/sit2stand/analyze", response_model=Sit2StandResult)
def analyze_sit2stand(body: Sit2StandInput):
    frames = [f.model_dump() for f in body.frames]

    try:
        metrics = analyze(frames, frame_rate=body.frame_rate)
    except ValueError as e:
        raise HTTPException(422, str(e))

    flags = []
    if metrics["mean_trunk_flexion_deg"] > _thresholds["max_trunk_flexion_deg"]:
        flags.append("excessive_trunk_flexion")
    if metrics["mean_movement_time_s"] > _thresholds["max_movement_time_s"]:
        flags.append("slow_movement")
    if metrics["mean_angular_velocity_deg_s"] < _thresholds["min_angular_velocity_deg_s"]:
        flags.append("low_angular_velocity")

    return Sit2StandResult(
        rep_count=metrics["rep_count"],
        mean_movement_time_s=metrics["mean_movement_time_s"],
        mean_trunk_flexion_deg=metrics["mean_trunk_flexion_deg"],
        mean_angular_velocity_deg_s=metrics["mean_angular_velocity_deg_s"],
        quality_flags=flags,
        reps=[RepMetrics(**r) for r in metrics["reps"]],
        disclaimer=DISCLAIMER,
        request_id=str(uuid.uuid4()),
    )
