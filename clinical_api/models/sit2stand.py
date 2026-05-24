from pydantic import BaseModel, Field


class Keyframe(BaseModel):
    hip: tuple[float, float]
    knee: tuple[float, float]
    ankle: tuple[float, float]
    shoulder: tuple[float, float]


class Sit2StandInput(BaseModel):
    frames: list[Keyframe] = Field(..., min_length=10)
    frame_rate: float = Field(30.0, gt=0, le=240)

    model_config = {"json_schema_extra": {"example": {
        "frames": [
            {"hip": [320, 400], "knee": [320, 500], "ankle": [320, 600], "shoulder": [320, 250]},
        ] * 60,
        "frame_rate": 30.0,
    }}}


class RepMetrics(BaseModel):
    duration_s: float
    peak_trunk_flexion_deg: float
    peak_angular_velocity_deg_s: float


class Sit2StandResult(BaseModel):
    rep_count: int
    mean_movement_time_s: float
    mean_trunk_flexion_deg: float
    mean_angular_velocity_deg_s: float
    quality_flags: list[str]
    reps: list[RepMetrics]
    disclaimer: str
    request_id: str
