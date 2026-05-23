import pandas as pd


def predict_proba_single(pipeline, record: dict, threshold: float) -> dict:
    df = pd.DataFrame([record])
    prob = float(pipeline.predict_proba(df)[:, 1][0])

    if hasattr(pipeline, "_platt_calibrator"):
        import numpy as np
        prob = float(pipeline._platt_calibrator.predict_proba(
            np.array([[prob]])
        )[:, 1][0])

    prediction = int(prob >= threshold)

    # uncertain band takes precedence over moderate/high in the overlap range [0.25, 0.45]
    if 0.25 <= prob <= 0.45:
        risk_level = "uncertain"
    elif prob < 0.2:
        risk_level = "low"
    elif prob <= 0.45:
        risk_level = "moderate"
    else:
        risk_level = "high"

    return {"prediction": prediction, "probability": round(prob, 4), "risk_level": risk_level}
