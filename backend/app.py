from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from xgboost import DMatrix, XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "backend" / "model"
MODEL_PATH = MODEL_DIR / "fraud_model.json"
FEATURES_PATH = MODEL_DIR / "features.json"

app = FastAPI(title="Fraud Analyst Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def to_plain_float(value: Any, name: str) -> float:
    # Handle values that arrived as strings, possibly wrapped in brackets
    # e.g. "[5.0025505E-1]" or "0.5"
    if isinstance(value, str):
        cleaned = value.strip().strip("[]").strip()
        try:
            return float(cleaned)
        except ValueError as exc:
            raise ValueError(f"{name} must be numeric, got {value!r}.") from exc

    # Handle single-element lists/tuples e.g. [0.5]
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} must be a single numeric value, got {value!r}.")
        return to_plain_float(value[0], name)

    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be a single numeric value, got shape {array.shape}.")

    try:
        return float(array.reshape(-1)[0].item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[XGBClassifier, list[str], Optional[shap.TreeExplainer]]:
    missing_paths = [
        str(path)
        for path in (MODEL_PATH, FEATURES_PATH)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing trained model artifacts: "
            + ", ".join(missing_paths)
            + ". Run `python3 backend/train_model.py` first."
        )

    feature_columns = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    model = XGBClassifier()
    model._estimator_type = "classifier"
    model.load_model(MODEL_PATH)
    try:
        explainer = shap.TreeExplainer(model)
    except ValueError as exc:
        if "base_score" not in str(exc) and "could not convert string to float" not in str(exc):
            raise
        explainer = None

    return model, feature_columns, explainer


def build_transaction_frame(
    transaction: dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    missing_features = [
        feature for feature in feature_columns if feature not in transaction
    ]
    extra_features = [
        feature for feature in transaction if feature not in feature_columns
    ]

    if missing_features or extra_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Transaction must contain exactly the trained feature columns.",
                "missing_features": missing_features,
                "extra_features": extra_features,
            },
        )

    try:
        values = [
            to_plain_float(transaction[feature], feature)
            for feature in feature_columns
        ]
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    transaction_frame = pd.DataFrame([values], columns=feature_columns, dtype=float)
    return transaction_frame


def get_fraud_shap_values(
    model: XGBClassifier,
    explainer: Optional[shap.TreeExplainer],
    transaction_frame: pd.DataFrame,
) -> np.ndarray:
    if explainer is None:
        booster = model.get_booster()
        dmatrix = DMatrix(
            transaction_frame,
            feature_names=list(transaction_frame.columns),
        )
        shap_values = booster.predict(dmatrix, pred_contribs=True)
    else:
        shap_values = explainer.shap_values(transaction_frame)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_array = np.asarray(shap_values, dtype=float)
    if shap_array.ndim == 3:
        if shap_array.shape[2] > 1:
            shap_array = shap_array[:, :, 1]
        else:
            shap_array = shap_array[:, :, 0]
    if shap_array.ndim != 2 or shap_array.shape[0] != 1:
        raise ValueError(f"Unexpected SHAP value shape: {shap_array.shape}")
    if shap_array.shape[1] == transaction_frame.shape[1] + 1:
        shap_array = shap_array[:, :-1]

    shap_row = shap_array[0].astype(float)
    return shap_row


@app.post("/score")
def score_transaction(transaction: dict[str, Any]) -> dict[str, Any]:
    try:
        model, feature_columns, explainer = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    transaction_frame = build_transaction_frame(transaction, feature_columns)

    fraud_probability = to_plain_float(
        model.predict_proba(transaction_frame)[0, 1],
        "fraud_probability",
    )
    shap_values = get_fraud_shap_values(model, explainer, transaction_frame)

    top_positive_features = sorted(
        (
            {
                "feature": feature,
                "value": to_plain_float(
                    transaction_frame.iloc[0][feature],
                    f"{feature} value",
                ),
                "shap_contribution": to_plain_float(
                    contribution,
                    f"{feature} SHAP contribution",
                ),
            }
            for feature, contribution in zip(feature_columns, shap_values)
            if to_plain_float(contribution, f"{feature} SHAP contribution") > 0
        ),
        key=lambda item: item["shap_contribution"],
        reverse=True,
    )[:5]

    return {
        "fraud_probability": fraud_probability,
        "top_increasing_features": top_positive_features,
    }


@app.post("/investigate")
def investigate_transaction(transaction: dict[str, Any]) -> dict[str, Any]:
    from backend.assistant import investigate

    try:
        return investigate(transaction)
    except (FileNotFoundError, EnvironmentError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
