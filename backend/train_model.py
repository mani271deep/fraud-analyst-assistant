from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "creditcard.csv"
MODEL_DIR = ROOT_DIR / "backend" / "model"
MODEL_PATH = MODEL_DIR / "fraud_model.json"
FEATURES_PATH = MODEL_DIR / "features.json"
LABEL_COLUMN = "Class"
RANDOM_STATE = 42


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {DATA_PATH}. "
            "Place creditcard.csv in the data/ directory before training."
        )

    df = pd.read_csv(DATA_PATH)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Dataset must include a '{LABEL_COLUMN}' label column.")

    feature_columns = [column for column in df.columns if column != LABEL_COLUMN]
    X = df[feature_columns]
    y = df[LABEL_COLUMN]

    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())
    if positive_count == 0:
        raise ValueError("Dataset contains no fraud rows where Class == 1.")

    scale_pos_weight = negative_count / positive_count

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_score)

    print("Test metrics")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved feature columns to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
