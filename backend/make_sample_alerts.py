from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "creditcard.csv"
OUTPUT_PATH = ROOT_DIR / "frontend" / "public" / "sample_alerts.json"
LABEL_COLUMN = "Class"
RANDOM_STATE = 42


def row_to_transaction(row: pd.Series) -> dict[str, float]:
    return {
        column: float(value)
        for column, value in row.items()
    }


def make_alerts() -> list[dict[str, Any]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {DATA_PATH}. "
            "Place creditcard.csv in the data/ directory first."
        )

    df = pd.read_csv(DATA_PATH)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Dataset must include a '{LABEL_COLUMN}' column.")

    fraud_rows = df[df[LABEL_COLUMN] == 1].sample(
        n=5,
        random_state=RANDOM_STATE,
    )
    legitimate_rows = df[df[LABEL_COLUMN] == 0].sample(
        n=3,
        random_state=RANDOM_STATE,
    )
    sample_rows = pd.concat([fraud_rows, legitimate_rows], ignore_index=True)
    sample_rows = sample_rows.drop(columns=[LABEL_COLUMN])

    return [
        {
            "id": f"ALERT-{index + 1:03d}",
            "transaction": row_to_transaction(row),
        }
        for index, (_, row) in enumerate(sample_rows.iterrows())
    ]


def main() -> None:
    alerts = make_alerts()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(alerts, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {len(alerts)} sample alerts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
