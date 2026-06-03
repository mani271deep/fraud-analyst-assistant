from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import OpenAI

from backend.app import score_transaction


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "backend" / "model"
POLICY_INDEX_PATH = MODEL_DIR / "policy_index.faiss"
POLICY_METADATA_PATH = MODEL_DIR / "policy_metadata.json"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """
You are a fraud analyst assistant. Explain in plain English why the transaction
was flagged using ONLY the provided SHAP features and retrieved policies.
Recommend exactly one action from APPROVE, DECLINE, or ESCALATE. Cite which
policies and features support the recommendation. Never invent reasons,
features, or policies that were not provided.

Return only valid JSON with this shape:
{
  "explanation": "plain English explanation",
  "recommended_action": "APPROVE | DECLINE | ESCALATE"
}
""".strip()


@lru_cache(maxsize=1)
def load_policy_artifacts() -> tuple[faiss.Index, dict[str, dict[str, str]]]:
    missing_paths = [
        str(path)
        for path in (POLICY_INDEX_PATH, POLICY_METADATA_PATH)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing policy index artifacts: "
            + ", ".join(missing_paths)
            + ". Run `python3 backend/build_index.py` first."
        )

    index = faiss.read_index(str(POLICY_INDEX_PATH))
    metadata = json.loads(POLICY_METADATA_PATH.read_text(encoding="utf-8"))

    return index, metadata


def get_openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is required.")

    return OpenAI()


def build_alert_description(
    fraud_probability: float,
    shap_features: list[dict[str, Any]],
) -> str:
    feature_summary = "; ".join(
        (
            f"{feature['feature']}={feature['value']} "
            f"contributed {feature['shap_contribution']:.6f}"
        )
        for feature in shap_features
    )

    if not feature_summary:
        feature_summary = "No positive SHAP features were returned."

    return (
        f"Fraud probability is {fraud_probability:.6f}. "
        f"Top increasing SHAP features: {feature_summary}"
    )


def embed_text(client: OpenAI, text: str) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return np.array([response.data[0].embedding], dtype=np.float32)


def retrieve_policies(
    query_embedding: np.ndarray,
    index: faiss.Index,
    metadata: dict[str, dict[str, str]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    distances, positions = index.search(query_embedding, limit)

    policies: list[dict[str, Any]] = []
    for position, distance in zip(positions[0], distances[0]):
        if position < 0:
            continue

        policy = metadata.get(str(int(position)))
        if not policy:
            continue

        policies.append(
            {
                "id": policy["id"],
                "text": policy["text"],
                "distance": float(distance),
            }
        )

    return policies


def generate_investigation(
    client: OpenAI,
    fraud_probability: float,
    shap_features: list[dict[str, Any]],
    policies: list[dict[str, Any]],
) -> dict[str, str]:
    user_payload = {
        "fraud_probability": fraud_probability,
        "shap_features": shap_features,
        "retrieved_policies": [
            {
                "id": policy["id"],
                "text": policy["text"],
            }
            for policy in policies
        ],
    }

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_payload, indent=2),
            },
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI returned an empty investigation response.")

    result = json.loads(content)
    action = result.get("recommended_action")
    if action not in {"APPROVE", "DECLINE", "ESCALATE"}:
        raise ValueError("OpenAI returned an invalid recommended_action.")

    explanation = result.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("OpenAI returned an invalid explanation.")

    return {
        "explanation": explanation,
        "recommended_action": action,
    }


def investigate(transaction: dict) -> dict:
    score = score_transaction(transaction)
    fraud_probability = score["fraud_probability"]
    shap_features = score["top_increasing_features"]

    client = get_openai_client()
    index, metadata = load_policy_artifacts()
    alert_description = build_alert_description(fraud_probability, shap_features)
    alert_embedding = embed_text(client, alert_description)
    policies = retrieve_policies(alert_embedding, index, metadata, limit=3)

    investigation = generate_investigation(
        client=client,
        fraud_probability=fraud_probability,
        shap_features=shap_features,
        policies=policies,
    )

    return {
        "fraud_probability": fraud_probability,
        "explanation": investigation["explanation"],
        "recommended_action": investigation["recommended_action"],
        "evidence": {
            "shap_features": shap_features,
            "retrieved_policies": policies,
        },
    }
