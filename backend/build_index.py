from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
POLICIES_PATH = ROOT_DIR / "backend" / "knowledge" / "policies.json"
MODEL_DIR = ROOT_DIR / "backend" / "model"
INDEX_PATH = MODEL_DIR / "policy_index.faiss"
METADATA_PATH = MODEL_DIR / "policy_metadata.json"
EMBEDDING_MODEL = "text-embedding-3-small"


def load_policies() -> list[dict[str, str]]:
    if not POLICIES_PATH.exists():
        raise FileNotFoundError(f"Could not find policy file at {POLICIES_PATH}.")

    policies: Any = json.loads(POLICIES_PATH.read_text(encoding="utf-8"))
    if not isinstance(policies, list):
        raise ValueError("policies.json must contain a JSON array.")

    for index, policy in enumerate(policies):
        if not isinstance(policy, dict):
            raise ValueError(f"Policy at position {index} must be an object.")
        if not policy.get("id") or not policy.get("text"):
            raise ValueError(f"Policy at position {index} must include id and text.")

    return policies


def embed_policy_texts(policy_texts: list[str]) -> np.ndarray:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is required.")

    client = OpenAI()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=policy_texts,
    )

    embeddings_by_index = sorted(response.data, key=lambda item: item.index)
    embeddings = np.array(
        [item.embedding for item in embeddings_by_index],
        dtype=np.float32,
    )

    if embeddings.ndim != 2 or embeddings.shape[0] != len(policy_texts):
        raise ValueError("OpenAI returned an unexpected embedding shape.")

    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def main() -> None:
    policies = load_policies()
    policy_texts = [policy["text"] for policy in policies]
    embeddings = embed_policy_texts(policy_texts)
    index = build_faiss_index(embeddings)

    metadata = {
        str(position): {
            "id": policy["id"],
            "text": policy["text"],
        }
        for position, policy in enumerate(policies)
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Indexed {len(policies)} policies.")
    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved policy metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
