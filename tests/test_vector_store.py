"""Tests for FAISS-backed and NumPy-backed retrieval."""

from __future__ import annotations

import numpy as np

from src.vector_store import FaissVectorStore


def test_vector_store_retrieves_ranked_matches_with_metadata(tmp_path) -> None:
    store = FaissVectorStore(tmp_path / "candidate_index.faiss")
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store.build(
        embeddings,
        item_ids=["cand_a", "cand_b", "cand_c"],
        metadata=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
    )

    matches = store.retrieve(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)

    assert [match.item_id for match in matches] == ["cand_a", "cand_b"]
    assert matches[0].metadata == {"name": "A"}
    assert matches[0].score >= matches[1].score


def test_vector_store_save_and_load_round_trips_metadata(tmp_path) -> None:
    path = tmp_path / "candidate_index.faiss"
    store = FaissVectorStore(path)
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    store.build(
        embeddings,
        item_ids=["cand_a", "cand_b"],
        metadata=[{"rank": 1}, {"rank": 2}],
    )
    store.save()

    loaded = FaissVectorStore(path)
    loaded.load()
    matches = loaded.retrieve(np.asarray([0.0, 1.0], dtype=np.float32), top_k=1)

    assert loaded.item_ids == ["cand_a", "cand_b"]
    assert loaded.metadata == [{"rank": 1}, {"rank": 2}]
    assert matches[0].item_id == "cand_b"
    assert matches[0].metadata == {"rank": 2}
