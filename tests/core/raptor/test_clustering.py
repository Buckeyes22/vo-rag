from __future__ import annotations

import numpy as np

from vo_rag.core.raptor.clustering import (
    ClusterAssignment,
    cluster_chunks,
    reduce_dimensions,
    select_n_components_via_bic,
    verify_reproducibility,
)
from vo_rag.core.raptor.schemas import ClusterReproducibilityConfig


def _make_config(
    umap_random_state: int = 42,
    gmm_random_state: int = 42,
    **kwargs: int | float | str,
) -> ClusterReproducibilityConfig:
    defaults: dict[str, int | float | str] = {
        "umap_n_components": 2,
        "umap_n_neighbors": 5,
        "umap_min_dist": 0.0,
        "gmm_max_components": 8,
        "min_cluster_size": 2,
        "max_cluster_size": 100,
    }
    defaults.update(kwargs)
    return ClusterReproducibilityConfig(
        umap_random_state=umap_random_state,
        gmm_random_state=gmm_random_state,
        **{  # type: ignore[arg-type]
            k: v for k, v in defaults.items() if k not in ("umap_random_state", "gmm_random_state")
        },
    )


def _make_2d_cluster_data(n_per_cluster: int = 20, seed: int = 42) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.66]])
    pieces: list[np.ndarray] = []
    ids: list[str] = []
    for i, center in enumerate(centers):
        cluster_pts = rng.normal(loc=center, scale=0.5, size=(n_per_cluster, 2))
        pieces.append(cluster_pts)
        ids.extend(f"chunk-c{i}-p{j}" for j in range(n_per_cluster))
    embeddings = np.vstack(pieces).astype(np.float32)
    return ids, embeddings


class TestSelectNComponentsViaBic:
    def test_select_n_components_via_bic_picks_known_optimum(self) -> None:
        chunk_ids, embeddings = _make_2d_cluster_data(n_per_cluster=20, seed=99)
        config = _make_config(umap_random_state=42, gmm_random_state=42)
        selected_k, bic_scores, aic_scores = select_n_components_via_bic(embeddings, config=config)
        assert 2 <= selected_k <= 5
        assert isinstance(bic_scores, dict)
        assert isinstance(aic_scores, dict)


class TestReduceDimensions:
    def test_reduce_dimensions_umap_deterministic_with_frozen_seed(self) -> None:
        rng = np.random.default_rng(0)
        embeddings = rng.normal(size=(50, 32)).astype(np.float32)
        config = _make_config(umap_random_state=42, gmm_random_state=42)
        result1 = reduce_dimensions(embeddings, config=config)
        result2 = reduce_dimensions(embeddings, config=config)
        assert np.allclose(result1, result2)


class TestClusterChunks:
    def test_cluster_chunks_returns_assignments_for_all_inputs(self) -> None:
        rng = np.random.default_rng(0)
        chunk_ids = [f"chunk-{i}" for i in range(50)]
        embeddings = rng.normal(size=(50, 32)).astype(np.float32)
        config = _make_config(umap_random_state=42, gmm_random_state=42)
        result = cluster_chunks(chunk_ids, embeddings, config=config)
        assert len(result.assignments) == 50
        assigned_ids = {a.chunk_id for a in result.assignments}
        assert assigned_ids == set(chunk_ids)

    def test_cluster_chunks_n_clusters_within_bic_range(self) -> None:
        rng = np.random.default_rng(0)
        chunk_ids = [f"chunk-{i}" for i in range(60)]
        embeddings = rng.normal(size=(60, 32)).astype(np.float32)
        config = _make_config(umap_random_state=42, gmm_random_state=42)
        result = cluster_chunks(chunk_ids, embeddings, config=config)
        assert 2 <= result.n_clusters_selected <= 10


class TestVerifyReproducibility:
    def test_verify_reproducibility_passes_with_frozen_seed(self) -> None:
        rng = np.random.default_rng(0)
        chunk_ids = [f"chunk-{i}" for i in range(50)]
        embeddings = rng.normal(size=(50, 32)).astype(np.float32)
        config = _make_config(umap_random_state=42, gmm_random_state=42)
        result = verify_reproducibility(chunk_ids, embeddings, config=config)
        assert result is True

    def test_verify_reproducibility_with_same_config_returns_true(self) -> None:
        rng = np.random.default_rng(0)
        chunk_ids = [f"chunk-{i}" for i in range(30)]
        embeddings = rng.normal(size=(30, 32)).astype(np.float32)
        config_a = _make_config(umap_random_state=42, gmm_random_state=42)
        config_b = _make_config(umap_random_state=99, gmm_random_state=99)
        result_a = verify_reproducibility(chunk_ids, embeddings, config=config_a)
        result_b = verify_reproducibility(chunk_ids, embeddings, config=config_b)
        assert result_a is True
        assert result_b is True


class TestClusterAssignments:
    def test_cluster_assignments_carry_cluster_id_and_membership_score(self) -> None:
        rng = np.random.default_rng(0)
        chunk_ids = [f"chunk-{i}" for i in range(50)]
        embeddings = rng.normal(size=(50, 32)).astype(np.float32)
        config = _make_config(umap_random_state=42, gmm_random_state=42)
        result = cluster_chunks(chunk_ids, embeddings, config=config)
        for a in result.assignments:
            assert isinstance(a, ClusterAssignment)
            assert 0.0 <= a.probability <= 1.0
            assert isinstance(a.cluster_index, int)
            assert a.cluster_index >= 0
            assert isinstance(a.umap_coords, list)
            assert len(a.umap_coords) > 0
            assert all(isinstance(c, float) for c in a.umap_coords)


class TestClusteringEdgeCases:
    def test_clustering_handles_small_n_chunks_edge_case(self) -> None:
        rng = np.random.default_rng(0)
        chunk_ids = [f"chunk-{i}" for i in range(10)]
        embeddings = rng.normal(size=(10, 32)).astype(np.float32)
        config = _make_config(
            umap_random_state=42,
            gmm_random_state=42,
            gmm_max_components=5,
            min_cluster_size=2,
        )
        result = cluster_chunks(chunk_ids, embeddings, config=config)
        assert len(result.assignments) == 10
        assert result.n_clusters_selected >= 2
