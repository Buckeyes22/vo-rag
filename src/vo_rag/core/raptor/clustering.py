from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast

import numpy as np
import structlog
from numpy.typing import NDArray

from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.raptor.schemas import ClusterReproducibilityConfig

log = structlog.get_logger(__name__)

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
IntArray = NDArray[np.int_]


class _UMAPReducer(Protocol):
    def fit_transform(self, x: Float32Array) -> Float32Array: ...


class _UMAPFactory(Protocol):
    def __call__(
        self,
        *,
        n_components: int,
        n_neighbors: int,
        min_dist: float,
        random_state: int,
    ) -> _UMAPReducer: ...


class _GaussianMixtureModel(Protocol):
    def fit(self, x: Float32Array) -> _GaussianMixtureModel: ...

    def predict(self, x: Float32Array) -> IntArray: ...

    def predict_proba(self, x: Float32Array) -> Float64Array: ...

    def bic(self, x: Float32Array) -> float: ...

    def aic(self, x: Float32Array) -> float: ...


class _GaussianMixtureFactory(Protocol):
    def __call__(
        self,
        *,
        n_components: int,
        covariance_type: str,
        random_state: int,
    ) -> _GaussianMixtureModel: ...


class ClusterAssignment(VoBaseModel):
    chunk_id: str
    cluster_index: int
    probability: float
    umap_coords: list[float]


class ClusteringResult(VoBaseModel):
    assignments: list[ClusterAssignment]
    n_clusters_selected: int
    bic_scores: dict[int, float]
    aic_scores: dict[int, float]
    umap_random_state: int
    gmm_random_state: int


def _load_umap_factory() -> _UMAPFactory:
    try:
        umap_module = import_module("umap")
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for RAPTOR dimensionality reduction. "
            "Install it with `uv add umap-learn` or ensure the executor image "
            "includes the package."
        ) from exc

    return cast(_UMAPFactory, umap_module.UMAP)


def _load_gaussian_mixture_factory() -> _GaussianMixtureFactory:
    try:
        mixture_module = import_module("sklearn.mixture")
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for RAPTOR GMM clustering. "
            "Install it with `uv add scikit-learn` or ensure the executor image "
            "includes the package."
        ) from exc

    return cast(_GaussianMixtureFactory, mixture_module.GaussianMixture)


def _as_float32_matrix(values: Float32Array, *, name: str) -> Float32Array:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    return matrix


def _fit_gmm(
    reduced: Float32Array,
    *,
    n_components: int,
    config: ClusterReproducibilityConfig,
) -> _GaussianMixtureModel:
    gaussian_mixture = _load_gaussian_mixture_factory()
    gmm = gaussian_mixture(
        n_components=n_components,
        covariance_type=config.gmm_covariance_type,
        random_state=config.gmm_random_state,
    )
    return gmm.fit(reduced)


def reduce_dimensions(
    embeddings: Float32Array,
    *,
    config: ClusterReproducibilityConfig,
) -> Float32Array:
    embeddings_matrix = _as_float32_matrix(embeddings, name="embeddings")

    umap_factory = _load_umap_factory()
    reducer = umap_factory(
        n_components=config.umap_n_components,
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        random_state=config.umap_random_state,
    )
    reduced = reducer.fit_transform(embeddings_matrix)
    reduced_matrix = np.asarray(reduced, dtype=np.float32)

    if reduced_matrix.shape != (embeddings_matrix.shape[0], config.umap_n_components):
        raise ValueError(
            "UMAP returned unexpected shape "
            f"{reduced_matrix.shape}; expected "
            f"{(embeddings_matrix.shape[0], config.umap_n_components)}"
        )

    return reduced_matrix


def select_n_components_via_bic(
    reduced: Float32Array,
    *,
    config: ClusterReproducibilityConfig,
) -> tuple[int, dict[int, float], dict[int, float]]:
    reduced_matrix = _as_float32_matrix(reduced, name="reduced")
    n_chunks = reduced_matrix.shape[0]
    max_components = min(config.gmm_max_components, n_chunks - 1)

    if max_components < 2:
        raise ValueError("At least 3 chunks are required for GMM component selection")

    bic_scores: dict[int, float] = {}
    aic_scores: dict[int, float] = {}
    valid_candidates: list[tuple[int, float]] = []

    for k in range(2, max_components + 1):
        gmm = _fit_gmm(reduced_matrix, n_components=k, config=config)
        labels = gmm.predict(reduced_matrix)
        cluster_sizes = np.bincount(labels, minlength=k)
        bic = float(gmm.bic(reduced_matrix))
        aic = float(gmm.aic(reduced_matrix))

        bic_scores[k] = bic
        aic_scores[k] = aic

        if bool(np.any(cluster_sizes < config.min_cluster_size)):
            log.debug(
                "raptor_gmm_candidate_rejected",
                n_components=k,
                min_cluster_size=config.min_cluster_size,
                cluster_sizes=cluster_sizes.astype(int).tolist(),
            )
            continue

        score = bic if config.gmm_cluster_selection_criterion == "bic" else aic
        valid_candidates.append((k, score))

    if not valid_candidates:
        raise ValueError(
            f"No GMM component candidate satisfied min_cluster_size={config.min_cluster_size}"
        )

    selected_k = min(valid_candidates, key=lambda candidate: (candidate[1], candidate[0]))[0]
    log.info(
        "raptor_gmm_components_selected",
        n_components=selected_k,
        criterion=config.gmm_cluster_selection_criterion,
        candidate_count=len(bic_scores),
    )

    return selected_k, bic_scores, aic_scores


def cluster_chunks(
    chunk_ids: list[str],
    embeddings: Float32Array,
    *,
    config: ClusterReproducibilityConfig,
) -> ClusteringResult:
    embeddings_matrix = _as_float32_matrix(embeddings, name="embeddings")
    if len(chunk_ids) != embeddings_matrix.shape[0]:
        raise ValueError(
            "chunk_ids length must match embeddings rows: "
            f"{len(chunk_ids)} != {embeddings_matrix.shape[0]}"
        )
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk_ids must be unique")

    reduced = reduce_dimensions(embeddings_matrix, config=config)
    selected_k, bic_scores, aic_scores = select_n_components_via_bic(reduced, config=config)
    final_gmm = _fit_gmm(reduced, n_components=selected_k, config=config)
    probabilities = final_gmm.predict_proba(reduced)
    cluster_indices = np.argmax(probabilities, axis=1)

    assignments = [
        ClusterAssignment(
            chunk_id=chunk_id,
            cluster_index=int(cluster_index),
            probability=float(probabilities[row_index, int(cluster_index)]),
            umap_coords=[float(coord) for coord in reduced[row_index].tolist()],
        )
        for row_index, (chunk_id, cluster_index) in enumerate(
            zip(chunk_ids, cluster_indices, strict=True)
        )
    ]

    return ClusteringResult(
        assignments=assignments,
        n_clusters_selected=selected_k,
        bic_scores=bic_scores,
        aic_scores=aic_scores,
        umap_random_state=config.umap_random_state,
        gmm_random_state=config.gmm_random_state,
    )


def verify_reproducibility(
    chunk_ids: list[str],
    embeddings: Float32Array,
    *,
    config: ClusterReproducibilityConfig,
) -> bool:
    first = cluster_chunks(chunk_ids, embeddings, config=config)
    second = cluster_chunks(chunk_ids, embeddings, config=config)

    first_assignments = [
        (assignment.chunk_id, assignment.cluster_index) for assignment in first.assignments
    ]
    second_assignments = [
        (assignment.chunk_id, assignment.cluster_index) for assignment in second.assignments
    ]
    return first_assignments == second_assignments


__all__ = [
    "ClusterAssignment",
    "ClusteringResult",
    "cluster_chunks",
    "reduce_dimensions",
    "select_n_components_via_bic",
    "verify_reproducibility",
]
