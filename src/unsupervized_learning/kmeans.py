from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional
import random

from src.lib.frame import Frame
from .metrics import ClusteringMetric, NotImplementedMetric


class AlgorithmInitializationMethod(str, Enum):
    RANDOM = "random"
    KMEANS_PLUS_PLUS = "kmeans++"


@dataclass(frozen=True)
class AlgorithmConfiguration:
    cluster_count: int
    initialization_method: AlgorithmInitializationMethod = AlgorithmInitializationMethod.KMEANS_PLUS_PLUS
    random_state: Optional[int] = None


def _numeric_columns(frame: Frame) -> list[str]:
    numeric_columns = frame.numeric_columns()
    if not numeric_columns:
        raise ValueError("No numeric columns available for K-means.")
    return numeric_columns


def _row_as_dict(frame: Frame, row_index: int, columns: Iterable[str]) -> dict[str, float]:
    return {column: float(frame[column][row_index]) for column in columns}


def _squared_distance(row: dict[str, float], centroid: dict[str, float], columns: Iterable[str]) -> float:
    return sum((row[column] - centroid[column]) ** 2 for column in columns)


def randomly_init_centroids(frame: Frame, cluster_count: int, *, seed: Optional[int] = None) -> list[dict[str, float]]:
    if cluster_count <= 0:
        raise ValueError("k must be positive")
    row_count, _ = frame.shape
    if cluster_count > row_count:
        raise ValueError("k cannot be greater than the number of data points")
    random_generator = random.Random(seed)
    columns = _numeric_columns(frame)
    indices = random_generator.sample(range(row_count), cluster_count)
    return [_row_as_dict(frame, index, columns) for index in indices]


def kmpp_init_centroids(frame: Frame, cluster_count: int, *, seed: Optional[int] = None) -> list[dict[str, float]]:
    if cluster_count <= 0:
        raise ValueError("k must be positive")
    row_count, _ = frame.shape
    if cluster_count > row_count:
        raise ValueError("k cannot be greater than the number of data points")
    random_generator = random.Random(seed)
    columns = _numeric_columns(frame)

    first_index = random_generator.randrange(row_count)
    centroids: list[dict[str, float]] = [_row_as_dict(frame, first_index, columns)]

    for _ in range(cluster_count - 1):
        squared_distances: list[float] = []
        for index in range(row_count):
            row = _row_as_dict(frame, index, columns)
            squared_distance = min(
                _squared_distance(row, centroid, columns)
                for centroid in centroids
            )
            squared_distances.append(squared_distance)

        total = sum(squared_distances)
        if total == 0.0:
            remaining_indices = [
                index for index in range(row_count)
            ]
            centroids.append(_row_as_dict(frame, random_generator.choice(remaining_indices), columns))
            continue

        choice = random_generator.random() * total
        cumulative = 0.0
        chosen_index = 0
        for index, squared_distance in enumerate(squared_distances):
            cumulative += squared_distance
            if cumulative >= choice:
                chosen_index = index
                break
        centroids.append(_row_as_dict(frame, chosen_index, columns))

    return centroids


class KMeans:
    def __init__(
            self,
            cluster_count: int,
            *,
            init: AlgorithmInitializationMethod = AlgorithmInitializationMethod.KMEANS_PLUS_PLUS,
            random_state: Optional[int] = None,
            metric: Optional[ClusteringMetric] = None,
    ) -> None:
        if cluster_count <= 1:
            raise ValueError("k must be greater than 1 for K-means")
        self._config = AlgorithmConfiguration(
            cluster_count=cluster_count,
            initialization_method=init,
            random_state=random_state
        )
        self._metric: ClusteringMetric = metric or NotImplementedMetric()
        self._centroids: Optional[list[dict[str, float]]] = None

    @property
    def centroids(self) -> Optional[list[dict[str, float]]]:
        return self._centroids

    @property
    def config(self) -> AlgorithmConfiguration:
        return self._config

    def fit(self, data: Frame) -> KMeans:
        init_method = None
        match self._config.initialization_method:
            case AlgorithmInitializationMethod.RANDOM:
                init_method = randomly_init_centroids
            case AlgorithmInitializationMethod.KMEANS_PLUS_PLUS:
                init_method = kmpp_init_centroids
        assert init_method is not None
        self._centroids = init_method(
            data,
            self._config.cluster_count,
            seed=self._config.random_state
        )
        return self

    def predict(self, data: Frame) -> list[int]:
        raise NotImplementedError("KMeans.predict is not implemented yet.")

    def fit_predict(self, data: Frame) -> list[int]:
        self.fit(data)
        return self.predict(data)
