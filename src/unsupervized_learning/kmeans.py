from dataclasses import dataclass
from typing import Iterable, Optional
import random

from lib.data import load_data
from src.lib.frame import Frame
from unsupervized_learning.metrics import ClusteringMetric, NotImplementedMetric


@dataclass(frozen=True)
class AlgorithmConfiguration:
    cluster_count: int
    random_state: Optional[int] = None


def _numeric_columns(dataframe: Frame) -> list[str]:
    numeric_columns = dataframe.numeric_columns()
    if not numeric_columns:
        raise ValueError("No numeric columns available for K-means.")
    return numeric_columns


def _row_as_dict(dataframe: Frame, row_index: int, columns: Iterable[str]) -> dict[str, float]:
    return {
        column: float(dataframe[column][row_index])
        for column in columns
    }


def _squared_distance(row: dict[str, float], centroid: dict[str, float], columns: Iterable[str]) -> float:
    return sum(
        (row[column] - centroid[column]) ** 2
        for column in columns
    )


def randomly_init_centroids(dataframe: Frame, cluster_count: int, *, seed: Optional[int] = None) \
        -> list[dict[str, float]]:
    if cluster_count <= 0:
        raise ValueError("Cluster count must be positive")
    row_count, _ = dataframe.shape
    if cluster_count > row_count:
        raise ValueError("Cluster count cannot be greater than the number of data points")
    random_generator = random.Random(seed)
    columns = _numeric_columns(dataframe)
    indices = random_generator.sample(range(row_count), cluster_count)
    centroids = [
        _row_as_dict(dataframe, index, columns)
        for index in indices
    ]
    print("Initial centroids:")
    print('\n'.join(
        f'{index}: {centroid}'
        for index, centroid in enumerate(centroids)
    ))
    return centroids


class KMeans:
    def __init__(
            self,
            cluster_count: int,
            *,
            random_state: Optional[int] = None,
            metric: Optional[ClusteringMetric] = None,
    ) -> None:
        if cluster_count <= 1:
            raise ValueError("Cluster count must be greater than 1 for K-means")
        self._config = AlgorithmConfiguration(
            cluster_count=cluster_count,
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
        self._centroids = randomly_init_centroids(
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


if __name__ == "__main__":
    frame: Frame = load_data()
    randomly_init_centroids(frame, 5)
