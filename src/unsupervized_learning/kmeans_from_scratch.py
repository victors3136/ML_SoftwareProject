import random
from typing import List, Iterable

import numpy as np

from unsupervized_learning.metrics import euclidean_distance
from unsupervized_learning.node import Node


class KMeansFromScratch:
    def __init__(self, cluster_count: int, max_iterations: int = 100, tolerance: float = 1e-4):
        self.k = cluster_count
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids: List[Node] = []
        self.clusters: List[List[Node]] = []

    def fit(self, data: np.ndarray):

        self.centroids = [
            data[index].copy()
            for index in random.sample(range(len(data)), self.k)
        ]

        for iteration in range(self.max_iterations):
            self.clusters = [[] for _ in range(self.k)]

            for point in data:
                distances: List[float] = [
                    euclidean_distance(point, centroid)
                    for centroid in self.centroids
                ]
                closest_index = int(np.argmin(distances))
                self.clusters[closest_index].append(point)

            previous_centroids: List[Node] = [
                centroid.copy()
                for centroid in self.centroids
            ]

            for index in range(self.k):
                cluster_points: List[Node] = self.clusters[index]

                if len(cluster_points) > 0:
                    self.centroids[index] = np.mean(cluster_points, axis=0)
                else:
                    self.centroids[index] = data[random.randint(0, len(data) - 1)].copy()

            total_centroid_shift: float = np.sum(
                np.fromiter(
                    (
                        euclidean_distance(previous, current)
                        for previous, current in zip(previous_centroids, self.centroids)
                    ), dtype=float
                )
            )

            if total_centroid_shift < self.tolerance:
                break

    def predict(self, pt: Node | Iterable[Node]) -> int | Iterable[int]:
        if isinstance(pt, Node):
            return int(
                np.argmin([
                    euclidean_distance(pt, centroid)
                    for centroid in self.centroids
                ]) + 1
            )
        return np.array([
            self.predict(point) for point in pt
        ])
