import numpy as np
import random
from typing import List

from unsupervized_learning.node import Node


class KMeans:
    def __init__(self, cluster_count: int, max_iterations: int = 100, tolerance: float = 1e-4):
        self.k = cluster_count
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids: List[Node] = []
        self.clusters: List[List[Node]] = []

    def fit(self, data: List[Node]):
        self.centroids = [
            data[index].copy()
            for index in random.sample(range(len(data)), self.k)
        ]

        for iteration in range(self.max_iterations):
            self.clusters = [[] for _ in range(self.k)]

            for point in data:
                distances: List[float] = [
                    self._euclidean_distance(point, centroid)
                    for centroid in self.centroids
                ]
                closest_index = np.argmin(distances)
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

            total_centroid_shift: float = sum(
                self._euclidean_distance(previous, current)
                for previous, current in zip(previous_centroids, self.centroids)
            )

            if total_centroid_shift < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break

    def predict(self, point: Node) -> int:
        distances: list[float] = [
            self._euclidean_distance(point, centroid)
            for centroid in self.centroids
        ]
        return int(np.argmin(distances) + 1)

    @staticmethod
    def _euclidean_distance(first_node: Node, second_node: Node) -> float:
        return np.sqrt(np.sum((first_node - second_node) ** 2))
