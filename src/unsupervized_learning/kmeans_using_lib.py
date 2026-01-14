import time
from typing import List
import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
from unsupervized_learning.node import Node


class KMeansLibrary:
    def __init__(self, cluster_count: int, max_iterations: int = 100, tolerance: float = 1e-4):
        self.k = cluster_count
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids: List[Node] = []
        self.clusters: List[List[Node]] = []
        self._sklearn_model = None

    def fit(self, data: np.ndarray):
        nodes = np.stack([node.view(np.ndarray) for node in data])
        self._sklearn_model = SKLearnKMeans(
            n_clusters=self.k,
            max_iter=self.max_iterations,
            tol=self.tolerance,
            n_init=10,
            random_state=int(time.time())
        )
        self._sklearn_model.fit(nodes)

        self.centroids = [
            Node(center)
            for center in self._sklearn_model.cluster_centers_
        ]

        self.clusters = [[] for _ in range(self.k)]
        labels = self._sklearn_model.labels_

        for index, label in enumerate(labels):
            self.clusters[label].append(data[index])

    def predict(self, point: Node) -> int:
        node = point.view(np.ndarray).reshape(1, -1)
        return int(self._sklearn_model.predict(node)[0] + 1)
