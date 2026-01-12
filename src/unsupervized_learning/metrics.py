from typing import List, Dict

import numpy as np

from unsupervized_learning.node import Node


def purity_score(clusters: List[List[Node]], id_to_label_map: Dict[int, str]) -> float:
    total_correct = 0
    total_samples = 0

    for cluster in clusters:
        if not cluster:
            continue

        label_counts = {}
        for node in cluster:
            true_label = id_to_label_map.get(node.id)
            if true_label:
                label_counts[true_label] = label_counts.get(true_label, 0) + 1

        if label_counts:
            most_frequent_count = max(label_counts.values())
            total_correct += most_frequent_count

        total_samples += len(cluster)

    return total_correct / total_samples if total_samples > 0 else 0.0


def dunn_index(clusters: List[List[Node]], centroids: List[Node]) -> float:
    cluster_count = len(centroids)
    if cluster_count < 2:
        return 0.0

    max_diameter = 0.0
    for centroid_index, points in enumerate(clusters):
        if not points:
            continue

        radii = [
            euclidean_distance(point, centroids[centroid_index])
            for point in points
        ]

        current_diameter = 2 * max(radii) if radii else 0.0

        if current_diameter > max_diameter:
            max_diameter = current_diameter

    if max_diameter == 0:
        return 0.0

    min_separation = float('inf')
    for centroid_index in range(cluster_count):
        for other_centroid_index in range(centroid_index + 1, cluster_count):
            dist = euclidean_distance(centroids[centroid_index], centroids[other_centroid_index])
            if dist < min_separation:
                min_separation = dist

    return float(min_separation / max_diameter)


def davies_bouldin_index(clusters: List[List[Node]], centroids: List[Node]) -> float:
    cluster_count = len(clusters)
    if cluster_count < 2:
        return 0.0

    avg_distances = []
    for cluster_id in range(cluster_count):
        points = clusters[cluster_id]
        if not points:
            avg_distances.append(0.0)
            continue

        total_dist = sum(
            euclidean_distance(point, centroids[cluster_id])
            for point in points
        )

        avg_distances.append(total_dist / len(points))

    max_r_values = []
    for cluster_id in range(cluster_count):
        r_values = []
        for other_cluster_id in range(cluster_count):
            if cluster_id == other_cluster_id:
                continue

            dist_centroids = euclidean_distance(centroids[cluster_id], centroids[other_cluster_id])

            if dist_centroids == 0:
                val = 0.0
            else:
                val = (avg_distances[cluster_id] + avg_distances[other_cluster_id]) / dist_centroids
            r_values.append(val)

        max_r_values.append(
            max(r_values) if r_values else 0.0
        )

    return float(np.mean(max_r_values))


def euclidean_distance(first_node: Node, second_node: Node) -> float:
    return np.sqrt(np.sum((first_node - second_node) ** 2))
