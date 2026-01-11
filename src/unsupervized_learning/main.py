from typing import List, Any

from lib.features import BaseFeatures
from lib.frame import Frame
from unsupervized_learning import KMeans
from unsupervized_learning.node import Node
from lib.data import load_data
from lib.pipeline import make_pipeline, apply_pipeline

_relevant_features: list[str] = [
    'id',
    'acousticness',
    'danceability',
    'energy',
    'instrumentalness',
    'key',
    'liveness',
    'loudness',
    'mode',
    'speechiness',
    'tempo',
    'time_signature',
    'valence',
    'duration_ms',
]

_irrelevant_features: list[str] = list(set(BaseFeatures.keys()) - set(_relevant_features))

_pipeline = make_pipeline(
    _irrelevant_features,
    "drop duplicates",
    "drop columns",
    "drop na",
    "encode ordinals",
    "normalize",
)

if __name__ == "__main__":
    print("Loading Data...")
    raw_data: Frame = load_data(shuffle=True)
    print(f"Loaded frame has shape: {raw_data.shape}")

    ground_truth_map = {}
    for node_id in range(len(raw_data)):
        row: dict[str, Any] = raw_data.get_row(node_id)
        ground_truth_map[row['id']] = row.get('playlist_genre')
    print(f"Ground truth map has {len(ground_truth_map)} items.")

    print("Processing Data...")
    processed_data: Frame = apply_pipeline(raw_data, _pipeline)
    print(f"After preprocessing, frame has shape: {processed_data.shape}")

    dataset: List[Node] = [
        Node.from_frame_row(row)
        for row in processed_data.head(len(processed_data))
    ]

    print(f"Dataset prepared with {len(dataset)} items.")

    results: List[tuple[int, float, float, float]] = []

    for K in [3, 5, 10, 21, 35, 100]:
        print(f"K-Means with K={K}...")
        kmeans = KMeans(cluster_count=K)
        kmeans.fit(dataset)
        db_index = KMeans.davies_bouldin_index(kmeans.clusters, kmeans.centroids)
        dunn = KMeans.dunn_index(kmeans.clusters, kmeans.centroids)
        purity = KMeans.purity_score(kmeans.clusters, ground_truth_map)

        print(f"Davies-Bouldin Index: {db_index:.4f} (Lower is better)")
        print(f"Dunn Index:           {dunn:.4f} (Higher is better)")
        print(f"Purity Score:         {purity:.4f} (Higher is better)")
        results.append((K, db_index, dunn, purity))

    print(f"Results: {results}")
