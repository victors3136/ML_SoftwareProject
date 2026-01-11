from typing import List

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
    "encode ordinals",
    "normalize",
)

if __name__ == "__main__":
    print("Loading Data...")
    raw_data: Frame = load_data(shuffle=True)

    print(f"Loaded frame has shape: {raw_data.shape}")

    print("Processing Data...")
    processed_data: Frame = apply_pipeline(raw_data, _pipeline)
    print(f"After preprocessing, frame has shape: {processed_data.shape}")

    dataset: List[Node] = [
        Node.from_frame_row(row)
        for row in processed_data.head(len(processed_data))
    ]

    print(f"Dataset prepared with {len(dataset)} items.")

    K = 21
    print(f"Initializing K-Means with K={K}...")

    kmeans = KMeans(cluster_count=K)

    print("Fitting model...")
    kmeans.fit(dataset)

    test_point = dataset[0]
    cluster_idx = kmeans.predict(test_point)
    print(f"First data point assigned to Cluster: {cluster_idx}")

    print("Algorithm execution complete.")
