import time
from typing import List, Any

import numpy as np

from lib.data import load_data
from lib.features import BaseFeatures
from lib.frame import Frame
from lib.pipeline import make_pipeline, apply_pipeline
from unsupervized_learning import KMeansFromScratch
from unsupervized_learning.kmeans_using_lib import KMeansLibrary
from unsupervized_learning.metrics import davies_bouldin_index, dunn_index, purity_score
from unsupervized_learning.node import Node

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

K_VALUES = [3, 5, 10, 21, 35, 50, 100]
SAMPLE_COUNT = 10

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

    experiment_results = []
    print(
        f"\n{'K':<4} | {'Model':<8} | {'Time (s)':<18} | {'DB Index':<18} | {'Dunn Index':<18} | {'Purity':<18}"
    )
    print("-" * 105)

    for K in K_VALUES:
        metrics = {
            'Scratch': {'db': [], 'dunn': [], 'purity': [], 'time': []},
            'Library': {'db': [], 'dunn': [], 'purity': [], 'time': []}
        }

        for _ in range(SAMPLE_COUNT):
            model_s = KMeansFromScratch(cluster_count=K)
            start_time = time.time()
            model_s.fit(dataset)
            end_time = time.time()

            metrics['Scratch']['time'].append(end_time - start_time)
            metrics['Scratch']['db'].append(
                davies_bouldin_index(model_s.clusters, model_s.centroids)
            )
            metrics['Scratch']['dunn'].append(
                dunn_index(model_s.clusters, model_s.centroids)
            )
            metrics['Scratch']['purity'].append(
                purity_score(model_s.clusters, ground_truth_map)
            )

            model_l = KMeansLibrary(cluster_count=K)
            start_time = time.time()
            model_l.fit(dataset)
            end_time = time.time()

            metrics['Library']['time'].append(end_time - start_time)
            metrics['Library']['db'].append(
                davies_bouldin_index(model_l.clusters, model_l.centroids))
            metrics['Library']['dunn'].append(
                dunn_index(model_l.clusters, model_l.centroids)
            )
            metrics['Library']['purity'].append(
                purity_score(model_l.clusters, ground_truth_map)
            )

        for model_name in ['Scratch', 'Library']:
            data = metrics[model_name]

            vals = {
                'ti_m': np.mean(data['time']), 'ti_s': np.std(data['time']),
                'db_m': np.mean(data['db']), 'db_s': np.std(data['db']),
                'du_m': np.mean(data['dunn']), 'du_s': np.std(data['dunn']),
                'pu_m': np.mean(data['purity']), 'pu_s': np.std(data['purity']),
            }

            print(f"{K:<4} | {model_name:<8} | "
                  f"{vals['ti_m']:2.4f} ± {vals['ti_s']:.4f} | "
                  f"{vals['db_m']:.4f} ± {vals['db_s']:.4f}      | "
                  f"{vals['du_m']:.4f} ± {vals['du_s']:.4f}        | "
                  f"{vals['pu_m']:.4f} ± {vals['pu_s']:.4f}")

            experiment_results.append({
                'k': K,
                'model': model_name,
                'time_mean': vals['ti_m'], 'time_std': vals['ti_s'],
                'db_mean': vals['db_m'], 'db_std': vals['db_s'],
                'dunn_mean': vals['du_m'], 'dunn_std': vals['du_s'],
                'purity_mean': vals['pu_m'], 'purity_std': vals['pu_s']
            })

        print("-" * 105)

    print(f"Results: {experiment_results}")
