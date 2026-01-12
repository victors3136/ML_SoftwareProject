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
        f"\n{'K':<4} | {'Model':<8} | {'Time (s)':<19} | {'DB Index':<19} | {'Dunn Index':<19} | {'Purity':<19}"
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
                  f"{vals['ti_m']:8.4f} ± {vals['ti_s']:8.4f} | "
                  f"{vals['db_m']:8.4f} ± {vals['db_s']:8.4f} | "
                  f"{vals['du_m']:8.4f} ± {vals['du_s']:8.4f} | "
                  f"{vals['pu_m']:8.4f} ± {vals['pu_s']:8.4f}")

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
"""
Sample output: 
    Loading Data...
    Loaded frame has shape: (4831, 29)
    Ground truth map has 4495 items.
    Processing Data...
    Applying pipeline step: 'drop columns'
    Applying pipeline step: 'drop na'
    Applying pipeline step: 'drop duplicates'
    Applying pipeline step: 'normalize'
    Applying pipeline step: 'encode ordinals'
    After preprocessing, frame has shape: (4494, 14)
    Dataset prepared with 4494 items.
    
    K    | Model    | Time (s)            | DB Index            | Dunn Index          | Purity             
    ---------------------------------------------------------------------------------------------------------
       3 | Scratch  |   3.7958 ±   1.8174 |   2.1862 ±   0.0770 |   0.0789 ±   0.0058 |   0.1817 ±   0.0110
       3 | Library  |   0.2469 ±   0.4003 |   2.2492 ±   0.0001 |   0.0742 ±   0.0000 |   0.1727 ±   0.0001
    ---------------------------------------------------------------------------------------------------------
       5 | Scratch  |   6.7734 ±   2.7410 |   2.0907 ±   0.0786 |   0.0758 ±   0.0049 |   0.1977 ±   0.0122
       5 | Library  |   0.1208 ±   0.0287 |   2.0534 ±   0.0071 |   0.0744 ±   0.0001 |   0.2137 ±   0.0136
    ---------------------------------------------------------------------------------------------------------
      10 | Scratch  |  23.8580 ±   6.7257 |   1.8912 ±   0.0784 |   0.1080 ±   0.0118 |   0.2251 ±   0.0062
      10 | Library  |   0.1980 ±   0.0382 |   1.7518 ±   0.0298 |   0.1155 ±   0.0049 |   0.2226 ±   0.0019
    ---------------------------------------------------------------------------------------------------------
      21 | Scratch  |  51.0267 ±  11.8616 |   1.9088 ±   0.0390 |   0.1078 ±   0.0071 |   0.2425 ±   0.0040
      21 | Library  |   0.2107 ±   0.0299 |   1.8598 ±   0.0216 |   0.1143 ±   0.0052 |   0.2443 ±   0.0032
    ---------------------------------------------------------------------------------------------------------
      35 | Scratch  |  75.2814 ±  14.4264 |   1.8540 ±   0.0422 |   0.1091 ±   0.0142 |   0.2675 ±   0.0057
      35 | Library  |   0.2576 ±   0.0190 |   1.8015 ±   0.0280 |   0.1186 ±   0.0083 |   0.2654 ±   0.0051
    ---------------------------------------------------------------------------------------------------------
      50 | Scratch  | 147.4839 ±  62.6964 |   1.8270 ±   0.0351 |   0.1033 ±   0.0118 |   0.2798 ±   0.0051
      50 | Library  |   0.3165 ±   0.0810 |   1.7971 ±   0.0294 |   0.1184 ±   0.0069 |   0.2797 ±   0.0031
    ---------------------------------------------------------------------------------------------------------
     100 | Scratch  | 223.9576 ±  59.0935 |   1.7764 ±   0.0272 |   0.0943 ±   0.0087 |   0.3051 ±   0.0047
     100 | Library  |   0.3874 ±   0.0219 |   1.7616 ±   0.0172 |   0.1103 ±   0.0097 |   0.3060 ±   0.0032
    ---------------------------------------------------------------------------------------------------------
    Results: [
        {
            'k': 3, 'model': 'Scratch',
            'time_mean': 3.795751690864563, 'time_std': 1.8174021841638726,
            'db_mean': 2.186224033813425, 'db_std': 0.07699893143248428,
            'dunn_mean': 0.07894647727660009, 'dunn_std': 0.005818766233534276,
            'purity_mean': 0.18166444147752556, 'purity_std': 0.011010167513177606
        }, {
            'k': 3, 'model': 'Library',
            'time_mean': 0.2469372034072876, 'time_std': 0.4003086966133257,
            'db_mean': 2.249177018973893, 'db_std': 7.226377042820026e-05,
            'dunn_mean': 0.07420002901269993, 'dunn_std': 3.419661674233164e-06,
            'purity_mean': 0.17265242545616374, 'purity_std': 6.675567423231331e-05
        }, {
            'k': 5, 'model': 'Scratch',
            'time_mean': 6.773355627059937, 'time_std': 2.740975600810442,
            'db_mean': 2.0906660933095993, 'db_std': 0.07857325353142756,
            'dunn_mean': 0.07577960059156344, 'dunn_std': 0.0048535480537639335,
            'purity_mean': 0.19773030707610148, 'purity_std': 0.01216639332431991
        }, {
            'k': 5, 'model': 'Library',
            'time_mean': 0.1208251953125, 'time_std': 0.028678411579864112,
            'db_mean': 2.0534166983082605, 'db_std': 0.007097080903199827,
            'dunn_mean': 0.07436784980827262, 'dunn_std': 0.00011493514425092465,
            'purity_mean': 0.21372941700044504, 'purity_std': 0.013577756972817042
        }, {
            'k': 10, 'model': 'Scratch',
            'time_mean': 23.858042931556703, 'time_std': 6.725713335113603,
            'db_mean': 1.8911691454856356, 'db_std': 0.07842507817904228,
            'dunn_mean': 0.10803646216771648, 'dunn_std': 0.011814915403894781,
            'purity_mean': 0.22510013351134844, 'purity_std': 0.0062195529222823145
        }, {
            'k': 10, 'model': 'Library',
            'time_mean': 0.19802825450897216, 'time_std': 0.03823996006265149,
            'db_mean': 1.7517978182898246, 'db_std': 0.029825660407106845,
            'dunn_mean': 0.11549354855283962, 'dunn_std': 0.004870832195143094,
            'purity_mean': 0.22258566978193145, 'purity_std': 0.0018777485046541868
        }, {
            'k': 21, 'model': 'Scratch',
            'time_mean': 51.02666077613831, 'time_std': 11.861636845189642,
            'db_mean': 1.9087713995690614, 'db_std': 0.03896036394481671,
            'dunn_mean': 0.10784672725485171, 'dunn_std': 0.007124927301374671,
            'purity_mean': 0.2425233644859813, 'purity_std': 0.003961149185352955
        }, {
            'k': 21, 'model': 'Library',
            'time_mean': 0.21071219444274902, 'time_std': 0.029878294451923634,
            'db_mean': 1.8598079349919185, 'db_std': 0.021648430533807366,
            'dunn_mean': 0.11429957387051433, 'dunn_std': 0.005217281514283651,
            'purity_mean': 0.24425901201602135, 'purity_std': 0.0032185343731974373
        }, {
            'k': 35, 'model': 'Scratch',
            'time_mean': 75.28139407634735, 'time_std': 14.426436248756367,
            'db_mean': 1.8540395080854786, 'db_std': 0.042231645667853285,
            'dunn_mean': 0.10907202966511845, 'dunn_std': 0.014236006548151823,
            'purity_mean': 0.2674677347574544, 'purity_std': 0.005711415123152856
        }, {
            'k': 35, 'model': 'Library',
            'time_mean': 0.25762689113616943, 'time_std': 0.018996151783260994,
            'db_mean': 1.8015448526725222, 'db_std': 0.027954953027327693,
            'dunn_mean': 0.11857138279141759, 'dunn_std': 0.008323565093538805,
            'purity_mean': 0.265376056964842, 'purity_std': 0.005140331406658398
        }, {
            'k': 50, 'model': 'Scratch',
            'time_mean': 147.4839099407196, 'time_std': 62.696405634006425,
            'db_mean': 1.8270332504888074, 'db_std': 0.03512947745163659,
            'dunn_mean': 0.10334966947834787, 'dunn_std': 0.011837661248685696,
            'purity_mean': 0.27975077881619936, 'purity_std': 0.005102237880895957
        }, {
            'k': 50, 'model': 'Library',
            'time_mean': 0.31647024154663084, 'time_std': 0.08103516736840378,
            'db_mean': 1.7971296048212786, 'db_std': 0.029400138910684517,
            'dunn_mean': 0.11835702612850652, 'dunn_std': 0.00694572841812708,
            'purity_mean': 0.2796617712505563, 'purity_std': 0.0030990102996980395
        }, {
            'k': 100, 'model': 'Scratch',
            'time_mean': 223.9576179742813, 'time_std': 59.09346415614707,
            'db_mean': 1.7764315093554999, 'db_std': 0.027186656979854913,
            'dunn_mean': 0.09432309712954226, 'dunn_std': 0.008713294192318243,
            'purity_mean': 0.30511793502447704, 'purity_std': 0.004734792571205815
        }, {
            'k': 100, 'model': 'Library',
            'time_mean': 0.38736739158630373, 'time_std': 0.021858068390536674,
            'db_mean': 1.7615545099142806, 'db_std': 0.01723954730135274,
            'dunn_mean': 0.11025206219001966, 'dunn_std': 0.00969069940936118,
            'purity_mean': 0.3060302625723187, 'purity_std': 0.003241528576591684
        }
    ]
"""