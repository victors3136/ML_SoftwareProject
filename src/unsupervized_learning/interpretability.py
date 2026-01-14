import random

import numpy as np
from lime import lime_tabular

from lib.data import load_data
from lib.features import BaseFeatures
from lib.frame import Frame
from lib.pipeline import make_pipeline, apply_pipeline
from unsupervized_learning import KMeansFromScratch
from unsupervized_learning.node import Node

if __name__ == "__main__":
    feature_names = ["danceability", "energy", "key", "loudness", "mode",
                     "speechiness", "acousticness", "instrumentalness",
                     "liveness", "valence", "tempo", "duration_ms", "time_signature", "id"]

    to_drop: list[str] = list(set(BaseFeatures.keys()) - set(feature_names))

    K = 21

    raw_data: Frame = load_data(shuffle=True)

    ground_truth_map = {}
    for node_id in range(len(raw_data)):
        row = raw_data.get_row(node_id)
        ground_truth_map[row['id']] = row.get('playlist_genre')

    processed_data: Frame = apply_pipeline(raw_data, make_pipeline(to_drop,
                                                                   "drop duplicates",
                                                                   "drop columns",
                                                                   "drop na",
                                                                   "normalize",
                                                                   )
                                           )
    dataset = np.array([
        Node.from_frame_row(row)
        for row in processed_data.head(len(processed_data))
    ])
    kmeans = KMeansFromScratch(cluster_count=K)
    print("Fitting KMeans...")
    kmeans.fit(dataset)


    def predict_cluster_probs(data_points):
        nodes = [Node(row) for row in data_points]

        predictions = kmeans.predict(nodes)

        zero_indexed_preds = np.array(predictions) - 1

        probabilities = np.zeros((data_points.shape[0], K))
        for i, cluster_id in enumerate(zero_indexed_preds):
            probabilities[i, int(cluster_id)] = 1.0

        return probabilities


    print("Creating explainer...")

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=dataset,
        feature_names=feature_names,
        class_names=[f"Cluster {i + 1}" for i in range(K)],
    )
    for index in random.sample(range(len(dataset)), 10):
        specific_track_values = Node(dataset[index])
        cluster = kmeans.predict(specific_track_values)
        label_to_explain = cluster - 1

        print(f"Explaining Cluster {label_to_explain + 1} for track at index {index}")

        exp = explainer.explain_instance(
            specific_track_values,
            predict_cluster_probs,
            num_features=5,
            labels=(label_to_explain,)
        )

        print(
            f"The top 5 features contributing to the prediction of cluster {label_to_explain + 1} are:\n {
            exp.as_list(label=label_to_explain)
            }"
        )
