from typing import LiteralString, List

import numpy as np

from lib.framework_types import Number, NumberType
from src.lib.data import load_data
from src.lib.features import BaseFeatures
from src.lib.pipeline import make_pipeline, pipeline_apply
from src.lib.frame import Frame

_relevant_features: List[LiteralString] = [
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
    'track_popularity'
]

_irrelevant_features: List[str] = list(set(BaseFeatures.keys()) - set(_relevant_features))

_pipeline = make_pipeline(
    _irrelevant_features,
    "drop duplicates",
    "drop columns"
)


class VectorNode:
    def __init__(self, frame: Frame):
        frame = pipeline_apply(frame, _pipeline)
        assert frame.shape[0] == 1, f"A node must represent a single row, not {frame.shape[0]}"
        row_data: dict[str, Number] = frame.get_row(0)
        assert all(
            key == 'id' or isinstance(value, NumberType)
            for key, value in row_data.items()
        )
        self.data = row_data

    def __getitem__(self, key: str) -> Number:
        return self.data[key]

    def __sub__(self, other: VectorNode) -> Number:
        return sum(
            (self[key] - other[key]) ** 2
            for key in self.data.keys()
        ) ** .5

    def keys(self):
        return self.data.keys()

    def __str__(self):
        return str(self.data)


class Cluster:
    def __init__(self, *, centroid: VectorNode | None, nodes: List[VectorNode] | None = None):
        self.nodes = nodes or list()
        self.centroid = centroid or self.compute_centroid()
        assert self.nodes is not None, "The list of nodes cannot be empty at the end of __init__"
        assert self.centroid is not None, "The centroid cannot be empty at the end of __init__"

    def __add__(self, other: VectorNode | Cluster) -> Cluster:
        if isinstance(other, VectorNode):
            self.nodes.append(other)
        elif isinstance(other, Cluster):
            self.nodes.extend(other.nodes)
        return self

    def compute_centroid(self) -> VectorNode:
        return VectorNode(
            Frame({
                key: [
                    np.average(
                        node[key]
                        for node in self.nodes
                    )
                ] for key in self.centroid.keys()
            })
        )

    def __str__(self):
        return f"Cluster(centroid={self.centroid}, nodes={self.nodes})"


if __name__ == "__main__":
    dataset: Frame = load_data()
    print(dataset)
    first_row = dataset.get_row(0)
    vector_node = VectorNode(
        Frame({
            key: [value] for key, value in first_row.items()
        })
    )
    print(vector_node)
