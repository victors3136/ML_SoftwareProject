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

    def __repr__(self):
        return str(self.data)


class Cluster:
    def __init__(self, *, centroid: VectorNode | None, nodes: List[VectorNode] | None = None):
        self.nodes = nodes or list()
        self.centroid = centroid
        if self.centroid is None:
            self.centroid = self.compute_centroid()
        assert self.nodes is not None, "The list of nodes cannot be empty at the end of __init__"
        assert self.centroid is not None, "The centroid cannot be empty at the end of __init__"
        assert all(node.keys() == self.centroid.keys() for node in self.nodes), "All nodes must have the same keys"

    def __add__(self, other: VectorNode | Cluster) -> Cluster:
        if isinstance(other, VectorNode):
            self.nodes.append(other)
        elif isinstance(other, Cluster):
            self.nodes.extend(other.nodes)
        return self

    @staticmethod
    def average_fields(key_set: set[str], nodes: list[VectorNode]):
        return VectorNode(
            Frame({
                key: [
                    np.average([
                        node[key]
                        for node in nodes
                    ]) if key != 'id' else None
                ] for key in key_set
            })
        )

    def compute_centroid(self) -> VectorNode:
        match (self.centroid, self.nodes):
            case None, None:
                raise ValueError("Cannot compute centroid of an empty cluster")
            case centroid, None:
                return centroid
            case None, nodes:
                if not nodes:
                    raise ValueError("Cannot compute centroid of an empty cluster")
                return Cluster.average_fields(set(nodes[0].keys()), nodes)
            case centroid, nodes:
                return Cluster.average_fields(set(centroid.keys()), nodes)

    def __str__(self):
        return f"Cluster(centroid={self.centroid},\n nodes={
        '\n'.join(str(node) for node in self.nodes)
        })"


if __name__ == "__main__":
    dataset: Frame = pipeline_apply(load_data(), _pipeline)

    first_row = dataset.get_row(0)
    vector_node = VectorNode(
        Frame({
            key: [value] for key, value in first_row.items()
        })
    )

    cluster = Cluster(centroid=None, nodes=[vector_node])

    print(cluster)
