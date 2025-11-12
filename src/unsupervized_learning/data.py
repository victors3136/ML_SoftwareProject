from __future__ import annotations

from typing import Iterable, List, LiteralString, Optional, Tuple

from lib.framework_types import NumberType
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
]

_irrelevant_features: List[str] = list(set(BaseFeatures.keys()) - set(_relevant_features))

_pipeline = make_pipeline(
    _irrelevant_features,
    "drop duplicates",
    "drop columns",
    "encode ordinals",
    "normalize",
)


class VectorNode:

    def __init__(self, frame: Frame):
        assert frame.shape[0] == 1, f"A node must represent a single row, not {frame.shape[0]}"
        row = frame.get_row(0)
        self.id: Optional[str] = row.get('id') \
            if isinstance(row.get('id'), str) \
            else None
        self.vector: dict[str, float] = {
            column_key: float(value)
            for column_key, value in row.items()
            if column_key != 'id' and isinstance(value, NumberType)
        }
        if not self.vector:
            raise ValueError("VectorNode requires at least one numeric feature")
        self._columns: Tuple[str, ...] = tuple(sorted(self.vector.keys()))

    def __getitem__(self, key: str) -> float:
        return self.vector[key]

    def keys(self) -> Tuple[str, ...]:
        return self._columns

    def distance(self, other: VectorNode, keys: Optional[Iterable[str]] = None) -> float:
        column_keys = tuple(keys) if keys is not None else self._columns
        if keys is None and self._columns != other._columns:
            raise ValueError("Vector key mismatch between nodes")
        return sum((self.vector[c] - other.vector[c]) ** 2 for c in column_keys) ** 0.5

    def __str__(self):
        return f"VectorNode(id={self.id}, \nkeys={list(self._columns)}, \nvalues={list(self.vector.values())})"

    def __repr__(self):
        return str(self)


class Cluster:
    def __init__(self, *, centroid: VectorNode | None, nodes: List[VectorNode] | None = None):
        self.nodes: List[VectorNode] = nodes or []
        self.centroid: Optional[VectorNode] = centroid
        if self.centroid is None:
            self.centroid = self.compute_centroid()
        if self.centroid is None:
            raise ValueError("The centroid cannot be empty at the end of __init__")
        assert all(node.keys() == self.centroid.keys() for node in self.nodes), "All nodes must have the same keys"

    def __add__(self, other: VectorNode | Cluster) -> Cluster:
        if isinstance(other, VectorNode):
            self.nodes.append(other)
        elif isinstance(other, Cluster):
            self.nodes.extend(other.nodes)
        return self

    def __iadd__(self, other: VectorNode | Cluster) -> Cluster:
        return self.__add__(other)

    def add_node(self, node: VectorNode) -> None:
        if self.centroid and node.keys() != self.centroid.keys():
            raise ValueError("Node keys do not match cluster centroid keys")
        self.nodes.append(node)

    @staticmethod
    def average_fields(key_set: Iterable[str], nodes: list[VectorNode]) -> VectorNode:
        column_keys = sorted(key_set)
        node_count = len(nodes)
        if node_count == 0:
            raise ValueError("Cannot average fields of empty node list")
        column_sums = {
            column_key: 0.0
            for column_key in column_keys
        }
        for node in nodes:
            for column_key in column_keys:
                column_sums[column_key] += node[column_key]
        means = {
            column_key: [column_sums[column_key] / node_count]
            for column_key in column_keys
        }
        return VectorNode(Frame(means))

    def compute_centroid(self) -> Optional[VectorNode]:
        if not self.nodes:
            return self.centroid
        key_set = self.nodes[0].keys()
        return Cluster.average_fields(key_set, self.nodes)

    def recompute_centroid(self) -> None:
        self.centroid = self.compute_centroid()

    def __str__(self):
        nodes_str = "\n".join(str(node) for node in self.nodes)
        return f"Cluster(centroid={self.centroid},\n nodes=\n{nodes_str}\n)"


def build_clustering_frame(
        frame: Frame,
        features: Optional[Iterable[str]] = None,
        *,
        normalize: bool = True,
        encode_ordinals: bool = True,
) -> Frame:
    selected = list(features) \
        if features is not None \
        else list(_relevant_features)
    discard = list(set(BaseFeatures.keys()) - set(selected))
    options: list[str] = ["drop duplicates", "drop columns"]
    if encode_ordinals:
        options.append("encode ordinals")
    if normalize:
        options.append("normalize")
    steps = make_pipeline(discard, *options)
    return pipeline_apply(frame, steps)


def frame_to_nodes(frame: Frame) -> List[VectorNode]:
    row_count, _ = frame.shape
    nodes: List[VectorNode] = []
    for index in range(row_count):
        row = frame.get_row(index)
        if any(
                value is None
                for value in row.values()):
            print(f"Skipping row {index} because it contains null values")
            continue
        node_frame = Frame({
            key: [value]
            for key, value in row.items()
        })
        nodes.append(VectorNode(node_frame))
    return nodes


if __name__ == "__main__":
    raw: Frame = load_data(shuffle=True)
    dataset: Frame = build_clustering_frame(raw)

    processed_nodes = frame_to_nodes(dataset)
    if not processed_nodes:
        raise RuntimeError("No data available to create VectorNode instances.")
    cluster = Cluster(centroid=None, nodes=[processed_nodes[0]])
    print(cluster)
