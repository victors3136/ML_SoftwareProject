from typing import LiteralString, List

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


class Node:
    def __init__(self, frame: Frame):
        frame = pipeline_apply(frame, _pipeline)
        assert frame.shape[0] == 1, f"A node must represent a single row, not {frame.shape[0]}"
        row_data: dict[str, Number] = frame.get_row(0)
        assert all(isinstance(value, NumberType) for value in row_data.values())
        self.data = row_data

    def __sub__(self, other: Node):
        ...


if __name__ == "__main__":
    dataset: Frame = load_data()
    print(dataset)
