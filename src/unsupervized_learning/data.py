from typing import LiteralString, List, Dict

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


class Cluster:
    def __init__(self, frame: Frame):
        self._data: Frame = pipeline_apply(frame, _pipeline)

    def centroid(self) -> Dict[str, float]:
        return self._data.column_means()

    def head(self):
        return self._data.head()

    @property
    def data(self) -> Frame:
        return self._data


if __name__ == "__main__":
    dataset: Frame = load_data()
    print(dataset)

    data = Cluster(dataset)
    print(data.data)
