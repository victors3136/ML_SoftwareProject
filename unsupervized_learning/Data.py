from typing import LiteralString, List

from pandas import DataFrame, Series

from lib.Data import LoadData
from lib.Features import BaseFeatures
from lib.SpotifyDatasetPipeline import Pipeline, apply

_relevant_features: List[LiteralString] = [
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

_irrelevant_features: List[LiteralString] = BaseFeatures.keys() - _relevant_features

# noinspection PyArgumentList
_pipeline = Pipeline(
    _irrelevant_features,
    "drop duplicates",
    "normalize",
    "encode ordinals",
    "drop columns"
)


class Cluster:
    def __init__(self, data: DataFrame):
        self._data: DataFrame = apply(data, _pipeline)

    def centroid(self) -> Series:
        pass

    def head(self):
        return self._data.head()


def main():
    data = Cluster(LoadData())
    print(data.head())


if __name__ == "__main__":
    main()
