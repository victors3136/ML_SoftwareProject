from typing import LiteralString, List

import pandas
from pandas import DataFrame, Series

from src.lib.Data import load_data
from src.lib.Features import BaseFeatures
from src.lib.SpotifyDatasetPipeline import make_pipeline, pipeline_apply

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

_irrelevant_features: List[LiteralString] = BaseFeatures.keys() - _relevant_features

# noinspection PyArgumentList
_pipeline = make_pipeline(
    _irrelevant_features,
    "drop duplicates",
    "normalize",
    "encode ordinals",
    "drop columns"
)


class Cluster:
    def __init__(self, data: DataFrame):
        self._data: DataFrame = pipeline_apply(data, _pipeline)

    def centroid(self) -> Series:
        pass

    def head(self):
        return self._data.head()

    @property
    def data(self):
        return self._data


def main():
    dataset: DataFrame = load_data()
    initial_max_col_disp_sz = pandas.get_option('display.max_columns')
    try:
        pandas.set_option('display.max_columns', 29)
        print(dataset.head(10))
    finally:
        pandas.set_option('display.max_columns', initial_max_col_disp_sz)

    data = Cluster(dataset)
    initial_max_col_disp_sz = pandas.get_option('display.max_columns')
    try:
        pandas.set_option('display.max_columns', 29)
        print(data.data.head(10))
    finally:
        pandas.set_option('display.max_columns', initial_max_col_disp_sz)


if __name__ == "__main__":
    main()
