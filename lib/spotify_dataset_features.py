from numbers import Number
from typing import Dict, Tuple, LiteralString, Optional, TypedDict, Literal, List, Iterable, Callable, Annotated

from pandas import DataFrame, isna

from lib.ContinuousRangeNormalizer import ContinuousRangeNormalizer
from lib.OrdinalEncoder import OrdinalEncoder

_BASE_FEATURES: Dict[LiteralString, Tuple[Literal["continuous", "discrete"], Optional[Tuple[Number, Number]]]] = {
    'acousticness': ("continuous", (0, 1)),
    'analysis_url': ("discrete", None),
    'danceability': ("continuous", (0, 1)),
    'duration_ms': ("continuous", None),
    'energy': ("continuous", (0, 1)),
    'id': ("discrete", None),
    'instrumentalness': ("continuous", (0, 1)),
    'key': ("discrete", (0, 11)),
    'liveness': ("continuous", (0, 1)),
    'loudness': ("continuous", None),
    'mode': ("discrete", (0, 1)),
    'playlist_genre': ("discrete", None),
    'playlist_id': ("discrete", None),
    'playlist_name': ("discrete", None),
    'playlist_subgenre': ("discrete", None),
    'speechiness': ("continuous", (0, 1)),
    'tempo': ("continuous", None),
    'time_signature': ("discrete", (1, 5)),
    'track_album_id': ("discrete", None),
    'track_album_name': ("discrete", None),
    'track_album_release_date': ("discrete", None),
    'track_artist': ("discrete", None),
    'track_href': ("discrete", None),
    'track_id': ("discrete", None),
    'track_name': ("discrete", None),
    'track_popularity': ("discrete", (0, 100)),
    'type': ("discrete", None),
    'uri': ("discrete", None),
    'valence': ("continuous", (0, 1)),
}


class FeatureRow(TypedDict):
    column_name: LiteralString
    column_type: Literal["continuous", "discrete"]
    lower_bound: Optional[Number]
    upper_bound: Optional[Number]


_FEATURES_AS_LIST: List[FeatureRow] = [
    {
        'column_name': name,
        'column_type': feature_type,
        'lower_bound': bounds[0] if bounds else None,
        'upper_bound': bounds[1] if bounds else None
    }
    for name, (feature_type, bounds) in _BASE_FEATURES.items()
]

SPOTIFY_DATASET_FEATURES: Annotated[DataFrame, FeatureRow] = DataFrame(_FEATURES_AS_LIST)
"""             column_name column_type  lower_bound  upper_bound
0               acousticness  continuous          0.0          1.0
1               analysis_url    discrete          NaN          NaN
2               danceability  continuous          0.0          1.0
3                duration_ms  continuous          NaN          NaN
4                     energy  continuous          0.0          1.0
5                         id    discrete          NaN          NaN
6           instrumentalness  continuous          0.0          1.0
7                        key    discrete          0.0         11.0
8                   liveness  continuous          0.0          1.0
9                   loudness  continuous          NaN          NaN
10                      mode    discrete          0.0          1.0
11            playlist_genre    discrete          NaN          NaN
12               playlist_id    discrete          NaN          NaN
13             playlist_name    discrete          NaN          NaN
14         playlist_subgenre    discrete          NaN          NaN
15               speechiness  continuous          0.0          1.0
16                     tempo  continuous          NaN          NaN
17            time_signature    discrete          1.0          5.0
18            track_album_id    discrete          NaN          NaN
19          track_album_name    discrete          NaN          NaN
20  track_album_release_date    discrete          NaN          NaN
21              track_artist    discrete          NaN          NaN
22                track_href    discrete          NaN          NaN
23                  track_id    discrete          NaN          NaN
24                track_name    discrete          NaN          NaN
25          track_popularity    discrete          0.0        100.0
26                      type    discrete          NaN          NaN
27                       uri    discrete          NaN          NaN
28                   valence  continuous          0.0          1.0
"""


def get_full_feature_type(row: FeatureRow) -> Literal["ordinal", "numerical"] | None:

    name, feature_type, lower_bound, upper_bound = row.values()
    # print(name, feature_type, lower_bound, upper_bound)
    lower_bound = None if isna(lower_bound) else lower_bound
    upper_bound = None if isna(upper_bound) else upper_bound

    match (feature_type, lower_bound, upper_bound):
        case ("discrete", None, None):
            return "ordinal"

        case ("discrete", Number(), Number()) \
             | ("continuous", None, None) \
             | ("continuous", Number(), Number()):
            return "numerical"


def get_normaliser_func(row: FeatureRow) -> (Callable[[Iterable[Number]], Iterable[Number]]
                                             | Callable[[Iterable[LiteralString]], Iterable[Number]]
                                             | None):
    match get_full_feature_type(row):
        case "ordinal":
            return lambda collection: OrdinalEncoder().fit(collection)

        case "numerical":
            return lambda collection: ContinuousRangeNormalizer.fit_to_0_1(collection)

        case _:
            return None


def check_feature_rows(dataframe: Annotated[DataFrame, FeatureRow]) -> None:
    for _, row in dataframe.iterrows():
        # noinspection PyTypeChecker
        feature_row: FeatureRow = row.to_dict()
        # acousticness: numerical
        # analysis_url: ordinal
        # danceability: numerical
        # duration_ms: numerical
        # energy: numerical
        # id: ordinal
        # instrumentalness: numerical
        # key: numerical
        # liveness: numerical
        # loudness: numerical
        # mode: numerical
        # playlist_genre: ordinal
        # playlist_id: ordinal
        # playlist_name: ordinal
        # playlist_subgenre: ordinal
        # speechiness: numerical
        # tempo: numerical
        # time_signature: numerical
        # track_album_id: ordinal
        # track_album_name: ordinal
        # track_album_release_date: ordinal
        # track_artist: ordinal
        # track_href: ordinal
        # track_id: ordinal
        # track_name: ordinal
        # track_popularity: numerical
        # type: ordinal
        # uri: ordinal
        # valence: numerical
        print(f"{feature_row['column_name']}: {get_full_feature_type(feature_row)}")


if __name__ == "__main__":
    print(SPOTIFY_DATASET_FEATURES)
    check_feature_rows(SPOTIFY_DATASET_FEATURES)
