from numbers import Number
from typing import Dict, Tuple, LiteralString, Optional, TypedDict, Literal, List, Iterable, Callable

from pandas import DataFrame, isna

type ColumnType = Literal["continuous", "discrete"]
type FormatterType = Literal["ordinal", "numerical"]
type NormalizerMap = Callable[[Iterable[Number]], Iterable[Number]]
type EncoderMap = Callable[[Iterable[LiteralString]], Iterable[Number]]
type FormatterMap = NormalizerMap | EncoderMap

BaseFeatures: Dict[LiteralString, Tuple[ColumnType, Optional[Tuple[Number, Number]]]] = {
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
    column_type: ColumnType
    lower_bound: Optional[Number]
    upper_bound: Optional[Number]


_features_as_list: List[FeatureRow] = [
    {
        'column_name': name,
        'column_type': feature_type,
        'lower_bound': bounds[0] if bounds else None,
        'upper_bound': bounds[1] if bounds else None
    }
    for name, (feature_type, bounds) in BaseFeatures.items()
]

_spotify_dataset_features: DataFrame = DataFrame(_features_as_list)


def _feature_type(row: FeatureRow) -> Optional[FormatterType]:
    name = row["column_name"]

    feature_type = row["column_type"]
    lower_bound = row["lower_bound"]
    upper_bound = row["upper_bound"]

    lower_bound = None if isna(lower_bound) else lower_bound
    upper_bound = None if isna(upper_bound) else upper_bound

    match (name, feature_type, lower_bound, upper_bound):
        case ("id" | "track_id" | "playlist_id" | "uri", _, _, _):
            return None

        case (_, "discrete", None, None):
            return "ordinal"

        case (_, "continuous" | "discrete", Number(), Number()) | (_, "continuous", None, None):
            return "numerical"


def get_feature_info(name: LiteralString) -> Optional[FeatureRow]:
    # noinspection PyTypeChecker
    return _spotify_dataset_features[_spotify_dataset_features["column_name"] == name].iloc[0].to_dict()


# noinspection PyTypeChecker
def ordinal_columns(dataframe: DataFrame) -> List[LiteralString]:
    return [
        row["column_name"]
        for _, row in _spotify_dataset_features.iterrows()
        if _feature_type(row) == "ordinal" and row["column_name"] in dataframe.columns
    ]


# noinspection PyTypeChecker
def normalizable_columns(dataframe: DataFrame) -> List[LiteralString]:
    return [
        row["column_name"]
        for _, row in _spotify_dataset_features.iterrows()
        if _feature_type(row) == "numerical" and row["column_name"] in dataframe.columns
    ]


def main() -> None:
    print(_spotify_dataset_features)
    for _, row in _spotify_dataset_features.iterrows():
        # noinspection PyTypeChecker
        feature_row: FeatureRow = row.to_dict()
        print(f"{feature_row['column_name']}: {_feature_type(feature_row)}")


if __name__ == "__main__":
    main()
