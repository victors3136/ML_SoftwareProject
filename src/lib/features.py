from typing import Dict, Tuple, LiteralString, Optional, List

from lib.frame import Frame
from lib.framework_types import ColumnType, Number, FeatureRow, FormatterType, NumberType

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

_features_as_list: List[FeatureRow] = [
    {
        'column_name': name,
        'column_type': feature_type,
        'lower_bound': bounds[0] if bounds else None,
        'upper_bound': bounds[1] if bounds else None
    }
    for name, (feature_type, bounds) in BaseFeatures.items()
]

_spotify_dataset_features: List[FeatureRow] = _features_as_list


def _feature_type(row_data: FeatureRow) -> Optional[FormatterType]:
    name = row_data["column_name"]

    feature_type = row_data["column_type"]
    lower_bound = row_data["lower_bound"]
    upper_bound = row_data["upper_bound"]

    match (name, feature_type, lower_bound, upper_bound):
        case ("id" | "track_id" | "playlist_id" | "uri", _, _, _):
            return None
        case (_, "discrete", None, None):
            return "ordinal"
        case (_, "continuous" | "discrete", lb, ub) if isinstance(lb, NumberType) and isinstance(ub, NumberType):
            return "numerical"
        case (_, "continuous", None, None):
            return "numerical"
    return None


def get_feature_info(name: LiteralString) -> Optional[FeatureRow]:
    for row_data in _spotify_dataset_features:
        if row_data["column_name"] == name:
            return row_data
    return None


def ordinal_columns(dataframe: Frame) -> List[str]:
    return [
        row_data["column_name"]
        for row_data in _spotify_dataset_features
        if _feature_type(row_data) == "ordinal" and row_data["column_name"] in dataframe.columns
    ]


def normalizable_columns(dataframe: Frame) -> List[str]:
    return [
        row_daya["column_name"]
        for row_daya in _spotify_dataset_features
        if _feature_type(row_daya) == "numerical" and row_daya["column_name"] in dataframe.columns
    ]

if __name__ == "__main__":
    for row in _spotify_dataset_features:
        print(f"{row['column_name']}: {_feature_type(row)}")
