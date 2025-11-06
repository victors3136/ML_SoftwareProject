from numbers import Number
from typing import Dict, Tuple, LiteralString, Optional

from pandas import DataFrame

_BASE_FEATURES: Dict[LiteralString, Tuple[LiteralString, Optional[Tuple[Number, Number]]]] = {
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

_FEATURES_AS_LIST = [
    {
        'column_name': name,
        'column_type': feature_type,
        'lower_bound': bounds[0] if bounds else None,
        'upper_bound': bounds[1] if bounds else None
    }
    for name, (feature_type, bounds) in _BASE_FEATURES.items()
]

SPOTIFY_DATASET_FEATURES = DataFrame(_FEATURES_AS_LIST)

if __name__ == "__main__":
    print(SPOTIFY_DATASET_FEATURES)
