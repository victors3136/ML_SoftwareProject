from lib.data import load_data
from src.lib.features import BaseFeatures
from src.lib.pipeline import make_pipeline, apply_pipeline
from unsupervized_learning.node import Node

_relevant_features: list[str] = [
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

_irrelevant_features: list[str] = list(set(BaseFeatures.keys()) - set(_relevant_features))

_pipeline = make_pipeline(
    _irrelevant_features,
    "drop duplicates",
    "drop columns",
    "encode ordinals",
    "normalize",
)

if __name__ == "__main__":
    data = load_data(shuffle=True)
    print(data.head())
    sanitized = apply_pipeline(data, _pipeline)
    print("Original:")
    print(data.head())
    print("Processed:")
    print(sanitized.head())
    print("As nodes:")
    for item in sanitized.head():
        n = Node.from_frame_row(item)
        print(n)
