from typing import Iterable

from lib.load_dataset import load_spotify_dataset

filter_columns = lambda columns_to_filter, dataframe: [column for column in columns_to_filter if
                                                       column in dataframe.columns]

drop_columns = lambda dataframe, columns_to_drop: dataframe.drop(columns=filter_columns(columns_to_drop, dataframe))


def make_pipeline(columns_to_drop: Iterable[str]):
    return [
        ("drop duplicates", lambda element: element.drop_duplicates(ignore_index=True)),
        ("drop columns", lambda element: element.drop(columns=list(columns_to_drop))),
    ]


def main() -> None:
    print("Downloading and loading Spotify dataset...")
    dataframe = load_spotify_dataset()
    print(f"Loaded DataFrame with shape: {dataframe.shape}")

    example_drop = ['energy', 'tempo', 'danceability', 'playlist_genre', 'loudness', 'liveness']

    pipeline = make_pipeline(example_drop)
    for name, action in pipeline:
        print(f"Performing '{name}' step...")
        dataframe = action(dataframe)
        print("shape:", dataframe.shape)


if __name__ == "__main__":
    main()
