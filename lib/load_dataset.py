from pandas import concat, DataFrame, read_csv
from kagglehub import dataset_download
from pathlib import Path

_DATASET_ID = "solomonameh/spotify-music-dataset"

_download_dataset = lambda: Path(dataset_download(_DATASET_ID))

_find_csv_files = lambda root: sorted(root.rglob("*.csv"))


def load_spotify_dataset() -> DataFrame:
    root = _download_dataset()
    csv_files = _find_csv_files(root)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {root}")

    dataframes = []
    for file in csv_files:
        try:
            dataframe = read_csv(file)
            dataframes.append(dataframe)
        except Exception as e:
            print(f"Warning: failed to read {file}: {e}")
    if not dataframes:
        raise RuntimeError("No readable CSV files found in the dataset")
    if len(dataframes) == 1:
        return dataframes[0]
    return concat(dataframes, ignore_index=True, sort=False)
