from typing import List

from pandas import concat, DataFrame, read_csv
from kagglehub import dataset_download
from pathlib import Path

from tqdm import tqdm

_dataset_id = "solomonameh/spotify-music-dataset"


def _download_dataset() -> Path:
    return Path(dataset_download(_dataset_id))


def _find_csv_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.csv"))


def load_data() -> DataFrame:
    root = _download_dataset()
    csv_files = _find_csv_files(root)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {root}")

    dataframes = []
    for file in tqdm(csv_files, "Reading CSV files... "):
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
