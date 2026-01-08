from typing import List

from kagglehub import dataset_download
from pathlib import Path

from lib.frame import Frame

_kaggle_dataset_id = "solomonameh/spotify-music-dataset"


def _download_dataset() -> Path:
    return Path(dataset_download(_kaggle_dataset_id))


def _find_csv_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.csv"))


def load_data(*, shuffle: bool = False) -> Frame:
    root: Path = _download_dataset()
    csv_files: List[Path] = _find_csv_files(root)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {root}")

    frames: List[Frame] = []
    for file in csv_files:
        try:
            frame = Frame.from_csv(file)
            frames.append(frame)
        except Exception as e:
            print(f"Warning: failed to read {file}: {e}")

    if not frames:
        raise RuntimeError("No readable CSV files found in the dataset")

    if len(frames) == 1:
        return frames[0]

    return Frame.concatenate(frames, shuffle=shuffle)
