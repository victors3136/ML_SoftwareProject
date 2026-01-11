from typing import Iterable, Callable, List, Optional

from lib.data import load_data
from lib.frame import Frame
from lib.frame_encoder import DataFrameEncoder
from lib.normalization import DataFrameNormalizer

type PipelineAction = Callable[[Frame], Frame]
type PipelineStep = tuple[str, PipelineAction]
type PipelineOptions = Iterable[str]
type PipelineFactory = Callable[[Optional[Iterable[str]], PipelineOptions], List[PipelineStep]]


def _has_column(frame: Frame) -> Callable[[str], bool]:
    return lambda column: column in frame.columns


def _full_options_included(options: Iterable[str]) -> bool:
    return "full" in options


def make_pipeline(discard: Optional[Iterable[str]], *options) -> List[PipelineStep]:
    steps: List[PipelineStep] = []
    full_options = _full_options_included(options)
    if full_options or "drop columns" in options:
        def drop_columns_step(frame: Frame) -> Frame:
            column_keys = [column_key for column_key in (discard or []) if column_key in frame.columns]
            if not column_keys:
                return frame
            return frame.drop(columns_to_drop=column_keys)

        steps.append(("drop columns", drop_columns_step))
    if full_options or "drop na" in options:
        steps.append(("drop na", lambda frame: frame.dropna()))
    if full_options or "drop duplicates" in options:
        steps.append(("drop duplicates", lambda frame: frame.drop_duplicates()))
    if full_options or "normalize" in options:
        steps.append(("normalize", lambda frame: DataFrameNormalizer.fit(frame)))
    if full_options or "encode ordinals" in options:
        steps.append(("encode ordinals", lambda frame: DataFrameEncoder().fit(frame)))
    return steps


def apply_pipeline(frame: Frame, steps: List[PipelineStep]) -> Frame:
    for step_name, apply_step in steps:
        frame = apply_step(frame)
    return frame


def _view_frame(frame: Frame) -> None:
    print(frame)


if __name__ == "__main__":
    print("Downloading and loading Spotify dataset...")
    dataframe = load_data()
    print(f"Loaded DataFrame with shape: {dataframe.shape}")
    print("===========================================")
    print("Head of the DataFrame before any processing:")
    _view_frame(dataframe)
    print("===========================================")
    pipeline = make_pipeline(None, "full")
    for name, action in pipeline:
        print(f"Performing '{name}' step. Dataframe has the dimensions: {dataframe.shape}")
        dataframe = action(dataframe)
        print(f"Step '{name}' finished. Dataframe now has the dimensions: {dataframe.shape}")
    print("===========================================")
    print("Head of the DataFrame after processing:")
    _view_frame(dataframe)
    print("===========================================")
