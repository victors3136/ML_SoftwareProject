from typing import Iterable, Callable, LiteralString, List, Tuple, Optional
import pandas
from pandas import DataFrame

from .Encoder import DataFrameEncoder
from .Normalizer import DataFrameNormalizer
from .Data import LoadData

type PipelineAction = Callable[[DataFrame], DataFrame]
type PipelineStep = Tuple[LiteralString, PipelineAction]
type PipelineOptions = Iterable[LiteralString]
type PipelineFactory = Callable[[Optional[Iterable[str]], PipelineOptions], List[PipelineStep]]

_has_column = lambda dataframe: lambda column: column in dataframe.columns


def full_options_included(*options) -> bool:
    return "full" in options[0][0]


def _make_pipeline(discard: Optional[Iterable[str]], *options) -> List[PipelineStep]:
    steps = []
    print(options)
    full_options = full_options_included(options)
    if full_options or "drop columns" in options:
        columns_to_drop_from = lambda dataframe: filter(
            _has_column(dataframe),
            discard or []
        )
        steps.append(("drop columns", lambda dataframe: dataframe.drop(columns=columns_to_drop_from(dataframe))))
    if full_options or "drop duplicates" in options:
        steps.append(("drop duplicates", lambda dataframe: dataframe.drop_duplicates(ignore_index=True)))
    if full_options or "normalize" in options:
        steps.append(("normalize", lambda dataframe: DataFrameNormalizer.fit(dataframe)))
    if full_options or "encode ordinals" in options:
        steps.append(("encode ordinals", lambda dataframe: DataFrameEncoder().fit(dataframe)))
    return steps


Pipeline: PipelineFactory = lambda discard=None, *options: _make_pipeline(discard, options)


def apply(dataframe: DataFrame, pipeline: List[PipelineStep]) -> DataFrame:
    for name, action in pipeline:
        dataframe = action(dataframe)
    return dataframe


def _view_dataframe(dataframe: DataFrame, max_col: int) -> None:
    initial_max_col_disp_sz = pandas.get_option('display.max_columns')
    try:
        pandas.set_option('display.max_columns', max_col)
        print(dataframe.head())
    finally:
        pandas.set_option('display.max_columns', initial_max_col_disp_sz)


def main() -> None:
    print("Downloading and loading Spotify dataset...")
    dataframe = LoadData()
    print(f"Loaded DataFrame with shape: {dataframe.shape}")
    print("===========================================")
    print("Head of the DataFrame before any processing:")
    _view_dataframe(dataframe, 29)
    print("===========================================")
    pipeline = Pipeline(None, "full")
    for name, action in pipeline:
        print(f"Performing '{name}' step...")
        dataframe = action(dataframe)
        print(f"shape: {dataframe.shape}")
    print("===========================================")
    print("Head of the DataFrame after processing:")
    _view_dataframe(dataframe, 29)
    print("===========================================")


if __name__ == "__main__":
    main()
