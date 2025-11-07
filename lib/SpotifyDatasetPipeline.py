from typing import Iterable, Callable, LiteralString, List, Tuple, Optional
import pandas
from pandas import DataFrame

from .Encoder import DataFrameEncoder
from .Normalizer import DataFrameNormalizer
from .Data import LoadData

type PipelineAction = Callable[[DataFrame], DataFrame]
type PipelineStep = Tuple[LiteralString, PipelineAction]
type PipelineFactory = Callable[[Optional[Iterable[str]]], List[PipelineStep]]

_has_column = lambda dataframe: lambda column: column in dataframe.columns


def _make_pipeline(columns_to_drop: Optional[Iterable[str]]) -> List[PipelineStep]:
    columns_to_drop_from = lambda dataframe: filter(
        _has_column(dataframe),
        columns_to_drop or []
    )
    return [
        ("drop duplicates", lambda dataframe: dataframe.drop_duplicates(ignore_index=True)),
        ("drop columns", lambda dataframe: dataframe.drop(columns=columns_to_drop_from(dataframe))),
        ("normalize", lambda dataframe: DataFrameNormalizer.fit(dataframe)),
        ("encode ordinals", lambda dataframe: DataFrameEncoder().fit(dataframe)),
    ]


Pipeline: PipelineFactory = lambda columns_to_drop=None: _make_pipeline(columns_to_drop)


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
    pipeline = Pipeline()
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
