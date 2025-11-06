from typing import Iterable, Callable
import pandas
from pandas import DataFrame

from lib.OrdinalEncoder import DataFrameOrdinalEncoder
from lib.load_dataset import load_spotify_dataset

has_column = lambda dataframe: lambda column: column in dataframe.columns


def make_pipeline(columns_to_drop: Iterable[str] = None) -> list[tuple[str, Callable[[DataFrame], DataFrame]]]:
    columns_to_drop_from = lambda dataframe: filter(has_column(dataframe),
                                                    list(columns_to_drop) if columns_to_drop else [])
    numerical_columns = lambda dataframe: dataframe.select_dtypes(exclude=['object']).columns.tolist()
    categorical_columns = lambda dataframe: dataframe.select_dtypes(include=['object']).columns.tolist()
    return [
        ("drop duplicates", lambda dataframe: dataframe.drop_duplicates(ignore_index=True)),
        ("drop columns", lambda dataframe: dataframe.drop(columns=columns_to_drop_from(dataframe))),
        # TODO Add normalization for continuous columns
        ("encode categorical columns", lambda dataframe: DataFrameOrdinalEncoder().fit(
            dataframe, categorical_columns(dataframe)
        ))
    ]


def view_df(dataframe: DataFrame, max_col: int) -> None:
    print("Head of the DataFrame before any processing:")
    initial_max_col_disp_sz = pandas.get_option('display.max_columns')
    pandas.set_option('display.max_columns', max_col)
    print(dataframe.head())
    pandas.set_option('display.max_columns', initial_max_col_disp_sz)


def main() -> None:
    print("Downloading and loading Spotify dataset...")
    dataframe = load_spotify_dataset()
    print(f"Loaded DataFrame with shape: {dataframe.shape}")
    view_df(dataframe, 20)
    pipeline = make_pipeline()
    for name, action in pipeline:
        print(f"Performing '{name}' step...")
        dataframe = action(dataframe)
        print(f"shape: {dataframe.shape}")
    view_df(dataframe, 20)


if __name__ == "__main__":
    main()
