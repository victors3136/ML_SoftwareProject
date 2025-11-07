from numbers import Number
from typing import Iterable, Optional

import numpy as np
from pandas import DataFrame

from .Features import NormalizableColumns


class Normalizer:
    @staticmethod
    def fit_to_0_1(collection: Iterable[Number]) -> Iterable[Number]:
        collection_min = np.min(collection)
        collection_max = np.max(collection)
        collection_range = collection_max - collection_min
        mapping = lambda number: (number - collection_min) / collection_range
        return collection if collection_range == 0 else list(map(mapping, collection))

    @staticmethod
    def fit_to_neg_1_pos_1(collection: Iterable[Number]) -> Iterable[Number]:
        collection_min = np.min(collection)
        collection_max = np.max(collection)
        collection_range = collection_max - collection_min
        mapping = lambda number: (2 * (number - collection_min) / collection_range) - 1
        return collection if collection_range == 0 else list(map(mapping, collection))


class DataFrameNormalizer:

    @staticmethod
    def fit(dataframe: DataFrame, columns: Optional[Iterable[str]] = None) -> DataFrame:
        if columns is None:
            columns = NormalizableColumns()

        transformed = dataframe.copy()
        for col in columns:
            transformed[col] = Normalizer.fit_to_0_1(dataframe[col])
        return transformed
