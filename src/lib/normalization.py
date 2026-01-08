from numbers import Number
from typing import Iterable, Optional, LiteralString

import numpy as np

from lib.frame import Frame
from lib.features import normalizable_columns, get_feature_info
from lib.framework_types import NumberType


class Normalizer:
    @staticmethod
    def get_column_features(collection: Iterable[Number | None]) -> Optional[tuple[Number, Number, Number]]:
        values = [value
                  for value in collection
                  if isinstance(value, NumberType)]
        if not values:
            return None
        collection_min = float(np.min(values))
        collection_max = float(np.max(values))
        collection_range = collection_max - collection_min

        return collection_min, collection_max, collection_range

    @staticmethod
    def fit_to_0_1(collection: Iterable[Number | None]) -> Iterable[Number | None]:
        features = Normalizer.get_column_features(collection)

        if features is None:
            return list(collection)

        collection_min, collection_max, collection_range = features

        if collection_range == 0:
            return list(collection)

        def mapping(number: Number | None) -> Number | None:
            if not isinstance(number, NumberType):
                return number
            # noinspection PyTypeChecker
            return (float(number) - collection_min) / collection_range

        return list(map(mapping, collection))

    @staticmethod
    def fit_to_normal_distribution(collection: Iterable[Number | None]) -> Iterable[Number | None]:
        values = [value for value in collection if isinstance(value, NumberType)]
        if not values:
            return list(collection)

        mean = float(np.mean(values))
        standard_deviation = float(np.std(values))

        if standard_deviation == 0:
            return list(collection)

        def mapping(number: Number | None) -> Number | None:
            if not isinstance(number, NumberType):
                return number
            # noinspection PyTypeChecker
            return (float(number) - mean) / standard_deviation

        return list(map(mapping, collection))

    @staticmethod
    def fit_to_neg_1_pos_1(collection: Iterable[Number | None]) -> Iterable[Number | None]:
        features = Normalizer.get_column_features(collection)

        if features is None:
            return list(collection)

        collection_min, collection_max, collection_range = features

        if collection_range == 0:
            return list(collection)

        def mapping(number: Number | None) -> Number | None:
            if not isinstance(number, NumberType):
                return number
            # noinspection PyTypeChecker
            return (2 * (float(number) - collection_min) / collection_range) - 1

        return list(map(mapping, collection))


class DataFrameNormalizer:

    @staticmethod
    def column_already_normalized(column_name: LiteralString) -> bool:
        info = get_feature_info(column_name)
        assert info is not None, f"Invalid row info {info}"
        if info['lower_bound'] is None or info['upper_bound'] is None:
            return False
        return info['lower_bound'] >= 0 and info['upper_bound'] <= 1

    @staticmethod
    def fit(dataframe: Frame, columns: Optional[Iterable[str]] = None) -> Frame:
        if columns is None:
            columns = normalizable_columns(dataframe)

        transformed = dataframe

        for column in columns:
            # noinspection PyTypeChecker
            transformed[column] = Normalizer.fit_to_normal_distribution(dataframe[column])
        return transformed
