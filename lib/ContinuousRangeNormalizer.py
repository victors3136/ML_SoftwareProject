from numbers import Number
from typing import Iterable

import numpy as np


class ContinuousRangeNormalizer:
    @staticmethod
    def fit_to_0_1(collection: Iterable[Number]) -> Iterable[Number]:
        collection_min = np.min(collection)
        collection_max = np.max(collection)
        collection_range = collection_max - collection_min
        mapping = lambda number: (number - collection_min) / collection_range
        return map(mapping, collection)

    @staticmethod
    def fit_to_neg_1_pos_1(collection: Iterable[Number]) -> Iterable[Number]:
        collection_min = np.min(collection)
        collection_max = np.max(collection)
        collection_range = collection_max - collection_min
        mapping = lambda number: (2 * (number - collection_min) / collection_range) - 1
        return map(mapping, collection)
