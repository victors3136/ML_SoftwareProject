from typing import MutableMapping, Any, Iterable, Literal, Callable, TypedDict, Optional

import numpy as np

type FrameData = dict[str, list[Any]]
type Number = (int | float | np.number)

NumberType = (int, float, np.number)

type ColumnType = Literal["continuous", "discrete"]
type FormatterType = Literal["ordinal", "numerical"]
type NormalizerMap = Callable[[Iterable[Number]], Iterable[Number]]
type EncoderMap = Callable[[Iterable[str]], Iterable[Number]]
type FormatterMap = NormalizerMap | EncoderMap

type CodeMap = MutableMapping[str, Number]
type ReverseCodeMap = MutableMapping[Number, str]
type EncodableColumn = Iterable[str]
type EncodedColumn = Iterable[Number]
type DecodedColumn = EncodableColumn
type ColumnNameCollection = Iterable[str]


class FeatureRow(TypedDict):
    column_name: str
    column_type: ColumnType
    lower_bound: Optional[Number]
    upper_bound: Optional[Number]

