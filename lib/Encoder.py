from numbers import Number
from typing import Iterable, MutableMapping, LiteralString, List, Optional

from pandas import DataFrame

from .Features import ordinal_columns

type CodeMap = MutableMapping[LiteralString, Number]
type ReverseCodeMap = MutableMapping[Number, LiteralString]
type EncodableColumn = Iterable[LiteralString]
type EncodedColumn = Iterable[Number]
type DecodedColumn = EncodableColumn
type ColumnNameCollection = Iterable[str]


class Encoder:
    def __init__(self, reversible: bool = False):
        self.map: CodeMap = {}
        self.reversible = reversible
        self.reverse_map: Optional[ReverseCodeMap] = {} if reversible else None

    def fit(self, collection: EncodableColumn) -> EncodedColumn:
        fitted = []
        for key in collection:
            if key not in self.map:
                value = len(self.map)
                self.map[key] = value
                if self.reverse_map is not None:
                    self.reverse_map[value] = key
            fitted.append(self.map[key])
        return fitted

    def reverse(self, collection: EncodedColumn) -> DecodedColumn:
        if not self.reversible:
            raise TypeError("Encoder is not reversible")
        return [self.reverse_map[value] for value in collection]


class DataFrameEncoder:

    def __init__(self, reversible: bool = False):
        self.reversible = reversible
        self.encoders: dict[str, Encoder] = {}

    def fit(self, dataframe: DataFrame, columns: Optional[ColumnNameCollection] = None) -> DataFrame:
        if columns is None:
            columns = ordinal_columns(dataframe)

        transformed = dataframe.copy()
        for col in columns:
            enc = Encoder(reversible=self.reversible)
            transformed[col] = enc.fit(dataframe[col])
            self.encoders[col] = enc
        return transformed

    def reverse(self, dataframe: DataFrame) -> DataFrame:
        if not self.reversible:
            raise TypeError("Encoder is not reversible")
        restored = dataframe.copy()
        for col, enc in self.encoders.items():
            restored[col] = enc.reverse(dataframe[col])
        return restored


def main() -> None:
    x: List[LiteralString] = ['man', 'woman', 'child', 'man', 'man', 'woman', 'child', 'man', 'child']
    print(f"Before encoding: {x}")
    one_way_encoder = Encoder()
    new_x = one_way_encoder.fit(x)
    print(f"Encoded with one-way encoder: {new_x}")
    two_way_encoder = Encoder(reversible=True)
    new_x = two_way_encoder.fit(x)
    print(f"Encoded with two-way encoder: {new_x}")
    old_x = two_way_encoder.reverse(new_x)
    print(f"Decoded with two-way decoder: {old_x}")
    print(f"Encoding some new values with our trusty encoder: {one_way_encoder.fit(['man', 'woman', 'elder'])}")


if __name__ == "__main__":
    main()
