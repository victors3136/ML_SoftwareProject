from numbers import Number
from typing import Iterable, Any, MutableMapping
from pandas import DataFrame

class OrdinalEncoder:
    def __init__(self, reversible: bool = False):
        self.map: MutableMapping[Any, Number] = {}
        self.reversible = reversible
        self.reverse_map: MutableMapping[Number, Any] | None = {} if reversible else None

    def fit(self, collection: Iterable[Any]) -> Iterable[Number]:
        fitted = []
        for key in collection:
            if key not in self.map:
                value = len(self.map)
                self.map[key] = value
                if self.reverse_map is not None:
                    self.reverse_map[value] = key
            fitted.append(self.map[key])
        return fitted

    def reverse(self, collection: Iterable[Number]) -> Iterable[Any]:
        if not self.reversible:
            raise TypeError("Encoder is not reversible")
        return [self.reverse_map[value] for value in collection]

class DataFrameOrdinalEncoder:

    def __init__(self, reversible: bool = False):
        self.reversible = reversible
        self.encoders: dict[str, OrdinalEncoder] = {}

    def fit(self, dataframe: DataFrame, columns: Iterable[str] | None = None) -> DataFrame:
        if columns is None:
            columns = dataframe.columns

        transformed = dataframe.copy()
        for col in columns:
            enc = OrdinalEncoder(reversible=self.reversible)
            transformed[col] = enc.fit(dataframe[col])
            self.encoders[col] = enc
        return transformed

    def reverse(self, df: DataFrame) -> DataFrame:
        if not self.reversible:
            raise TypeError("Encoder is not reversible")
        restored = df.copy()
        for col, enc in self.encoders.items():
            restored[col] = enc.reverse(df[col])
        return restored

def main() -> None:
    x = ['man', 'woman', 'child', 'man', 'man', 'woman', 'child', 'man', 'child']
    print(f"Before encoding: {x}")
    one_way_encoder = OrdinalEncoder()
    new_x = one_way_encoder.fit(x)
    print(f"Encoded with one-way encoder: {new_x}")
    two_way_encoder = OrdinalEncoder(reversible=True)
    new_x = two_way_encoder.fit(x)
    print(f"Encoded with two-way encoder: {new_x}")
    old_x = two_way_encoder.reverse(new_x)
    print(f"Decoded with two-way decoder: {old_x}")
    print(f"Encoding some new values with our trusty encoder: {one_way_encoder.fit(['man', 'woman', 'elder'])}")

    data = DataFrame({
        "color": ["red", "blue", "green", "red", "blue"],
        "size": ["S", "M", "L", "S", "XL"]
    })

    encoder = DataFrameOrdinalEncoder(reversible=True)
    encoded = encoder.fit(data)
    print("Encoded:\n", encoded)

    decoded = encoder.reverse(encoded)
    print("\nDecoded:\n", decoded)

if __name__ == "__main__":
    main()
