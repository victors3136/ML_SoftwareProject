from typing import LiteralString, List, Optional

from lib.framework_types import CodeMap, ReverseCodeMap, EncodableColumn, EncodedColumn, DecodedColumn


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


if __name__ == "__main__":
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
