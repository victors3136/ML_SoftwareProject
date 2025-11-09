from typing import Optional

from lib.encoding import Encoder
from lib.features import ordinal_columns
from lib.frame import Frame
from lib.framework_types import ColumnNameCollection


class DataFrameEncoder:

    def __init__(self, reversible: bool = False):
        self.reversible = reversible
        self.encoders: dict[str, Encoder] = {}

    def fit(self, dataframe: Frame, columns: Optional[ColumnNameCollection] = None) -> Frame:
        if columns is None:
            columns = ordinal_columns(dataframe)

        transformed = dataframe
        for column in columns:
            encoder = Encoder(reversible=self.reversible)
            transformed[column] = encoder.fit(dataframe[column])
            self.encoders[column] = encoder
        return transformed

    def reverse(self, dataframe: Frame) -> Frame:
        if not self.reversible:
            raise TypeError("Encoder is not reversible")
        restored = dataframe
        for column, encoder in self.encoders.items():
            restored[column] = encoder.reverse(dataframe[column])
        return restored
