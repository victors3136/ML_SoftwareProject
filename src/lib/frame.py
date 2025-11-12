from typing import Iterable, Optional, Any
import csv
import random
from pathlib import Path

from lib.framework_types import FrameData, Number, NumberType


class Frame:

    def __init__(self, data: Optional[FrameData] = None, shuffle: bool = False):
        data: FrameData = data or {}
        lengths: set[int] = set(len(col) for col in data.values())
        if len(lengths) > 1:
            raise ValueError("All columns must have the same length")
        self._data: FrameData = dict(data)
        self._column_keys: list[str] = list(data.keys())
        self._row_count: int = next(iter(lengths)) if lengths else 0
        if shuffle:
            self._inplace_shuffle()

    @property
    def columns(self) -> list[str]:
        return list(self._column_keys)

    @property
    def shape(self) -> tuple[Number, Number]:
        return self._row_count, len(self._column_keys)

    def __getitem__(self, key: str | int | range) -> dict[str, Any] | list[dict[str, Any]]:
        match key:
            case str():
                return self._data[key]
            case int():
                return self.get_row(key)
            case range():
                return [
                    self.get_row(index) for index in range(key.start, key.stop, key.step)
                ]

    def get_row(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self._row_count:
            raise IndexError("Index out of range")
        return {
            column_key: values[index]
            for column_key, values in self._data.items()
        }

    def __setitem__(self, column: str, values: Iterable[Any]) -> None:
        values_list = list(values)
        if self._column_keys:
            if len(values_list) != self._row_count:
                raise ValueError("Assigned column length must match number of rows")
        else:
            self._row_count = len(values_list)
        if column not in self._column_keys:
            self._column_keys.append(column)
        self._data[column] = values_list

    def head(self, row_count: int = 5) -> Frame:
        row_count = max(0, min(row_count, self._row_count))
        return Frame({
            column_key: self._data[column_key][:row_count]
            for column_key in self._column_keys
        })

    def drop(self, *, columns_to_drop: Iterable[str]) -> Frame:
        columns_to_drop = set(columns_to_drop)
        return Frame({
            column_key: value
            for column_key, value in self._data.items()
            if column_key not in columns_to_drop
        })

    def drop_duplicates(self) -> Frame:
        if not self._column_keys:
            return Frame({})
        seen: set[tuple[Any, ...]] = set()
        keep_indices: list[int] = []
        for row_id in range(self._row_count):
            row_tuple = tuple(
                self._data[column_key][row_id]
                for column_key in self._column_keys
            )
            if row_tuple not in seen:
                seen.add(row_tuple)
                keep_indices.append(row_id)
        return Frame({
            column_key: [
                self._data[column_key][index]
                for index in keep_indices
            ] for column_key in self._column_keys
        })

    def numeric_columns(self) -> list[str]:
        numeric_column_keys: list[str] = []
        for column_key in self._column_keys:
            column_data = self._data[column_key]
            values = [
                value for value in column_data
                if value is not None
            ]
            if values and all(isinstance(value, NumberType)
                              for value in values):
                numeric_column_keys.append(column_key)
        return numeric_column_keys

    def column_means(self, columns: Optional[Iterable[str]] = None) -> dict[str, float]:
        column_keys = list(columns) or self.numeric_columns()
        means: dict[str, float] = {}
        for column_key in column_keys:
            values = [
                value
                for value in self._data[column_key]
                if isinstance(value, NumberType)
            ]
            if values:
                means[column_key] = sum(values) / len(values)
        return means

    def __str__(self) -> str:
        row_count, column_count = self.shape
        max_previewed_rows = 5
        rows_to_show = min(row_count, max_previewed_rows)

        column_widths = {
            column_key: max(len(column_key), *(
                len(str(value))
                for value in self._data[column_key][:rows_to_show]
            )) for column_key in self._column_keys}

        header = " | ".join(
            f"{column_key:{column_widths[column_key]}}"
            for column_key in self._column_keys
        )

        separator = "-+-".join("-" * column_widths[column_key]
                               for column_key in self._column_keys)

        lines = [header, separator]
        for row_index in range(rows_to_show):
            line = " | ".join(
                f"{str(self._data[column_key][row_index]):{column_widths[column_key]}}"
                for column_key in self._column_keys
            )
            lines.append(line)
        if row_count > rows_to_show:
            lines.append(f"... and {row_count - rows_to_show} more rows")
        return "\n".join(lines)

    def _inplace_shuffle(self) -> None:
        if self._row_count <= 1 or not self._column_keys:
            return

        indices = list(range(self._row_count))
        random.shuffle(indices)

        for column_key in self._column_keys:
            column = self._data[column_key]
            self._data[column_key] = [
                column[index] for index in indices
            ]

    def shuffle(self) -> Frame:
        if self._row_count == 0 or not self._column_keys:
            return Frame({})
        indices = list(range(self._row_count))
        random.shuffle(indices)
        shuffled_data: FrameData = {
            column_key: [
                self._data[column_key][index]
                for index in indices
            ]
            for column_key in self._column_keys
        }
        return Frame(shuffled_data)

    @staticmethod
    def concatenate(frames: list[Frame], *, shuffle: bool = False) -> Frame:
        if not frames:
            return Frame()
        columns: list[str] = []
        seen_columns = set()
        for frame in frames:
            for column in frame.columns:
                if column not in seen_columns:
                    seen_columns.add(column)
                    columns.append(column)
        data: FrameData = {
            column_keys: [] for column_keys in columns
        }
        for frame in frames:
            row_count = frame.shape[0]
            for column in columns:
                column_data = frame._data.get(column)
                if column_data is None:
                    data[column].extend([None] * row_count)
                else:
                    data[column].extend(column_data)
        return Frame(data, shuffle=shuffle)

    @staticmethod
    def from_csv(path: Path) -> Frame:
        with path.open(encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            rows: list[dict[str, Any]] = []
            for row in reader:
                rows.append({
                    column_key: _parse_value(value)
                    for column_key, value in row.items()
                })
        column_keys: list[str] = []
        column_set: set[str] = set()
        for row in rows:
            for column_key in row.keys():
                if column_key not in column_set:
                    column_set.add(column_key)
                    column_keys.append(column_key)

        data: FrameData = {column_key: [] for column_key in column_keys}
        for row in rows:
            for column_key in column_keys:
                data[column_key].append(row.get(column_key))
        return Frame(data)


def _parse_value(to_parse: str) -> Any:
    if to_parse == "" or to_parse is None:
        return None
    try:
        if to_parse.isdigit() or (
                to_parse.startswith('-') and to_parse[1:].isdigit()):
            return int(to_parse)
        return float(to_parse)
    except ValueError:
        return to_parse
