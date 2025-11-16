import numpy as np


class Node(np.ndarray):
    KEYS: list[str] = [
        'acousticness',
        'danceability',
        'energy',
        'instrumentalness',
        'key',
        'liveness',
        'loudness',
        'mode',
        'speechiness',
        'tempo',
        'time_signature',
        'valence',
        'duration_ms'
    ]

    def __new__(cls, input_array, *, identifier=None):
        array = np.asarray(input_array)

        if array.shape[0] != len(cls.KEYS):
            raise ValueError(
                f"Input array length {array.shape[0]} does not match number of KEYS {len(cls.KEYS)}"
            )

        node_array = array.view(cls)
        node_array.id = identifier
        return node_array

    @classmethod
    def from_frame_row(cls, row):

        row = list(row.values())
        identifier = row[-1]

        values = row[:-1]

        if len(values) != len(cls.KEYS):
            raise ValueError(
                f"Row contains {len(values)} feature values but Node expects {len(cls.KEYS)}."
            )

        return cls(values, identifier=identifier)

    def __array_finalize__(self, node_array):
        if node_array is None:
            return
        self.id = getattr(node_array, "id", None)

    def __getitem__(self, item):
        if isinstance(item, str):
            try:
                index = self.KEYS.index(item)
            except ValueError:
                raise KeyError(f"Invalid key '{item}'. Valid keys: {self.KEYS}")
            return super().__getitem__(index)
        return super().__getitem__(item)

    def __setitem__(self, item, value):
        if isinstance(item, str):
            try:
                index = self.KEYS.index(item)
            except ValueError:
                raise KeyError(f"Invalid key '{item}'. Valid keys: {self.KEYS}")
            return super().__setitem__(index, value)
        return super().__setitem__(item, value)

    def __getattr__(self, name):
        if name in self.KEYS:
            return self[name]
        raise AttributeError(f"'Node' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self.KEYS:
            self[name] = value
        else:
            super().__setattr__(name, value)

    def as_dict(self):
        return {
            key: self[key]
            for key in self.KEYS
        } | {
            "id": self.id
        }

    @classmethod
    def _wrap(cls, array, *, id_value=None):
        array = np.asarray(array)
        if array.ndim == 1 and array.shape[0] == len(cls.KEYS):
            return cls(array, identifier=id_value)
        # Fallback for multidimensional arrays
        return array

    def _unary_apply(self, operation):
        result = operation(self.view(np.ndarray))
        return Node._wrap(result, id_value=self.id)

    def _binary_apply(self, rhs, operation):
        result = operation(self.view(np.ndarray), rhs)
        return Node._wrap(result, id_value=self.id)

    def _reverse_binary_apply(self, lhs, operation):
        result = operation(lhs, self.view(np.ndarray))
        return Node._wrap(result, id_value=self.id)

    def _inplace_binary_apply(self, rhs, operation):
        operation(self.view(np.ndarray), rhs)
        return self

    def __add__(self, rhs):
        return self._binary_apply(rhs, np.add)

    def __radd__(self, lhs):
        return self._reverse_binary_apply(lhs, np.add)

    def __iadd__(self, rhs):
        return self._inplace_binary_apply(rhs, np.add)

    def __sub__(self, rhs):
        return self._binary_apply(rhs, np.subtract)

    def __rsub__(self, lhs):
        return self._reverse_binary_apply(lhs, np.subtract)

    def __isub__(self, rhs):
        return self._inplace_binary_apply(rhs, np.subtract)

    def __mul__(self, rhs):
        return self._binary_apply(rhs, np.multiply)

    def __rmul__(self, lhs):
        return self._reverse_binary_apply(lhs, np.multiply)

    def __imul__(self, rhs):
        return self._inplace_binary_apply(rhs, np.multiply)

    def __truediv__(self, rhs):
        return self._binary_apply(rhs, np.divide)

    def __rtruediv__(self, lhs):
        return self._reverse_binary_apply(lhs, np.divide)

    def __itruediv__(self, rhs):
        return self._inplace_binary_apply(rhs, np.divide)

    def __floordiv__(self, rhs):
        return self._binary_apply(rhs, np.floor_divide)

    def __rfloordiv__(self, lhs):
        return self._reverse_binary_apply(lhs, np.floor_divide)

    def __ifloordiv__(self, rhs):
        return self._inplace_binary_apply(rhs, np.floor_divide)

    def __pow__(self, rhs, **kwargs):
        return self._binary_apply(rhs, np.power)

    def __rpow__(self, lhs, **kwargs):
        return self._reverse_binary_apply(lhs, np.power)

    def __neg__(self):
        return self._unary_apply(np.negative)

    def __abs__(self):
        return self._unary_apply(np.abs)

    def distance(self, other, metric="euclidean"):
        lhs = self.view(np.ndarray)
        rhs = np.asarray(other)

        if metric == "euclidean":
            return np.linalg.norm(lhs - rhs)
        elif metric == "manhattan":
            return np.sum(np.abs(lhs - rhs))
        elif metric == "cosine":
            return 1 - (np.dot(lhs, rhs) / (np.linalg.norm(lhs) * np.linalg.norm(rhs)))
        else:
            raise ValueError(f"Unknown metric '{metric}'")

    @staticmethod
    def get_mean(cluster, id_value=None):
        arr = np.mean([
            clusterNode.view(np.ndarray)
            for clusterNode in cluster
        ], axis=0)
        return Node(arr, identifier=id_value)

    def __str__(self):
        keys = sorted(self.KEYS + ["id"])
        column_widths = {
                            column_key: max(len(column_key), len(str(self[column_key])))
                            for column_key in self.KEYS
                        } | {
                            "id": max(len("id"), len(str(self.id)))
                        }

        header = " | ".join(
            f"{column_key:{column_widths[column_key]}}"
            for column_key in keys
        )

        separator = "-+-".join("-" * column_widths[column_key]
                               for column_key in keys)

        values = " | ".join(
            f"{str(self[column_key] if column_key in self.KEYS else self.id):{column_widths[column_key]}}"
            for column_key in keys
        )
        lines = [header, separator, values, '']
        return "\n".join(lines)

    def __repr__(self):
        return str(self.as_dict())


if __name__ == "__main__":
    node = Node([1., 2., 3., 4., 5., 6., 7., 8., .9, .10, 1.1, 1 / 2, 1 ** 3], identifier="AAAAAAAAAAAAAAA")
    print(node.acousticness)
    node.acousticness = 2
    print(node[0])
    node[0] = 3
    print(node['acousticness'])
    node['acousticness'] = 4
    print(node * 3)
    print(node)
    print(node * node)
    print(node + node)
    print(node + 20)
