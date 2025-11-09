from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable, Optional, Iterable

from src.lib.frame import Frame


@runtime_checkable
class ClusteringMetric(Protocol):

    @abstractmethod
    def score(
            self,
            data: Frame,
            labels: Iterable[int],
            *,
            centroids: Optional[list[dict[str, float]]] = None,
    ) -> float:
        ...


class NotImplementedMetric:
    def score(
            self,
            data: Frame,
            labels: Iterable[int],
            *,
            centroids: Optional[list[dict[str, float]]] = None,
    ) -> float:
        ...
