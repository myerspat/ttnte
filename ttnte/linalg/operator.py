from abc import ABC, abstractmethod
from typing import List

import torch as tn

from ttnte.linalg.linear_operator import LinearOperator


class Operator(ABC):
    def __init__(self):
        self._scale = 1.0

    # ========================================================================
    # Public methods

    @abstractmethod
    def apply(self, x: tn.Tensor) -> tn.Tensor:
        """"""
        pass

    @abstractmethod
    def cuda(self, idx: int):
        """"""
        pass

    @abstractmethod
    def cpu(self):
        """"""
        pass

    @abstractmethod
    def clone(self) -> object:
        """"""
        pass

    @abstractmethod
    def add_(self, other):
        """"""
        raise RuntimeError("This operator does not support addition")

    def add(self, other) -> object:
        """"""
        # Create clone
        new_self = self.clone()
        new_self.add_(other)
        return new_self

    def __add__(self, other) -> LinearOperator:
        """"""
        return LinearOperator(self, other)

    def __sub__(self, other) -> LinearOperator:
        """"""
        return LinearOperator(self, -other)

    def __neg__(self):
        """"""
        new_self = self.clone()
        new_self.scale *= -1.0
        return new_self

    # ========================================================================
    # Getters / Setters

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, other):
        self._scale = other

    @property
    @abstractmethod
    def input_shape(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def output_shape(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def nelements(self) -> int:
        pass

    @property
    @abstractmethod
    def compression(self) -> float:
        pass
