from abc import ABC, abstractmethod
from typing import List

import torch as tn

from ttnte.linalg.linear_operator import LinearOperator


class Operator(ABC):
    """
    Abstract operator class.

    Attributes
    ----------
    scale: float
        Scale result of ``Operator.apply(x)``.
    input_shape: list of int
        Shape of input vector.
    output_shape: list of int
        Shape of output vector.
    nelements: int
        Number of floating point numbers required to hold the operator.
    compression: double
        Compression ratio.
    """

    def __init__(self):
        """
        Initialize abstract Operator.
        """
        self._scale = 1.0

    # ========================================================================
    # Public methods

    @abstractmethod
    def apply(self, x: tn.Tensor) -> tn.Tensor:
        """
        Apply operator to a vector.

        Parameters
        ----------
        x: torch.Tensor
            Input vector.

        Returns
        -------
        y: torch.Tensor
            Output vector.
        """
        pass

    @abstractmethod
    def cuda(self, idx: int):
        """
        Put operator on GPU.

        Parameters
        ----------
        idx: int
            GPU index.
        """
        pass

    @abstractmethod
    def cpu(self):
        """
        Take operator off GPU.
        """
        pass

    @abstractmethod
    def clone(self) -> object:
        """
        Clone operator class. This is a shallow clone.

        Returns
        -------
        clone: ttnte.linalg.Operator
            The new clone.
        """
        pass

    @abstractmethod
    def add_(self, other):
        """
        Add in-place two operators.

        Parameters
        ----------
        other: ttnte.linalg.Operator
            The other operator.
        """
        raise RuntimeError("This operator does not support addition")

    def add(self, other) -> object:
        """"""
        # Create clone
        new_self = self.clone()
        new_self.add_(other)
        return new_self

    def __add__(self, other) -> LinearOperator:
        """
        Add two operators. The result is a ``ttnte.linalg.LinearOperator``
        in which both operators are applied to the same vector and
        summed. To combine common operators run
        ``ttnte.linalg.LinearOperator.combine()``.

        Parameters
        ----------
        other: ttnte.linalg.Operator
            Other operator.

        Returns
        -------
        op: ttnte.linalg.LinearOperator
            New 'summed' operator.
        """
        return LinearOperator(self, other)

    def __sub__(self, other) -> LinearOperator:
        """
        Subtract two operators. The result is a ``ttnte.linalg.LinearOperator``
        in which both operators are applied to the same vector, multiplied by -1,
        and summed. To combine common operators run
        ``ttnte.linalg.LinearOperator.combine()``.

        Parameters
        ----------
        other: ttnte.linalg.Operator
            Other operator.

        Returns
        -------
        op: ttnte.linalg.LinearOperator
            New 'summed' operator.
        """
        return LinearOperator(self, -other)

    def __neg__(self):
        """
        Negate operator.
        """
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
