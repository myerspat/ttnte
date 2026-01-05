from functools import reduce
from typing import Optional
import sys

import numpy as np
import torch as tn


class LinearOperator:
    """
    Linear operator class.

    Attributes
    ----------
    operators: list of ttnte.linalg.Operator
        List of operators in linear operator.
    input_shape: list of int
        Shape of input vector.
    output_shape: list of int
        Shape of output vector.
    shape: list of int
        Shape of the scattering operator.
    nelements: int
        Number of floating point numbers required to hold the operator.
    compression: double
        Compression ratio.
    dtype: torch.dtype
        Data type of operator.
    device: torch.device
        Device the operator is on.
    """

    def __init__(self, *operators):
        """
        Initialize linear operator.

        *operators: list of ttnte.linalg.Operator
            List of operators in linear operator.
        """
        # Set operators
        self._operators = list(operators)

        # Check the size of input and output
        input_size = np.prod(self._operators[0].input_shape)
        output_size = np.prod(self._operators[0].output_shape)

        for op in self._operators[1:]:
            if input_size != np.prod(op.input_shape):
                raise RuntimeError("Size of input must be the same for each operator")
            if output_size != np.prod(op.output_shape):
                raise RuntimeError("Size of output must be the same for each operator")

    # ========================================================================
    # Public methods

    def append(self, op):
        """
        Add operator(s) to the back of the list.

        Parameters
        ----------
        op: ttnte.linalg.Operator
            Operator to be added.
        """
        op = op if isinstance(op, list) else [op]
        self._operators += op

    def prepend(self, op):
        """
        Add operator(s) to the front of the list.

        Parameters
        ----------
        op: ttnte.linalg.Operator
            Operator to be added.
        """
        op = op if isinstance(op, list) else [op]
        self._operators = op + self._operators

    def apply(self, x: tn.Tensor):
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
        result = self._operators[0].apply(x)
        for op in self._operators[1:]:
            result += op.apply(x)
        return result.reshape((-1, 1))

    # Other aliases
    matvec = apply
    __matmul__ = apply

    def cuda(self, idx: int):
        """
        Put operator on GPU.

        Parameters
        ----------
        idx: int
            GPU index.
        """
        for op in self._operators:
            op.cuda(idx)

    def cpu(self):
        """
        Take operator off GPU.
        """
        for op in self._operators:
            op.cpu()

    def combine(self):
        """
        Combine common operators.

        Returns
        -------
        op: ttnte.linalg.LinearOperator
            Combined operator.
        """
        # Group operators by type
        grouped = {}
        for op in self._operators:
            op_type = type(op)
            if op_type in grouped:
                grouped[op_type].append(op)
            else:
                grouped[op_type] = [op]

        # Sum operators
        new_ops = []
        for ops in grouped.values():
            if len(ops) == 1:
                new_ops.append(ops[0])
            else:
                new_ops.append(reduce(lambda lop, rop: lop.add(rop), ops))

        return LinearOperator(*new_ops)

    def round(
        self, eps: float = 1e-12, max_rank=sys.maxsize, gpu_idx: Optional[int] = None
    ):
        """
        Combine and round all ``ttnte.linalg.TTOperator``(s) in this linear operator.

        Parameters
        ----------
        eps: float, default=1e-12
            Tolerance for SVD truncation.
        max_rank: int, default=sys.maxsize
            Maximum rank in SVD truncation.
        gpu_idx: int or None, default=None
            Device index to run round on.

        Returns
        -------
        op: ttnte.linalg.LinearOperator
            Combined and rounded linear operator.
        """
        from ttnte.linalg.tt_operator import TTOperator

        new_ops = []
        tt = None

        for op in self._operators:
            if isinstance(op, TTOperator):
                tt = op if tt is None else tt.add(op)
            else:
                new_ops.append(op)

        # Round TT
        if tt is not None:
            new_ops.append(tt.round(eps, max_rank, gpu_idx))

        return LinearOperator(*new_ops)

    def clone(self):
        """
        Clone operator class. This is a shallow clone.

        Returns
        -------
        clone: ttnte.linalg.LinearOperator
            The new clone.
        """
        return LinearOperator(*[op.clone() for op in self._operators])

    def type(self, dtype: tn.dtype):
        """
        Cast cores to a different type.

        Parameters
        ----------
        dtype: torch.dtype
            Type to cast to.

        Returns
        -------
        op: ttnte.linalg.FissionOperator
            New operator with casted cores.
        """
        ops = []

        for op in self._operators:
            ops.append(op.type(dtype))

        return LinearOperator(ops)

    def set_scale(self, scale):
        """
        Set scale of operators.

        Parameters
        ----------
        scale: float
            Scaler to multiply operators by.
        """
        # Set scale for operators
        for i in range(len(self._operators)):
            self._operators[i].scale *= scale

    # ========================================================================
    # Overloads

    def __add__(self, other):
        """
        Add operators.

        Parameters
        ----------
        other: ttnte.linalg.Operator or ttnte.linalg.LinearOperator
            Add operators into one linear operator.

        Returns
        -------
        op: ttnte.linalg.LinearOperator
            New linear operator.
        """
        # Check if the other is another linear operator
        if isinstance(other, LinearOperator):
            # Clone operators and append
            self._operators += [op.clone() for op in other.operators]
        else:
            # Append operators
            self._operators.append(other)

        return self

    def __sub__(self, other):
        """
        Subtract operators.

        Parameters
        ----------
        other: ttnte.linalg.Operator or ttnte.linalg.LinearOperator
            Subtract operators into one linear operator.

        Returns
        -------
        op: ttnte.linalg.LinearOperator
            New linear operator.
        """
        # Check if the other is another linear operator
        if isinstance(other, LinearOperator):
            # Clone ops, flip scale, and append
            for op in other.operators:
                new_op = op.clone()
                new_op.set_scale(-1.0)
                self._operators.append(new_op)

        else:
            # Clone and flip scale
            new_op = other.clone()
            new_op.scale *= -1.0
            self._operators.append(new_op)

        return self

    def __neg__(self):
        """
        Negate operator.
        """
        new_self = self.clone()
        new_self.set_scale(-1)
        return new_self

    # ========================================================================
    # Getters / Setters

    @property
    def operators(self):
        return self._operators

    @property
    def input_shape(self):
        return [np.prod(self._operators[0].input_shape)]

    @property
    def output_shape(self):
        return [np.prod(self._operators[0].output_shape)]

    @property
    def shape(self):
        return self.output_shape + self.input_shape

    @property
    def nelements(self):
        return sum([op.nelements for op in self._operators])

    @property
    def compression(self):
        return np.prod(self.input_shape + self.output_shape) / self.nelements

    @property
    def dtype(self):
        return self._operators[0].dtype

    @property
    def device(self):
        return self._operators[0].device
