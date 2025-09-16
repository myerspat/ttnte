import sys
from typing import Optional

import numpy as np
import torch as tn
import torchtt as tntt
import cotengra as ctg

from ttnte.linalg.utils._gen_expr import gen_expr
from ttnte.linalg.operator import Operator


class TTOperator(Operator):
    """
    Tensor train (TT) operator class. This class handles the interface used
    by GMRES. Particularly the operator-vector apply using Cotengra for
    an optimized contraction path.

    Attributes
    ----------
    num_cores: int
        Number of cores in the TT.
    cores: list of torch.Tensor
        The cores in the TT.
    ranks: list of int
        The ranks between the cores.
    output_shape: list of int
        The shape of the resulting vector.
    input_shape: list of int
        The shape of the input vector.
    shape: list of tuple of int
        Mode sizes of the cores.
    nelements: int
        Number of numbers stored in the TT format.
    compression: float
        The compression ratio of the TT format.
    """

    def __init__(self, tt: tntt.TT):
        """
        Build TTOperator.

        Parameters
        ----------
        tt: torchtt.TT
            The tensor train that is used. All cores should be 4-D.
        """
        super().__init__()
        assert tt.is_ttm

        self._cores = tt.clone().cores
        self._cores[0].squeeze_(0)
        self._cores[-1].squeeze_(-1)

        # Make all tensors contiguous
        for i in range(len(self._cores)):
            self._cores[i] = self._cores[i].contiguous()

        # Get shapes
        shape = [tuple(core.shape) for core in self._cores] + [
            (self._cores[0].shape[1], *(core.shape[2] for core in self._cores[1:]))
        ]

        # Generate einsum expression
        self._matvec = ctg.einsum_expression(
            gen_expr(len(self._cores)),
            *shape,
            autojit=True,
            cache=True,
            sort_contraction_indices=True,
        )

    # ========================================================================
    # Public methods

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
        shape = x.shape
        return (
            self._matvec(
                (
                    (self._cores[0])
                    if self.scale == 1.0
                    else (self.scale * self._cores[0])
                ),
                *self._cores[1:],
                x.reshape(
                    (
                        self._cores[0].shape[1],
                        *(core.shape[2] for core in self._cores[1:]),
                    )
                ),
            )
            .reshape(shape)
            .contiguous()
        )

    # Aliases for this method
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
        assert tn.cuda.is_available() and tn.cuda.device_count() > 0

        # Send cores to GPU
        for i in range(len(self._cores)):
            self._cores[i] = self._cores[i].cuda(idx)

    def cpu(self):
        """
        Take operator off GPU.
        """
        # Get cores from GPU
        for i in range(len(self._cores)):
            self._cores[i] = self._cores[i].cpu()

    def round(
        self, eps: float = 1e-12, max_rank=sys.maxsize, gpu_idx: Optional[int] = None
    ):
        """
        Round TT Operator.

        Parameters
        ----------
        eps: float, default=1e-12
            Tolerance for SVD rank truncation.
        max_rank: int, default=sys.maxsize
            Maximum rank for SVD rank truncaation.
        gpu_idx: int or None, default=None
            GPU index for rounding. If it's ``None`` then the rounding is done
            on CPU.
        """
        # Add back rank one ends
        self._cores[0].unsqueeze_(0)
        self._cores[-1].unsqueeze_(3)

        # Place on GPU if requested
        if gpu_idx != None:
            self.cuda(gpu_idx)

        # Create torchtt object
        tt = tntt.TT(self._cores)

        # Run rounding
        tt = tt.round(eps, max_rank)

        # Make new operator
        op = TTOperator(tt)

        # Remove from GPU
        if gpu_idx != None:
            op.cpu()
            self.cpu()

        # Remove rank one ends
        self._cores[0].squeeze_(0)
        self._cores[-1].squeeze_(3)

        return op

    def clone(self):
        """
        Clone operator class. This is a shallow clone.

        Returns
        -------
        clone: ttnte.linalg.TTOperator
            The new clone.
        """
        cores = self._cores
        cores[0].unsqueeze_(0)
        cores[-1].unsqueeze_(3)
        return TTOperator(tntt.TT(cores))

    def add_(self, other):
        """
        Add in-place two operators.

        Parameters
        ----------
        other: ttnte.linalg.TTOperator
            The other operator.
        """
        if not isinstance(other, TTOperator) or self.shape != other.shape:
            raise RuntimeError("Both operators must be TTOperators of the same shape")

        # Create tts
        lcores = self.cores
        rcores = other.cores

        # Add single dimension
        if lcores[0].ndim != 4:
            lcores[0].unsqueeze_(0)
        if rcores[0].ndim != 4:
            rcores[0].unsqueeze_(0)
        if lcores[-1].ndim != 4:
            lcores[-1].unsqueeze_(3)
        if rcores[-1].ndim != 4:
            rcores[-1].unsqueeze_(3)

        self._cores = (
            self.scale * tntt.TT(lcores) + other.scale * tntt.TT(rcores)
        ).cores

        self._cores[0].squeeze_(0)
        self._cores[-1].squeeze_(3)
        self._scale = 1.0

    def to_dense(self):
        """
        Convert TTOperator to a dense tensor.

        Returns
        -------
        result: torch.Tensor
            Resulting tensor.
        """
        # Add back rank one ends
        self._cores[0].unsqueeze_(0)
        self._cores[-1].unsqueeze_(3)

        # Create torchtt object
        tt = tntt.TT(self._cores)
        result = tt.full()

        # Remove rank one ends
        self._cores[0].squeeze_(0)
        self._cores[-1].squeeze_(3)

        return result

    # ========================================================================
    # Getters / Setters

    @property
    def num_cores(self):
        return len(self._cores)

    @property
    def cores(self):
        return self._cores

    @property
    def ranks(self):
        return [core.shape[0] for core in self._cores[1:]]

    @property
    def output_shape(self):
        return [self._cores[0].shape[0]] + [core.shape[1] for core in self._cores[1:]]

    @property
    def input_shape(self):
        return [self._cores[0].shape[1]] + [core.shape[2] for core in self._cores[1:]]

    @property
    def shape(self):
        return [(o, i) for o, i in zip(self.output_shape, self.input_shape)]

    @property
    def nelements(self):
        return sum([c.numel() for c in self._cores])

    @property
    def compression(self):
        return float(
            np.prod([o * i for o, i in zip(self.output_shape, self.input_shape)])
            / self.nelements
        )
