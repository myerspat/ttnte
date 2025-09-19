from typing import List
import torch as tn
import numpy as np
import cotengra as ctg

from ttnte.linalg.operator import Operator


class ScatterOperator(Operator):
    """
    Scatter operator class.

    Attributes
    ----------
    S: torch.Tensor
        Spatial and energy group potion of the operator.
    Y: torch.Tensor
        Moment information for higher order scattering.
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

    def __init__(
        self, S: List[tn.Tensor], Y: tn.Tensor, w_mu: tn.Tensor, w_eta: tn.Tensor
    ):
        """
        Initialize scattering operator.

        Parameters
        ----------
        S: torch.Tensor
            Spatial and energy group potion of the operator.
        Y: torch.Tensor
            Moment information for higher order scattering.
        w_mu: torch.Tensor
            Weights along polar cosine.
        w_eta: torch.Tensor
            Weights along azimuthal angle.
        """
        super().__init__()
        assert Y.ndim == 4 and w_mu.ndim == 1 and w_eta.ndim == 1

        self._S = [s.clone() for s in S]
        self._Y = Y.clone()
        self._w_mu = w_mu.clone()
        self._w_eta = w_eta.clone()

        for s in S:
            assert s.ndim == 2

            # Check layout
            if s.is_sparse and s.layout != tn.sparse_coo and s.layout != tn.sparse_csr:
                raise RuntimeError("S must be a list of dense, COO, or CSR 2-D tensors")

    # ========================================================================
    # Public methods

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
        shape = x.shape

        # Apply spherical harmonics and angular integration
        result = ctg.einsum(
            "abcd,labc,b,c->ld",
            x.reshape((*self._Y.shape[1:], -1)),
            self._Y,
            self._w_mu,
            self._w_eta,
        )

        # Calculate first moment
        result[0,] = self._S[0] @ result[0,]

        # Compute remaining moments
        i = 1
        for n in range(1, len(self._S)):
            for _ in range(n + 1):
                result[i,] = self._S[n] @ result[i,]
                i += 1

        # Outer product with spherical harmonics
        result = (
            ctg.einsum("ld,labc->abcd", result, self._Y).reshape(shape).contiguous()
        )
        return result if self.scale == 1.0 else (self.scale * result)

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
        self._S = [s.cuda(idx) for s in self._S]
        self._Y = self._Y.cuda(idx)
        self._w_mu = self._w_mu.cuda(idx)
        self._w_eta = self._w_eta.cuda(idx)

    def cpu(self):
        """
        Take operator off GPU.
        """
        self._S = [s.cpu() for s in self._S]
        self._Y = self._Y.cpu()
        self._w_mu = self._w_mu.cpu()
        self._w_eta = self._w_eta.cpu()

    def clone(self):
        """
        Clone operator class. This is a shallow clone.

        Returns
        -------
        clone: ttnte.linalg.ScatterOperator
            The new clone.
        """
        return ScatterOperator(self._S, self._Y, self._w_mu, self._w_eta)

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
        S = []

        for s in self._S:
            S.append(s.to(dtype))

        return ScatterOperator(
            S, self._Y.to(dtype), self._w_mu.to(dtype), self._w_eta.to(dtype)
        )

    def add_(self, other):
        """
        Not implemented for scattering operator.
        """
        raise RuntimeError("This operator does not support addition")

    # ========================================================================
    # Getters / Setters

    @property
    def S(self):
        return self._S

    @property
    def Y(self):
        return self._Y

    @property
    def output_shape(self):
        return [int(np.prod(self._Y.shape[1:]) * self._S[0].shape[0])]

    @property
    def input_shape(self):
        return self.output_shape

    @property
    def shape(self):
        return self.output_shape + self.input_shape

    @property
    def nelements(self):
        nelements = 0

        for s in self._S:
            if not s.is_sparse:
                nelements += s.numel()

            elif s.layout == tn.sparse_csr:
                nelements += (
                    s.values().numel()
                    + s.crow_indices().numel()
                    + s.col_indices().numel()
                )

            else:
                nelements += s.values().numel() + s.indices().numel()

        return nelements + self._Y.numel() + self._w_mu.numel() + self._w_eta.numel()

    @property
    def compression(self):
        return float(self.output_shape[0] * self.input_shape[0] / self.nelements)

    @property
    def dtype(self):
        return self._Y.dtype

    @property
    def device(self):
        return self._Y.device
