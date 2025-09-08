from typing import List
import torch as tn
import numpy as np
import cotengra as ctg


class ScatterOperator:
    def __init__(
        self, S: List[tn.Tensor], Y: tn.Tensor, w_mu: tn.Tensor, w_eta: tn.Tensor
    ):
        """"""
        assert Y.ndim == 4 and w_mu.ndim == 1 and w_eta.ndim == 1

        self._S = S
        self._Y = Y
        self._w_mu = w_mu
        self._w_eta = w_eta

        for s in S:
            assert s.ndim == 2

            # Check layout
            if s.is_sparse and s.layout != tn.sparse_coo and s.layout != tn.sparse_csr:
                raise RuntimeError("S must be a list of dense, COO, or CSR 2-D tensors")

    # ========================================================================
    # Public methods

    def matvec(self, x: tn.Tensor):
        """"""
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
        i = 0
        for n in range(1, len(self._S)):
            for m in range(n + 1):
                result[i,] = self._S[n] @ result[i,]
                i += 1

        # Outer product with spherical harmonics
        return (
            ctg.einsum("ld,labc->abcd", result, self._Y).reshape(x.shape).contiguous()
        )

    def cuda(self, idx: int):
        """"""
        self._S = [s.cuda(idx) for s in self._S]
        self._Y = self._Y.cuda(idx)
        self._w_mu = self._w_mu.cuda(idx)
        self._w_eta = self._w_eta.cuda(idx)

    def cpu(self):
        """"""
        self._S = [s.cpu() for s in self._S]
        self._Y = self._Y.cpu()
        self._w_mu = self._w_mu.cpu()
        self._w_eta = self._w_eta.cpu()

    # ========================================================================
    # Overloads

    def __matmul__(self, x: tn.Tensor):
        return self.matvec(x)

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
        return np.prod(self._Y[1:]) * self._S[0].shape[0]

    @property
    def input_shape(self):
        return self.output_shape

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
        return self.output_shape * self.input_shape / self.nelements
