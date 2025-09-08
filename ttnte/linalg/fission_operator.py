import torch as tn
import numpy as np
import cotengra as ctg


class FissionOperator:
    def __init__(self, F: tn.Tensor, w_mu: tn.Tensor, w_eta: tn.Tensor):
        """"""
        assert F.ndim == 2 and w_mu.ndim == 1 and w_eta.ndim == 1

        self._F = F
        self._w_mu = w_mu
        self._w_eta = w_eta

        # Check layout
        if (
            self._F.is_sparse
            and self._F.layout != tn.sparse_coo
            and self._F.layout != tn.sparse_csr
        ):
            raise RuntimeError("F must be a dense, COO, or CSR 2-D tensor")

    # ========================================================================
    # Public methods

    def matvec(self, x: tn.Tensor):
        """"""
        # Angular integration
        result = ctg.einsum(
            "abcd,b,c->d",
            x.reshape((4, self._w_mu.shape[0], self._w_eta.shape[0], -1)),
            self._w_mu,
            self._w_eta,
        )

        # Apply fission operator
        return (
            ctg.einsum(
                "abc,d->abcd",
                tn.ones(
                    (4, self._w_mu.shape[0], self._w_eta.shape[0]), device=x.device
                ),
                self._F @ result,
            )
            .reshape(x.shape)
            .contiguous()
        )

    def cuda(self, idx: int):
        """"""
        self._F = self._F.cuda(idx)
        self._w_mu = self._w_mu.cuda(idx)
        self._w_eta = self._w_eta.cuda(idx)

    def cpu(self):
        """"""
        self._F = self._F.cpu()
        self._w_mu = self._w_mu.cpu()
        self._w_eta = self._w_eta.cpu()

    # ========================================================================
    # Overloads

    def __matmul__(self, x: tn.Tensor):
        return self.matvec(x)

    # ========================================================================
    # Getters / Setters

    @property
    def F(self):
        return self._F

    @property
    def output_shape(self):
        return 4 * self._w_mu.shape[0] * self._w_eta.shape[0] * self._F.shape[0]

    @property
    def input_shape(self):
        return self.output_shape

    @property
    def nelements(self):
        nelements = 0

        if not self._F.is_sparse:
            nelements += self._F.numel()

        elif self._F.layout == tn.sparse_csr:
            nelements += (
                self._F.values().numel()
                + self._F.crow_indices().numel()
                + self._F.col_indices().numel()
            )

        else:
            nelements += self._F.values().numel() + self._F.indices().numel()

        return nelements + self._w_mu.numel() + self._w_eta.numel()

    @property
    def compression(self):
        return self.output_shape * self.input_shape / self.nelements
