import numpy as np
import torch as tn

from ttnte.linalg.operator import Operator


class SparseOperator(Operator):
    def __init__(self, tensor: tn.Tensor):
        """"""
        super().__init__()
        if tensor.ndim > 2:
            raise RuntimeError("Tensors must be 2-D for SparseOperator")

        if not tensor.is_sparse:
            self._tensor = tensor.clone().to_sparse_csr()
        elif tensor.layout == tn.sparse_coo or tensor.layout == tn.sparse_csr:
            self._tensor = tensor.clone()
        else:
            raise RuntimeError("The tensor must be dense, COO, or CSR format")

    # ========================================================================
    # Public methods

    def apply(self, x: tn.Tensor):
        """"""
        shape = x.shape
        result = (self._tensor @ x.flatten()).reshape(shape).contiguous()
        return result if self.scale == 1.0 else self.scale * result

    def matvec(self, x: tn.Tensor):
        """"""
        return self.apply(x)

    def cuda(self, idx):
        """"""
        self._tensor = self._tensor.cuda(idx)

    def cpu(self):
        """"""
        self._tensor = self._tensor.cpu()

    def clone(self):
        return SparseOperator(self._tensor)

    def add_(self, other):
        csr = False
        if (
            self._tensor.layout == tn.sparse_csr
            and other.tensor.layout == tn.sparse_csr
        ):
            self._tensor = self._tensor.to_sparse_coo()
            other._tensor = other._tensor.to_sparse_coo()
            csr = True

        assert self._tensor.layout == other._tensor.layout

        self._tensor = self.scale * self._tensor + other.scale * other.tensor
        self.scale = 1.0

        if csr:
            self._tensor = self._tensor.to_sparse_csr()

    def to_dense(self):
        return self._tensor.to_dense()

    # ========================================================================
    # Overloads

    def __matmul__(self, x: tn.Tensor):
        return self.apply(x)

    # ========================================================================
    # Getters / Setters

    @property
    def tensor(self):
        return self._tensor

    @property
    def output_shape(self):
        return [self._tensor.shape[0]]

    @property
    def input_shape(self):
        return [self._tensor.shape[1]]

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def nnz(self):
        return self._tensor.values().numel()

    @property
    def nelements(self):
        if self._tensor.layout == tn.sparse_coo:
            return int(self.nnz + self._tensor.indices().numel())
        else:
            return int(
                self.nnz
                + self._tensor.crow_indices().numel()
                + self._tensor.col_indices().numel()
            )

    @property
    def compression(self):
        return float(np.prod(self._tensor.shape) / self.nelements)

    @property
    def is_sparse(self):
        return True
