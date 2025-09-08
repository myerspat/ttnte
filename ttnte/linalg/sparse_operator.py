import numpy as np
import torch as tn


class SparseOperator:
    def __init__(self, tensor: tn.Tensor):
        """"""
        if tensor.ndim > 2:
            raise RuntimeError("Tensors must be 2-D for SparseOperator")

        if not tensor.is_sparse:
            self._tensor = tensor.to_sparse_csr()
        elif tensor.layout == tn.sparse_coo or tensor.layout == tn.sparse_csr:
            self._tensor = tensor
        else:
            raise RuntimeError("The tensor must be dense, COO, or CSR format")

        self._tensor = tensor if tensor.is_sparse_csr else tensor.to_sparse_csr()

    # ========================================================================
    # Public methods

    def matvec(self, x: tn.Tensor):
        """"""
        return self._tensor @ x.flatten()

    def cuda(self, idx):
        """"""
        self._tensor = self._tensor.cuda(idx)

    def cpu(self):
        """"""
        self._tensor = self._tensor.cpu()

    # ========================================================================
    # Overloads

    def __matmul__(self, x: tn.Tensor):
        return self.matvec(x)

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
            return self.nnz + self._tensor.indices()
        else:
            return (
                self.nnz
                + self._tensor.crow_indices().numel()
                + self._tensor.col_indices().numel()
            )

    @property
    def compression(self):
        return np.prod(self._tensor.shape) / self.nelements

    @property
    def is_sparse(self):
        return True
