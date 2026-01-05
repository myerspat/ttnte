import numpy as np
import torch as tn

from ttnte.linalg.operator import Operator


class SparseOperator(Operator):
    """
    Sparse operator class for dense, CSR, and COO pytorch tensors.

    Attributes
    ----------
    tensor: torch.Tensor
        The operator in pytorch.
    output_shape: list of int
        Output shape of operator.
    input_shape: list of int
        Input shape of operator
    shape: list of int
        Shape of the operator
    nnz: int
        Number of non-zeros.
    nelements: int
        Number of numbers used to store the operator.
    compression: float
        The compression ratio of this operator.
    is_sparse: bool
        Whether the operator is in a sparse format or not.
    dtype: torch.dtype
        Data type of tensor.
    device: torch.device
        Device the tensor is on.
    """

    def __init__(self, tensor: tn.Tensor):
        """
        Build SparseOperator.

        Parameters
        ----------
        tensor: torch.Tensor
            The operator as a pytorch tensor.
        """
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
        result = (self._tensor @ x.flatten()).reshape(shape).contiguous()
        return result if self.scale == 1.0 else self.scale * result

    # Aliases for this method
    matvec = apply
    __matmul__ = apply

    def cuda(self, idx):
        """
        Put operator on GPU.

        Parameters
        ----------
        idx: int
            GPU index.
        """
        self._tensor = self._tensor.cuda(idx)

    def cpu(self):
        """
        Take operator off GPU.
        """
        self._tensor = self._tensor.cpu()

    def clone(self):
        """
        Clone operator class. This is a shallow clone.

        Returns
        -------
        clone: ttnte.linalg.SpaarseOperator
            The new clone.
        """
        return SparseOperator(self._tensor)

    def add_(self, other):
        """
        Add in-place two operators.

        Parameters
        ----------
        other: ttnte.linalg.SparseOperator
            The other operator.
        """
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
        """
        Convert SparseOperator to a dense tensor.

        Returns
        -------
        result: torch.Tensor
            Resulting tensor.
        """
        return self._tensor.to_dense()

    def type(self, dtype: tn.dtype):
        """
        Cast cores to a different type.

        Parameters
        ----------
        dtype: torch.dtype
            Type to cast to.

        Returns
        -------
        op: ttnte.linalg.SparseOperator
            New operator with casted cores.
        """
        return SparseOperator(self._tensor.to(dtype))

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

    @property
    def dtype(self):
        return self._tensor.dtype

    @property
    def device(self):
        return self._tensor.device
