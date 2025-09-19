import torch as tn
import cotengra as ctg

from ttnte.linalg.operator import Operator


class FissionOperator(Operator):
    """
    Fission operator class.

    Attributes
    ----------
    F: torch.Tensor
        Spatial and energy group tensor for fission operator.
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

    def __init__(self, F: tn.Tensor, w_mu: tn.Tensor, w_eta: tn.Tensor):
        """
        Initialize fission operator.

        Parameters
        ----------
        F: torch.Tensor
            Spatial and energy group tensor for fission operator.
        w_mu: torch.Tensor
            Weights along polar cosine.
        w_eta: torch.Tensor
            Weights along azimuthal angle.
        """
        super().__init__()
        assert F.ndim == 2 and w_mu.ndim == 1 and w_eta.ndim == 1

        dtype = F.dtype
        device = F.device

        self._F = F.clone()
        self._w_mu = w_mu.clone()
        self._w_eta = w_eta.clone()

        if (
            self._w_mu.dtype != dtype
            or self._w_eta.dtype != dtype
            or self._w_mu.device != device
            or self._w_eta.device != device
        ):
            raise RuntimeError(
                "F, w_mu, and w_eta must be the same type and on the same device"
            )

        # Check layout
        if (
            self._F.is_sparse
            and self._F.layout != tn.sparse_coo
            and self._F.layout != tn.sparse_csr
        ):
            raise RuntimeError("F must be a dense, COO, or CSR 2-D tensor")

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

        # Angular integration
        result = ctg.einsum(
            "abcd,b,c->d",
            x.reshape((4, self._w_mu.shape[0], self._w_eta.shape[0], -1)),
            self._w_mu,
            self._w_eta,
        )

        # Apply fission operator
        result = (
            ctg.einsum(
                "abc,d->abcd",
                tn.ones(
                    (4, self._w_mu.shape[0], self._w_eta.shape[0]), device=x.device
                ),
                self._F @ result,
            )
            .reshape(shape)
            .contiguous()
        )
        return result if self.scale == 1.0 else self.scale * result

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
        self._F = self._F.cuda(idx)
        self._w_mu = self._w_mu.cuda(idx)
        self._w_eta = self._w_eta.cuda(idx)

    def cpu(self):
        """
        Take operator off GPU.
        """
        self._F = self._F.cpu()
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
        return FissionOperator(self._F, self._w_mu, self._w_eta)

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
        return FissionOperator(
            self._F.to(dtype), self._w_mu.to(dtype), self._w_eta.to(dtype)
        )

    def add_(self, other):
        """
        Not implemented for fission operator.
        """
        raise RuntimeError("This operator does not support addition")

    # ========================================================================
    # Getters / Setters

    @property
    def F(self):
        return self._F

    @property
    def output_shape(self):
        return [4 * self._w_mu.shape[0] * self._w_eta.shape[0] * self._F.shape[0]]

    @property
    def input_shape(self):
        return self.output_shape

    @property
    def shape(self):
        return self.output_shape + self.input_shape

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
        return float(self.output_shape[0] * self.input_shape[0] / self.nelements)

    @property
    def dtype(self):
        return self._F.dtype

    @property
    def device(self):
        return self._F.device
