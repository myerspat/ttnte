import numpy as np
import torch as tn
import torchtt as tntt
import cotengra as ctg

from ttnte.linalg.utils._gen_expr import gen_expr


class TTOperator:
    def __init__(self, tt: tntt.TT):
        """"""
        assert tt.is_ttm

        self._cores = tt.cores
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

    def matvec(self, x: tn.Tensor):
        """"""
        return self._matvec(*self._cores, x).contiguous()

    def cuda(self, idx):
        assert tn.cuda.is_available() and tn.cuda.device_count() > 0

        # Send cores to GPU
        for i in range(len(self._cores)):
            self._cores[i] = self._cores[i].cuda(idx)

    def cpu(self):
        # Get cores from GPU
        for i in range(len(self._cores)):
            self._cores[i] = self._cores[i].cpu()

    # ========================================================================
    # Overloads

    def __matmul__(self, x: tn.Tensor):
        return self.matvec(x)

    # ========================================================================
    # Getters / Setters

    @property
    def num_cores(self):
        return len(self._cores)

    @property
    def cores(self):
        return self._cores

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
        return (
            np.prod([o * i for o, i in zip(self.output_shape, self.input_shape)])
            / self.nelements
        )
