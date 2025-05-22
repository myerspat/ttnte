from typing import List, Optional, Tuple, Union

import cotengra as ctg
import numpy as np
import torchtt as tntt

from ttnte.assemblers.operators import SparseOperator


class LinearOperator(object):
    """
    Linear operator class for ``(N, M)`` operator.

    Attributes
    ----------
    N: tuple of int
        Output shape.
    M: tuple of int
        Input shape.
    shape: list of tuple of int
        Shape of operator.
    """

    def __init__(
        self,
        ops: List[Union[tntt.TT, SparseOperator]],
        N: Tuple[int],
        M: Tuple[int],
        P: Optional[Union[tntt.TT, SparseOperator]] = None,
    ):
        """
        Initialize linear operator class.

        Parameters
        ----------
        ops: list of torchtt.TT or ttnte.assemblers.operators.SparseOperator
            List of operators. Each is applied and the results are summed.
        N: tuple of int
            Output shape.
        M: tuple of int
            Input shape.
        P: torchtt.TT, ttnte.assemblers.operators.SparseOperator, or None
            Preconditioner if applicable.
        """
        assert len(M) == len(N)
        self._ops = ops
        self._N = N
        self._M = M

        # Get correct linear operator method
        if P is not None:
            self._ops += [P]
            self._matvec_method = LinearOperator._preconditioned
        else:
            self._matvec_method = LinearOperator._nonpreconditioned

        # Create expressions
        self._update_expressions()

    # ========================================================================
    # Methods

    def matvec(self, x):
        """
        Apply operator to ``x``.

        Parameters
        ----------
        x: torch.Tensor
            Input vector.

        Returns
        Ax: torch.Tensor
            Flattened output vector.
        """
        return self._matvec_method(self._exprs, x.reshape(self._N)).reshape((-1, 1))

    def cuda(self, device):
        """
        Send operator to GPU.

        Parameters
        ----------
        device: int
            Index of device to send to.

        Returns
        -------
        operator: ttnte.linalg.LinearOperator
            Operator on GPU.
        """
        self._ops = [op.cuda(device) for op in self._ops]
        self._update_expressions()
        return self

    def cpu(self):
        """
        Get operator from GPU.

        Returns
        -------
        operator: ttnte.linalg.LinearOperator
            Operator taken from GPU.
        """
        self._ops = [op.cpu() for op in self._ops]
        self._update_expressions()
        return self

    def _update_expressions(self):
        """
        Create operator application method.

        This uses traditional ``@`` for
        ``ttnte.assemblers.operators.SparseOperator``s and uses ``cotengra``
        for ``torch.TT``s.
        """
        self._exprs = []
        for i in range(len(self._ops)):
            if isinstance(self._ops[i], tntt.TT):
                self._exprs.append(
                    ctg.contract_expression(
                        self._gen_expression(self._ops[i]),
                        *(
                            [self._ops[i].cores[0][0, ...]]
                            + self._ops[i].cores[1:-1]
                            + [self._ops[i].cores[-1][..., 0]]
                            + [tuple(s[0] for s in self._ops[i].shape)]
                        ),
                        constants=np.arange(len(self._ops[i].cores)),
                    )
                )

            elif isinstance(self._ops[i], SparseOperator):
                self._exprs.append(lambda x: self._ops[i] @ x)

            else:
                raise RuntimeError(
                    "Operator must be tntt.TT, tn.Tensor, ScatteringOperator, "
                    + "FissionOperator"
                )

    @staticmethod
    def _gen_expression(op):
        """
        Generate einsum expression for TT operator.

        Parameters
        ----------
        op: torch.TT
            TT operator.

        Returns
        -------
        ex: str
            Einsum expression.
        """
        # Generate matrix-vector contraction expressions
        ex = ""
        inn = ""
        out = ""
        idx = 0
        for _ in op.cores:
            # Add core to expression
            chars = [ctg.get_symbol(i) for i in range(idx, idx + 4)]
            ex += "".join(chars) + ","

            # Save indices for hitting vector
            inn += chars[2]
            out += chars[1]

            idx += 3

        ex = ex[1:-2] + ","
        ex += f"{inn}->{out}"

        return ex

    @staticmethod
    def _preconditioned(exprs, x):
        """
        Compute preconditioned application of this operator.

        Parameters
        ----------
        x: tn.Tensor
            Input tensor.

        Returns
        -------
        Ax: tn.Tensor
            Flattened output tensor.
        """
        return exprs[-1](LinearOperator._nonpreconditioned(exprs[:-1], x))

    @staticmethod
    def _nonpreconditioned(exprs, x):
        """
        Compute nonpreconditioned application of this operator.

        Parameters
        ----------
        x: tn.Tensor
            Input tensor.

        Returns
        -------
        Ax: tn.Tensor
            Flattened output tensor.
        """
        return sum(expr(x).flatten() for expr in exprs)

    # ========================================================================
    # Operations

    def __matmul__(self, x):
        """
        Apply operator to ``x``.

        Parameters
        ----------
        x: torch.Tensor
            Input vector.

        Returns
        Ax: torch.Tensor
            Flattened output vector.
        """
        return self.matvec(x)

    # ========================================================================
    # Properties

    @property
    def N(self):
        return self._N

    @property
    def M(self):
        return self._M

    @property
    def shape(self):
        return [(self._N[i], self._M[i]) for i in range(len(self._N))]
