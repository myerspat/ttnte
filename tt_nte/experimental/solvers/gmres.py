"""
gmres.py.

GMRES for TNLinearOperators.
"""

from typing import Union, Optional, Callable

import numpy as np
import cupy as cu
from scipy.sparse.linalg import gmres as sc_gmres
from cupyx.scipy.sparse.linalg import gmres as cu_gmres

from tt_nte.experimental.solvers.linear_operator import TNLinearOperator


class GMRES(object):
    def __init__(self, use_gpu: bool = False):
        """"""
        self._use_gpu = use_gpu
        self._gmres = GMRES._sc_gmres if self._use_gpu == False else GMRES._cu_gmres

    # =============================================================
    # Methods

    def solve(
        self,
        A: TNLinearOperator,
        B: Union[np.ndarray, cu.ndarray],
        x0: Optional[Union[np.ndarray, cu.ndarray]] = None,
        rtol: float = 1e-5,
        restart: Optional[int] = None,
        max_iter: Optional[int] = None,
        M: Optional[TNLinearOperator] = None,
        callback: Optional[Callable] = None,
        callback_type: Optional[str] = None,
    ):
        """"""
        return self._gmres(
            A, B, x0, rtol, restart, max_iter, M, callback, callback_type
        )

    # =============================================================
    # Static methods

    @staticmethod
    def _sc_gmres(
        A: TNLinearOperator,
        b: Union[np.ndarray, cu.ndarray],
        x0: Optional[Union[np.ndarray, cu.ndarray]] = None,
        rtol: float = 1e-5,
        restart: Optional[int] = None,
        max_iter: Optional[int] = None,
        M: Optional[TNLinearOperator] = None,
        callback: Optional[Callable] = None,
        callback_type: Optional[str] = None,
    ):
        x, not_converged = sc_gmres(
            A=A,
            b=b,
            x0=x0,
            rtol=rtol,
            restart=restart,
            maxiter=max_iter,
            M=M,
            callback=callback,
            callback_type=callback_type,
        )
        return x, not not_converged

    @staticmethod
    def _cu_gmres(
        A: TNLinearOperator,
        b: Union[np.ndarray, cu.ndarray],
        x0: Optional[Union[np.ndarray, cu.ndarray]] = None,
        rtol: float = 1e-5,
        restart: Optional[int] = None,
        max_iter: Optional[int] = None,
        M: Optional[TNLinearOperator] = None,
        callback: Optional[Callable] = None,
        callback_type: Optional[str] = None,
    ):
        x, not_converged = cu_gmres(
            A=A,
            b=b,
            x0=x0,
            tol=rtol,
            restart=restart,
            maxiter=max_iter,
            M=M,
            callback=callback,
            callback_type=callback_type,
        )
        return x, not not_converged

    # =============================================================
    # Properties

    @property
    def op_type(self):
        return TNLinearOperator
