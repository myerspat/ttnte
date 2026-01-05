import sys
import time
from dataclasses import dataclass
from typing import Optional, Union, Literal, Callable

import numpy as np
import torch as tn

from .operator import Operator
from .linear_operator import LinearOperator
from .gmres import gmres


@dataclass
class LinearSolverOptions:
    """
    Linear solver options for GMRES.

    Attributes
    ----------
    gpu_idx: int or None, default=None
        If ``gpu_idx`` is not None then GMRES is run on GPU with the given
        index.
    tol: float, default=1e-10
        Tolerance of GMRES with convergence condition
        ``norm(residual) <= max(tol*norm(b), atol)``.
    atol: float, default=0.0
        Tolerance of GMRES with convergence condition
        ``norm(residual) <= max(tol*norm(b), atol)``.
    restart: int, default=100
        Size of the Krylov subspace ("number of iterations") built between
        restarts.
    maxiter: int or None, default=5
        Maximum number of times to rebuild the size-``restart`` Krylov space
        starting from the solution found at the last iteration.
    solve_method: "batched" or "incremental", default="batched"
        The "incremental" solve method builds a QR decomposition for the
        Krylov subspace incrementally using Givens rotation. This enables
        early stopping and is overall more stable than the "batched"
        method. The "batched" solution method solves a least squares problem
        from scratch at the end of each GMRES iteration. This has less overhead
        on GPUs. The "batched" method is only implemented in the C++ backend.
    callback: callable or None, default=None
        A callable called after each restart. This callable takes three inputs:
        iteration index ``i``, the residual norm for that iteration ``rnorm``,
        and the solution for that iteration ``x``.
    callback_frequency: int, default=1
        How often updates are printed to terminal and how often the callback
        is called.
    verbose: bool, default=False
        Whether to print the progress of GMRES.
    """

    tol: float = 1e-10
    atol: float = 0.0
    restart: int = 100
    maxiter: int = 5
    solve_method: Literal["batched", "incremental"] = "batched"
    callback: Optional[Callable] = None
    callback_frequency: int = 1
    gpu_idx: Optional[int] = None
    verbose: bool = False


def power(
    T: Union[LinearOperator, Operator],
    F: Union[LinearOperator, Operator],
    psi0: Optional[tn.Tensor] = None,
    tol: float = 1e-8,
    maxiter: int = 100,
    gpu_idx: Optional[int] = None,
    callback: Optional[Callable] = None,
    callback_frequency: int = 1,
    verbose: bool = True,
    lsoptions: LinearSolverOptions = LinearSolverOptions(),
):
    """
    Find the largest eigenvalue and corresponding eigenvector for the
    k-eigenvalue neutron transport equation using power iteration.
    The linear solver is ``ttnte.linalg.gmres`` and solves

    .. math::
        T\\psi^{(n + 1)} = \\frac{1}{k^{(n)}}F\\psi^{(n)}.

    Parameters
    ----------
    T: ttnte.linalg.LinearOperator or ttnte.linalg.Operator
        The operator to invert.
    F: ttnte.linalg.LinearOperator or ttnte.linalg.Operator
        The fission operator.
    psi0: torch.Tensor or None, default=None
        An initial guess
    tol: float, default=1e-8
        The convergence criterion for the angular flux
        relative L2-error.
    maxiter: int, default=100
        Maximum number of power iterations.
    gpu_idx: int or None, default=None
        Which GPU to run on.
    callback: callable or None, default=None
        A callback function that runs at the end of each power
        iteration. This function takes three arguments:
        the iteration index ``i``, the relative angular flux L2-error
        ``error``, and the angular flux solution for that iteration
        ``psii``.
    callback_frequency: int, default=1
        How often progress is printed to terminal and the
        ``callback`` is ran.
    verbose: bool, default=True
        Print progress to terminal.
    lsoptions: ttnte.linalg.LinearSolverOptions, default=ttnte.linalg.LinearSolverOptions()
        Options to pass to ``ttnte.linalg.gmres``.

    Returns
    -------
    psi: torch.Tensor
        The eigenvector.
    k: float
        The eigenvalue.
    """
    # Get the size of the system
    n = np.prod(T.input_shape)

    # Get types and device
    dtype = T.dtype
    device = T.device

    # Get initial eigenvector
    psi = psi0 if psi0 is not None else tn.ones((n, 1), dtype=dtype, device=device)
    psi /= psi.norm(2)

    # Check data types and device
    if (
        F.dtype != dtype
        or F.device != device
        or psi.dtype != dtype
        or psi.device != device
    ):
        raise RuntimeError(
            "T, F, and x0 should be on the same device with the same data type"
        )

    # Check shapes
    if (
        n != np.prod(T.output_shape)
        or n != np.prod(F.input_shape)
        or n != np.prod(F.output_shape)
    ):
        raise RuntimeError(
            "The eigenvalue problem should be square with T and F matching in shape"
        )

    # Send to GPU
    if gpu_idx != None and tn.cuda.is_available() and gpu_idx < tn.cuda.device_count():
        T.cuda(gpu_idx)
        F.cuda(gpu_idx)
        psi = psi.cuda(gpu_idx)

    # Get initial eigenvalue by Rayleigh quotient
    k = psi.t() @ F.apply(psi) / (psi.t() @ T.apply(psi))

    # Get total fission
    ft = F.apply(psi).sum()

    i = 0
    error = sys.float_info.max
    start = time.time()
    if verbose:
        print(
            "Running power iteration on " + ("GPU " + str(gpu_idx))
            if gpu_idx is not None
            else "CPU"
        )

    while error >= tol and i < maxiter:
        # Run GMRES
        psii = gmres(
            T,
            (1 / k) * F.apply(psi),
            psi,
            lsoptions.gpu_idx,
            lsoptions.tol,
            lsoptions.atol,
            lsoptions.restart,
            lsoptions.maxiter,
            lsoptions.solve_method,
            lsoptions.callback,
            lsoptions.callback_frequency,
            lsoptions.verbose,
        )[0]

        # Calculate error
        error = ((psii - psi).norm(2) / psi.norm(2)).item()
        psi = psii

        # Calculate total fission source
        fti = F.apply(psi).sum()

        # Update the eigenvalue
        k *= fti / ft
        ft = fti

        # Flip sign if needed
        if ft < 0 and k > 0:
            psi *= -1
            ft *= -1

        # Print progress
        if i % callback_frequency == 0:
            if verbose:
                print(
                    "-- ({}):  k = {:.6f}, Angular Flux L2-Error = {:.12f}, Elapsed Time = {:.3f} s".format(
                        i, k.item(), error, time.time() - start
                    )
                )

            if callback is not None:
                callback(i, error, psi)

        i += 1

    # Remove data from GPU
    if gpu_idx is not None:
        T.cpu()
        F.cpu()
        psi = psi.cpu()

    # Print convergence
    if verbose:
        print("-- {}".format("Converged!" if error < tol else "Failed to Converge!"))

    return psi, k.item()
