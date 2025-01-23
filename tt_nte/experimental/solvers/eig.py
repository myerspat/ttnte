"""
eig.py.

Contains power iteration implementation for quimb.tensor.MatrixProductOperators and
TNLinearOperators.
"""

import warnings
import random
import math
import time
from typing import Optional, Callable, Union, Literal

import numpy as np
import cupy as cu
from quimb.tensor import (
    MatrixProductOperator,
    MatrixProductState,
    MPS_rand_state,
    tensor_network_apply_op_vec,
)

from tt_nte.geometry import Geometry
from tt_nte.methods import DiscreteOrdinates
from tt_nte.experimental.solvers.linear_operator import TNLinearOperator
from tt_nte.experimental.solvers.gpu_manager import GPUManager


def eig(
    H: MatrixProductOperator,
    S: MatrixProductOperator,
    F: MatrixProductOperator,
    geometry: Geometry,
    linear_solver: object = None,
    linear_solver_opts: dict = {},
    psi0: Optional[Union[MatrixProductState, np.ndarray]] = None,
    k0: Optional[float] = None,
    tol: float = 1e-5,
    error_norm: Literal["fro", "euc"] = "fro",
    max_iter: int = 100,
    rank: int = 2,
    cutoff: float = 1e-10,
    verbose: bool = False,
    print_every: int = 5,
    scatter_iter: bool = True,
    scatter_iter_opts: dict = {},
    use_gpu: bool = False,
):
    """"""
    # System parameters
    order = H.L
    phys_dims = [H.phys_dim(site=site) for site in H.sites]

    # Push operators to gpu
    gpu_manager = None
    if use_gpu:
        gpu_manager = GPUManager()
        gpu_manager.to_gpu((H, S, F))

    # Convert to TNLinearOperator if needed
    use_linop = linear_solver.op_type == TNLinearOperator
    if use_linop:
        H = _convert_to_linop(H, phys_dims)
        S = _convert_to_linop(S, phys_dims)
        F = _convert_to_linop(F, phys_dims)

        # Turn off compression
        compress = False
        cutoff = 0.0

        # Get initial guess
        assert psi0 is None or isinstance(psi0, np.ndarray)
        psi = psi0.copy() if psi0 is not None else np.random.randn(np.prod(phys_dims))
        psi /= np.linalg.norm(psi, 2)
        psi = cu.asarray(psi) if use_gpu else psi

    else:
        # Determine compression bool
        compress = cutoff > 0

        # Get initial guess
        assert psi0 is None or isinstance(psi0, MatrixProductState)
        psi = (
            psi0.copy()
            if psi0 is not None
            else MPS_rand_state(
                L=order,
                phys_dim=phys_dims,
                bond_dim=rank,
                dist="uniform",
                dtype=H.dtype,
                normalize=True,
            )
        )
        psi /= math.sqrt(psi.H @ psi)
        psi.permute_arrays("lpr")

    # Get functions
    _matvec = _get_matvec(use_linop, compress, cutoff)
    _norm = _get_norm(use_linop, error_norm, use_gpu)
    _integrate = _get_int(use_linop, order, phys_dims, geometry, cutoff, use_gpu)
    _round = (
        (lambda x, p: np.round(x, p)) if not use_gpu else (lambda x, p: cu.round(x, p))
    )
    _inner = _scatter_iter if scatter_iter else _no_scatter_iter

    # Initial k
    k0 = k0 if k0 else random.random()

    # Initialize power iteration variables
    k = None
    f0 = _matvec(F, psi)
    ft0 = _integrate(f0)
    error0 = np.inf

    # Begin power iteration
    if verbose:
        print("Starting power iteration")

    start = time.time()
    for i in range(max_iter):
        # Run scattering source iteration
        psi, converged = _inner(
            H=H,
            S=S,
            f0=f0 / k0,
            psi=psi,
            _norm=_norm,
            _matvec=_matvec,
            _round=_round,
            linear_solver=linear_solver,
            linear_solver_opts=linear_solver_opts,
            start_time=start,
            **scatter_iter_opts,
        )

        # Check scattering source iteration convergence
        if not converged and verbose:
            warnings.warn(
                f"Scattering source iteration failed to converge for iteration {i}"
            )

        # Calculated updated fission source
        f = _matvec(F, psi)
        ft = _integrate(f)

        # Calculate updated eigenvalue
        k = k0 * ft / ft0

        # Calculate error
        error = _norm(f - f0) / _norm(f0)

        # Print status
        if verbose and i % print_every == 0:
            print(
                "-- ({}): k = {}, |df|/|f| = {}, Elapsed Time = {}".format(
                    i, _round(k, 8), _round(error, 8), _round(time.time() - start, 5)
                )
            )

        # Check convergence
        if error < tol:
            if verbose:
                print(
                    "-- Converged: k = {}, Elapsed Time = {}".format(
                        _round(k, 8), _round(time.time() - start, 5)
                    )
                )

            psi /= math.sqrt(psi.H @ psi) if not use_linop else math.sqrt(psi.T @ psi)

            # Flip eigenvector if needed
            if ft < 0:
                psi *= -1

            # Take off GPU
            if use_gpu:
                k, psi = gpu_manager.from_gpu((k, psi))
                if use_linop:
                    gpu_manager.from_gpu((H._tensors, S._tensors, F._tensors))
                else:
                    gpu_manager.from_gpu((H, S, F))

            return float(k), psi, True

        elif error > error0:
            # Warn about potential instability
            warnings.warn(
                "Fission iteration {} error is greater than iteration {}".format(
                    i, i - 1
                )
            )

        # Update old
        f0 = f.copy()
        k0 = k
        ft0 = ft
        error0 = error

    # Normalize
    psi /= math.sqrt(psi.H @ psi) if not use_linop else math.sqrt(psi.T @ psi)

    # Flip eigenvector if needed
    if ft < 0:
        psi *= -1

    # Take off GPU
    if use_gpu:
        k, psi = gpu_manager.from_gpu((k, psi))
        if use_linop:
            gpu_manager.from_gpu((H._tensors, S._tensors, F._tensors))
        else:
            gpu_manager.from_gpu((H, S, F))

    return float(k), psi, False


def _scatter_iter(
    H,
    S,
    f0,
    psi,
    _norm,
    _matvec,
    _round,
    linear_solver,
    linear_solver_opts,
    start_time,
    tol=1e-5,
    max_iter=30,
    verbose=False,
):
    # Calculate initial scattering source
    s0 = _matvec(S, psi)

    # Start scattering source iteration
    for i in range(max_iter):
        # Solve linear system
        psi, converged = linear_solver.solve(
            A=H, B=s0 + f0, x0=psi, **linear_solver_opts
        )

        # Calculate updated scattering source
        s = _matvec(S, psi)

        # Calculate error
        error = _norm(s - s0) / _norm(s0)

        # Print status
        if verbose:
            if not converged:
                warnings.warn(
                    "Linear solver did not converge on iteration {}".format(i)
                )

            print(
                "  -- ({}): |ds|/|s| = {}, Elapsed Time = {} s".format(
                    i, _round(error, 8), _round(time.time() - start_time, 5)
                )
            )

        # Check convergence
        if error < tol:
            return psi, True

        # Update old
        s0 = s.copy()

    assert psi is not None
    return psi, False


def _no_scatter_iter(
    H,
    S,
    f0,
    psi,
    _norm,
    _matvec,
    _round,
    linear_solver,
    linear_solver_opts,
    start_time,
    tol=1e-5,
    max_iter=30,
    verbose=False,
):
    return linear_solver.solve(
        A=H, B=_matvec(S, psi) + f0, x0=psi, **linear_solver_opts
    )


def _convert_to_linop(op: MatrixProductState, phys_dims):
    # Get indices in correct order
    op.permute_arrays("ludr")

    # Get indices and physical dimensions for each core
    inds = {
        "left_inds": [
            op[0].inds[0],
        ]
        + [op[i].inds[1] for i in range(1, op.L)],
        "right_inds": [op[0].inds[1]] + [op[i].inds[2] for i in range(1, op.L)],
        "ldims": phys_dims,
        "rdims": phys_dims,
    }

    # Create linear operator
    return TNLinearOperator(op, **inds)


def _get_matvec(use_linop, compress, cutoff):
    return (
        (
            lambda A, x: tensor_network_apply_op_vec(
                A, x, contract=True, compress=compress, cutoff=cutoff
            )
        )
        if not use_linop
        else lambda A, x: A @ x
    )


def _get_norm(use_linop, error_norm, use_gpu):
    if not use_linop:
        return (
            (lambda x: x.norm())
            if error_norm == "fro"
            else (lambda x: math.sqrt(x.H @ x))
        )

    else:
        if use_gpu:
            return (
                (lambda x: cu.linalg.norm(x, "fro"))
                if error_norm == "fro"
                else (lambda x: cu.linalg.norm(x, 2))
            )

        else:
            return (
                (lambda x: np.linalg.norm(x, "fro"))
                if error_norm == "fro"
                else (lambda x: np.linalg.norm(x, 2))
            )


def _get_int(use_linop, order, phys_dims, geometry, cutoff, use_gpu):
    # Number of spatial dimensions
    num_dim = geometry.num_dim

    # Get direction spaces
    octants = DiscreteOrdinates._direction_spaces[num_dim - 1]
    num_octants = octants.shape[0]

    integral = None
    for i in range(num_octants):
        # Get first octant
        octant = octants[0, :]

        # Get spatial cores
        spatial_cores = []
        for i in reversed(range(num_dim)):
            core = np.zeros(geometry.diff[i].size + 1)
            if octant[i] > 0:
                core[1:] = geometry.diff[i].flatten()
            else:
                core[:-1] = geometry.diff[i].flatten()

            spatial_cores.append(core)

        # TT cores
        cores = [
            np.ones(phys_dims[i]) for i in range(len(phys_dims) - num_dim)
        ] + spatial_cores

        # Add or create MPS object
        if integral is None:
            integral = MatrixProductState(
                [cores[0][:, np.newaxis]]
                + [cores[j][np.newaxis, :, np.newaxis] for j in range(1, order - 1)]
                + [cores[-1][np.newaxis, :]],
                shape="lpr",
            )
        else:
            integral += MatrixProductState(
                [cores[0][:, np.newaxis]]
                + [cores[j][np.newaxis, :, np.newaxis] for j in range(1, order - 1)]
                + [cores[-1][np.newaxis, :]],
                shape="lpr",
            )

    assert isinstance(integral, MatrixProductState)

    # Convert to regular vector if using linear operators
    if not use_linop:
        # Compress
        if cutoff > 0:
            integral.compress(cutoff=cutoff)

        return lambda x: integral.H @ x

    else:
        integral = (integral ^ all).data.reshape((-1))

        if not use_gpu:
            return lambda x: np.inner(integral, x)

        else:
            integral = cu.asarray(integral)
            return lambda x: cu.inner(integral, x)


def _get_sum(use_linop, order, phys_dims):
    if not use_linop:
        tt_ones = MatrixProductState(
            [np.ones((phys_dims[0], 1))]
            + [np.ones((1, phys_dims[i], 1)) for i in range(1, order - 1)]
            + [np.ones((1, phys_dims[-1]))],
            shape="lpr",
        )
        return lambda x: x @ tt_ones

    else:
        return lambda x: x.sum()
