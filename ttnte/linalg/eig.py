import time
from typing import Optional, Tuple, Union

import numpy as np
import torch as tn
from torchtt._iterative_solvers import gmres_restart


def eig(
    LHS,
    RHS,
    tols: Optional[Union[Tuple[float], float]] = 1e-8,
    max_iters: Optional[Union[Tuple[int], int]] = 500,
    device: Optional[int] = None,
    linear_solver_opts: dict = {
        "max_iterations": 100,
        "threshold": 1e-10,
        "resets": 5,
    },
):
    """
    Compute largest eigenvalue and corresponding eigenvector of :math:`Ax=\\lambda B_1 x
    + B_2x + ...`. This solver uses power iteration with GMRES for the resulting linear
    system.

    Parameters
    ----------
    LHS: ttnte.linalg.LinearOperator
        Left hand side operator.
    RHS: tuple of ttnte.linalg.LinearOperator
        Right hand side operators.
    tols: tuple of float, default=(1e-8,)
        Tolerances for solvers of each ``RHS`` operator.
    max_iters: tuple of int, default=(500,)
        Maximum number of iterations for each ``RHS`` operator.
    device: int, default=None
        Device to compute problem on.
    linear_solver_opts: dict
        GMRES options including ``max_threshold``, ``threshold``, and ``resets``.
    """
    # Convert singles to tuples
    RHS = RHS if isinstance(RHS, tuple) or isinstance(RHS, list) else (RHS,)
    tols = (tols,) if isinstance(tols, float) else tols
    max_iters = (max_iters,) if isinstance(max_iters, int) else max_iters

    # Check shapes of operators
    assert (
        isinstance(RHS, tuple)
        and np.array(
            [
                np.prod(LHS.input_shape) == np.prod(RHS[i].input_shape)
                for i in range(len(RHS))
            ]
        ).all()
        and np.array(
            [
                np.prod(LHS.output_shape) == np.prod(RHS[i].output_shape)
                for i in range(len(RHS))
            ]
        ).all()
    )

    # Send operators to GPU
    if device is not None:
        LHS.cuda(device)
        for op in RHS:
            op.cuda(device)

    # Create initial guess
    k0 = tn.ones(1, device=device)
    psi0 = tn.ones(LHS.input_shape, device=device)
    ft0 = (RHS[0] @ psi0).sum()
    psi = None

    print("Starting power iteration")
    error = np.inf
    start = time.time()
    for i in range(max_iters[0]):
        # Run GMRES
        psi = gmres_restart(
            LinOp=LHS,
            b=1 / k0 * (RHS[0] @ psi0).reshape((-1, 1)),
            x0=psi0.reshape((-1, 1)),
            N=np.prod(LHS.output_shape),
            **linear_solver_opts
        )[0].reshape(LHS.output_shape)

        # Calculate total fission source
        ft = (RHS[0] @ psi).sum()

        # Calculate new eigenvalue
        k = k0 * ft / ft0

        # Ensure eigenvector is pointing in the right direction
        if ft < 0 and k > 0:
            psi *= -1
            ft *= -1

        # Calculate error
        error = tn.linalg.norm((psi - psi0).flatten(), 2) / tn.linalg.norm(
            psi0.flatten(), 2
        )

        # Print progress
        print(
            "-- ({}): k = {}, Angular Flux L2-Error = {}, Elapsed Time = {}".format(
                i,
                round(k.item(), 8),
                round(error.item(), 8),
                round(time.time() - start, 3),
            )
        )

        # Exit iteration if we've converged
        if error < tols[0]:
            k0 = k
            print(
                "-- Converged: k = {}, Elapsed Time = {}".format(
                    round(k0.item(), 8), round(time.time() - start, 3)
                ),
            )
            break

        # Update old
        psi0 = psi
        k0 = k
        ft0 = ft

    # Move operators back to CPU
    LHS.cpu()
    for op in RHS:
        op.cpu()
    psi0 = psi0.cpu()
    psi = psi.cpu()

    return k0.item(), psi
