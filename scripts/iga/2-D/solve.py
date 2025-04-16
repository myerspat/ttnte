import time

import numpy as np
import torch as tn
from operator_builder import TT
from torchtt._iterative_solvers import gmres_restart


def eig(
    H,
    S,
    F,
    Int_N,
    Int_I,
    tol,
    eps,
    max_fission_iter=500,
    max_scatter_iter=10,
    single_precision=False,
    sparsity_frac=0.2,
):
    dtype = tn.float32 if single_precision else tn.float64

    # Physical dimensions
    phys_dim = H.M

    # Create linear operators
    if not isinstance(H, TT):
        H, S, F, Int_N, Int_I = TT2LinOp(
            [H, S, F, Int_N, Int_I],
            sparsity_frac=sparsity_frac,
        )

    size_H = 0
    size_S = 0
    size_F = 0

    for i in range(len(phys_dim)):
        size_H += (
            tn.numel(H.cores[i])
            if not H.cores[i].is_sparse
            else 5 * H.cores[i].coalesce().indices().shape[1]
        )
        size_S += (
            tn.numel(S.cores[i])
            if not S.cores[i].is_sparse
            else 5 * S.cores[i].coalesce().indices().shape[1]
        )
        size_F += (
            tn.numel(F.cores[i])
            if not F.cores[i].is_sparse
            else 5 * F.cores[i].coalesce().indices().shape[1]
        )

    # Initial guess
    k0 = 1.0
    psi = tn.ones(phys_dim, dtype=dtype)
    psi /= tn.linalg.norm(psi.flatten(), 2)

    # Calculate initial scattering and fission sources
    s0 = S @ psi
    f0 = F @ psi
    ft0 = (Int_I @ f0.cpu()).sum()

    print("Starting power iteration")
    error = np.inf
    start = time.time()

    # Outer fission loop
    for i in range(max_fission_iter):
        # Inner scattering loop
        for _ in range(max_scatter_iter):
            # Solve linear system
            psi = gmres_restart(
                LinOp=H,
                b=(s0 + 1 / k0 * f0).reshape((-1, 1)),
                x0=psi.reshape((-1, 1)),
                N=np.prod(phys_dim),
                max_iterations=2,
                threshold=1e-7,
                resets=5,
            )[0].reshape(phys_dim)

            # Calculate updated scattering source
            s = S @ psi

            # Check convergence
            if (
                tn.linalg.norm((s - s0).flatten(), 2) / tn.linalg.norm(s0.flatten(), 2)
                < tol
            ):
                s0 = s
                break

            # Update old
            s0 = s

        # Calculate updated fission source
        f = F @ psi
        ft = (Int_I @ f.cpu()).sum()

        # Calculate updated eigenvalue
        k = k0 * ft / ft0

        # Ensure eigenvector is pointing in the right direction
        if ft < 0 and k > 0:
            psi *= -1
            f *= -1
            ft *= -1

        # Calculate error
        error = tn.linalg.norm((f - f0).flatten(), 2) / tn.linalg.norm(f0.flatten(), 2)

        # Print progress
        print(
            "-- ({}): k = {}, |df| / |f| = {}, Elapsed Time = {}".format(
                i,
                round(k.item(), 8),
                round(error.item(), 8),
                round(time.time() - start, 3),
            )
        )

        # Exit iteration if we've converged
        if error < tol:
            k0 = k
            print(
                "-- Converged: k = {}, Elapsed Time = {}".format(
                    round(k0.item(), 8), round(time.time() - start, 3)
                ),
            )
            break

        # Update old
        f0 = f
        k0 = k
        ft0 = ft

    # Apply angular integration operator
    return k0.item(), (Int_N @ psi).numpy()[0, ...]


def TT2LinOp(ops, sparsity_frac=0.2):
    # Physical dimensions
    order = len(ops[0].cores)

    def _create_linop(op):
        # Convert cores with sparsity_frac to COO matrices
        for i in range(order):
            op.cores[i] = (
                op.cores[i].to_sparse()
                if op.cores[i].nonzero().shape[0] / tn.numel(op.cores[i])
                < sparsity_frac
                else op.cores[i]
            )

        return TT(op.cores, eps=0)

    return list(map(_create_linop, ops))
