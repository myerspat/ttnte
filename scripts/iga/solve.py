import time

import numpy as np
import torch as tn
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from torchtt._iterative_solvers import gmres_restart
from tt import TT


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
    phys_dim = [H.phys_dim(site=site) for site in H.sites]

    # Create linear operators
    if not isinstance(H, TT):
        H, S, F, Int_N, Int_I = TT2LinOp(
            [H, S, F, Int_N, Int_I],
            eps=eps,
            single_precision=single_precision,
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

    print(size_H)
    print(size_S)
    print(size_F)

    print(H)
    print(S)
    print(F)

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


def TT2LinOp(ops, eps=1e-10, single_precision=False, sparsity_frac=0.2):
    dtype = tn.float32 if single_precision else tn.float64

    # Physical dimensions
    order = ops[0].L

    def _create_linop(op):
        # Get MPS in right shape
        op.permute_arrays("ludr")

        # Convert to torchTT object
        op = TT(
            [tn.tensor(op[0].data[np.newaxis, ...], dtype=dtype)]
            + [tn.tensor(op[i].data, dtype=dtype) for i in range(1, order - 1)]
            + [tn.tensor(op[-1].data[..., np.newaxis], dtype=dtype)],
            eps=eps,
        ).round(eps)

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


def fixed_source(
    H,
    S,
    q,
    Int_N,
    tol=1e-6,
    eps=1e-10,
    max_iter=500,
    single_precision=False,
    sparsity_frac=0.2,
    use_sparse_mat=False,
):
    dtype = tn.float32 if single_precision else tn.float64

    # Physical dimensions
    phys_dim = [H.phys_dim(site=site) for site in H.sites]

    # Convert q to tntt and get full
    q.permute_arrays("lpr")
    q = (
        TT(
            [tn.tensor(q[0].data[np.newaxis, :, np.newaxis, :], dtype=dtype)]
            + [
                tn.tensor(q[i].data[:, :, np.newaxis, :], dtype=dtype)
                for i in range(1, H.L - 1)
            ]
            + [tn.tensor(q[-1].data[:, :, np.newaxis, np.newaxis], dtype=dtype)]
        )
        .full()
        .reshape(phys_dim)
    )

    if use_sparse_mat:
        H, S, Int_N = TT2LinOp(
            [H, S, Int_N],
            eps=eps,
            single_precision=single_precision,
            sparsity_frac=0,
        )
        H = csr_matrix(H.full().numpy().reshape(2 * [np.prod(phys_dim)]))
        S = csr_matrix(S.full().numpy().reshape(2 * [np.prod(phys_dim)]))

        psi = spsolve(H - S, q.numpy().flatten())

        for i in range(max_iter):
            psi = spsolve(H, S @ psi + q.numpy().flatten())

        return (Int_N @ tn.tensor(psi.reshape(phys_dim), dtype=dtype)).numpy()[0, ...]

    else:
        # Create linear operators
        H, S, Int_N = TT2LinOp(
            [H, S, Int_N],
            eps=eps,
            single_precision=single_precision,
            sparsity_frac=sparsity_frac,
        )

        # Initial guess
        psi = tn.ones(phys_dim, dtype=dtype)

    # Calculate initial scattering source
    s0 = S @ psi

    start = time.time()
    for i in range(max_iter):
        # Solve linear system
        psi = gmres_restart(
            LinOp=H,
            b=(s0 + q).reshape((-1, 1)),
            x0=psi.reshape((-1, 1)),
            N=np.prod(phys_dim),
            max_iterations=10,
            threshold=1e-7,
            resets=5,
        )[0].reshape(phys_dim)

        # Calculate updated scattering source
        s = S @ psi

        # Calculate error
        error = tn.linalg.norm((s - s0).flatten(), 2) / tn.linalg.norm(s0.flatten(), 2)

        # Print progress
        print(
            "-- ({}): |ds| / |s| = {}, Elapsed Time = {}".format(
                i,
                round(error.item(), 8),
                round(time.time() - start, 3),
            )
        )

        # Check convergence
        if error < tol:
            print("-- Converged!")
            break

        # Update old
        s0 = s

    # Apply angular integration operator
    return (Int_N @ psi).numpy()[0, ...]
