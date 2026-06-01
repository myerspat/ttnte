import time

import torch

from ttnte.linalg import TTEngine, mm, amen_solve


def power(A: TTEngine, F: TTEngine, Bin: TTEngine | None = None):
    device = A.device
    dtype = A.dtype
    tol = 1e-5 if dtype == torch.float32 else 1e-8
    eps = 1e-6 if dtype == torch.float32 else 1e-10

    # Create an initial guess
    psi = TTEngine.ones(A.m_modes, device=device, dtype=dtype)

    # Get the total fission source
    f = mm(F, psi)
    tf = f.sum()

    # Compute the initial eigenvalue
    k = (mm(psi.transpose(), f) / mm(psi.transpose(), mm(A, psi))).to_dense().item()

    start = time.time()
    for i in range(100):
        # Run AMEn solver
        psii = amen_solve(
            A,
            ((1 / k) * f) if Bin == None else ((1 / k) * f + mm(Bin, psi).round(eps)),
            x0=psi,
            nswp=4,
            eps=eps,
            kickrank=4,
            kick2=2,
            resets=4,
            local_iterations=60,
        )

        # Compute the relative L2 error
        error = (psii - psi).norm() / psi.norm()
        psi = psii

        # Calculate the new total fission source
        f = mm(F, psi)
        tfi = f.sum()

        # Update the eigenvalue and total fission source
        k *= tfi / tf
        tf = tfi

        # Flip sign if needed
        if tf < 0 and k > 0:
            psi *= -1.0
            tf *= -1.0

        print(
            f"({i}): k = {k:6f}, Angular Flux L2-Error = {error:3e}, Elapsed Time = {(time.time() - start):3f} s"
        )

        if error < tol:
            break

    return k, psi
