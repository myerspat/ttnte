import numpy as np
import pytest
import torch as tn

from ttnte.assemblers import MatrixAssembler, TTAssembler
from ttnte.iga import IGAMesh
from ttnte.linalg import LinearOperator, eig
from ttnte.xs import Server

tn.set_default_dtype(tn.float64)


def run_problem(mesh: IGAMesh, xs_server: Server, k_ref: float, num_ordinates: int):
    # Create operators in sparse format
    assembler = MatrixAssembler(
        mesh=mesh, xs_server=xs_server, num_ordinates=num_ordinates
    )
    H, S, F, B_in, B_out = assembler.build()

    # Run eigenvalue problem
    k, psi = eig(
        LHS=LinearOperator([H + B_out - B_in, -S], N=assembler.N, M=assembler.M),
        RHS=LinearOperator([F], N=assembler.N, M=assembler.M),
        tols=1e-8,
        max_iters=500,
        device=0 if tn.cuda.is_available() and tn.cuda.device_count() > 0 else None,
    )
    assert isinstance(k, float)
    assert isinstance(psi, tn.Tensor)

    # Check eignvalue is within 50 pcm of OpenMC
    assert abs(k - k_ref) * 1e5 < 50

    # Check shapes of angular flux
    assert psi.shape == tuple(
        [4]
        + 2 * [int(np.sqrt(num_ordinates / 4))]
        + ([xs_server.num_groups] if xs_server.num_groups > 1 else [])
        + ([mesh.num_patches] if mesh.num_patches > 1 else [])
        + [mesh.patches[0].ctrlpts_size_u, mesh.patches[0].ctrlpts_size_v]
    )

    # Calculate scalar flux and check shape
    phi = assembler.angular_integral(psi)
    assert phi.shape == tuple(
        ([xs_server.num_groups] if xs_server.num_groups > 1 else [])
        + ([mesh.num_patches] if mesh.num_patches > 1 else [])
        + [mesh.patches[0].ctrlpts_size_u, mesh.patches[0].ctrlpts_size_v]
    )
    assert (phi > 0).all()
    del H, S, F, B_in, B_out, k, psi, phi

    # ===============================================
    # Check with TTs
    # ===============================================
    # Discretization
    num_ordinates = 16

    # Create operators in sparse format
    assembler = MatrixAssembler(
        mesh=mesh, xs_server=xs_server, num_ordinates=num_ordinates
    )
    H_m, S_m, F_m, B_in_m, B_out_m = assembler.build()

    # Create operators in TT format
    assembler = TTAssembler(mesh=mesh, xs_server=xs_server, num_ordinates=num_ordinates)
    assembler.interp_jacobian = False
    assembler.interp_jacobian_det = False
    assembler.interp_boundary_jacobian_det = False
    H_tt, S_tt, F_tt, B_in_tt, B_out_tt = assembler.build(use_tt=True, eps=1e-10)

    # Check loss and boundary operators
    assert (tn.abs(H_m.to_dense() - H_tt.full().reshape(H_m.shape)) < 1e-10).all()
    assert (
        tn.abs(B_out_m.to_dense() - B_out_tt.full().reshape(B_out_m.shape)) < 1e-10
    ).all()
    assert (
        tn.abs(B_in_m.to_dense() - B_in_tt.full().reshape(B_in_m.shape)) < 1e-10
    ).all()

    # Run eigenvalue problem
    k_m, psi_m = eig(
        LHS=LinearOperator(
            [H_m + B_out_m - B_in_m, -S_m], N=assembler.N, M=assembler.M
        ),
        RHS=LinearOperator([F_m], N=assembler.N, M=assembler.M),
        tols=1e-8,
        max_iters=500,
    )

    # Run eigenvalue problem
    k_tt, psi_tt = eig(
        LHS=LinearOperator(
            [H_tt, -S_tt, B_out_tt, -B_in_tt], N=assembler.N, M=assembler.M
        ),
        RHS=LinearOperator([F_tt], N=assembler.N, M=assembler.M),
        tols=1e-8,
        max_iters=500,
    )

    # Check solution
    assert abs(k_m - k_tt) * 1e5 < 1
    assert (tn.abs(psi_m - psi_tt) < 1e-8).all()
