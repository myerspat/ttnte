import numpy as np
import torch as tn

from ttnte.assemblers import MatrixAssembler, TTAssembler
from ttnte.iga import IGAMesh
from ttnte.linalg import eig, gmres
from ttnte.xs import Server


def run_eig(mesh: IGAMesh, xs_server: Server, k_ref: float, num_ordinates: int):
    # Create operators in sparse format
    assembler = MatrixAssembler(
        mesh=mesh, xs_server=xs_server, num_ordinates=num_ordinates, max_processes=2
    )
    mats = assembler.build()

    # Run eigenvalue problem
    k, psi = eig(
        LHS=(
            (mats.H + mats.B_out - mats.B_in - mats.S)
            if mats.B_in is not None
            else (mats.H + mats.B_out - mats.S)
        ),
        RHS=mats.F,
        tols=1e-8,
        max_iters=500,
        device=0 if tn.cuda.is_available() and tn.cuda.device_count() > 0 else None,
    )
    assert isinstance(k, float)
    assert isinstance(psi, tn.Tensor)

    # Reshape for discretization
    psi = psi.reshape(assembler.discretization)

    # Check eignvalue is within 50 pcm of OpenMC
    assert abs(k - k_ref) * 1e5 < 50

    # Check shapes of angular flux
    assert psi.shape == tuple(
        [4]
        + 2 * [int(np.sqrt(num_ordinates / 4))]
        + [xs_server.num_groups, mesh.num_patches]
        + list(mesh.patches[mesh.patch_ids[0]].shape)
    )

    # Calculate scalar flux and check shape
    phi = assembler.angular_integral(psi).squeeze()
    assert phi.shape == tuple(
        ([xs_server.num_groups] if xs_server.num_groups > 1 else [])
        + ([mesh.num_patches] if mesh.num_patches > 1 else [])
        + list(mesh.patches[mesh.patch_ids[0]].shape)
    )
    assert (phi > 0).all()
    del mats, k, psi, phi, assembler

    # ===============================================
    # Check with TTs
    # ===============================================
    # Discretization
    num_ordinates = 16

    # Create operators in sparse format
    assembler = MatrixAssembler(
        mesh=mesh, xs_server=xs_server, num_ordinates=num_ordinates, max_processes=2
    )
    mats = assembler.build()

    # Create operators in TT format
    assembler = TTAssembler(
        mesh=mesh, xs_server=xs_server, num_ordinates=num_ordinates, max_processes=2
    )
    tts = assembler.build(use_tt=True, eps=1e-10)

    # Check loss and boundary operators
    assert (
        tn.abs(mats.H.to_dense() - tts.H.to_dense().reshape(mats.H.shape)) < 1e-10
    ).all()
    assert (
        tn.abs(mats.B_out.to_dense() - tts.B_out.to_dense().reshape(mats.B_out.shape))
        < 1e-10
    ).all()
    if mats.B_in is not None:
        assert (
            tn.abs(mats.B_in.to_dense() - tts.B_in.to_dense().reshape(mats.B_in.shape))
            < 1e-10
        ).all()

    # Run eigenvalue problem
    k_m, psi_m = eig(
        LHS=(
            (mats.H + mats.B_out - mats.B_in - mats.S).combine()
            if mats.B_in is not None
            else (mats.H + mats.B_out - mats.S).combine()
        ),
        RHS=mats.F,
        tols=1e-8,
        max_iters=500,
        device=0 if tn.cuda.is_available() and tn.cuda.device_count() > 0 else None,
    )
    psi_m = psi_m.reshape(assembler.discretization)

    # Run eigenvalue problem
    k_tt, psi_tt = eig(
        LHS=(
            (tts.H + tts.B_out - tts.B_in - tts.S)
            if tts.B_in is not None
            else (tts.H + tts.B_out - tts.S)
        ),
        RHS=tts.F,
        tols=1e-8,
        max_iters=500,
        device=0 if tn.cuda.is_available() and tn.cuda.device_count() > 0 else None,
    )
    psi_tt = psi_tt.reshape(assembler.discretization)

    # Check solution
    assert abs(k_m - k_tt) * 1e5 < 1
    assert (tn.abs(psi_m - psi_tt) < 1e-8).all()


def run_fixed_source(
    mesh: IGAMesh, xs_server: Server, leakage_frac: float, num_ordinates: int
):
    # Create operators in sparse format
    assembler = MatrixAssembler(
        mesh=mesh, xs_server=xs_server, num_ordinates=num_ordinates, max_processes=2
    )
    mats = assembler.build()

    # Run eigenvalue problem
    psi = gmres(
        A=(
            (mats.H + mats.B_out - mats.B_in)
            if mats.B_in is not None
            else (mats.H + mats.B_out)
        )
        - mats.S,
        b=mats.q,
        tol=1e-8,
        gpu_idx=0 if tn.cuda.is_available() and tn.cuda.device_count() > 0 else None,
    )[0]
    psi = psi.reshape(assembler.discretization)
    assert isinstance(psi, tn.Tensor)

    # Calculate leakage fraction
    leakage = assembler.outward_current(psi)
    production = assembler.total_production()
    assert (leakage / production - leakage_frac) * 1e5 < 50

    # Check shapes of angular flux
    assert psi.shape == tuple(
        [4]
        + 2 * [int(np.sqrt(num_ordinates / 4))]
        + [xs_server.num_groups, mesh.num_patches]
        + list(mesh.patches[mesh.patch_ids[0]].shape)
    )

    # Calculate scalar flux and check shape
    phi = assembler.angular_integral(psi).squeeze()
    assert phi.shape == tuple(
        ([xs_server.num_groups] if xs_server.num_groups > 1 else [])
        + ([mesh.num_patches] if mesh.num_patches > 1 else [])
        + list(mesh.patches[mesh.patch_ids[0]].shape)
    )
    assert (phi > 0).all()
    del mats, psi, phi

    # ===============================================
    # Check with TTs
    # ===============================================
    # Discretization
    num_ordinates = 16

    # Create operators in sparse format
    assembler_m = MatrixAssembler(
        mesh=mesh, xs_server=xs_server, num_ordinates=num_ordinates
    )
    mats = assembler_m.build()

    # Create operators in TT format
    assembler = TTAssembler(mesh=mesh, xs_server=xs_server, num_ordinates=num_ordinates)
    tts = assembler.build(use_tt=True, eps=1e-10)

    # Check loss and boundary operators
    assert (
        tn.abs(mats.H.to_dense() - tts.H.to_dense().reshape(mats.H.shape)) < 1e-10
    ).all()
    assert (
        tn.abs(mats.B_out.to_dense() - tts.B_out.to_dense().reshape(mats.B_out.shape))
        < 1e-10
    ).all()
    if mats.B_in is not None:
        assert (
            tn.abs(mats.B_in.to_dense() - tts.B_in.full().reshape(mats.B_in.shape))
            < 1e-10
        ).all()

    # Run eigenvalue problem
    psi_m = gmres(
        A=(
            (mats.H + mats.B_out - mats.B_in)
            if mats.B_in is not None
            else (mats.H + mats.B_out)
        )
        - mats.S,
        b=mats.q,
        tol=1e-10,
        restart=50,
        gpu_idx=0 if tn.cuda.is_available() and tn.cuda.device_count() > 0 else None,
    )[0]
    psi_m = psi_m.reshape(assembler.discretization)
    leakage_frac_m = assembler_m.outward_current(psi_m) / assembler_m.total_production()

    # Run eigenvalue problem
    psi_tt = gmres(
        A=(
            (tts.H + tts.B_out - tts.B_in)
            if tts.B_in is not None
            else (tts.H + tts.B_out)
        )
        - tts.S,
        b=tts.q,
        tol=1e-10,
        restart=50,
        gpu_idx=0 if tn.cuda.is_available() and tn.cuda.device_count() > 0 else None,
    )[0]
    psi_tt = psi_tt.reshape(assembler.discretization)
    leakage_frac_tt = (
        assembler_m.outward_current(psi_tt) / assembler_m.total_production()
    )

    # Check solution
    assert abs(leakage_frac_m - leakage_frac_tt) * 1e5 < 1
    assert (tn.abs(psi_m - psi_tt) < 1e-8).all()
