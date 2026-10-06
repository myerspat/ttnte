"""Partial (specular) albedo on REFLECTIVE faces.

Incoming angular flux on a REFLECTIVE face with albedo a is a * (mirrored outgoing
angular flux): the reflective inflow operator is scaled by a, so the assembled LHS is
... + B_out - a * B_refl. a = 1 is plain REFLECTIVE (bit-for-bit), a = 0 is VACUUM (to
roundoff), and the particle balance's REFLECTIVE `incoming` is the same scaled inflow
term, so global balance closes for any a in [0, 1].

Exactness checks use the dense direct solution of the single patch's assembled
linear system (small problem) so they are not polluted by iterative-solver error.
"""

import math

import pytest
import torch
from igakit.cad import line, refine, ruled

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.driver import IGATransportDriver2D
from ttnte.linalg import State, TTEngine
from ttnte.math import ProductQuadrature
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BCPlane,
    BoundaryType,
    DGTransportAssemblerConfig,
    DIGAFirstOrderTransportAssembler2D,
    FixedSource,
)
from ttnte.solvers import (
    AMEnSolver,
    BlockJacobiStrategy,
    CommMode,
    DDSolverConfig,
    ExecMode,
    IGADDSolver,
    MemoryPolicy,
)
from ttnte.xs import Material, Server

DTYPE = torch.float64
DEVICE = torch.device("cpu")
L = 2.0
Q = 1.0


def _server():
    server = Server()
    mat = Material("Scatterer")
    mat.chi = torch.zeros(1, dtype=DTYPE)
    mat.total = torch.tensor([1.0], dtype=DTYPE)
    mat.nu_fission = torch.zeros(1, dtype=DTYPE)
    mat.fission = torch.zeros(1, dtype=DTYPE)
    mat.absorption = torch.tensor([0.4], dtype=DTYPE)
    mat.scatter_gtg = torch.tensor([[[0.6]]], dtype=DTYPE)
    mat.finalize()
    server.add_material(mat)
    server.finalize()
    return mat.label, server


def _config():
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-13
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10)
    config.cross_jacobian_inverse = True
    return config


def _qset():
    qset = ProductQuadrature.gauss_legendre_chebyshev(2, 4, 2)
    qset.to_(DEVICE, DTYPE)
    return qset


def _mesh(lower_type, albedo=1.0):
    """Homogeneous square, isotropic source; (x_min, y_min) get `lower_type` (with
    `albedo`), (x_max, y_max) are VACUUM."""
    mpi_context.init()
    torch.set_default_dtype(DTYPE)
    torch.autograd.set_grad_enabled(False)
    fill, server = _server()
    patch = Patch.from_igakit(
        refine(ruled(line((0, 0), (L, 0)), line((0, L), (L, L))), [4, 4], [2, 2]),
        device=DEVICE,
        dtype=DTYPE,
        fill=fill,
    )
    patch.source = FixedSource(isotropic_strength=torch.tensor([Q], dtype=DTYPE))
    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True), lower_type, albedo=albedo
    )
    mesh.set_axis_aligned_conditions(
        BCPlane(x_max=True, y_max=True), BoundaryType.VACUUM
    )
    mesh.finalize()
    return mesh, patch, server


def _dense_system(lower_type, albedo=1.0):
    mesh, patch, server = _mesh(lower_type, albedo)
    assembler = DIGAFirstOrderTransportAssembler2D(patch, _qset(), server, _config())
    assembler.assemble()
    sys = assembler.get_linear_system()
    A = sys.interior_op.as_tt().to_dense()
    b = sys.source.state.to_dense()
    n = b.numel()
    return (
        assembler,
        A.reshape(n, n),
        b.reshape(n),
        list(sys.source.state.as_tt().m_modes),
    )


def _solve(lower_type, albedo=1.0):
    assembler, A, b, modes = _dense_system(lower_type, albedo)
    return assembler, torch.linalg.solve(A, b), modes


def _balance(assembler, x, modes):
    psi = State(TTEngine.from_dense(x.reshape(modes), 1e-14))
    return assembler.compute_balance(psi, 1e-14, 10**9)


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_albedo_one_is_reflective():
    """Albedo == 1 skips the scaling entirely, so the code path is plain REFLECTIVE.

    Assembly itself is only reproducible to ~1e-14 run to run (two plain
    REFLECTIVE assemblies already differ at that level), so compare to roundoff.
    """
    _, A_ref, b_ref, _ = _dense_system(BoundaryType.REFLECTIVE)
    _, A_one, b_one, _ = _dense_system(BoundaryType.REFLECTIVE, albedo=1.0)
    assert _rel(A_one, A_ref) < 1e-13
    assert _rel(b_one, b_ref) < 1e-13


def test_albedo_zero_is_vacuum():
    _, x_vac, _ = _solve(BoundaryType.VACUUM)
    _, x_zero, _ = _solve(BoundaryType.REFLECTIVE, albedo=0.0)
    assert _rel(x_zero, x_vac) < 1e-12


def test_albedo_balance_closes_and_leakage_monotone():
    leakages = []
    for albedo in (0.0, 0.25, 0.5, 0.75, 1.0):
        assembler, x, modes = _solve(BoundaryType.REFLECTIVE, albedo=albedo)
        pb = _balance(assembler, x, modes)
        gain = pb.fixed_source + pb.scatter_in
        loss = pb.absorption + pb.scatter_out + pb.leakage
        assert ((gain - loss).abs() / gain).item() < 1e-9, albedo

        reflective = [f for f in pb.faces if f.type == BoundaryType.REFLECTIVE]
        assert len(reflective) == 2
        for f in reflective:
            # Weak-form inflow term == albedo * mirrored outgoing current.
            torch.testing.assert_close(
                f.incoming, albedo * f.outgoing, rtol=1e-10, atol=1e-14
            )
        leakages.append(pb.leakage.item())

    assert all(a > b for a, b in zip(leakages[:-1], leakages[1:])), leakages


def test_albedo_end_to_end_dd_solve_balance():
    """Same problem through the driver/DD/AMEn path: global balance closes to solver
    accuracy for a partial albedo."""
    mesh, patch, server = _mesh(BoundaryType.REFLECTIVE, albedo=0.5)
    assert patch.get_boundary_albedo(0, False) == 0.5
    assert patch.get_boundary_albedo(0, True) == 1.0
    assert patch.get_boundary_info(1, False).albedo == 0.5

    config = _config()
    config.rounding.eps = 1e-10
    driver = IGATransportDriver2D(mesh, server, mpi_context)
    driver.assemble(_qset(), config)
    dd_config = DDSolverConfig(
        tol=1e-8,
        max_iter=10,
        use_gpu=False,
        memory_policy=MemoryPolicy.OUT_OF_CORE,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=False,
    )
    strategy = BlockJacobiStrategy(dd_config)
    strategy.set_local_solver(
        AMEnSolver(nswp=6, eps=1e-9, kickrank=4, local_iterations=200, resets=6)
    )
    result = driver.solve_fixed_source(
        IGADDSolver(driver.mesh, strategy),
        tol=1e-7,
        max_iter=50,
        verbose=False,
        clear_assemblers=False,
    )
    gb = result.global_balance(driver.get_assemblers(), eps=1e-12, max_rank=10**9)
    gain = gb.fixed_source + gb.scatter_in
    loss = gb.absorption + gb.scatter_out + gb.leakage
    assert ((gain - loss).abs() / gain).item() < 1e-3


@pytest.mark.parametrize("bad", [-0.1, 1.5, math.nan])
def test_set_albedo_out_of_range_throws(bad):
    mesh, patch, _ = _mesh(BoundaryType.REFLECTIVE)
    with pytest.raises(RuntimeError, match="albedo"):
        patch.set_boundary_albedo(0, False, bad)
    with pytest.raises(RuntimeError, match="albedo"):
        patch.get_boundary_info(0, False).albedo = bad


def test_axis_aligned_albedo_requires_reflective():
    mpi_context.init()
    fill, _ = _server()
    patch = Patch.from_igakit(
        refine(ruled(line((0, 0), (L, 0)), line((0, L), (L, L))), [2, 2], [2, 2]),
        device=DEVICE,
        dtype=DTYPE,
        fill=fill,
    )
    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    with pytest.raises(RuntimeError, match="REFLECTIVE"):
        mesh.set_axis_aligned_conditions(
            BCPlane(x_min=True), BoundaryType.VACUUM, albedo=0.5
        )
    with pytest.raises(RuntimeError, match="albedo"):
        mesh.set_axis_aligned_conditions(
            BCPlane(x_min=True), BoundaryType.REFLECTIVE, albedo=2.0
        )
    # albedo == 1 is the default and valid for any type.
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True), BoundaryType.VACUUM, albedo=1.0
    )


def test_albedo_on_non_reflective_face_rejected_at_assembly():
    mesh, patch, server = _mesh(BoundaryType.VACUUM)
    patch.set_boundary_albedo(0, False, 0.5)  # allowed to set any time...
    assembler = DIGAFirstOrderTransportAssembler2D(patch, _qset(), server, _config())
    with pytest.raises(RuntimeError, match="only valid on REFLECTIVE"):
        assembler.assemble()  # ...but rejected when assembled on a VACUUM face
