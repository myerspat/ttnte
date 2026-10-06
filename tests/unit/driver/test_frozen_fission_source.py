"""Freeze a converged k-eigenvalue fission source into a fixed source.

Scaling convention verified here (TransportDriver::solve_eigenvalue + EigenSource):

* At the top of outer iteration i the driver calls ``set_eigval(1 / k_{i-1})`` and
  ``scale()``, so DURING the inner (DD) solve ``source.state == F psi_{i-1} /
  k_{i-1}`` -- this is what a DDSolver callback sees.
* After the inner solve the driver calls ``update(psi_i)``, so ``source.state ==
  F psi_i`` (UNSCALED) and ``source.total_source == sum(F psi_i)`` (a plain sum over
  every TT entry, ordinates included). ``k_i = sum over patches/ranks of
  total_source`` -- this is what a driver (outer-iteration) callback sees, with
  ``driver.last_k() == k_i``.
* No ``scale()`` follows the last ``update()``, so after ``solve_eigenvalue()``
  returns, ``source.state == F psi_final`` (unscaled) and ``result.k_eff == k_final ==
  sum(total_source)``.

Hence the frozen right-hand side is ``q = source.state / result.k_eff`` (since
``L psi = F psi / k`` with ``k = sum(F psi)`` at convergence). F applies the
angular integral (weights summing to 1), so ``q`` is identical on every ordinate and
equals the projected load b_{i,g} = int chi_g sum_g' nu_f psi_g' R_i dV / k; that
single-ordinate slice is exactly what ``FixedSource(projected_isotropic_source=...)``
expects. Solving the NON-fissile (otherwise identical) problem with it must
reproduce the eigenvalue angular flux.
"""

import torch
from igakit.cad import line, refine, ruled

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.driver import IGATransportDriver2D
from ttnte.math import ProductQuadrature
from ttnte.mesh import IGAMesh
from ttnte.physics import DGTransportAssemblerConfig, FixedSource
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
NUM_ANGULAR = 2
SOLVER_TOL = 1e-8


def _server(fissile):
    server = Server()
    mat = Material("Fuel")
    z = torch.zeros(2, dtype=DTYPE)
    mat.chi = torch.tensor([1.0, 0.0], dtype=DTYPE) if fissile else z.clone()
    mat.nu_fission = torch.tensor([0.02, 0.45], dtype=DTYPE) if fissile else z.clone()
    mat.fission = torch.tensor([0.008, 0.18], dtype=DTYPE) if fissile else z.clone()
    mat.kappa_fission = (
        torch.tensor([0.008, 0.18], dtype=DTYPE) * 200.0 if fissile else z.clone()
    )
    mat.total = torch.tensor([0.6, 1.4], dtype=DTYPE)
    mat.absorption = torch.tensor([0.02, 0.3], dtype=DTYPE)
    mat.scatter_gtg = torch.tensor([[[0.55, 0.03], [0.0, 1.1]]], dtype=DTYPE)
    mat.finalize()
    server.add_material(mat)
    server.finalize()
    return mat.label, server


def _driver(fissile, loads=None):
    """Two 2-D patches side by side (x in [0, 2] split at 1, y in [0, 2]), VACUUM outer
    boundary.

    `loads` maps gid -> projected_isotropic_source (dense).
    """
    mpi_context.init()
    torch.set_default_dtype(DTYPE)
    torch.autograd.set_grad_enabled(False)
    fill, server = _server(fissile)
    mesh = IGAMesh(mpi_context)
    patches = []
    for x0, x1 in [(0.0, 1.0), (1.0, 2.0)]:
        p = Patch.from_igakit(
            refine(
                ruled(line((x0, 0), (x1, 0)), line((x0, 2), (x1, 2))), [3, 5], [2, 2]
            ),
            device=DEVICE,
            dtype=DTYPE,
            fill=fill,
        )
        mesh.add_block(p)
        patches.append(p)
    mesh.connect()
    mesh.finalize()
    if loads is not None:
        for p in mesh.blocks:
            p.source = FixedSource(projected_isotropic_source=loads[p.gid])

    qset = ProductQuadrature.gauss_legendre_chebyshev(2, 4, 2)
    qset.to_(DEVICE, DTYPE)
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-12
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10)
    config.cross_jacobian_inverse = True
    driver = IGATransportDriver2D(mesh, server, mpi_context)
    driver.assemble(qset, config)

    dd_config = DDSolverConfig(
        tol=1e-10,
        max_iter=30,
        use_gpu=False,
        memory_policy=MemoryPolicy.OUT_OF_CORE,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=False,
    )
    strategy = BlockJacobiStrategy(dd_config)
    strategy.set_local_solver(
        AMEnSolver(nswp=6, eps=1e-11, kickrank=4, local_iterations=200, resets=6)
    )
    return driver, IGADDSolver(driver.mesh, strategy)


def _projected_load(state, scale):
    """Dense (space x NumDim, energy) load on ordinate 0, multiplied by `scale`."""
    num_cores = len(state.as_tt().cores)
    # to_dense() gives the row modes followed by a size-1 column mode per core.
    dense = state.to_dense()
    dense = dense.reshape(dense.shape[:num_cores])
    return scale * dense[(0,) * NUM_ANGULAR].clone()


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_frozen_fission_source_reproduces_eigen_flux():
    eig_driver, eig_dd = _driver(fissile=True)
    eig = eig_driver.solve_eigenvalue(
        eig_dd,
        tol=SOLVER_TOL,
        max_iter=200,
        k_tol=1e-10,
        verbose=False,
        clear_assemblers=False,
    )
    k = eig.k_eff
    systems = {s.gid: s for s in eig_dd.local_systems}

    # k == sum over patches of the stored (unscaled) F psi.
    total = sum(s.source.total_source for s in systems.values())
    assert abs(total - k) <= 1e-12 * abs(k)
    for s in systems.values():
        dense = s.source.state.to_dense()
        assert abs(dense.sum().item() - s.source.total_source) <= 1e-10 * abs(
            s.source.total_source
        )
        # F psi is isotropic: every ordinate carries the same load.
        assert _rel(dense, dense[:1, :1].expand_as(dense)) < 1e-10

    loads = {
        gid: _projected_load(s.source.state, 1.0 / k) for gid, s in systems.items()
    }

    fs_driver, fs_dd = _driver(fissile=False, loads=loads)
    fs = fs_driver.solve_fixed_source(
        fs_dd, tol=SOLVER_TOL, max_iter=200, verbose=False, clear_assemblers=False
    )
    fs_systems = {s.gid: s for s in fs_dd.local_systems}

    for gid, s in systems.items():
        # Same transport operator, and RHS == F psi / k exactly.
        A_eig = s.interior_op.as_tt().to_dense()
        A_fs = fs_systems[gid].interior_op.as_tt().to_dense()
        assert _rel(A_fs, A_eig) < 1e-12
        expected = s.source.state.to_dense() / k
        assert _rel(fs_systems[gid].source.state.to_dense(), expected) < 1e-12

    num = den = 0.0
    for gid in systems:
        a = eig.get_local_field(gid).to_dense()
        b = fs.get_local_field(gid).to_dense()
        num += ((a - b) ** 2).sum().item()
        den += (a**2).sum().item()
    err = (num / den) ** 0.5
    print(f"k = {k:.10f}, frozen-source angular flux rel L2 diff = {err:.3e}")
    # A few times the outer solver tolerance (observed ~2e-9).
    assert err < 5 * SOLVER_TOL
