import torch
from igakit.cad import refine, line

from ttnte import mpi_context
from ttnte.xs import Server, Material
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import BoundaryType, DGTransportAssemblerConfig, FixedSource
from ttnte.math import QuadratureSet1D
from ttnte.driver import IGATransportDriver1D
from ttnte.solvers import (
    DDSolverConfig,
    MemoryPolicy,
    AMEnSolver,
    BlockJacobiStrategy,
    IGADDSolver,
    ExecMode,
    CommMode,
)


def _pure_absorber(sigma_t, device, dtype):
    """A 1-group material with no scattering/fission -- every direction's transport
    equation decouples (mu dpsi/dx + sigma_t psi = 0), so the exact per-ordinate
    solution is a plain exponential and the only discretization error left is spatial
    (DG/IGA basis), not angular quadrature."""
    server = Server()
    mat = Material("Absorber")
    mat.chi = torch.zeros(1, dtype=dtype, device=device)
    mat.total = torch.tensor([sigma_t], dtype=dtype, device=device)
    mat.nu_fission = torch.zeros(1, dtype=dtype, device=device)
    mat.fission = torch.zeros(1, dtype=dtype, device=device)
    mat.absorption = torch.tensor([sigma_t], dtype=dtype, device=device)
    mat.scatter_gtg = torch.zeros((1, 1, 1), dtype=dtype, device=device)
    mat.finalize()
    server.add_material(mat)
    server.finalize()
    return mat.label, server


def test_incident_beam_matches_exact_attenuation():
    """Single-patch, pure-absorber slab with a prescribed incident flux Q on the x_min
    face (VACUUM at x_max): since sigma_s = 0, every discrete.

    ordinate mu solves mu dpsi/dx + sigma_t psi = 0 independently, giving the
    closed form psi(x, mu) = Q * exp(-sigma_t x / mu) for incoming directions
    (mu > 0) and psi(x, mu) = 0 for mu < 0 (VACUUM at x_max forces psi(L, mu) =
    0 for those, and the homogeneous ODE with a zero boundary value and no
    source is zero everywhere along that characteristic). Reducing with the SAME
    quadrature points/weights the assembler itself uses (QuadratureSet1D's
    weights are normalized to sum to 1 -- see src/math/quadrature_set.cpp) gives
    an exact reference scalar flux to compare solve_fixed_source()'s output
    against, isolating spatial discretization error only.
    """
    device = torch.device("cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    mpi_context.init()

    sigma_t = 0.75
    Q = 2.0
    L = 3.0

    fill, xs_server = _pure_absorber(sigma_t, device, dtype)

    patch = Patch.from_igakit(
        refine(line((0, 0), (L, 0)), 12, 3),
        device=device,
        dtype=dtype,
        fill=fill,
    )

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    patch.set_boundary_source(
        0, False, FixedSource(isotropic_strength=torch.tensor([Q], dtype=dtype))
    )
    patch.set_boundary_type(0, True, BoundaryType.VACUUM)

    assert patch.get_boundary_info(0, False).type == BoundaryType.INCIDENT
    assert patch.get_boundary_info(0, True).type == BoundaryType.VACUUM

    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(device, dtype)

    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = 0
    config.cross_jacobian_inverse = True

    driver = IGATransportDriver1D(mesh, xs_server, mpi_context)
    driver.assemble(qset, config)

    dd_config = DDSolverConfig(
        tol=1e-7,
        max_iter=10,
        use_gpu=False,
        memory_policy=MemoryPolicy.OUT_OF_CORE,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=True,
    )
    strategy = BlockJacobiStrategy(dd_config)
    strategy.set_local_solver(
        AMEnSolver(nswp=4, eps=1e-8, kickrank=4, local_iterations=80, resets=4)
    )
    dd_solver = IGADDSolver(driver.mesh, strategy)

    # DDSolver.set_callback()/TransportDriver.set_callback(): the callback is
    # handed the solver/driver itself (not a bespoke snapshot struct), so it
    # can pull whatever it needs off their existing getters.
    dd_calls = []

    def dd_callback(solver):
        dd_calls.append(
            {
                "j": solver.last_num_iterations(),
                "eps": solver.eps,
                "states_defined": all(s.state.defined() for s in solver.local_systems),
            }
        )

    dd_solver.set_callback(dd_callback)

    outer_calls = []

    def outer_callback(driver_arg, solver_arg):
        outer_calls.append(
            {
                "i": driver_arg.last_num_outer_iterations(),
                "k": driver_arg.last_k(),
                "eps": solver_arg.eps,
            }
        )

    driver.set_callback(outer_callback)

    result = driver.solve_fixed_source(dd_solver, tol=1e-6, max_iter=50)
    assert result.k_eff is None

    # Both callbacks must have fired, with eps > 0 (the tolerance that
    # actually produced that iteration's state, not a stale/unset value),
    # and every observed patch state defined.
    assert len(dd_calls) > 0
    assert all(c["eps"] > 0 for c in dd_calls)
    assert all(c["states_defined"] for c in dd_calls)

    # Fixed-source has no eigenvalue -- last_k() must stay None throughout.
    assert len(outer_calls) > 0
    assert all(c["k"] is None for c in outer_calls)
    assert all(c["eps"] > 0 for c in outer_calls)
    assert [c["i"] for c in outer_calls] == list(range(1, len(outer_calls) + 1))
    assert driver.last_num_outer_iterations() == len(outer_calls)

    scalar_result = result.compute_scalar_flux()
    gid = patch.gid
    field = scalar_result.get_local_field(gid).to_dense().reshape(-1)

    # evaluate_field()/evaluate() both take PARAMETRIC coordinates in [0, 1],
    # not physical ones -- map through the patch's own geometry evaluation to
    # get the physical x used by the exact attenuation formula below.
    plot_resolution = 9
    t = torch.linspace(0.0, 1.0, plot_resolution, dtype=dtype)
    values = patch.evaluate_field(field.reshape(-1, 1), t.reshape(-1, 1))[..., 0]
    x_phys = patch.evaluate(t.reshape(-1, 1))[..., 0]

    mu = qset.points.reshape(-1)
    w = qset.weights.reshape(-1)
    incoming = mu > 0
    mu_in = mu[incoming]
    w_in = w[incoming]

    exact = torch.stack(
        [(w_in * Q * torch.exp(-sigma_t * xi / mu_in)).sum() for xi in x_phys]
    )

    # A boundary-layer effect right at the incident face (x=0) makes that one
    # sample point converge more slowly with mesh refinement than the rest of
    # the slab; check it separately with a looser tolerance.
    torch.testing.assert_close(values[1:], exact[1:], atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(values[:1], exact[:1], atol=2e-2, rtol=2e-2)
