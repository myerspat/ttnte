import numpy as np
from igakit import cad

from ttnte.diagnostics import Timer
from ttnte.iga import IGAMesh
from ttnte.xs.benchmarks import c5g7


def mesh_pincell(factor, degree):
    # Create quarter circle NURBS surface
    radius = 0.54  # cm
    pitch = 1.26  # cm
    c0 = cad.circle(radius=radius, angle=np.pi / 2)
    c1 = c0.slice(0, 0, 0.5)
    c2 = c0.slice(0, 0.5, 1)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))
    fuel1 = cad.ruled(l0, c1)
    fuel2 = cad.ruled(l0, c2)

    # Create water patch
    l1 = cad.line(p0=(pitch / 2, 0), p1=(pitch / 2, pitch / 2))
    l2 = cad.line(p0=(pitch / 2, pitch / 2), p1=(0, pitch / 2))
    mod1 = cad.ruled(c1, l1)
    mod2 = cad.ruled(c2, l2)

    # NURBS surfaces
    patches = {fuel1: "UO2", fuel2: "UO2", mod1: "Water", mod2: "Water"}

    # Create IGA mesh object
    mesh = IGAMesh(patches)

    # Refine mesh
    for p in range(mesh.num_patches):
        mesh.refine(p, factor, degree)

    # Finalize mesh
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_condition(("left", "bottom", "top", "right"))

    # Finalize mesh
    mesh.finalize()

    return mesh


if __name__ == "__main__":
    # Get XS information
    xs_server = c5g7()

    # Monte Carlo solution
    k_mc = 1.325593
    phi_mc = np.load("../../../notebooks/pincell/openmc/data/mesh_flux.npy")

    # Scaling values
    num_ordinates_list = [16, 64, 256, 1024, 4096, 16384]
    eps_list = [1e-10]
    factor_list = [16]
    degree_list = [2, 3, 6]

    # ===========================================
    # Warmup CPU

    # Build scaling class
    timer = Timer(
        xs_server=xs_server,
        mesh_generator=mesh_pincell,
        num_ordinates_list=[num_ordinates_list[0]],
        eps_list=[eps_list[0]],
        factor_list=[factor_list[0]],
        degree_list=[degree_list[0]],
    )

    # Configure for CPU
    timer.device = None
    timer.info_prefix = "/dev/null/"
    timer.results_prefix = "/dev/null"
    timer.data_prefix = "/dev/null/"
    timer.linear_solver_opts["resets"] = 10
    timer.linear_solver_opts["max_iterations"] = 200

    # Run scaler
    timer.run()

    # ===========================================
    # Run CPU scaling

    # Build scaling class
    timer = Timer(
        xs_server=xs_server,
        mesh_generator=mesh_pincell,
        num_ordinates_list=num_ordinates_list,
        eps_list=eps_list,
        factor_list=factor_list,
        degree_list=degree_list,
    )

    # Configure for CPU
    timer.device = None
    timer.info_prefix = "./cpu/info/"
    timer.results_prefix = "./cpu/"
    timer.data_prefix = "./cpu/data/"
    timer.linear_solver_opts["resets"] = 10
    timer.linear_solver_opts["max_iterations"] = 200

    # Run scaler
    timer.run(k_mc, phi_mc)

    # ===========================================
    # Warmup GPU

    # Build scaling class
    timer = Timer(
        xs_server=xs_server,
        mesh_generator=mesh_pincell,
        num_ordinates_list=[num_ordinates_list[0]],
        eps_list=[eps_list[0]],
        factor_list=[factor_list[0]],
        degree_list=[degree_list[0]],
    )

    # Configure for CPU
    timer.device = 0
    timer.info_prefix = "/dev/null/"
    timer.results_prefix = "/dev/null"
    timer.data_prefix = "/dev/null/"
    timer.linear_solver_opts["resets"] = 10
    timer.linear_solver_opts["max_iterations"] = 200

    # Run scaler
    timer.run()

    # ===========================================
    # Run GPU

    # Build scaling class
    timer = Timer(
        xs_server=xs_server,
        mesh_generator=mesh_pincell,
        num_ordinates_list=num_ordinates_list,
        eps_list=eps_list,
        factor_list=factor_list,
        degree_list=degree_list,
    )

    # Configure for CPU
    timer.device = 0
    timer.info_prefix = "./gpu/info/"
    timer.results_prefix = "./gpu/"
    timer.data_prefix = "./gpu/data/"
    timer.linear_solver_opts["resets"] = 10
    timer.linear_solver_opts["max_iterations"] = 200

    # Run scaler
    timer.run(k_mc, phi_mc)
