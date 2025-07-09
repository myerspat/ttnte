import numpy as np
from igakit import cad

from ttnte.diagnostics import Timer
from ttnte.iga import IGAMesh
from ttnte.xs.benchmarks import pu239


def mesh_quarter_circle(factor, degree):
    # Create NURBS surface
    rc = 4.279960  # Critical radius (cm)
    c0 = cad.circle(radius=rc, angle=np.pi / 2)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))
    patches = {cad.ruled(l0, c0): "Pu-239"}

    # Create mesh
    mesh = IGAMesh(patches)

    # Refine mesh resolution
    mesh.refine(0, factor=factor, degree=degree)

    # Connect patches
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_condition(("left", "bottom"))

    # Finalize mesh
    mesh.finalize()

    return mesh


if __name__ == "__main__":
    # Get XS information
    xs_server = pu239(num_groups=2)

    # Monte Carlo solution
    k_mc = 1.0

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
        mesh_generator=mesh_quarter_circle,
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

    # Run scaler
    timer.run()

    # ===========================================
    # Run CPU scaling

    # Build scaling class
    timer = Timer(
        xs_server=xs_server,
        mesh_generator=mesh_square,
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

    # Run scaler
    timer.run(k_mc)

    # ===========================================
    # Warmup GPU

    # Build scaling class
    timer = Timer(
        xs_server=xs_server,
        mesh_generator=mesh_square,
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

    # Run scaler
    timer.run()

    # ===========================================
    # Run GPU

    # Build scaling class
    timer = Timer(
        xs_server=xs_server,
        mesh_generator=mesh_square,
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

    # Run scaler
    timer.run(k_mc)
