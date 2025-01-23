import numpy as np

import tt_nte.benchmarks as benchmarks
import tt_nte.solvers as solvers
from tt_nte.methods import DiscreteOrdinates


def test_pu_brick():
    # Set numpy random seed
    np.random.seed(42)

    # Get single-media problem geometry and XS server
    server, geometry = benchmarks.pu_brick(1024)

    # ----------------------------------------------------------------
    # SN in TT format

    # Initialize SN solver
    SN = DiscreteOrdinates(
        xs_server=server,
        geometry=geometry,
        num_ordinates=2,
        tt_fmt="tt",
    )

    # Assertions
    assert SN.H.matricize().shape == (1024 * 2, 1024 * 2)
    assert SN.S.matricize().shape == (1024 * 2, 1024 * 2)
    assert SN.F.matricize().shape == (1024 * 2, 1024 * 2)

    # Ordinates and solvers to test
    num_ordinates = [2, 8]
    tt_solvers = [solvers.Matrix, solvers.ALS, solvers.AMEn]

    # Expected solutions
    expected_k = [0.80418, 0.99176]
    expected_psi = []

    for i in range(len(num_ordinates)):
        # Change number of ordinates in SN
        SN.update_settings(num_ordinates=num_ordinates[i])

        # Run matrix generalized eigenvalue solver
        solver = solvers.Matrix(method=SN, verbose=True)
        solver.ges()

        # Assertions
        assert abs(expected_k[i] - solver.k) < 0.00005
        assert solver.psi.shape == (1024 * num_ordinates[i],)

        # Save psi
        expected_psi.append(solver.psi)

        for solver in tt_solvers:
            # Initialize solver
            solver = solver(method=SN, verbose=True)

            # Run power iteration
            solver.power()

            # Assertions
            assert abs(expected_k[i] - solver.k) < 0.00005
            assert (
                solver.psi.shape
                if solver.__class__.__name__ == "Matrix"
                else solver.psi.matricize().shape == (1024 * num_ordinates[i],)
            )
            assert (
                np.linalg.norm(
                    expected_psi[i] - solver.psi
                    if solver.__class__.__name__ == "Matrix"
                    else expected_psi[i] - solver.psi.matricize()
                )
                < 0.002
            )

    # ----------------------------------------------------------------
    # SN in QTT format
    qtt_solvers = [solvers.MALS]
    qtt_solver_configs = [{"max_rank": 20}, {}]

    for i in range(len(num_ordinates)):
        # Change number of ordinates in SN
        SN.update_settings(num_ordinates=num_ordinates[i], tt_fmt="qtt")

        for j in range(len(qtt_solvers)):
            # Initialize solver
            solver = qtt_solvers[j](method=SN, verbose=True, **qtt_solver_configs[j])

            # Run power iteration
            solver.power(ranks=4, tol=3e-5)

            # Assertions
            assert abs(expected_k[i] - solver.k) < 0.00005
            assert (
                solver.psi.shape
                if solver.__class__.__name__ == "Matrix"
                else solver.psi.matricize().shape == (1024 * num_ordinates[i],)
            )
            assert (
                np.linalg.norm(
                    expected_psi[i] - solver.psi
                    if solver.__class__.__name__ == "Matrix"
                    else expected_psi[i] - solver.psi.matricize()
                )
                < 0.002
            )


def test_research_reactor_multi_region():
    # Set numpy random seed
    np.random.seed(42)

    # Get research reactor XS server and geometry
    xs_server, geometry = benchmarks.research_reactor_multi_region(
        [133, 761, 132], "vacuum"
    )

    # Initialize SN solver
    SN = DiscreteOrdinates(
        xs_server=xs_server,
        geometry=geometry,
        num_ordinates=2,
    )

    # Ordinates to test
    num_ordinates = [2, 8]

    # Expected solutions
    expected_k = [0.99055, 0.99975]

    for i in range(len(num_ordinates)):
        N = num_ordinates[i]

        # Change number of ordinates in SN
        SN.update_settings(num_ordinates=N, tt_fmt="qtt")

        # Run solver
        solver = solvers.AMEn(method=SN, verbose=True)
        solver.power(max_iter=2000)

        print(f"N = {N}, k = {solver.k}")

        # Assertions
        assert SN.H.matricize().shape == (1024 * N * 2, 1024 * N * 2)
        assert SN.S.matricize().shape == (1024 * N * 2, 1024 * N * 2)
        assert SN.F.matricize().shape == (1024 * N * 2, 1024 * N * 2)
        assert abs(expected_k[i] - solver.k) < 0.00010

    # Get research reactor XS server and geometry
    xs_server, geometry = benchmarks.research_reactor_multi_region(
        [133, 380], "reflective"
    )

    # Initialize SN solver
    SN = DiscreteOrdinates(
        xs_server=xs_server,
        geometry=geometry,
        num_ordinates=2,
    )

    for i in range(len(num_ordinates)):
        N = num_ordinates[i]

        # Change number of ordinates in SN
        SN.update_settings(num_ordinates=N, tt_fmt="qtt")

        # Run solver
        solver = solvers.AMEn(method=SN, verbose=True)
        solver.power(max_iter=2000)

        print(f"N = {N}, k = {solver.k}")

        # Assertions
        assert SN.H.matricize().shape == (512 * N * 2, 512 * N * 2)
        assert SN.S.matricize().shape == (512 * N * 2, 512 * N * 2)
        assert SN.F.matricize().shape == (512 * N * 2, 512 * N * 2)
        assert abs(expected_k[i] - solver.k) < 0.00010


def test_research_reactor_multi_region_infinite():
    # Set numpy random seed
    np.random.seed(42)

    # Get single-media problem geometry and XS server
    xs_server, geometry = benchmarks.research_reactor_multi_region(
        [300, 725], infinite=True
    )

    # ----------------------------------------------------------------
    # SN in TT format

    # Initialize SN solver
    SN = DiscreteOrdinates(
        xs_server=xs_server,
        geometry=geometry,
        num_ordinates=2,
        tt_fmt="tt",
    )

    num_ordinates = [8]
    correct_k = 1.365821
    current_k = 1.25

    for i in range(len(num_ordinates)):
        # Change number of ordinates in SN
        SN.update_settings(num_ordinates=num_ordinates[i])

        solver = solvers.Matrix(method=SN, verbose=True)
        solver.ges()

        print(f"N = {num_ordinates[i]}, k = {solver.k}")

        # Assertions
        assert round(solver.k, 6) >= current_k and round(solver.k, 6) <= correct_k

        current_k = round(solver.k, 6)


def test_research_reactor_anisotropic():
    # Set numpy random seed
    np.random.seed(42)

    # Get research reactor XS server and geometry
    xs_server, geometry = benchmarks.research_reactor_anisotropic(1024, "reflective")

    # Initialize SN solver
    SN = DiscreteOrdinates(
        xs_server=xs_server,
        geometry=geometry,
        num_ordinates=2,
    )

    # Ordinates to test
    num_ordinates = [2, 8]

    # Expected solutions
    expected_k = [0.97220, 0.99939]

    for i in range(len(num_ordinates)):
        N = num_ordinates[i]

        # Change number of ordinates in SN
        SN.update_settings(num_ordinates=N, tt_fmt="qtt")

        # Run solver
        solver = solvers.AMEn(method=SN, verbose=True)
        solver.power(max_iter=2000)

        # Assertions
        assert SN.H.matricize().shape == (1024 * N * 2, 1024 * N * 2)
        assert SN.S.matricize().shape == (1024 * N * 2, 1024 * N * 2)
        assert SN.F.matricize().shape == (1024 * N * 2, 1024 * N * 2)
        assert abs(expected_k[i] - solver.k) < 0.00010


def test_pu_brick_infinite_2d():
    # Set numpy random seed
    np.random.seed(42)

    # Get single-media problem geometry and XS server
    xs_server, geometry = benchmarks.pu_brick_2d(512, 16, infinite=True)

    # ----------------------------------------------------------------
    # SN in TT format

    # Initialize SN solver
    SN = DiscreteOrdinates(
        xs_server=xs_server,
        geometry=geometry,
        num_ordinates=4,
        tt_fmt="tt",
    )

    num_ordinates = [4, 16]
    correct_k = 2.612903
    current_k = 2.5

    for i in range(len(num_ordinates)):
        # Change number of ordinates in SN
        SN.update_settings(num_ordinates=num_ordinates[i])

        solver = solvers.AMEn(method=SN, verbose=True)
        solver.power(ranks=4, tol=1e-4)

        print(f"N = {num_ordinates[i]}, k = {solver.k}")

        # Assertions
        assert round(solver.k, 6) >= current_k and round(solver.k, 6) <= correct_k

        current_k = round(solver.k, 6)


def test_research_reactor_anisotropic_2d():
    # Set numpy random seed
    np.random.seed(42)

    # Get single-media problem geometry and XS server
    server, geometry = benchmarks.research_reactor_anisotropic_2d(
        512, 4, right_bc="vacuum"
    )

    # Initialize SN solver
    SN = DiscreteOrdinates(
        xs_server=server,
        geometry=geometry,
        num_ordinates=4,
    )
    assert SN.H.row_dims == SN.H.col_dims
    assert SN.S.row_dims == SN.S.col_dims
    assert SN.F.row_dims == SN.F.col_dims
    assert SN.H.row_dims == [4, 2, 4, 512]
    assert SN.S.row_dims == [4, 2, 4, 512]
    assert SN.F.row_dims == [4, 2, 4, 512]

    # Ordinates to test
    num_ordinates = [4, 16]

    # Old k
    old_k = 0.85

    for i in range(len(num_ordinates)):
        N = num_ordinates[i]

        # Change number of ordinates in SN
        SN.update_settings(num_ordinates=N, tt_fmt="qtt")

        # Run solver
        solver = solvers.AMEn(method=SN, verbose=True)
        solver.power(max_iter=2000)

        # Assertions
        assert solver.k > old_k and solver.k < 1.0

        old_k = solver.k


def test_research_reactor_multi_region_2d():
    # Set numpy random seed
    np.random.seed(42)

    # Get research reactor XS server and geometry
    xs_server, geometry = benchmarks.research_reactor_multi_region_2d(
        [133, 380], 4, "reflective"
    )

    # Initialize SN solver
    SN = DiscreteOrdinates(
        xs_server=xs_server,
        geometry=geometry,
        num_ordinates=4,
    )
    assert SN.H.row_dims == SN.H.col_dims
    assert SN.S.row_dims == SN.S.col_dims
    assert SN.F.row_dims == SN.F.col_dims
    assert SN.H.row_dims == [4, 2, 4, 512]
    assert SN.S.row_dims == [4, 2, 4, 512]
    assert SN.F.row_dims == [4, 2, 4, 512]

    # Ordinates to test
    num_ordinates = [4, 16]

    # Old k
    old_k = 0.85

    for i in range(len(num_ordinates)):
        N = num_ordinates[i]

        # Change number of ordinates in SN
        SN.update_settings(num_ordinates=N, tt_fmt="qtt")

        # Run solver
        solver = solvers.AMEn(method=SN, verbose=True)
        solver.power(max_iter=2000)

        # Assertions
        assert solver.k > old_k and solver.k < 1.0

        old_k = solver.k
