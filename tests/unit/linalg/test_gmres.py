import torch as tn
import pytest

try:
    from ttnte.cpp.linalg import (
        gmres,
        SparseOperator,
        Operator,
    )

    cpp_available = True

except ImportError:
    cpp_available = False


@pytest.mark.skipif(not cpp_available, reason="C++ backend failed to import")
def test_gmres_cpp():
    tn.set_default_dtype(tn.float64)

    # Create basic Laplace equation problem
    A = (
        1
        / (0.1**2)
        * (
            tn.diag(-2 * tn.ones(200))
            + tn.diag(tn.ones(199), diagonal=-1)
            + tn.diag(tn.ones(199), diagonal=1)
        )
    )
    b = tn.tensor([-1] + 198 * [0] + [-2], dtype=tn.float64).reshape((-1, 1))
    x_exac = tn.linalg.solve(A, b)

    # Create random operator and vector
    A = SparseOperator(A)
    assert isinstance(A, Operator)

    # ==================================================================
    # Test on CPU (batched)
    x_calc, rnorms = gmres(
        A, b, gpu_idx=None, tol=1e-8, restart=100, maxiter=200, solve_method="batched"
    )

    # Check solution
    assert not x_calc.is_cuda
    assert not A.tensor.is_cuda
    assert not b.is_cuda
    assert x_calc.shape == (200, 1)
    assert rnorms.shape[0] <= 200 + 1
    tn.testing.assert_close(x_calc, x_exac)

    # ==================================================================
    # Test on GPU (batched)
    if tn.cuda.is_available() and tn.cuda.device_count() > 0:
        # Run gmres
        x_calc, rnorms = gmres(
            A, b, gpu_idx=0, tol=1e-8, restart=100, maxiter=200, solve_method="batched"
        )

        # Check solution
        assert not x_calc.is_cuda
        assert not A.tensor.is_cuda
        assert not b.is_cuda
        assert rnorms.shape[0] <= 200 + 1
        assert x_calc.shape == (200, 1)
        tn.testing.assert_close(x_calc, x_exac)

    # ==================================================================
    # Test on GPU (incremental)
    x_calc, rnorms = gmres(
        A,
        b,
        gpu_idx=None,
        tol=1e-8,
        restart=100,
        maxiter=200,
        solve_method="incremental",
    )

    # Check solution
    assert not x_calc.is_cuda
    assert not A.tensor.is_cuda
    assert not b.is_cuda
    assert rnorms.shape[0] <= 200 + 1
    assert x_calc.shape == (200, 1)
    tn.testing.assert_close(x_calc, x_exac)

    # ==================================================================
    # Test on GPU (incremental)
    if tn.cuda.is_available() and tn.cuda.device_count() > 0:
        # Run gmres
        x_calc, rnorms = gmres(
            A,
            b,
            gpu_idx=0,
            tol=1e-8,
            restart=100,
            maxiter=200,
            solve_method="incremental",
        )

        # Check solution
        assert not x_calc.is_cuda
        assert not A.tensor.is_cuda
        assert not b.is_cuda
        assert rnorms.shape[0] <= 200 + 1
        assert x_calc.shape == (200, 1)
        tn.testing.assert_close(x_calc, x_exac)
