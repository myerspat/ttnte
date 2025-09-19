import torch as tn
import pytest

from ttnte.linalg.operator import Operator as O_python
from ttnte.linalg.sparse_operator import SparseOperator as SO_python
from ttnte.linalg.gmres import gmres as gmres_python

try:
    from ttnte.cpp.linalg import Operator as O_cpp
    from ttnte.cpp.linalg import SparseOperator as SO_cpp
    from ttnte.cpp.linalg import gmres as gmres_cpp

    cpp_available = True

except ImportError:
    cpp_available = False


def run_gmres(Operator, SparseOperator, gmres):
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
        A=A,
        b=b,
        gpu_idx=None,
        tol=1e-8,
        restart=100,
        maxiter=200,
        solve_method="batched",
    )

    # Check solution
    assert not x_calc.is_cuda
    assert not A.tensor.is_cuda
    assert not b.is_cuda
    assert x_calc.shape == (200, 1)
    if rnorms is not None:
        assert rnorms.shape[0] <= 200 + 1
    tn.testing.assert_close(x_calc, x_exac)

    # ==================================================================
    # Test on GPU (batched)
    if tn.cuda.is_available() and tn.cuda.device_count() > 0:
        # Run gmres
        x_calc, rnorms = gmres(
            A=A,
            b=b,
            gpu_idx=0,
            tol=1e-8,
            restart=100,
            maxiter=200,
            solve_method="batched",
        )

        # Check solution
        assert not x_calc.is_cuda
        assert not A.tensor.is_cuda
        assert not b.is_cuda
        if rnorms is not None:
            assert rnorms.shape[0] <= 200 + 1
        assert x_calc.shape == (200, 1)
        tn.testing.assert_close(x_calc, x_exac)

    # ==================================================================
    # Test on GPU (incremental)
    x_calc, rnorms = gmres(
        A=A,
        b=b,
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
    if rnorms is not None:
        assert rnorms.shape[0] <= 200 + 1
    assert x_calc.shape == (200, 1)
    tn.testing.assert_close(x_calc, x_exac)

    # ==================================================================
    # Test on GPU (incremental)
    if tn.cuda.is_available() and tn.cuda.device_count() > 0:
        # Run gmres
        x_calc, rnorms = gmres(
            A=A,
            b=b,
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
        if rnorms is not None:
            assert rnorms.shape[0] <= 200 + 1
        assert x_calc.shape == (200, 1)
        tn.testing.assert_close(x_calc, x_exac)


def test_gmres_python():
    run_gmres(O_python, SO_python, gmres_python)


@pytest.mark.skipif(not cpp_available, reason="C++ backend failed to import")
def test_gmres_cpp():
    run_gmres(O_cpp, SO_cpp, gmres_cpp)
