import os

# from .eig import eig
# from .linear_operator import LinearOperator
# from .fixed_source import fixed_source


def import_python():
    from .tt_operator import TTOperator
    from .sparse_operator import SparseOperator
    from .scatter_operator import ScatterOperator
    from .fission_operator import FissionOperator


# Import ttnte.cpp.linalg if we can
cpp_available = False
if bool(os.environ.get("TTNTE_CPP_BACKEND", True)):
    try:
        from ttnte.cpp.linalg import (
            TTOperator,
            SparseOperator,
            ScatterOperator,
            FissionOperator,
        )

        cpp_available = True

    except ImportError:
        print("C++ backend was not configured, falling back to Python")
        import_python()

else:
    import_python()
