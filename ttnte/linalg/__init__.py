import os
import warnings


# Import ttnte.cpp.linalg if we can
cpp_available = False

# Get environment variable
TTNTE_CPP_BACKEND = os.environ.get("TTNTE_CPP_BACKEND", True)

if isinstance(TTNTE_CPP_BACKEND, str):
    TTNTE_CPP_BACKEND = (
        True if TTNTE_CPP_BACKEND.lower() in ("1", "true", "on", "yes") else False
    )

if TTNTE_CPP_BACKEND:
    try:
        from ttnte.cpp.linalg import (
            Operator,
            TTOperator,
            SparseOperator,
            ScatterOperator,
            FissionOperator,
            LinearOperator,
            gmres,
        )

        cpp_available = True

    except ImportError:
        warnings.warn("C++ backend was not configured, falling back to Python")
        from .tt_operator import TTOperator
        from .sparse_operator import SparseOperator
        from .scatter_operator import ScatterOperator
        from .fission_operator import FissionOperator
        from .linear_operator import LinearOperator
        from .operator import Operator
        from .gmres import gmres

else:
    from .tt_operator import TTOperator
    from .sparse_operator import SparseOperator
    from .scatter_operator import ScatterOperator
    from .fission_operator import FissionOperator
    from .linear_operator import LinearOperator
    from .operator import Operator
    from .gmres import gmres

from .eig import eig

# from .fixed_source import fixed_source
