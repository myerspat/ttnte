import os

from .eig import eig
from .linear_operator import LinearOperator
from .fixed_source import fixed_source

# Import ttnte.cpp.linalg if we can
cpp_available = False
if bool(os.environ.get("TTNTE_CPP_BACKEND", True)):
    try:
        from ttnte.cpp.linalg import TTOperator
        from ttnte.cpp.linalg import CSROperator

        cpp_available = True

    except ImportError:
        print("C++ backend was not configured, falling back to Python")
        from .tt_operator import TTOperator
        from .csr_operator import CSROperator

else:
    from .tt_operator import TTOperator
    from .csr_operator import CSROperator
