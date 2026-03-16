# Determine if display is terminal or notebook
try:
    import IPython

    IS_NOTEBOOK = "Terminal" not in IPython.get_ipython().__class__.__name__

except (NameError, ImportError):
    IS_NOTEBOOK = False

# Make sure double precision is used everywhere
import torch as tn

import warnings

# Suppress sparse tensor warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Sparse CSR tensor support is in beta state.*",
)

tn.set_default_dtype(tn.float64)

# Start MPI context
import atexit

from ttnte.utils import mpi_context

# Initialize MPI
mpi_context.init()

# Register finalize method to run on Python exit
atexit.register(mpi_context.finalize)

# This should always be the last line of this file
__version__ = "0.0.0b0"
