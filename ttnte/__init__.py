# Determine if display is terminal or notebook
try:
    import IPython

    IS_NOTEBOOK = "Terminal" not in IPython.get_ipython().__class__.__name__

except (NameError, ImportError):
    IS_NOTEBOOK = False

# Make sure double precision is used everywhere
import torch as tn
import numpy as np

import warnings

# Suppress sparse tensor warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Sparse CSR tensor support is in beta state.*",
)

tn.set_default_dtype(tn.float64)

# =====================================================================
# NumPy 2.0 Compatibility Patch for upstream `igakit` dependency.
# igakit looks for np.in1d and np.setmember1d, which were removed in 2.0.
# We inject them back into the numpy namespace before igakit loads.
# =====================================================================
if not hasattr(np, "in1d"):
    np.in1d = lambda ar1, ar2, assume_unique=False, invert=False: np.isin(
        ar1, ar2, assume_unique=assume_unique, invert=invert
    )
if not hasattr(np, "setmember1d"):
    np.setmember1d = np.in1d

# Start MPI context
import atexit

from ttnte.parallel import mpi_context

# Register finalize method to run on Python exit
atexit.register(mpi_context.finalize)

# This should always be the last line of this file
__version__ = "0.0.0b0"
