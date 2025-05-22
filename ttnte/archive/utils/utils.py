import numpy as np


def get_degree(dim_size):
    """Get degree where the dimension size is 2 ** degree."""
    return int(np.round(np.log(dim_size) / np.log(2)))


def check_dim_size(name, dim_size):
    """Check dimension sizes are powers of 2."""
    if not ((dim_size & (dim_size - 1) == 0) and dim_size != 0):
        raise RuntimeError(f"Number of {name} must be a power of 2")
