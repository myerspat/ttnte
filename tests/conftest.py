import multiprocessing
import pytest


@pytest.fixture(scope="session", autouse=True)
def always_spawn():
    multiprocessing.set_start_method("spawn")


import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = lambda ar1, ar2, assume_unique=False, invert=False: np.isin(
        ar1, ar2, assume_unique=assume_unique, invert=invert
    )
if not hasattr(np, "setmember1d"):
    np.setmember1d = np.in1d
