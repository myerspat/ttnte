import time

import numpy as np

from tt_nte.benchmarks import bwr_core
from tt_nte.methods import DiscreteOrdinates
from tt_nte.solvers import ALS, AMEn

print("hello")
# Load unreflected data
xs_server, geometry, ordinates, regions = bwr_core(256, reflected=False)

# Initialize SN solver
start = time.time()
SN = DiscreteOrdinates(
    xs_server=xs_server,
    geometry=geometry,
    num_ordinates=int(4 * ordinates.shape[0]),
    tt_fmt="qtt",
    regions=regions,
    qtt_threshold=1e-10,
)
print(f"H = {SN.H}")
print(f"F = {SN.F}")
print(f"S = {SN.S}")
unref_setup = time.time() - start

# AMEn preconditioner (ALS solver)
start = time.time()
solver = ALS(method=SN, verbose=True)
solver.power(ranks=10, tol=1e-2, max_iter=250)
k0 = solver.k
psi0 = solver.psi

# AMEn solver
solver = AMEn(method=SN, verbose=True)
solver.power(max_iter=1000, k0=k0, psi0=psi0)
unref_k = solver.k
unref_solve = time.time() - start

# Load reflected data
xs_server, geometry, ordinates, regions = bwr_core(256, reflected=True)

# Initialize SN solver
start = time.time()
SN = DiscreteOrdinates(
    xs_server=xs_server,
    geometry=geometry,
    num_ordinates=int(4 * ordinates.shape[0]),
    tt_fmt="qtt",
    regions=regions,
    qtt_threshold=1e-10,
)
print(f"H = {SN.H}")
print(f"F = {SN.F}")
print(f"S = {SN.S}")
ref_setup = time.time() - start

# AMEn preconditioner (ALS solver)
start = time.time()
solver = ALS(method=SN, verbose=True)
solver.power(ranks=10, tol=1e-2, max_iter=250)
k0 = solver.k
psi0 = solver.psi

# AMEn solver
solver = AMEn(method=SN, verbose=True)
solver.power(max_iter=1000, k0=k0, psi0=psi0)
ref_k = solver.k
ref_solve = time.time() - start

print("\nSolutions")
print(
    f"Unreflected k = {np.round(unref_k, 5)}, "
    + f"Setup time = {np.round(unref_setup, 3)} s, "
    + f"Solve time = {np.round(unref_solve, 3)} s"
)
print(
    f"Reflected k = {np.round(ref_k, 5)}, "
    + f"Setup time = {np.round(ref_setup, 3)} s, "
    + f"Solve time = {np.round(ref_solve, 3)} s"
)
print(f"Delta = {abs(ref_k - unref_k) * 1e5} pcm")
