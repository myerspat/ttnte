import time

import numpy as np

from tt_nte.benchmarks import bwr_assembly
from tt_nte.methods import DiscreteOrdinates
from tt_nte.solvers import ALS, AMEn

# Load unreflected data
xs_server, geometry, ordinates = bwr_assembly(264, control_rod=False)

# Initialize SN solver
start = time.time()
SN = DiscreteOrdinates(
    xs_server=xs_server,
    geometry=geometry,
    num_ordinates=int(4 * ordinates.shape[0]),
    tt_fmt="qtt",
    qtt_threshold=1e-10,
)
print(f"H = {SN.H}")
print(f"F = {SN.F}")
print(f"S = {SN.S}")
nocont_setup = time.time() - start

# AMEn preconditioner (ALS solver)
start = time.time()
solver = ALS(method=SN, verbose=True)
solver.power(ranks=10, tol=1e-2, max_iter=250)
k0 = solver.k
psi0 = solver.psi

# AMEn solver
solver = AMEn(method=SN, verbose=True)
solver.power(max_iter=1000, k0=k0, psi0=psi0)
nocont_k = solver.k
nocont_solve = time.time() - start

# Load reflected data
xs_server, geometry, ordinates = bwr_assembly(256, control_rod=True)

# Initialize SN solver
start = time.time()
SN = DiscreteOrdinates(
    xs_server=xs_server,
    geometry=geometry,
    num_ordinates=int(4 * ordinates.shape[0]),
    tt_fmt="qtt",
    qtt_threshold=1e-10,
)
print(f"H = {SN.H}")
print(f"F = {SN.F}")
print(f"S = {SN.S}")
cont_setup = time.time() - start

# AMEn preconditioner (ALS solver)
start = time.time()
solver = ALS(method=SN, verbose=True)
solver.power(ranks=10, tol=1e-2, max_iter=250)
k0 = solver.k
psi0 = solver.psi

# AMEn solver
solver = AMEn(method=SN, verbose=True)
solver.power(max_iter=1000, k0=k0, psi0=psi0)
cont_k = solver.k
cont_solve = time.time() - start

print("\nSolutions")
print(
    f"No control rod k = {np.round(nocont_k, 5)}, "
    + f"Setup time = {np.round(nocont_setup, 3)} s, "
    + f"Solve time = {np.round(nocont_solve, 3)} s"
)
print(
    f"Control Rod k = {np.round(cont_k, 5)}, "
    + f"Setup time = {np.round(cont_setup, 3)} s, "
    + f"Solve time = {np.round(cont_solve, 3)} s"
)
