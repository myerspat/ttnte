import itertools
import pickle
import time
from typing import Callable, List, Optional, Union

import numpy as np
import torch as tn
import torchtt as tntt

from ttnte.assemblers import MatrixAssembler, TTAssembler
from ttnte.iga import IGAMesh
from ttnte.linalg import LinearOperator, eig
from ttnte.xs import Server


class Timer(object):
    """
    Class for timing and collecting information as discretization changes for a given
    problem.

    Attributes
    ----------
    info_prefix: str, default="./info/"
        Directory for saving TT information. Each run will be named
        ``info_prefix + f"tt_info_{num_ordinates}_{factor}_{degree}.csv"``
        where ``num_ordinates``, ``factor``, and ``degree`` is the
        current discretization.
    results_prefix: str, default="./"
        Directory for final results. The final file will be
        ``results_prefix + "results.pkl"``.
    data_prefix: str, default="./data/"
        Directory for scalar flux data. The files are saved as
        ``data_prefix + f"phi_{format}_{num_ordinates}_{factor}_{degree}.npy"``
        where ``format`` is ``"m"`` for matrix, ``"tt"`` for tensor trains,
        and ``"mix"`` for mixed. Mixed includes loss, fission, and scattering
        operators in TT format while the boundary operators are in COOrdinate
        sparse matrix format.
    device: int or None, default=None
        Device to solve the problem on.
    tol: float, default=1e-8
        Tolerance for the power iteration.
    max_iter: int, default=500
        Maximum number of power iterations.
    linear_solver_opts: dict, default={"max_iterations": 100,
    "threshold": 1e-10, "resets": 5}
        GMRES options.
    """

    info_prefix = "./info/"
    results_prefix = "./"
    data_prefix = "./data/"
    device = None
    tol = 1e-8
    max_iter = 500
    linear_solver_opts = {
        "max_iterations": 100,
        "threshold": 1e-10,
        "resets": 5,
    }

    def __init__(
        self,
        xs_server: Server,
        mesh_generator: Callable,
        num_ordinates_list: List[int],
        eps_list: List[float],
        factor_list: Union[List[List[int]], List[int]],
        degree_list: Union[List[List[int]], List[int]],
    ):
        """
        Initialize Timer.

        Parameters
        ----------
        xs_server: ttnte.xs.Server
            XS server containing XS information.
        mesh_generator: callable
            Function that has two arguments: ``factor`` and ``degree``
            which are passed to ``ttnte.iga.IGAMesh.refine()``. The
            result of the function if the ``ttnte.iga.IGAMesh`` for
            that problem.
        num_ordinates_list: list of int
            Possible discretizations for discrete ordinates.
        eps_list: list of float
            List of possible rounding eps for TT rounding.
        factor_list: list of int or list of list of int
            Possible number of elements along each parametric direction.
        degree_list: list of int or list of list of int
            Possible order of basis functions along each parametric direction.
        """
        # Build data
        self._xs_server = xs_server
        self._mesh_generator = mesh_generator

        # Sequences
        self._num_ordinates_list = num_ordinates_list
        self._eps_list = eps_list
        self._factor_list = factor_list
        self._degree_list = degree_list
        tn.set_default_dtype(tn.float64)

    # ========================================================================
    # Refinement methods

    def run(self, k_ref: Optional[float] = None, phi_ref: Optional[np.ndarray] = None):
        """"""
        # Initialize results data
        self._results = {
            "format": [],
            "num_ordinates": [],
            "eps": [],
            "factor": [],
            "degree": [],
            "shape": [],
            "avg_element_area": [],
            "H": {"ranks": [], "num_entries": [], "compression": []},
            "S": {"ranks": [], "num_entries": [], "compression": []},
            "F": {"ranks": [], "num_entries": [], "compression": []},
            "B_in": {"ranks": [], "num_entries": [], "compression": []},
            "B_out": {"ranks": [], "num_entries": [], "compression": []},
            "LHS": {"num_entries": [], "compression": []},
            "assembly_time": [],
            "H_MV_avg_time": [],
            "H_MV_std_time": [],
            "solve_time": [],
            "k": [],
        }

        # Add arrays for eigenvalue and scalar flux error
        if k_ref:
            self._results["k_relative_error"] = []
        if phi_ref:
            self._results["phi_relative_l2_error"] = []

            # Generate coarse mesh
            mesh = self._mesh_generator(
                factor=self._factor_list[0], degree=self._degree_list[0]
            )

            # Map rectangular mesh
            self._pids, self._coords = mesh.map_regular_mesh(
                shape=phi_ref.shape[1:], N=(5, 5)
            )

        # Begin iterating
        for num_ordinates, factor, degree in itertools.product(
            self._num_ordinates_list, self._factor_list, self._degree_list
        ):
            factor = factor if isinstance(factor, list) else 2 * [factor]
            degree = degree if isinstance(degree, list) else 2 * [degree]

            # Ensure assembly wont fail
            if (degree[0] - factor[0]) > 1 or (degree[1] - factor[1]) > 1:
                continue

            # Append current config
            self._results["format"].append("COO")
            self._results["num_ordinates"].append(num_ordinates)
            self._results["factor"].append(factor)
            self._results["degree"].append(degree)
            self._results["eps"].append(None)

            # Create IGA mesh
            mesh = self._mesh_generator(factor=factor, degree=degree)

            # Build COOrdinate matrix config
            start = time.time()
            assembler = MatrixAssembler(
                mesh=mesh,
                xs_server=self._xs_server,
                num_ordinates=num_ordinates,
            )
            H, S, F, B_in_m, B_out_m = assembler.build()
            self._results["assembly_time"].append(time.time() - start)
            self._results["shape"].append(assembler.M)

            # Calculate average mesh element size
            self._results["avg_element_area"].append(assembler.avg_element_size)

            # Append operator information
            self._append_operator_info("H", H)
            self._append_operator_info("S", S)
            self._append_operator_info("F", F)
            self._append_operator_info("B_in", B_in_m)
            self._append_operator_info("B_out", B_out_m)

            # Run solver
            LHS = LinearOperator(
                [H + B_out_m - B_in_m, -S], N=assembler.N, M=assembler.M
            )
            self._results["LHS"]["num_entries"].append(LHS.num_entries)
            self._results["LHS"]["compression"].append(LHS.compression)
            start = time.time()
            k, psi = eig(
                LHS=LHS,
                RHS=LinearOperator([F], N=assembler.N, M=assembler.M),
                tols=self.tol,
                max_iters=self.max_iter,
                device=self.device,
                linear_solver_opts=self.linear_solver_opts,
            )
            self._results["solve_time"].append(time.time() - start)
            self._results["k"].append(k)

            # Append matrix-vector product timing
            self._results["H_MV_avg_time"].append(LHS.mv_time[0])
            self._results["H_MV_std_time"].append(LHS.mv_time[1])

            # Integrate angular component
            phi = assembler.angular_integral(psi).numpy()

            # Save scalar flux
            np.save(
                self.data_prefix + f"phi_m_{num_ordinates}_{factor}_{degree}.npy", phi
            )

            # Calculate errors
            if k_ref:
                self._results["k_relative_error"].append(abs(k - k_ref) / k_ref)
            if phi_ref:
                self._results["phi_relative_l2_error"].append(
                    self._phi_error(mesh, phi, phi_ref)
                )

            # Run TT and mixed solvers
            for eps in self._eps_list:
                # Append all relavent info
                self._results["format"].append("TT")
                self._results["num_ordinates"].append(num_ordinates)
                self._results["factor"].append(factor)
                self._results["degree"].append(degree)
                self._results["eps"].append(eps)
                self._results["avg_element_area"].append(
                    self._results["avg_element_area"][-1]
                )

                # Run TT config
                start = time.time()
                assembler = TTAssembler(
                    mesh=mesh,
                    xs_server=self._xs_server,
                    num_ordinates=num_ordinates,
                )
                assembler.interp_jacobian = False
                assembler.interp_jacobian_det = False
                assembler.interp_boundary_jacobian_det = False
                H, S, F, B_in_tt, B_out_tt = assembler.build(eps=eps)
                self._results["assembly_time"].append(time.time() - start)
                self._results["shape"].append(assembler.M)
                assembler.save_tt_info(
                    self.info_prefix + f"tt_info_{num_ordinates}_{factor}_{degree}.csv"
                )

                # Append operator information
                self._append_operator_info("H", H)
                self._append_operator_info("S", S)
                self._append_operator_info("F", F)
                self._append_operator_info("B_in", B_in_tt)
                self._append_operator_info("B_out", B_out_tt)

                # Run solver
                LHS = LinearOperator(
                    [H, B_out_tt, -B_in_tt, -S], N=assembler.N, M=assembler.M
                )
                self._results["LHS"]["num_entries"].append(LHS.num_entries)
                self._results["LHS"]["compression"].append(LHS.compression)
                start = time.time()
                k, psi = eig(
                    LHS=LHS,
                    RHS=LinearOperator([F], N=assembler.N, M=assembler.M),
                    tols=self.tol,
                    max_iters=self.max_iter,
                    device=self.device,
                    linear_solver_opts=self.linear_solver_opts,
                )
                self._results["solve_time"].append(time.time() - start)
                self._results["k"].append(k)

                # Append matrix-vector product timing
                self._results["H_MV_avg_time"].append(LHS.mv_time[0])
                self._results["H_MV_std_time"].append(LHS.mv_time[1])

                # Integrate angular component
                phi = assembler.angular_integral(psi).numpy()

                # Save scalar flux
                np.save(
                    self.data_prefix + f"phi_tt_{num_ordinates}_{factor}_{degree}.npy",
                    phi,
                )

                # Calculate errors
                if k_ref:
                    self._results["k_relative_error"].append(abs(k - k_ref) / k_ref)
                if phi_ref:
                    self._results["phi_relative_l2_error"].append(
                        self._phi_error(mesh, phi, phi_ref)
                    )

                # Mixed run
                self._results["format"].append("Mixed")
                self._results["num_ordinates"].append(num_ordinates)
                self._results["factor"].append(factor)
                self._results["degree"].append(degree)
                self._results["eps"].append(eps)
                self._results["avg_element_area"].append(
                    self._results["avg_element_area"][-1]
                )
                self._results["shape"].append(assembler.M)
                self._results["assembly_time"].append(None)

                # Append operator information
                self._append_operator_info("H", H)
                self._append_operator_info("S", S)
                self._append_operator_info("F", F)
                self._append_operator_info("B_in", B_in_tt)
                self._append_operator_info("B_out", B_out_tt)

                # Run solver
                LHS = LinearOperator(
                    [H, B_out_m - B_in_m, -S], N=assembler.N, M=assembler.M
                )
                self._results["LHS"]["num_entries"].append(LHS.num_entries)
                self._results["LHS"]["compression"].append(LHS.compression)
                start = time.time()
                k, psi = eig(
                    LHS=LHS,
                    RHS=LinearOperator([F], N=assembler.N, M=assembler.M),
                    tols=self.tol,
                    max_iters=self.max_iter,
                    device=self.device,
                    linear_solver_opts=self.linear_solver_opts,
                )
                self._results["solve_time"].append(time.time() - start)
                self._results["k"].append(k)

                # Append matrix-vector product timing
                self._results["H_MV_avg_time"].append(LHS.mv_time[0])
                self._results["H_MV_std_time"].append(LHS.mv_time[1])

                # Integrate angular component
                phi = assembler.angular_integral(psi).numpy()

                # Save scalar flux
                np.save(
                    self.data_prefix + f"phi_mix_{num_ordinates}_{factor}_{degree}.npy",
                    phi,
                )

                # Calculate errors
                if k_ref:
                    self._results["k_relative_error"].append(abs(k - k_ref) / k_ref)
                if phi_ref:
                    self._results["phi_relative_l2_error"].append(
                        self._phi_error(mesh, phi, phi_ref)
                    )

        # Save results
        with open(self.results_prefix + "results.pkl", "wb") as f:
            pickle.dump(self._results, f)

    def _append_operator_info(self, name, op):
        """"""
        if isinstance(op, tntt.TT):
            self._results[name]["ranks"].append(op.R[1:-1])
            self._results[name]["num_entries"].append(
                sum([tn.numel(c) for c in op.cores])
            )
            self._results[name]["compression"].append(
                (np.prod(op.N) if not op.is_ttm else np.prod(op.N) * np.prod(op.M))
                / self._results[name]["num_entries"][-1]
            )
        else:
            self._results[name]["ranks"].append(None)
            self._results[name]["num_entries"].append(op.num_entries)
            self._results[name]["compression"].append(op.compression)

    def _phi_error(self, mesh: IGAMesh, phi: np.ndarray, phi_ref: np.ndarray):
        """"""
        # Iterate through groups and plot
        phi_avg = np.zeros(phi_ref.shape)
        for g in range(self._xs_server.num_groups):
            # Set control points
            mesh.set_phi(phi[g,])

            # Calculate regular mesh
            phi_avg[g,] = mesh.regular_mesh(self._pids, self._coords)

        # Normalize average
        phi_avg /= np.linalg.norm(phi_avg.flatten(), 2)

        # Calculate relative L2-errors
        error = np.zeros(self._xs_server.num_groups)
        for g in range(self._xs_server.num_groups):
            error[g] = np.linalg.norm(
                (phi_avg[g,] - phi_ref[g,]).flatten(), 2
            ) / np.linalg.norm(
                phi_ref[g,].flatten(), 2
            )

        return error
