import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

from tt_nte.solvers import Matrix
from tt_nte.utils.utils import get_degree


def plot_ordinate_perturbation(
    method,
    num_ordinates,
    solvers,
    expected_k=1.0,
    plot_mat_ges=True,
    ground_truth_solver=None,
    fig=None,
    verbose=True,
):
    if fig is None:
        plt.clf()
        fig = plt.figure()

    # Construct subplot array
    gs = fig.add_gridspec(3, 2)
    axs = [fig.add_subplot(gs[0, :])] + [
        fig.add_subplot(gs[int(x / 2) + 1, x % 2]) for x in range(4)
    ]

    # Plot expected eigenvalue
    axs[0].axhline(expected_k, color="k", ls="--", label="Ground Truth")

    # Run through matrix or ground truth solver
    gt_name = "Mat/GES" if plot_mat_ges else ground_truth_solver

    assert isinstance(gt_name, str)
    if verbose:
        print(f"-- {gt_name}")

    configs = (
        (Matrix, {"tt_fmt": "tt"}, {}, {}, "o-")
        if plot_mat_ges
        else solvers.pop(ground_truth_solver)
    )

    gt_ks = []
    gt_psis = []
    exec_times = []
    for N in num_ordinates:
        if verbose:
            print(f"--   N = {N},", end=" ")

        # Update method
        method.update_settings(num_ordinates=N, **configs[1])

        solver = configs[0](method, **configs[2])

        # Run solver
        start = time.time()
        if plot_mat_ges:
            solver.ges(**configs[3])
        else:
            solver.power(**configs[3])
        stop = time.time()

        # Save k and psi
        gt_ks.append(solver.k)
        gt_psis.append(
            solver.psi if isinstance(solver.psi, np.ndarray) else solver.psi.matricize()
        )
        exec_times.append(stop - start)

        if verbose:
            print(
                f"k = {np.round(solver.k, 5)},"
                + f" exec_time = {np.round(exec_times[-1], 3)}"
            )

    axs[0].plot(num_ordinates, gt_ks, configs[-1], label=gt_name)
    axs[3].plot(num_ordinates, exec_times, configs[-1], label=gt_name)

    # Plot compression ratio
    _create_compress_plot(method, num_ordinates, axs[4])

    # Run given solvers
    for name, configs in solvers.items():
        if verbose:
            print(f"-- {name}")

        ks = []
        delta_psis = []
        exec_times = []

        for i in range(len(num_ordinates)):
            if verbose:
                print(f"--   N = {num_ordinates[i]},", end=" ")

            # Update method
            method.update_settings(num_ordinates=num_ordinates[i], **configs[1])

            solver = configs[0](method, **configs[2])

            # Run solver
            start = time.time()
            solver.power(**configs[3])
            stop = time.time()

            # Save k and psi
            ks.append(solver.k)
            delta_psis.append(
                np.linalg.norm(
                    gt_psis[i]
                    - (
                        solver.psi
                        if isinstance(solver.psi, np.ndarray)
                        else solver.psi.matricize()
                    ),
                )
            )

            exec_times.append(stop - start)

            if verbose:
                print(
                    f"k = {np.round(solver.k, 5)},"
                    + f" exec_time = {np.round(exec_times[-1], 3)}"
                )

        axs[0].plot(num_ordinates, ks, configs[-1], label=name)
        axs[1].plot(
            num_ordinates,
            [abs(gt_ks[i] - ks[i]) * 1e5 for i in range(len(num_ordinates))],
            configs[-1],
            label=f"$|k_{'{' + gt_name + '}'} - k_{'{' + name + '}'}|$",
        )
        axs[2].plot(
            num_ordinates,
            delta_psis,
            configs[-1],
            label=f"$\\|\\psi_{'{' + gt_name + '}'} - \\psi_{'{' + name + '}'}\\|$",
        )
        axs[3].plot(num_ordinates, exec_times, configs[-1], label=name)

    # Change visuals of plot
    axs[0].set(ylabel="Eigenvalue $k$", xlabel="Number of Ordinates ($N$)")
    axs[1].set(ylabel="Relative Eigenvalue Error ($pcm$)")
    axs[2].set(ylabel="Relative Eigenvector Error")
    axs[3].set(ylabel="Execution Time ($s$)", xlabel="Number of Ordinates ($N$)")

    axs[1].set_yscale("log")
    axs[3].set_yscale("log")

    for ax in axs:
        ax.legend(fontsize=8)

    return fig


def _create_compress_plot(method, num_ordinates, ax):
    # Initialize size arrays
    full_mat_size = np.zeros(len(num_ordinates))
    csc_mat_size = np.zeros(len(num_ordinates))
    tt_size = np.zeros(len(num_ordinates))
    qtt_size = np.zeros(len(num_ordinates))

    for i in range(len(num_ordinates)):
        method.update_settings(num_ordinates=num_ordinates[i], tt_fmt="tt")

        for tt in [method.H, method.F, method.S]:
            # Full matrix elements
            full_mat_size[i] += tt.matricize().todense().size

            # CSC matrix elements
            sp_mat = tt.matricize()
            csc_mat_size[i] += (
                sp_mat.data.size + sp_mat.indices.size + sp_mat.indptr.size
            )

            # TT elements
            for core in tt.cores:
                tt_size[i] += core.size

            # QTT elements
            tt.tt2qtt()
            for core in tt.cores:
                qtt_size[i] += core.size

    ax.plot(num_ordinates, csc_mat_size / full_mat_size, "o-", label="CSC Matrix")
    ax.plot(num_ordinates, tt_size / full_mat_size, "o-", label="TT")
    ax.plot(num_ordinates, qtt_size / full_mat_size, "o-", label="QTT")
    ax.set(ylabel="Compression Ratio", xlabel="Number or Ordinates ($N$)")
    ax.set_yscale("log")


def plot_qtt_svd(method, axs=None, figsize=(10, 15)):
    """Plot SVD for all cores in each of the three operators when converting from TT to
    QTT format."""
    if axs is None:
        plt.clf()
        _, axs = plt.subplots(3, figsize=figsize)

    tts = [method.H.train(), method.F.train(), method.S.train()]
    tt_names = ["LHS Operator", "Fission Operator", "Scattering Operator"]

    for tt_idx in range(len(tts)):
        tt = tts[tt_idx]
        tt_tensor = tt.copy()

        # QTT shaping
        new_dims = []
        for dim_size in tt.row_dims:
            new_dims.append([2] * get_degree(dim_size))

        for i in range(tt.order):
            # Get core features
            core = tt_tensor.cores[i]
            rank = tt_tensor.ranks[i]
            row_dim = tt_tensor.row_dims[i]
            col_dim = tt_tensor.col_dims[i]

            c = plt.cm.rainbow(np.linspace(0, 1, tt.order))[i]

            for j in range(len(new_dims[i]) - 1):
                # Set new row_dim and col_dim for reshape
                row_dim = int(row_dim / new_dims[i][j])
                col_dim = int(col_dim / new_dims[i][j])

                # Reshape and transpose core
                core = core.reshape(
                    rank,
                    new_dims[i][j],
                    row_dim,
                    new_dims[i][j],
                    col_dim,
                    tt_tensor.ranks[i + 1],
                ).transpose([0, 1, 3, 2, 4, 5])

                # Apply SVD to split core
                [_, s, v] = svd(
                    core.reshape(
                        rank * new_dims[i][j] ** 2,
                        row_dim * col_dim * tt_tensor.ranks[i + 1],
                    ),
                    full_matrices=False,
                    overwrite_a=True,
                    check_finite=False,
                    lapack_driver="gesvd",
                )

                # Plot SVD singular values
                if j == 0:
                    axs[tt_idx].plot(s, c=c, label=f"Core {i}")
                else:
                    axs[tt_idx].plot(s, c=c)

                # Update residual core and rank
                core = np.diag(s).dot(v)
                rank = s.shape[0]

        axs[tt_idx].set(xlabel="Singular Value Index", title=tt_names[tt_idx])
        axs[tt_idx].legend()
        axs[tt_idx].set_yscale("log")

    return axs
