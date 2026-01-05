import matplotlib.pyplot as plt
import numpy as np
import openmc

num_groups = 1

# Load statepoint file
sp = openmc.StatePoint("./statepoint.1000.h5")

print(
    r"Leakage Fraction: {} +\- {}".format(
        sp.global_tallies[-1][-2], sp.global_tallies[-1][-1]
    )
)

# Calculate source area
sa = 2 * (np.pi * 0.5 * 0.25) / 4 + (2 + 0.5) ** 2 - np.pi

# Get regular mesh flux tally
flux = (
    np.transpose(
        sp.get_tally(name="Regular Mesh")
        .get_values(scores=["flux"])
        .reshape((128, 128, num_groups))[..., ::-1],
        axes=(2, 0, 1),
    )
    * sa
    / (5 * 5 / (128 * 128))
)
stdev = (
    np.transpose(
        sp.get_tally(name="Regular Mesh")
        .get_values(scores=["flux"], value="std_dev")
        .reshape((128, 128, num_groups))[..., ::-1],
        axes=(2, 0, 1),
    )
    * sa
    / (5 * 5 / (128 * 128))
)
print(f"Mesh flux shape: {flux.shape}")

# Save data
np.save(open("./data/mesh_flux.npy", "wb"), flux)
np.save(open("./data/mesh_stdev.npy", "wb"), stdev)

# Plot fluxes
X, Y = np.meshgrid(np.linspace(0, 5, 129), np.linspace(0, 5, 129))
for g in range(num_groups):
    plt.clf()
    heatmap = plt.pcolormesh(X, Y, flux[g,])
    plt.xlabel("$x (cm)$")
    plt.ylabel("$y (cm)$")

    # Get colorbar
    cbar = plt.colorbar(heatmap)
    cbar.ax.set_ylabel(f"$\\phi_{g + 1}(x, y)$")

    plt.savefig(f"./figs/phi_{g + 1}.png", dpi=300)
