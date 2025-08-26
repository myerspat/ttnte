import matplotlib.pyplot as plt
import numpy as np
import openmc

num_groups = 1

# Load statepoint file
sp = openmc.StatePoint("./statepoint.100.h5")

print(
    r"Leakage Fraction: {} +\- {}".format(
        sp.global_tallies[-1][-2], sp.global_tallies[-1][-1]
    )
)

# Get cell flux tally
flux = (
    sp.get_tally(name="Cell")
    .get_values(scores=["flux"])
    .reshape((1, num_groups))[..., ::-1]
    .T
)
print(f"Cell flux shape: {flux.shape}")

# Save data
np.save(open("./data/cell_flux.npy", "wb"), flux)

# Get regular mesh flux tally
flux = np.transpose(
    sp.get_tally(name="Regular Mesh")
    .get_values(scores=["flux"])
    .reshape((128, 128, num_groups))[..., ::-1],
    axes=(2, 0, 1),
)
print(f"Mesh flux shape: {flux.shape}")

# Save data
np.save(open("./data/mesh_flux.npy", "wb"), flux)

# Plot fluxes
X, Y = np.meshgrid(np.linspace(0, 6, 129), np.linspace(0, 6, 129))
for g in range(num_groups):
    plt.clf()
    heatmap = plt.pcolormesh(X, Y, flux[g,])
    plt.xlabel("$x (cm)$")
    plt.ylabel("$y (cm)$")

    # Get colorbar
    cbar = plt.colorbar(heatmap)
    cbar.ax.set_ylabel(f"$\\phi_{g + 1}(x, y)$")

    plt.savefig(f"./figs/phi_{g + 1}.png", dpi=300)
