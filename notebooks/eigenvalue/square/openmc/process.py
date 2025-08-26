import openmc
import matplotlib.pyplot as plt
import numpy as np

# Load statepoint file
sp = openmc.StatePoint("./statepoint.20000.h5")

# Print eigenvalue
print(f"keff = {sp.keff}")

# Get cell flux tally
flux = (
    sp.get_tally(name="Cell").get_values(scores=["flux"]).reshape((1, 2))[..., ::-1].T
)
print(f"Cell flux shape: {flux.shape}")

# Save data
np.save(open("./data/cell_flux.npy", "wb"), flux)

# Get regular mesh flux tally
flux = np.transpose(
    sp.get_tally(name="Regular Mesh")
    .get_values(scores=["flux"])
    .reshape((128, 128, 2))[..., ::-1],
    axes=(2, 0, 1),
)
print(f"Mesh flux shape: {flux.shape}")

# Save data
np.save(open("./data/mesh_flux.npy", "wb"), flux)

# Plot fluxes
X, Y = np.meshgrid(np.linspace(0, 6.5, 129), np.linspace(0, 6.5, 129))
for g in range(2):
    plt.clf()
    heatmap = plt.pcolormesh(X, Y, flux[g,])
    plt.xlabel("$x (cm)$")
    plt.ylabel("$y (cm)$")

    # Get colorbar
    cbar = plt.colorbar(heatmap)
    cbar.ax.set_ylabel(f"$\\phi_{g + 1}(x, y)$")

    plt.savefig(f"./figs/phi_{g + 1}.png", dpi=300)
