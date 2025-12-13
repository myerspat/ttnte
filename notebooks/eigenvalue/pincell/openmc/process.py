import matplotlib.pyplot as plt
import numpy as np
import openmc

num_groups = 7

# Load statepoint file
sp = openmc.StatePoint("./statepoint.20000.h5")

# Print eigenvalue
print(f"keff = {sp.keff.nominal_value} +/- {sp.keff.std_dev}")

# Get cell flux tally
flux = (
    sp.get_tally(name="Cell").get_values(scores=["flux"]).reshape((2, 7))[..., ::-1].T
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
stdev = np.transpose(
    sp.get_tally(name="Regular Mesh")
    .get_values(scores=["flux"], value="std_dev")
    .reshape((128, 128, num_groups))[..., ::-1],
    axes=(2, 0, 1),
)
print(f"Mesh flux shape: {flux.shape}")

# Save data
np.save(open("./data/mesh_flux.npy", "wb"), flux)
np.save(open("./data/mesh_stdev.npy", "wb"), stdev)

# Plot fluxes
X, Y = np.meshgrid(np.linspace(0, 0.63, 129), np.linspace(0, 0.63, 129))
for g in range(7):
    plt.clf()
    heatmap = plt.pcolormesh(X, Y, flux[g,])
    plt.xlabel("$x (cm)$")
    plt.ylabel("$y (cm)$")

    # Get colorbar
    cbar = plt.colorbar(heatmap)
    cbar.ax.set_ylabel(f"$\\phi_{g + 1}(x, y)$")

    plt.savefig(f"./figs/phi_{g + 1}.png", dpi=300)
