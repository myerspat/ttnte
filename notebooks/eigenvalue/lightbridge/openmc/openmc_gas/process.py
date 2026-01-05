import openmc
import matplotlib.pyplot as plt
import numpy as np

X = 1.36  # channel pitch
num_groups = 7

# Load statepoint file
sp = openmc.StatePoint("./statepoint.300.h5")

# Print eigenvalue
print(f"keff = {sp.keff.nominal_value} +/- {sp.keff.std_dev}")

# Get cell flux tally
flux = (
    sp.get_tally(name="Cell").get_values(scores=["flux"]).reshape((4, 7))[..., ::-1].T
)
print(f"Cell flux shape: {flux.shape}")

# Save data
np.save(open("./data/cell_flux.npy", "wb"), flux)

# Get regular mesh flux tally
flux = np.transpose(
    sp.get_tally(name="Regular Mesh")
    .get_values(scores=["flux"])
    .reshape((128, 128, 7))[..., ::-1],
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
X, Y = np.meshgrid(np.linspace(0, X / 2, 129), np.linspace(0, X / 2, 129))
for g in range(7):
    plt.clf()
    heatmap = plt.pcolormesh(X, Y, flux[g,])
    plt.xlabel("$x (cm)$")
    plt.ylabel("$y (cm)$")

    # Get colorbar
    cbar = plt.colorbar(heatmap)
    cbar.ax.set_ylabel(f"$\\phi_{g + 1}(x, y)$")

    plt.savefig(f"./figs/phi_{g + 1}.png", dpi=300)
