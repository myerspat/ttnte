import numpy as np
import openmc

from ttnte.xs.benchmarks import pu239

# Get XS data from ttnte
xs_server = pu239(num_groups=2)

# Instantiate the energy group data
group_edges = np.logspace(-5, 7, xs_server.num_groups + 1)
groups = openmc.mgxs.EnergyGroups(np.logspace(-5, 7, xs_server.num_groups + 1))
print(group_edges)
# =======================================================
# Creating mg_cross_sections.xml file
mats = ["Pu-239"]
xsdata = []

for mat in mats:
    xsdata.append(openmc.XSdata(mat, groups))
    xsdata[-1].order = xs_server.num_moments - 1

    # Fill xs object
    xsdata[-1].set_chi(xs_server.chi)
    xsdata[-1].set_total(xs_server.total(mat))
    xsdata[-1].set_fission(xs_server.fission(mat))
    xsdata[-1].set_nu_fission(xs_server.nu_fission(mat))
    xsdata[-1].set_absorption(xs_server.absorption(mat))
    xsdata[-1].set_scatter_matrix(
        np.transpose(xs_server.scatter_gtg(mat), axes=(2, 1, 0))
    )

# Write XS data
mg_cross_sections_file = openmc.MGXSLibrary(groups)
mg_cross_sections_file.add_xsdatas(xsdata)
mg_cross_sections_file.export_to_hdf5()

# =======================================================
# Create materials.xml
materials = {}
for mat in mats:
    materials[mat] = openmc.Material(name=mat)
    materials[mat].set_density("macro", 1.0)
    materials[mat].add_macroscopic(openmc.Macroscopic(mat))

materials_file = openmc.Materials(materials.values())
materials_file.cross_sections = "./mgxs.h5"
materials_file.export_to_xml()

# =======================================================
# Create geometry.xml
# CSG
left = openmc.XPlane(x0=0)
right = openmc.XPlane(x0=6.5)
bottom = openmc.YPlane(y0=0)
top = openmc.YPlane(y0=6.5)

# Boundary conditions
left.boundary_type = "vacuum"
right.boundary_type = "vacuum"
bottom.boundary_type = "vacuum"
top.boundary_type = "vacuum"

# Cells
fuel = openmc.Cell(fill=materials["Pu-239"], region=(+left & +bottom & -right & -top))

# Universes
universe = openmc.Universe(universe_id=0, name="Root", cells=[fuel])

# Instantiate a Geometry, register the root Universe, and export to XML
geometry = openmc.Geometry()
geometry.root_universe = universe
geometry.export_to_xml()

# =======================================================
# Create settings.xml
# Instantiate a Settings, set all runtime parameters, and export to XML
settings_file = openmc.Settings()
settings_file.energy_mode = "multi-group"
settings_file.run_mode = "fixed source"
settings_file.batches = 200
settings_file.inactive = 50
settings_file.particles = 10000
settings_file.output = {"tallies": True, "summary": True}
settings_file.source = openmc.Source(
    space=openmc.stats.Box([0, 0, -1.0], [6.5, 6.5, 1.0], only_fissionable=False),
    angle=openmc.stats.Isotropic(),
    energy=openmc.stats.Uniform(group_edges[-2], group_edges[-1]),
)
settings_file.entropy_lower_left = [0, 0, -1.0e50]
settings_file.entropy_upper_right = [6.5, 6.5, 1.0e50]
settings_file.entropy_dimension = [10, 10, 1]
settings_file.export_to_xml()

# =======================================================
# Create plots.xml
plot_1 = openmc.Plot(plot_id=1)
plot_1.filename = "./figs/geometry.png"
plot_1.origin = [6.5 / 2, 6.5 / 2, 0.0]
plot_1.width = [6.5, 6.5]
plot_1.pixels = [500, 500]
plot_1.color_by = "material"
plot_1.basis = "xy"

# Instantiate a Plots collection and export to XML
plot_file = openmc.Plots([plot_1])
plot_file.export_to_xml()

# =======================================================
# Create tallies.xml
tallies_file = openmc.Tallies()

# Define regular mesh
mesh = openmc.RegularMesh(mesh_id=0)
mesh.dimension = [128, 128]
mesh.lower_left = [0, 0]
mesh.upper_right = [6.5, 6.5]

# Define tallies
energy_filter = openmc.EnergyFilter(groups.group_edges)
tally = openmc.Tally(name="Regular Mesh")
tally.filters = [openmc.MeshFilter(mesh), energy_filter]
tally.scores = ["flux"]
tallies_file.append(tally)

tally = openmc.Tally(name="Cell")
tally.filters = [openmc.CellFilter([fuel]), energy_filter]
tally.scores = ["flux"]
tallies_file.append(tally)

tallies_file.export_to_xml()
