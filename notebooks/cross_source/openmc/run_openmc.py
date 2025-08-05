import numpy as np
import openmc

from ttnte.xs.benchmarks import kaist

## Initialize dimensional variables
X = 1.6  # Channel pitch

# Cruciform
R = 0.297  # Radius defining valleys of fixed source
delta = 0.306  # Width of lobes
d2 = delta * 0.5  # Half width of lobes
x = 0.18  # Portrusion of lobes

# Shielding
I = 0.7  # Inner radius
O = 0.75  # Outer radius

# ==========================================================
# XS data
xs_server = kaist()

# Instantiate the energy group data
groups = openmc.mgxs.EnergyGroups(np.logspace(-5, 7, xs_server.num_groups + 1))

# Creating mg_cross_sections.xml file
mats = ["Gas", "Water", "Guide Tube"]
xsdata = []

for mat in mats:
    xsdata.append(openmc.XSdata(mat, groups))
    xsdata[-1].order = xs_server.num_moments - 1

    # Fill xs object
    xsdata[-1].set_chi(xs_server.chi)
    xsdata[-1].set_total(xs_server.total(mat))
    xsdata[-1].set_fission((xs_server.nu_fission(mat)) / 2.5)
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

# ========================================================
# Source
cylinderIa = openmc.ZCylinder(R + d2, R + d2, R)

elipseIa = openmc.Quadric(
    d2**2,
    x**2,
    0,
    0,
    0,
    0,
    -2 * (d2**2) * (R + d2),
    0,
    0,
    (d2**2) * ((R + d2) ** 2) - (d2**2) * (x**2),
)
elipseIIa = openmc.Quadric(
    x**2,
    d2**2,
    0,
    0,
    0,
    0,
    0,
    -2 * (d2**2) * (R + d2),
    0,
    (d2**2) * ((R + d2) ** 2) - (d2**2) * (x**2),
)

square = (
    +openmc.XPlane(-(R + d2))
    & -openmc.XPlane((R + d2))
    & +openmc.YPlane(-(R + d2))
    & -openmc.YPlane((R + d2))
)
missingHolesa = +cylinderIa
crossa = square & missingHolesa
elipsesa = -elipseIa | -elipseIIa

source = (
    (crossa | elipsesa)
    & +openmc.XPlane(0, boundary_type="reflective")
    & +openmc.YPlane(0, boundary_type="reflective")
)


# =======================================================
# Cladding

cylinderIn = openmc.ZCylinder(0, 0, I)
cylinderOut = openmc.ZCylinder(0, 0, O)

cladding = (
    +cylinderIn
    & -cylinderOut
    & +openmc.XPlane(0, boundary_type="reflective")
    & +openmc.YPlane(0, boundary_type="reflective")
)


# =======================================================
# Void
right = openmc.XPlane(X / 2, boundary_type="reflective")
top = openmc.YPlane(X / 2, boundary_type="reflective")
left = openmc.XPlane(0, boundary_type="reflective")
bottom = openmc.YPlane(0, boundary_type="reflective")

coolant = -right & -top & +left & +bottom
coolant = coolant & ~cladding & ~source


# ======================================================
# Define cells and universe
cell1 = openmc.Cell()
cell1.region = source
cell1.fill = materials["Water"]
cell2 = openmc.Cell()
cell2.region = cladding
cell2.fill = materials["Guide Tube"]
cell3 = openmc.Cell()
cell3.region = coolant
cell3.fill = materials["Gas"]

# Universe
universe = openmc.Universe(cells=[cell1, cell2, cell3])
geometry = openmc.Geometry(universe)
geometry.export_to_xml()

# ======================================================
# Plots
plot = openmc.Plot()
plot.filename = "./figs/geometry.png"
plot.origin = (X / 4, X / 4, 0)
plot.width = (X / 2, X / 2)
plot.pixels = (500, 500)
plot.color_by = "material"
plot.basis = "xy"

plots = openmc.Plots()
plots.append(plot)
plots.export_to_xml()

# =======================================================
# OpenMC mission parameters
batches = 300
inactive = 50
particles = 500000

# Instantiate a Settings, set all runtime parameters, and export to XML
settings_file = openmc.Settings()
settings_file.energy_mode = "multi-group"
settings_file.batches = batches
settings_file.inactive = inactive
settings_file.particles = particles
settings_file.run_mode = "fixed source"
settings_file.output = {"tallies": True, "summary": True}
settings_file.source = openmc.IndependentSource(
    space=openmc.stats.Box([0, 0, -1.0], [X / 2, X / 2, 1.0]),
    constraints={"domains": [cell1]},
)
settings_file.entropy_lower_left = [0, 0, -1.0e50]
settings_file.entropy_upper_right = [X / 2, X / 2, 1.0e50]
settings_file.entropy_dimension = [10, 10, 1]
settings_file.export_to_xml()

# =======================================================
# Create tallies.xml
tallies_file = openmc.Tallies()

# Define regular mesh
mesh = openmc.RegularMesh(mesh_id=0)
mesh.dimension = [128, 128]
mesh.lower_left = [0, 0]
mesh.upper_right = [X / 2, X / 2]

# Define tallies
energy_filter = openmc.EnergyFilter(groups.group_edges)
tally = openmc.Tally(name="Regular Mesh")
tally.filters = [openmc.MeshFilter(mesh), energy_filter]
tally.scores = ["flux"]
tallies_file.append(tally)

tally = openmc.Tally(name="Cell")
tally.filters = [openmc.CellFilter([cell1, cell2, cell3]), energy_filter]
tally.scores = ["flux"]
tallies_file.append(tally)

tallies_file.export_to_xml()
