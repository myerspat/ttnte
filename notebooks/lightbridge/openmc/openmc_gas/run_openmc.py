import numpy as np
import openmc

from ttnte.xs.benchmarks import kaist

D = 1.26  # fuel width
D2 = D * 0.5
X = 1.36  # channel pitch
delta = 0.306  # thickness of lobes
y2 = delta * 0.5
d = 0.04  # thickness of cladding at valleys
dmax = 0.102  # thickness of cladding at ends of the lobes_______
R = 0.297  # radius defining outer curve of valleys
a = 0.156  # displacer width


y1 = y2 - d
x1 = D2 - R - y2 - dmax
x2 = x1 + dmax

displacer_material = "Gas"

# ==========================================================
# XS data
xs_server = kaist()
kaist_edges = [
    2 * (10**7),
    1.353 * (10**6),
    9119,
    3.928,
    0.6251,
    0.1457,
    0.05692,
    0,
]
# Instantiate the energy group data
groups = openmc.mgxs.EnergyGroups(kaist_edges[::-1])

# Creating mg_cross_sections.xml file
mats = ["UO2 3%", "Water", "Guide Tube", displacer_material]
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
# define outer edge of metallogically bonded barrier
cylinderIa = openmc.ZCylinder(D2 - x2, D2 - x2, R)

elipseIa = openmc.Quadric(
    y2**2,
    x2**2,
    0,
    0,
    0,
    0,
    -2 * (y2**2) * (D2 - x2),
    0,
    0,
    (y2**2) * ((D2 - x2) ** 2) - (y2**2) * (x2**2),
)
elipseIIa = openmc.Quadric(
    x2**2,
    y2**2,
    0,
    0,
    0,
    0,
    0,
    -2 * (y2**2) * (D2 - x2),
    0,
    (y2**2) * ((D2 - x2) ** 2) - (y2**2) * (x2**2),
)

square = (
    +openmc.XPlane(-(D2 - x2))
    & -openmc.XPlane((D2 - x2))
    & +openmc.YPlane(-(D2 - x2))
    & -openmc.YPlane((D2 - x2))
)
missingHolesa = +cylinderIa
crossa = square & missingHolesa
elipsesa = -elipseIa | -elipseIIa

cladding = (
    (crossa | elipsesa)
    & +openmc.XPlane(0, boundary_type="reflective")
    & +openmc.YPlane(0, boundary_type="reflective")
)


# =======================================================
# coolant
right = openmc.XPlane(X / 2, boundary_type="reflective")
top = openmc.YPlane(X / 2, boundary_type="reflective")
left = openmc.XPlane(0, boundary_type="reflective")
bottom = openmc.YPlane(0, boundary_type="reflective")

coolant = -right & -top & +left & +bottom
coolant = coolant & ~cladding

# =======================================================
# fuel core
cylinderIb = openmc.ZCylinder(D2 - x2, D2 - x2, R + d)
elipseIb = openmc.Quadric(
    y1**2,
    x1**2,
    0,
    0,
    0,
    0,
    -2 * (y1**2) * (D2 - x2),
    0,
    0,
    (y1**2) * ((D2 - x2) ** 2) - (y1**2) * (x1**2),
)
elipseIIb = openmc.Quadric(
    x1**2,
    y1**2,
    0,
    0,
    0,
    0,
    0,
    -2 * (y1**2) * (D2 - x2),
    0,
    (y1**2) * ((D2 - x2) ** 2) - (y1**2) * (x1**2),
)

missingHolesb = +cylinderIb
crossb = square & missingHolesb
elipsesb = -elipseIb | -elipseIIb

fuel = crossb | elipsesb
cladding = cladding & ~fuel & +bottom & +left

# ========================================================
# displacer
edgeI = openmc.Plane(1, 1, 0, (((a**2) / 2) ** 0.5))

displacer = -edgeI & +left & +bottom  # & slice
fuel = fuel & ~displacer & +left & +bottom

# ======================================================
# Define cells and universe
cell1 = openmc.Cell()
cell1.region = cladding
cell1.fill = materials["Guide Tube"]
cell2 = openmc.Cell()
cell2.region = fuel
cell2.fill = materials["UO2 3%"]
cell3 = openmc.Cell()
cell3.region = displacer
cell3.fill = materials[displacer_material]
cell4 = openmc.Cell()
cell4.region = coolant
cell4.fill = materials["Water"]

# Universe
universe = openmc.Universe(cells=[cell1, cell2, cell3, cell4])
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
settings_file.run_mode = "eigenvalue"
settings_file.output = {"tallies": True, "summary": True}
settings_file.source = openmc.IndependentSource(
    space=openmc.stats.Box([0, 0, -1.0], [X / 2, X / 2, 1.0]),
    constraints={"fissionable": True},
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
tally.filters = [openmc.CellFilter([cell1, cell2, cell3, cell4]), energy_filter]
tally.scores = ["flux"]
tallies_file.append(tally)

tallies_file.export_to_xml()
