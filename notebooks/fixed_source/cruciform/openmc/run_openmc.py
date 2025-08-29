import numpy as np
import openmc

from ttnte.xs import Server

xs_server = Server(
    {
        "Source": {
            "total": np.array([0.01]),
            "scatter_gtg": np.array([[[0.008]]]),
        },
        "Void": {
            "total": np.array([0]),
            "scatter_gtg": np.array([[[0]]]),
        },
        "Shield": {
            "total": np.array([3]),
            "scatter_gtg": np.array([[[0.5]]]),
        },
    }
)

# Instantiate the energy group data
groups = openmc.mgxs.EnergyGroups(np.logspace(-5, 7, xs_server.num_groups + 1))

# =======================================================
# Creating mg_cross_sections.xml file
mats = ["Source", "Void", "Shield"]
xsdata = []

for mat in mats:
    xsdata.append(openmc.XSdata(mat, groups))
    xsdata[-1].order = xs_server.num_moments - 1

    scatter = np.zeros((1, 1, 1))
    if mat != "Void":
        scatter = xs_server.scatter_gtg(mat)

    print(mat, scatter)

    # Fill xs object
    xsdata[-1].set_total(xs_server.total(mat))
    xsdata[-1].set_fission(np.zeros(1))
    xsdata[-1].set_nu_fission(np.zeros(1))
    xsdata[-1].set_absorption(xs_server.total(mat) - scatter.flatten())
    xsdata[-1].set_scatter_matrix((scatter))

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
D = 0  # fuel width
D2 = D * 0.5
X = 10  # channel pitch
delta = 1  # thickness of lobes
y2 = delta * 0.5
d = 0.04  # thickness of cladding at valleys
dmax = 0.102  # thickness of cladding at ends of the lobes_______
R = 2  # radius defining outer curve of valleys
a = 0.156  # displacer width


y1 = y2 - d
x1 = D2 - R - y2 - dmax
x2 = x1 + dmax

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

# # =======================================================
# # fuel core
# cylinderIb = openmc.ZCylinder(D2 - x2, D2 - x2, R + d)
# elipseIb = openmc.Quadric(
#     y1**2,
#     x1**2,
#     0,
#     0,
#     0,
#     0,
#     -2 * (y1**2) * (D2 - x2),
#     0,
#     0,
#     (y1**2) * ((D2 - x2) ** 2) - (y1**2) * (x1**2),
# )
# elipseIIb = openmc.Quadric(
#     x1**2,
#     y1**2,
#     0,
#     0,
#     0,
#     0,
#     0,
#     -2 * (y1**2) * (D2 - x2),
#     0,
#     (y1**2) * ((D2 - x2) ** 2) - (y1**2) * (x1**2),
# )
#
# missingHolesb = +cylinderIb
# crossb = square & missingHolesb
# elipsesb = -elipseIb | -elipseIIb
#
# fuel = crossb | elipsesb
# cladding = cladding & ~fuel & +bottom & +left

# ======================================================
# Define cells and universe
cell1 = openmc.Cell()
cell1.region = cladding
cell1.fill = materials["Source"]
cell4 = openmc.Cell()
cell4.region = coolant
cell4.fill = materials["Void"]

# Universe
universe = openmc.Universe(cells=[cell1, cell4])
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
# Create settings.xml
# Create uniform in volume cylindrical source
source = openmc.Source()
source.space = openmc.stats.CylindricalIndependent(
    r=openmc.stats.PowerLaw(a=0.0, b=5, n=1),
    phi=openmc.stats.Uniform(0.0, np.pi / 2),  # Uniform angle
    z=openmc.stats.Uniform(-0.5, 0.5),
    origin=(0, 0, 0),
)
source.angle = openmc.stats.Isotropic()
source.energy = openmc.stats.Discrete([1.0], [1.0])

# Instantiate a Settings, set all runtime parameters, and export to XML
settings_file = openmc.Settings()
settings_file.energy_mode = "multi-group"
settings_file.run_mode = "fixed source"
settings_file.batches = 1000
settings_file.particles = 500000
settings_file.output = {"tallies": True, "summary": True}
settings_file.source = source
settings_file.export_to_xml()

# =======================================================
# Create tallies.xml
tallies_file = openmc.Tallies()

# Define regular mesh
mesh = openmc.RegularMesh(mesh_id=0)
mesh.dimension = [128, 128]
mesh.lower_left = [0, 0]
mesh.upper_right = [6, 6]

# Define tallies
energy_filter = openmc.EnergyFilter(groups.group_edges)
tally = openmc.Tally(name="Regular Mesh")
tally.filters = [openmc.MeshFilter(mesh), energy_filter]
tally.scores = ["flux"]
tallies_file.append(tally)

# tally = openmc.Tally(name="Cell")
# tally.filters = [openmc.CellFilter([fuel]), energy_filter]
# tally.scores = ["flux"]
# tallies_file.append(tally)

tallies_file.export_to_xml()
