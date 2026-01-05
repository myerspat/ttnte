import openmc

num_groups = 1

# Load statepoint file
sp = openmc.StatePoint("./statepoint.1000.h5")

print(
    r"Leakage Fraction: {} +\- {}".format(
        sp.global_tallies[-1][-2], sp.global_tallies[-1][-1]
    )
)
