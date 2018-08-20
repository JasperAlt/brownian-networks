print("Importing...")
from sim import *
import networkx as nx

def series(measure, matrices):
    stdout.write("Computing series")
    stdout.flush()
    result = []
    for m in matrices:
        G = nx.from_numpy_matrix(m)
        result.append(measure(G))
        stdout.write(".")
        stdout.flush()
    print("")
    return result

################################################################################

# Parameters

P = 500
# number of points (int)
N = 200
# time slices (int)
T = 50.0
# total time (float)
D = 2
# dimensions (int)
delta = 1.0
# scaling parameter for motion (float)
d = 0.1
# radius of connections (float)
bound = 20
# boundary (None or number)
# this is a radius or in the torus case the side length
init_bound = bound
# boundary for initialization can be separate
handling = None
# "Exit", "Torus", None
init = "Point"
# Initialization ("Default", "Random" or "Point")
# # (Default is point if unbounded, random if bounded)
drift = None
# None or a length-D list
memory = False
# True or False

################################################################################

# simulate some motion
result = sim(P,N,T,D, bound=bound, init_bound = init_bound, handling=handling, init=init, drift=drift)

# apply boundary rules
bounded = handle_bounds(handling, bound, result)

# get matrices
s = scan(bounded, d, bound=bound, handling=handling, memory=memory)

a = series(nx.average_clustering, s)

plot(a)
show()

# plot_2D result
# net_anim(s, rate= 0.01)
