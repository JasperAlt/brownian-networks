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

def largest_component_series(measure, matrices):
    stdout.write("Computing series")
    stdout.flush()
    result = []
    for m in matrices:
        G = nx.from_numpy_matrix(m)
        G = max(nx.connected_component_subgraphs(G), key=len)
        result.append(measure(G))
        stdout.write(".")
        stdout.flush()
    print("")
    return result

def average_component_series(measure, matrices):
    stdout.write("Computing series")
    stdout.flush()
    result = []
    for m in matrices:
        G = nx.from_numpy_matrix(m)
        G = nx.connected_component_subgraphs(G)
        s = 0
        for g in G:
            s += measure(g)
        result.append(s/len(G))
        stdout.write(".")
        stdout.flush()
    print("")
    return result


################################################################################

# Parameters

P = 50
# number of points (int)
N = 90
# time slices (int)
T = 100.0
# total time (float)
D = 1
# dimensions (int)
delta = 1.0
# scaling parameter for motion (float)
r = 1.
# radius of connections (float)
bound = 20
# boundary (None or number)
# this is a radius or in the torus case the side length
init_bound = bound
# boundary for initialization can be separate
handling = None
# boundary handling: "Exit", "Torus", None
init = "Random"
# Initialization ("Default", "Random" or "Point")
# # (Default is point if unbounded, random if bounded)
drift = None
# None or a length-D list
memory = False
# do nodes stay connected (boolean)

################################################################################

# simulate some motion
result = sim(P,N,T,D, bound=bound, init_bound = init_bound, handling=handling, init=init, drift=drift)

# apply boundary rules
bounded = handle_bounds(handling, bound, result)

# get matrices
#s = scan(bounded, r, bound=bound, handling=handling, memory=memory)
s = scan(bounded, r, bound=bound, handling=handling, memory=memory)

a = series(nx.density, s)

plot(a)
show()

net_anim(s, rate= 0.02)
