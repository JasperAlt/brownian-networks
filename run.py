
print("Importing...")
from sim import *
from figs import *
from analysis import *
import networkx as nx
import time


# This file is for testing.

################################################################################

# Parameters

P = 100
# number of points (int)
N = 60
# time slices (int)
T = 20.0
# total time (float)
D = 2
# dimensions (int)
delta = .5
# scaling parameter for motion (float)
r = 1.0
# radius of connections (float)
bound = 10
# boundary (None or number)
# this is a radius or in the torus case the side length
init_bound = bound
# boundary for initialization can be separate
handling = "Torus"
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
result = sim(P,N,T,D, bound=bound, init_bound = init_bound, handling=handling, init=init, drift=drift, origin_point=True)

# build network from trajectories
G = scan_cells_multi(result, r, bound=bound, handling=handling, memory=memory)
#w = build_moving_window(G, 5)

#net_anim(w, rate=0.02)
#plot_2D(result, time=3, radius=None, edges=None, center="Square")
#show()
#plot_2D(result, time=3, radius=1, edges=None, center="Square")
#show()
#plot_2D(result, time=3, radius=1, edges=1, paths='1', center="Square")
#show()
#plot_2D(result, time=3, radius=None, edges=1, paths='1', center="Square")
#show()

#print(detection_time(G, "Origin", T, N))
#print(average_detection_time(G, T, N))
#print(coverage_time(G, T, N))
#print(broadcast_time(G, "Origin"))

