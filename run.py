
print("Importing...")
from sim import *
from figs import *
from analysis import *
import networkx as nx
import time


# This file is for testing.

################################################################################

# Parameters

P = 200
# number of points (int)
N = 100
# time slices (int)
T = (N/3.0)
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

# plot average CC for each time step
plot(series(nx.average_clustering, G))

# a hacky way to generate 100 random geometric graphs:
RGGs = []
for i in range(100):
    # simulate for a single step
    result = sim(P,1,(1/3.),D, bound=bound, init_bound = init_bound, handling=handling, init=init, drift=drift, origin_point=True)
    # add the first of the two resulting graphs = the initial RGG to our
    # list of RGGs
    RGGs.append(scan_cells_multi(result, r, bound=bound, handling=handling, memory=memory)[0])

# plot the average average CC over the RGGs for 101 time slices
plot([average_value(nx.average_clustering, RGGs)] * 101)
show()

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

