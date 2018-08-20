from math import sqrt
from random import seed, random
from operator import add
from sys import stdout
from scipy.stats import norm
from scipy.spatial import distance, KDTree
from numpy import asarray, empty, zeros, expand_dims, cumsum, tile, copy
from networkx import from_numpy_matrix, draw
from pylab import figure, pause, plot, xlim, ylim, show, close
from rtree import index

###############################################################################

# SIMULATION functions

###############################################################################

# brownian: brownian motion
# x0: initial condition, N: Slices, T: Time, out: Output variable,
# delta: scaling parameter
# drift: length-D [] of drift parameters or None
def brownian(x0, N, T, out=None, delta=1, drift=None):
    dt = T/N
    x0 = asarray(x0)

    dW = norm.rvs(size=x0.shape + (N,), scale=delta*sqrt(dt))

    if out is None:
        out = empty(dW.shape)

    cumsum(dW, axis = -1, out=out)
    out += expand_dims(x0, axis = -1)

    if drift is not None:
        for d in range(len(out)):
            for t in range(len(out[d])):
                out[d][t] += t * dt * drift[d]

    return out

# sim: initializes nodes and simulates motion
# uses boundary settings to determine how to initialize, but does not apply
# boundary behavior
# P: points, N: Slices, T: Time, D: Dimensions, delta: scaling parameter
# bound: None or a number
# handling: "Torus", None, "Exit"
# init: "Random", "Point", "Default"
#   "Random": random points in the boundary area or if none the [0, 1] interval
#   "Point": all points start at center of boundary area or else origin
#   "Default": If boundary, Random, if not, Point
# init_bound: side of square area within which points are initialized (if none,
# set to bound)
def sim(P, N, T, D, delta=1, bound=None, handling=None, init="Default", init_bound="Default", drift=None):
    print("Simulating motion...")
    out = [];
    origin = [0 for d in range(D)]
    seed()

    if init is "Default":
        if bound is None:
            init = "Point"
        else:
            init = "Random"

    if init_bound is "Default":
            init_bound = bound

    for p in range(P):
        # initialize trajectory
        t = empty((D,N+1))

        # uniform random starting point
        if init is "Random":
            if init_bound is not None and handling is "Torus":
                for d in range(D):
                    t[d,0] = random() * init_bound
            elif init_bound is not None:
                for d in range(D):
                    t[d,0] = 999999999
                while distance.euclidean(origin, [t[d, 0] for d in range(D)]) > init_bound:
                    for d in range(D):
                        t[d,0] = random() * init_bound * 2 - init_bound
            else:
                for d in range(D):
                    t[d,0] = random()

        # single point
        if init is "Point":
            if bound is not None and handling is "Torus":
                for d in range(D):
                    t[d,0] = bound / 2
            else:
                for d in range(D):
                    t[d,0] = 0

        # brownian motion
        brownian(t[:,0], N, T, out=t[:,1:], drift=drift)
        out.append(t)

    return out

# handle_bounds: Apply boundary behavior to output from sim
# If "Torus" handling, %= bound coordinates
# If "Exit" handling, map all out of bound coords to 999999999
def handle_bounds(handling, bound, result):
    # recover sim parameters
    N = range(len(result[0][0])) # "N" is N + 1, including the initial condition
    D = range(len(result[0]))
    P = range(len(result))

    if bound is not None and handling is "Torus":
        for p in P:
            for d in D:
                for n in N:
                    result[p][d][n] %= bound

    if bound is not None and handling is "Exit":
        origin = [0 for d in D]
        for p in P:
            flag = False
            for n in N:
                if flag is False:
                    coords = [result[p][d][n] for d in D]
                    if distance.euclidean(origin, coords) > bound:
                        flag = True
                if flag:
                    for d in D:
                        result[p][d][n] = 999999999
    return result

###############################################################################

# ANALYSIS functions

###############################################################################

# torus_distance: Checks distance between points on torus
# I, J: Lists of float coords
# bound: boundary distance
def torus_distance(I, J, bound):
    if bound is None:
        return distance.euclidean(I, J)

    D = len(I)
    squared_sides = 0

    for c in range(D):              # each coordinate
        dc = abs(I[c] - J[c])       # get difference
        dc = min(dc, bound - dc)    # crossing bound if shorter
        squared_sides += pow(dc, 2) # square and sum

    return sqrt(squared_sides)

# scan: sim result -> list of np array adjacency matrices
# self_edges toggles whether nodes are within distance of themselves
# if weights, A[ij] = d(ij), else A[ij] = (d(ij) < d)
# memory sets whether the network remembers previous edges
# [ should probably be broken up into more functions ]
def scan(result, d, self_edges=False, weight=False, bound=None, handling=None, memory=False):
    stdout.write("Assembling graphs")
    stdout.flush()
    # recover sim parameters
    N = len(result[0][0]) # "N" = N + 1
    D = len(result[0])
    P = len(result)

    p = index.Property(dimension=D)
    # using an R*-tree variant did not produced faster speeds for any n tested
    # (up to 10,000)

    S = []          # List of adjacency matrices
    if handling is "Torus":
        offsets = []
        # simulate the toroid by inserting extra points in the 'next box over'
        # in each direction.
        # we do this using the binary integers up to 2^D:
        for k in range(pow(2, D)):
            plus = [int(x) * bound for x in bin(k)[2:].zfill(D)]
            # eg, 1 -> 0001 -> 0,0,0,1 -> 0,0,0,5 if boundary is 5
            # then 2 -> ... -> 0,0,5,0 etc
            for l in range(pow(2,D)):
                minus = [-int(x) * bound for x in bin(l)[2:].zfill(D)]
                # eg, 1 -> 0001 -> 0,0,0,-5 if boundary is 5
                # add the positive and negative offsets...
                m = list(map(add, plus, minus))
                # the unique results are then the offsets in all directions
                if m not in offsets:
                    offsets.append(m)

    # Each time
    for t in range(N):
        # copy previous matrix or start fresh as appropriate
        if t is 0 or memory is False:
            S.append(zeros((P, P)))
        else:
            S.append(copy(S[t-1]))

        # build rtree of points unless they are out of bounds
        idx = index.Index(properties=p)
        for i in range(P):
            I = []
            # get point coordinates
            for coord in range(D):
                I.append(result[i][coord][t])

            if bound is "Torus" and I[0] < 999999999.0:
                # each offset (includes the non-offset)
                for o in offsets:
                    # add the offset
                    m = list(map(add, I, o))
                    # rtree coords are a tuple: (xmin, ymin, ... xmax, ymax)
                    # representing a bounding box
                    # for points, obviously, all min = max
                    m += m
                    tup = tuple(m)
                    print(tup)
                    idx.insert(i, tup)
            elif I[0] < 999999999.0:
                idx.insert(i, tuple(I + I)) # insert bounding box where min = max

        # now query the rtree with a d x d bounding box centered on each point
        # (unless the point is out of bounds)
        for i in range(P):
            I = []
            skip = False
            for coord in range(D):
                I.append(result[i][coord][t])
                if I[coord] is 999999999.0:
                    skip = True
            if skip is not True:
                minima = [x - d for x in I]
                maxima = [x + d for x in I]
                tup = tuple(minima + maxima)
                hits = set(idx.intersection(tup))
                # and check whether each point within the box is within d
                for j in hits:
                    J = []
                    for coord in range(D):
                        J.append(result[j][coord][t])
                    if handling is "Torus":
                        dist = torus_distance(I,J, bound)
                    else:
                        dist = distance.euclidean(I,J)
                    # connect nodes if in distance and appropriate re self edges
                    if dist < d:
                        S[t][i,j] = (self_edges or i != j)
                    if weight:
                        S[t][i,j] *= dist

        stdout.write(".")
        stdout.flush()
    print("")
    return S


##############################################################################

# DISPLAY functions

##############################################################################

# Write trajectories to output
def trajectories(result):
    print("Displaying results...")
    for p in result:
        print(p)

# Plot point trajectories from 2D sim result
def plot_2D(result):
    print("Displaying results...")
    for p in result:
        plot(p[0], p[1])

    xm = 0
    xM = 0
    ym = 0
    yM = 0
    for p in result:
        for x in p[0]:
            if x < 999999999 and x > xM:
                xM = x
            if x < xm:
                xm = x
        for y in p[1]:
            if y < 999999999 and y > yM:
                yM = y
            if y < ym:
                ym = y
    xlim(xm,xM)
    ylim(ym,yM)

    show()

# Animate point motion from 2D sim result
def anim_2D(result, rate=0.05):
    print("Displaying results...")
    xm = 0
    xM = 0
    ym = 0
    yM = 0
    for p in result:
        for x in p[0]:
            if x < 999999999 and x > xM:
                xM = x
            if x < xm:
                xm = x
        for y in p[1]:
            if y < 999999999 and y > yM:
                yM = y
            if y < ym:
                ym = y
    xlim(xm,xM)
    ylim(ym,yM)
    show()
    for t in range(1, len(result[0][0])):
        fig = figure()
        for p in result:
            plot(p[0][t], p[1][t], 'o')
        pause(rate)
        close(fig)

# Plot graphs from scan result
def net_anim(series, rate=0.05):
    print("Displaying results...")
    show()
    for A in series:
        fig = figure()
        G = from_numpy_matrix(A)
        draw(G, node_color="black", node_size=10)
        pause(rate)
        close(fig)

