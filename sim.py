from math import sqrt
from random import seed, random
from operator import add, itemgetter
from sys import stdout
from scipy.stats import norm
from scipy.spatial import distance
from numpy import asarray, empty, zeros, expand_dims, cumsum, tile, copy
import networkx as nx
from rtree import index
from bintrees import FastAVLTree
import multiprocessing as mp

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
def sim(P, N, T, D, delta=1, bound=None, handling=None, init="Default", init_bound="Default", drift=None, origin_point=None):
    print("Simulating motion...")
    result = [];
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
        t = empty((D,N+1))

        # uniform random starting point
        if init is "Random":
            if init_bound is not None and handling is "Torus":
                for d in range(D):
                    t[d,0] = random() * init_bound
            elif init_bound is not None:
                # randomize by rolling until within radius
                for d in range(D):
                    t[d,0] = 999999999.0
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
        result.append(t)

    if origin_point:
        result.append([[0 for t in range(N+1)] for d in range(D)])

    return handle_bounds(handling, bound, result)

# handle_bounds: Apply boundary behavior to output from sim
def sim(P, N, T, D, delta=1, bound=None, handling=None, init="Default", init_bound="Default", drift=None, origin_point=None):
    print("Simulating motion...")
    result = [];
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
        t = empty((D,N+1))

        # uniform random starting point
        if init is "Random":
            if init_bound is not None and handling is "Torus":
                for d in range(D):
                    t[d,0] = random() * init_bound
            elif init_bound is not None:
                # randomize by rolling until within radius
                for d in range(D):
                    t[d,0] = 999999999.0
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
        result.append(t)

    if origin_point:
        result.append([[0 for t in range(N+1)] for d in range(D)])

    return handle_bounds(handling, bound, result)
# If "Torus" handling, %= bound coordinates
# If "Exit" handling, map all out of bound coords to 999999999.0
def handle_bounds(handling, bound, result):
    if handling is None:
        return result

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
                        result[p][d][n] = 999999999.0

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

# Spatial tree for the toroid case and cell lists for all cases
# require a list of directions on each side of a D-cube
def offset(D, bound=1):
    offsets = []
    for k in range(pow(2, D)):
        # we do this using the binary integers up to 2^D:
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

    return offsets


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

    if(D is 1):
        return scan_1D(result, d, self_edges=self_edges, weight=weight, bound=bound, handling=handling, memory=memory)
    if(D < 3 or handling=="Torus"):
        return scan_cells(result, d, self_edges=self_edges, weight=weight, bound=bound, handling=handling, memory=memory)

    p = index.Property(dimension=D)
    # using an R*-tree variant did not produced faster speeds for any n tested
    # (up to 10,000)

    G = []

    if handling is "Torus":
        offsets = offset(D, bound)

    # Each time
    for t in range(N):
        # copy previous matrix or start fresh as appropriate
        if t is 0 or memory is False:
            G.append(nx.Graph())
        else:
            G.append(G[t-1].copy())

        G[t].add_nodes_from(range(P))

        # build rtree of points unless they are out of bounds
        idx = index.Index(properties=p)
        for i in range(P):
            I = []
            # get point coordinates
            for coord in range(D):
                I.append(result[i][coord][t])

            if I[0] < 999999999.0:
                # insert point as bounding box where min = max
                idx.insert(i, tuple(I + I))

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
                hits = set([])
                if handling is "Torus":
                    for o in offsets:
                        # add the offset
                        m = list(map(add, I, o))
                        # rtree coords are a tuple: (xmin, ymin, ... xmax, ymax)
                        # representing a bounding box
                        # for points, obviously, all min = max
                        minima = [x - d for x in m]
                        maxima = [x + d for x in m]
                        tup = tuple(minima + maxima)
                        hits = hits.union(set(idx.intersection(tup)))
                else:
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
                        if self_edges or i != j:
                            G[t].add_edge(i, j, weight=(dist if weight else 1))

        stdout.write(".")
        stdout.flush()
    print("")
    return G


# implementation of scan using cell lists
# faster than tree in open spaces of dimension 1 or 2
# and on toroids of any dimension
def scan_cells(result, d, self_edges=False, weight=False, bound=None, handling=None, memory=False):
    stdout.write("Assembling graphs")
    stdout.flush()
    # recover sim parameters
    N = len(result[0][0]) # "N" = N + 1
    D = len(result[0])
    P = len(result)

    offsets = offset(D)
    G = []

    for t in range(N):
        if t is 0 or memory is False:
            G.append(nx.Graph())
        else:
            G.append(G[t-1].copy())

        G[t].add_nodes_from(range(P))

        cells = {}

        # build hash table
        # each point hashed to key '(x,y,...,z)'
        for p in range(P):
            skip = False
            cell = "("
            for coord in range(D):
                c = result[p][coord][t]
                if c == 999999999.0:
                    skip = True
                else:
                    c = c // d
                    cell += str(int(c)) + ","

            cell = cell[:-1] + ')'
            if skip is not True and cell in cells:
                cells[cell].append(p)
            elif skip is not True:
                cells[cell] = [p]

        # then for each cell in the table, build list of contents + neighbors'
        # contents. Do this by decomposing the key string and adding offsets
        for cell in list(cells):
            coords = [int(coord) for coord in cell[1:-1].split(",")]
            neighborhood = [list(map(add, coords, o)) for o in offsets]

            if handling is "Torus":
                mod_neighbors = []
                for n in neighborhood:
                    m = [coord % int(bound//d) for coord in n]
                    mod_neighbors.append(m)
                neighborhood = []
                for m in mod_neighbors:
                    if m not in neighborhood:
                        neighborhood.append(m)

            # now that we have a list of neighbor cell coordinates, get the
            # particles they contain by rebuilding the (coord) strings...
            neighbors = []
            for n in neighborhood:
                ncell = "("
                for coord in range(D):
                    c = n[coord]
                    c = c // d
                    ncell += str(int(c)) + ","

                ncell = ncell[:-1] + ')'
                if ncell in cells:
                    neighbors += cells[ncell]

            # and for each particle in the current cell, check against all the
            # particles in the list of neighbors
            for i in cells[cell]:
                I = []
                for coord in range(D):
                    I.append(result[i][coord][t])
                for j in neighbors:
                    J = []
                    for coord in range(D):
                        J.append(result[j][coord][t])
                    if handling is "Torus":
                        dist = torus_distance(I,J, bound)
                    else:
                        dist = distance.euclidean(I,J)
                    # connect nodes if in distance and appropriate re self edges
                    if dist < d:
                        if self_edges or i != j:
                            G[t].add_edge(i, j, weight=(dist if weight else 1))

        stdout.write(".")
        stdout.flush()
    print("")
    return G

def scan_cells_multi(result, d, self_edges=False, weight=False, bound=None, handling=None, memory=False):
    stdout.write("Assembling graphs")
    stdout.flush()
    # recover sim parameters
    N = len(result[0][0]) # "N" = N + 1
    D = len(result[0])
    P = len(result)

    offsets = offset(D)

    # do the multi thing
    man = mp.Manager()
    output = man.Queue()
    S = []
    for chunk in range(max(N/10 + 1, 1)):
        start = chunk * 10
        end = min(start + 10, N)

        processes = [mp.Process(target=scan_cells_multi_t, args=(result, d, t, P, D, N, output, self_edges, weight, bound, handling, offsets)) for t in range(start, end)]

        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()

        s = [output.get() for proc in processes]
        s.sort(key=itemgetter(1))
        S.extend(s)

    S = [A for (A, t) in S]

    if memory is not False:
        for t in range(1, N):
            for i in range(P):
                for j in range(P):
                    S[t][i][j] = (S[t][i][j] or S[t - 1][i][j])

    print("")
    return S

def scan_cells_multi_t(result, d, t, P, D, N, output, self_edges, weight, bound, handling, offsets):
    G = nx.Graph()
    G.add_nodes_from(range(P))

    cells = {}

    # build hash table
    for p in range(P):
        skip = False
        cell = "("
        for coord in range(D):
            c = result[p][coord][t]
            if c == 999999999.0:
                skip = True
            else:
                c = c // d
                cell += str(int(c)) + ","
        cell = cell[:-1] + ")"

        if skip is not True and cell in cells:
            cells[cell].append(p)
        elif skip is not True:
            cells[cell] = [p]

        # then for each cell in the table, build list of contents + neighbors'
        # contents. Do this by decomposing the key string and adding offsets
    for cell in list(cells):
        coords = [int(coord) for coord in cell[1:-1].split(",")]
        neighborhood = [list(map(add, coords, o)) for o in offsets]

        if handling is "Torus":
            mod_neighbors = []
            for n in neighborhood:
                m = [coord % int(bound//d) for coord in n]
                mod_neighbors.append(m)
            neighborhood = []
            for m in mod_neighbors:
                if m not in neighborhood:
                    neighborhood.append(m)

        # now that we have a list of neighbor cell coordinates, get the
        # particles they contain by rebuilding the (coord) strings...
        neighbors = []
        for n in neighborhood:
            ncell = "("
            for coord in range(D):
                c = n[coord]
                c = c // d
                ncell += str(int(c)) + ","

            ncell = ncell[:-1] + ')'
            if ncell in cells:
                neighbors += cells[ncell]

        # and for each particle in the current cell, check against all the
        # particles in the list of neighbors
        for i in cells[cell]:
            I = []
            for coord in range(D):
                I.append(result[i][coord][t])
            for j in neighbors:
                J = []
                for coord in range(D):
                    J.append(result[j][coord][t])
                if handling is "Torus":
                    dist = torus_distance(I,J, bound)
                else:
                    dist = distance.euclidean(I,J)
                # connect nodes if in distance and appropriate re self edges
                if dist < d:
                    if self_edges or i != j:
                        G.add_edge(i, j, weight=(dist if weight else 1))

    stdout.write(".")
    stdout.flush()
    output.put((G, t))


# scan: sim result -> list of np array adjacency matrices
# self_edges toggles whether nodes are within distance of themselves
# if weights, A[ij] = d(ij), else A[ij] = (d(ij) < d)
# memory sets whether the network remembers previous edges
# [ should probably be broken up into more functions ]
# t is this iter
def scan_multi(result, d, self_edges=False, weight=False, bound=None, handling=None, memory=False):
    stdout.write("Assembling graphs")
    stdout.flush()
    # recover sim parameters
    N = len(result[0][0]) # "N" = N + 1
    D = len(result[0])
    P = len(result)

    if(D is 1):
        return scan_1D(result, d, self_edges=self_edges, weight=weight, bound=bound, handling=handling, memory=memory)
    if(D < 3 or handling=="Torus"):
        return scan_cells_multi(result, d, self_edges=self_edges, weight=weight, bound=bound, handling=handling, memory=memory)

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
    else:
        offsets = None

    # do the multi thing
    man = mp.Manager()
    output = man.Queue()
    S = []
    for chunk in range(max(N/10 + 1, 1)):
        start = chunk * 10
        end = min(start + 10, N)

        processes = [mp.Process(target=scan_multi_t, args=(result, d, t, P, D, N, output, self_edges, weight, bound, handling, offsets)) for t in range(start, end)]

        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()

        s = [output.get() for proc in processes]
        s.sort(key=itemgetter(1))
        S.extend(s)

    S = [A for (A, t) in S]

    if memory is not False:
        for t in range(1, N):
            for i in range(P):
                for j in range(P):
                    S[t][i][j] = (S[t][i][j] or S[t - 1][i][j])

    print("")
    return S

def scan_multi_t(result, d, t, P, D, N, output, self_edges, weight, bound, handling, offsets):
    # Each time
    # build rtree of points unless they are out of bounds
    p = index.Property(dimension=D)
    G = nx.Graph()
    G.add_nodes_from(range(P))
    # using an R*-tree variant did not produce faster speeds for any n tested
    # (up to 10,000)
    idx = index.Index(properties=p)

    for i in range(P):
        I = []
        # get point coordinates
        for coord in range(D):
            I.append(result[i][coord][t])

        if I[0] < 999999999.0:
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
            hits = set([])
            if handling is "Torus":
                for o in offsets:
                        # add the offset
                    m = list(map(add, I, o))
                        # rtree coords are a tuple: (xmin, ymin, ... xmax, ymax)
                        # representing a bounding box
                        # for points, obviously, all min = max
                    minima = [x - d for x in m]
                    maxima = [x + d for x in m]
                    tup = tuple(minima + maxima)
                    hits = hits.union(set(idx.intersection(tup)))
            else:
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
                    if self_edges or i != j:
                        G.add_edge(i, j, weight=(dist if weight else 1))

    stdout.write(".")
    stdout.flush()
    output.put((G, t))

# for the 1D case, use an AVL tree
def scan_1D(result, d, self_edges=False, weight=False, bound=None, handling=None, memory=False):
    # recover sim parameters
    N = len(result[0][0]) # "N" = N + 1
    D = len(result[0])
    P = len(result)

    G = []          # list of graphs

    # Each time
    for t in range(N):
        # copy previous matrix or start fresh as appropriate
        if t is 0 or memory is False:
            G.append(nx.Graph())
        else:
            G.append(G[t-1].copy())

        G[t].add_nodes_from(range(P))


        idx = FastAVLTree()
        # insert all points not out of bounds
        for i in range(P):
            I = result[i][0][t]
            if I < 999999999.0:
                idx.insert(I, i)


        for i in range(P):
            I = result[i][0][t]
            if I < 999999999.0:
                minimum = I - d
                maximum = I + d
                # get all results within range
                hits = [v for (k,v) in idx.item_slice(minimum,maximum)]
                if handling is "Torus" and maximum > bound:
                    hits += [v for (k,v) in idx.item_slice(0, maximum % bound)]
                if handling is "Torus" and minimum < 0:
                    hits += [v for (k,v) in idx.item_slice(minimum % bound, bound)]
                for j in hits:
                    if self_edges or i != j:
                        G[t].add_edge(i, j, weight=1)
                    # add something to handle weight case here
        stdout.write(".")
        stdout.flush()
    print("")
    return G

