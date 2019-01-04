from sim import *
from sys import stdout
from numpy import maximum
import networkx as nx

# for every g, f(g)
def series(measure, graphs):
    stdout.write("Computing series")
    stdout.flush()
    return [measure(g) for g in graphs]

# computes a measure on the temporal network, then takes the average
def average_value(measure, graphs):
    result = series(measure, graphs)
    stdout.write("Computing average")
    stdout.flush()
    L = len(result)
    result = sum(result) + 0.0
    result /= L
    print("")
    return result

# outputs a time series averaged over n runs
# TODO a version of this which computes multiple measures on each run
def average_run_series(n, measure, P, N, T, D, bound=None, handling=None, init="Default", init_bound="Default", drift=None, origin_point=None):
    S = []
    for i in range(n):
        stdout.write("Run ")
        stdout.write(str(i))
        stdout.flush()
        print("")

        result = sim(P, N, T, D, bound=bound, init_bound=init_bound, handling=handling, init=init, drift=drift, origin_point=origin_point)
        G = scan_multi(result, r, bound=bound, handling=handling, memory=memory)
        S.append(series(measure, G))

    out = []
    for i in range(len(S[0])):
        su = 0.0
        for s in S:
            su += s[i]
        out.append(su / len(S))

    return out

def average_run_average_value(n, measure, P, N, T, D, bound=None, handling=None, init="Default", init_bound ="Default", drift=None, origin_point=None):
    result = average_run_series(n, measure, P, N, T, D, bound=bound, init_bound=init_bound, handling=handling, init=init, drift=drift, origin_point=origin_point)
    L = len(result)
    result = sum(result) + 0.0
    result /= L
    print("")
    return result

def largest_component_series(measure, matrices):
    stdout.write("Computing series")
    stdout.flush()
    result = []
    for m in matrices:
        G = m
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
        G = m
        G = nx.connected_component_subgraphs(G)
        s = 0
        for g in G:
            s += measure(g)
        result.append(s/len(G))
        stdout.write(".")
        stdout.flush()
    print("")
    return result

def build_moving_window(matrices, d, truncate=True):
    stdout.write("Building moving window")
    stdout.flush()
    M = []
    graphs = []

    for t in range(len(matrices)):
        M.append(nx.to_numpy_matrix(matrices[t]))

    for t in range(len(matrices)):
        graphs.append(M[t].copy())
        start = max(0, t - d)
        for i in range(start, t):
            graphs[t] = maximum(graphs[t], M[i])
        stdout.write(".")
        stdout.flush()

    for t in range(len(matrices)):
        graphs[t] = nx.from_numpy_matrix(graphs[t])

    print("")
    if(truncate):
        return(graphs[d:])
    else:
        return(graphs)

def detection_time(graphs, p, T, N):
    if p is "Origin": # ASSUMES the origin point has been correctly added
        p = len(graphs[0]) - 1
    for t in range(len(graphs)):
        nbr = [i for i in graphs[t].neighbors(p)]
        if len(nbr):
            return t * (0.0 + T)/N

def detection_times(graphs, T, N):
    return [detection_time(graphs, p, T, N) for p in range(len(graphs[0]))]

def average_detection_time(graphs, T, N):
    detimes = detection_times(graphs, T, N)
    if None in detimes:
        return None
    return reduce(lambda x, y: x + y, detimes) / len(detimes)

def coverage_time(graphs, T, N):
    detection_times = [detection_time(graphs, p, T, N) for p in range(len(graphs[0]))]
    if None in detection_times:
        return None
    return max(detection_times)

def broadcast_time(graphs, p):
    if p is "Origin": # ASSUMES the origin point has been correctly added
        p = len(graphs[0]) - 1
    r = set([p]) # start with broadcast point
    for t in range(len(graphs)): # each t
        G = nx.connected_component_subgraphs(graphs[t]) # get components
        for g in G: # each component
            if bool(r & set(nx.nodes(g))): # if nonempty intersection with nodes that have been reached
                r = r | set(nx.nodes(g)) # reached nodes becomes union with this component
        if len(r) == len(nx.nodes(graphs[t])):
            return t * (0.0 + T)/N


