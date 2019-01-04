from sim import *
from matplotlib.pyplot import figure, subplots, axis, pause, plot, xlim, ylim, show, close, Circle
from matplotlib.collections import LineCollection
from networkx import from_numpy_matrix, draw, draw_shell, draw_circular, draw_spectral, draw_spring

##############################################################################

# DISPLAY functions

##############################################################################

# Write trajectories to output
def trajectories(result):
    print("Displaying results...")
    for p in result:
        print(p)

# Plot point trajectories from 2D sim result
# time: int, plot point at specific time along trajectory
# radius: float, show radius at given time or at all times if time=None
# edges: bool, show edges between points within radius of each other
def plot_2D(result, time=None, radius=None, edges=False, paths='black', center="Default"):
    from scipy.spatial import distance
    print("Displaying results...")
    fig, ax = subplots()

    for p in result:
        if paths is not None:
            plot(p[0], p[1], linewidth=1, color=paths, alpha=0.2, zorder=1)
        if time is not None:
            plot(p[0][time], p[1][time], 'o', markersize=2.5, color="black", zorder=3)
        if radius is not None and time is not None:
            ax.add_artist(Circle((p[0][time], p[1][time]), radius, linestyle='--', color='k', alpha=0.5, fill=False, zorder=2))

    if time is not None and edges is not None:
        edge = []
        for i in result:
            i_t = [i[0][time], i[1][time]]
            for j in result:
                j_t = [j[0][time], j[1][time]]
                if i is not j and distance.euclidean(i_t, j_t) < edges:
                    edge.append([tuple(i_t), tuple(j_t)])
        lines = LineCollection(edge, color='k', linewidth=1, zorder=3)
        ax.add_collection(lines)

    ax.autoscale()
    axis("equal")
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
            if x < 999999999.0 and x > xM:
                xM = x
            if x < xm:
                xm = x
        for y in p[1]:
            if y < 999999999.0 and y > yM:
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
def net_anim(series, rate=0.05, technique="default"):
    print("Displaying results...")
    show()
    for A in series:
        fig = figure()
        G = A
        if technique is "default":
            draw(G, node_color="black", node_size=10)
        elif technique is "Circular":
            draw_circular(G, node_color="black", node_size=10)
        elif technique is "Spectral":
            draw_spectral(G, node_color="black", node_size=10)
        elif technique is "Spring":
            draw_spring(G, node_color="black", node_size=10)
        elif technique is "Shell":
            draw_shell(G, node_color="black", node_size=10)
        pause(rate)
        close(fig)

