"""plotting functions"""
import numpy as np
import networkx as nx
from sklearn.utils import check_symmetric
import sklearn.datasets as skd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
import os
import sklearn.neighbors as skn
import sys


def plot_graph(graph, outfolder="", params=None, figsize=(5, 4)):
    """plot a graph in 2d or 3d"""
    if len(graph.nodes[list(graph)[0]]["pos"]) == 2:
        fig = plot_graph_2d(graph, figsize=figsize)

    elif len(graph.nodes[list(graph)[0]]["pos"]) == 3:
        # WIP
        # fig = plot_graph_3d(graph, node_colors=node_colors, params=params)
        pass

    fig.savefig(os.path.join(outfolder, graph.graph["name"] + ".png"))


def plot_graph_2d(graph, figsize=(5, 4)):
    """plot 2d graph"""
    pos = np.array([graph.nodes[u]["pos"] for u in graph])
    weights = np.log10(np.array([graph[u][v]["weight"] for u, v in graph.edges]))

    if "attribute" in graph.nodes[list(graph)[0]]:
        node_color = [graph.nodes[u]["attribute"] for u in graph]
    else:
        node_color = None

    fig = plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(
        graph, pos=pos, node_size=20, node_color=node_color, cmap=plt.get_cmap("tab20")
    )
    edges = nx.draw_networkx_edges(
        graph, pos=pos, width=1, edge_color=weights, edge_cmap=plt.get_cmap("plasma")
    )
    plt.axis("off")

    return fig


def plot_graph_3d(G, node_colors="custom", edge_colors=[], params=None):
    # WIP
    n = G.number_of_nodes()
    m = G.number_of_edges()

    pos = nx.get_node_attributes(G, "pos")
    pos = np.array([pos[i] for i in range(n)])
    node_colors = list(nx.get_node_attributes(G, "color").values())

    # node colors
    if node_colors == "degree":
        edge_max = max([G.degree(i) for i in range(n)])
        node_colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]
    else:
        node_colors = "k"

    # edge colors
    if edge_colors != []:
        edge_color = plt.cm.cool(edge_colors)
        width = np.exp(-(edge_colors - np.min(np.min(edge_colors), 0))) + 0.5
        norm = mpl.colors.Normalize(vmin=np.min(edge_colors), vmax=np.max(edge_colors))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.cool)
        cmap.set_array([])
    else:
        edge_color = ["b" for x in range(m)]
        width = [1 for x in range(m)]

    # 3D network plot
    with plt.style.context(("ggplot")):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=node_colors,
            s=200,
            edgecolors="k",
            alpha=0.7,
        )

        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            ax.plot(x, y, z, c=edge_color[i], alpha=0.3, linewidth=width[i])

    if edge_colors != []:
        fig.colorbar(cmap)

    if params == None:
        params = {"elev": 10, "azim": 290}
    elif params != None and "elev" not in params.keys():
        params["elev"] = 10
        params["azim"] = 290
    ax.view_init(elev=params["elev"], azim=params["azim"])

    ax.set_axis_off()

    return fig
