"""library of some relevant graphs for code testing"""
import numpy as np
import networkx as nx
from sklearn.utils import check_symmetric
import sklearn.datasets as skd
import pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
import os
import sklearn.neighbors as skn
import sys

from . import graph_generators
from .plotting import plot_graph


def generate(graph_name, params=None, plot=True, save=True, outfolder=""):
    """main function to generate a graph"""

    if params is None:
        params = {}

    if "seed" in params:
        np.random.seed(params["seed"])

    graph = getattr(graph_generators, "generate_%s" % graph_name)(params)
    graph.graph["name"] = graph_name

    if not nx.is_connected(graph):
        print("WARNING: Graph is disconnected!!")

    if plot:
        plot_graph(graph, outfolder=outfolder)

    if save:
        nx.write_gpickle(
            graph, os.path.join(outfolder, "graph_" + graph_name + ".gpickle")
        )

    return graph


def generate_graph_family(whichgraph, params={}, nsamples=2, plot=True, outfolder=""):
    # WIP
    counter = 0
    if "seed" not in params.keys():
        params["seed"] = 0

    disconnected = 0
    while counter < nsamples and disconnected == 0:
        params["seed"] += 1

        G = generate(
            whichgraph, params=None, plot=plot, save=False, outfolder=outfolder
        )

        if not nx.is_connected(G):
            disconnected = 1
            print("Graph is disconnected!!")
        else:
            disconnected = 0

            # create a folder
            if not os.path.isdir(outfolder):
                os.mkdir(outfolder)

            # save
            nx.write_gpickle(G, outfolder + whichgraph + ".gpickle")

            counter += 1
