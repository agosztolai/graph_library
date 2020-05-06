"""all graph generator functions"""
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

from . import utils


def generate_multiscale(params):
    """Multiscale graph.""" 
    block_sizes = params['block_sizes']
    block_p = params['block_p']

    adjacency_matrix = np.zeros([block_sizes[0], block_sizes[0]])
    for size, p in zip(block_sizes, block_p):
        for i in range(int(block_sizes[0] / size)): 
            adjacency_matrix[i*size:(i+1)*size, i*size:(i+1)*size] = np.random.binomial(1,  p, size**2).reshape(size, size)

    adjacency_matrix = 0.5 * (adjacency_matrix  + adjacency_matrix.T)
    adjacency_matrix -= np.diag(np.diag(adjacency_matrix))
    #plt.figure()
    #plt.imshow(adjacency_matrix)
    #plt.show()

    graph = nx.Graph(nx.from_numpy_matrix(adjacency_matrix))
    utils.set_force_position(graph)
    utils.set_constant_weights(graph)
    return graph


def generate_karate(params=None):
    """karate clube graph"""
    graph = nx.karate_club_graph()

    for i in graph:
        if graph.nodes[i]["club"] == "Mr. Hi":
            graph.nodes[i]["attribute"] = 0
        else:
            graph.nodes[i]["attribute"] = 1

    utils.set_force_position(graph)
    utils.set_constant_weights(graph)

    return graph


def generate_barbell(params={"m1": 7, "m2": 0}):
    """barbell graph"""
    graph = nx.barbell_graph(params["m1"], params["m2"])

    utils.set_force_position(graph)
    utils.set_constant_weights(graph)

    return graph


def generate_clique(params={"n": 5}):
    """clique graph"""
    graph = nx.complete_graph(params["n"])

    utils.set_force_position(graph)
    utils.set_constant_weights(graph)
    return graph


def generate_erdos_renyi(params={"n": 50, "p": 0.1}):
    """erdos_renyi graph"""
    graph = nx.erdos_renyi_graph(params["n"], params["p"], seed=params["seed"])

    utils.set_force_position(graph)
    utils.set_constant_weights(graph)
    return graph


def generate_SBM(params={"sizes": [20, 20], "probs": [[1.0, 0.1,], [0.1, 1.0,]]}):

    graph = nx.stochastic_block_model(
        params["sizes"],
        np.array(params["probs"]) / params["sizes"][0],
        seed=params["seed"],
    )

    utils.set_force_position(graph)
    utils.set_constant_weights(graph)
    return graph


def generate_clique_of_cliques(
    params={"m": 5, "n": 3, "L": 500, "w": [1, 10, 100], "p": [0.01, 0.1, 1]}
):
    """multiscale symmetryc graph"""

    m = params["m"]
    levels = params["n"]
    N = m ** levels
    L = params["L"]

    A = np.zeros([N, N])
    for l in range(levels):
        for i in range(0, N, N // (m ** l)):
            for j in range(N // (m ** l)):
                for k in range(j + 1, N // (m ** l)):
                    A[i + j, i + k] = 0

        for i in range(0, N, N // (m ** l)):
            for j in range(N // (m ** l)):
                for k in range(j + 1, N // (m ** l)):
                    A[i + j, i + k] = params["w"][l] * np.random.binomial(
                        1, params["p"][l]
                    )

    pos = np.zeros([N, 2])
    for i in range(levels):
        for k in range(m ** (i + 1)):
            pos[k * N // (m ** (i + 1)) : (k + 1) * N // (m ** (i + 1)), :] += [
                L / (3 ** i) * np.cos((k % m) * 2 * np.pi / m),
                L / (3 ** i) * np.sin((k % m) * 2 * np.pi / m),
            ]

    graph = nx.from_numpy_matrix(A + A.T)

    for u in graph:
        graph.nodes[u]["pos"] = pos[u]

    return graph


def generate_powergrid(params={}):
    """powergrid graph"""

    edges = np.genfromtxt("../datasets/UCTE_edges.txt")
    location = np.genfromtxt("../datasets/UCTE_nodes.txt")
    posx = location[:, 1]
    posy = location[:, 2]
    pos = {}

    edges = np.array(edges, dtype=np.int32)
    graph = nx.Graph()
    graph.add_edges_from(edges)

    graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_label")
    for u in graph.nodes:
        graph.nodes[u]["pos"] = [
            posx[graph.nodes[u]["original_label"] - 1],
            posy[graph.nodes[u]["original_label"] - 1],
        ]
    utils.set_constant_weights(graph)
    return graph


def generate_dolphin(params={}):
    """social network of dolphin"""
    graph = nx.read_gml("../datasets/dolphins.gml")
    graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_label")

    utils.set_force_position(graph)
    utils.set_constant_weights(graph)
    return graph


def generate_football(params={}):
    """americal football graph"""

    graph = nx.read_gml("../datasets/football.gml")
    graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_label")

    utils.set_force_position(graph)
    utils.set_constant_weights(graph)
    return graph


def generate_grid(params={"n": 10, "m": 10}):
    """2d grid"""
    graph = nx.grid_2d_graph(params["n"], params["m"], periodic=False)
    graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_label")

    for u in graph:
        graph.nodes[u]["pos"] = np.array(graph.nodes[u]["original_label"])

    utils.set_constant_weights(graph)

    return graph


def generate_miserable(params={}):
    """characters from les miserables"""
    graph = nx.read_gml("../datasets/lesmis.gml")
    graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_label")

    utils.set_force_position(graph)
    utils.set_constant_weights(graph)
    return graph


def generate_netscience(params={}):
    """network of network scientists"""
    graph = nx.read_gml("../datasets/netscience.gml")
    graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_label")
    graph = graph.subgraph(max(nx.connected_components(graph), key=len))
    graph = nx.convert_node_labels_to_integers(graph)

    utils.set_force_position(graph)
    utils.set_constant_weights(graph)
    return graph


###### WIP below
def generate_celegans(params={"undirected": True}):

    from datasets.celegans.create_graph import create_celegans

    G, pos, labels, neuron_type, colors = create_celegans(location="datasets/celegans/")

    for i in G:
        G.nodes[i]["block"] = G.nodes[i]["labels"]

    if params["undirected"]:
        G = G.to_undirected()

    return G, pos


def generate_email(params={}):

    edges = np.loadtxt("../../datasets/email-Eu-core.txt").astype(int)
    G = nx.DiGraph()
    G.add_edges_from(edges)
    labels = np.loadtxt("../../datasets/email-Eu-core-department-labels.txt").astype(
        int
    )
    for i in G:
        G.nodes[i]["block"] = labels[i]

    G = nx.convert_node_labels_to_integers(G)

    return G, None


def generate_barbell_noisy(params={"m1": 7, "m2": 0, "noise": 0.5}, seed=None):

    if seed is not None:
        params["seed"] = seed

    np.random.seed(params["seed"])
    G = nx.barbell_graph(params["m1"], params["m2"])

    for i in G:
        G.nodes[i]["block"] = np.mod(i, params["m1"])
    for i, j in G.edges():
        G[i][j]["weight"] = abs(np.random.normal(1, params["noise"]))

    return G, None


def generate_barbell_asy(params={"m1": 7, "m2": 0}):

    A = np.block(
        [
            [
                np.ones([params["m1"], params["m1"]]),
                np.zeros([params["m1"], params["m2"]]),
            ],
            [
                np.zeros([params["m2"], params["m1"]]),
                np.ones([params["m2"], params["m2"]]),
            ],
        ]
    )
    A = A - np.eye(params["m1"] + params["m2"])
    A[params["m1"] - 1, params["m1"]] = 1
    A[params["m1"], params["m1"] - 1] = 1
    G = nx.from_numpy_matrix(A)

    for i in G:
        G.nodes[i]["block"] = np.mod(i, params["m1"])

    return G, None


def generate_dumbbell_of_stars(params={"n": 15, "m": 10}):

    G = nx.Graph()
    G.add_star(np.arange(params["m"]))
    G.add_star(np.arange(params["m"] - 1, params["m"] + params["n"]))

    return G, None


def generate_Fan(
    params={"w_in": 1.5, "l": 4, "g": 32, "p_in": 0.125, "p_out": 0.125}, seed=None
):

    if seed is not None:
        params["seed"] = seed

    G = nx.planted_partition_graph(
        params["l"], params["g"], params["p_in"], params["p_out"], params["seed"]
    )

    for i, j in G.edges:
        if G.nodes[i]["block"] == G.nodes[j]["block"]:
            G[i][j]["weight"] = params["w_in"]
        else:
            G[i][j]["weight"] = 2 - params["w_in"]

    labels_gt = []
    for i in range(params["l"]):
        labels_gt = np.append(labels_gt, i * np.ones(params["g"]))

    for n in G.nodes:
        G.nodes[n]["block"] = labels_gt[n - 1]

    G = nx.convert_node_labels_to_integers(G, label_attribute="old_label")

    return G, None


def generate_frucht(params={}):

    G = nx.frucht_graph()

    return G, None


def generate_GN(params={"l": 4, "g": 32, "p_in": 0.4, "p_out": 0.2}, seed=0):

    if seed is not None:
        params["seed"] = seed

    G = nx.planted_partition_graph(
        params["l"], params["g"], params["p_in"], params["p_out"], seed=params["seed"]
    )

    labels_gt = []
    for i in range(params["l"]):
        labels_gt = np.append(labels_gt, i * np.ones(params["g"]))

    for n in G.nodes:
        G.nodes[n]["block"] = labels_gt[n - 1]

    return G, None


def generate_geometric(params={"n": 50, "p": 0.3}, seed=None):

    if seed is not None:
        params["seed"] = seed

    G = nx.random_geometric_graph(params["n"], params["p"])

    return G, None


def generate_2grid(params={"n": 10}):

    F = nx.grid_2d_graph(params["n"], params["n"], periodic=False)
    F = nx.convert_node_labels_to_integers(F, label_attribute="old_label")

    pos = {}
    for i in F:
        pos[i] = np.array(F.nodes[i]["old_label"])

    H = nx.grid_2d_graph(params["n"], params["n"], periodic=False)
    H = nx.convert_node_labels_to_integers(
        H, first_label=len(F), label_attribute="old_label"
    )

    for i in H:
        pos[i] = np.array(H.nodes[i]["old_label"]) + np.array([params["n"] + 5, 0])

    G = nx.compose(F, H)
    G.add_edge(
        int(params["n"] ** 2 - params["n"] / 2 + 1),
        int(params["n"] ** 2 + params["n"] / 2 + 1),
    )
    G.add_edge(
        int(params["n"] ** 2 - params["n"] / 2), int(params["n"] ** 2 + params["n"] / 2)
    )
    G.add_edge(
        int(params["n"] ** 2 - params["n"] / 2 - 1),
        int(params["n"] ** 2 + params["n"] / 2 - 1),
    )

    return G, pos


def generate_delaunay_grid(params={"n": 10}, seed=None):

    from scipy.spatial import Delaunay

    if seed is not None:
        params["seed"] = seed

    np.random.seed(params["seed"])

    x = np.linspace(0, 1, params["n"])

    pos = []
    for i in range(params["n"]):
        for j in range(params["n"]):
            pos.append([x[j], x[i]])

    pos = np.array(pos)

    tri = Delaunay(pos)

    edge_list = []
    for t in tri.simplices:
        edge_list.append([t[0], t[1]])
        edge_list.append([t[0], t[2]])
        edge_list.append([t[1], t[2]])

    G = nx.Graph()
    G.add_nodes_from(np.arange(params["n"]))
    G.add_edges_from(edge_list)

    return G, pos


def generate_gnr(param={"n": 20, "p": 0.2}):

    G = nx.gnr_graph(params["n"], params["p"])

    return G, None


def generate_grid_delaunay_nonunif(params={"n": 10}, seed=None):

    from scipy.spatial import Delaunay

    if seed is not None:
        params["seed"] = seed

    np.random.seed(params["seed"])
    x = np.linspace(0, 1, params["n"])

    pos = []
    for i in range(params["n"]):
        for j in range(params["n"]):
            pos.append([x[j], x[i]])

    pos = np.array(pos)

    gauss_pos = [0.5, 0.5]
    gauss_pos2 = [0.7, 0.7]
    gauss_var = [0.05, 0.05]
    new_points = np.random.normal(gauss_pos, gauss_var, [20, 2])
    # new_points = np.concatenate( (new_points, np.random.normal( gauss_pos2, gauss_var, [50,2])) )

    for p in new_points:
        if p[0] > 0 and p[0] < 1.0 and p[1] > 0 and p[1] < 1:
            pos = np.concatenate((pos, [p,]))

    # pos = np.concatenate( (pos, np.random.normal(.5,.1, [200,2])) )

    tri = Delaunay(pos)

    edge_list = []
    for t in tri.simplices:
        edge_list.append([t[0], t[1]])
        edge_list.append([t[0], t[2]])
        edge_list.append([t[1], t[2]])

    G = nx.Graph()
    G.add_nodes_from(np.arange(len(pos)))
    G.add_edges_from(edge_list)

    return G, pos


def generate_krackhardt(params={}):

    G = nx.Graph(nx.krackhardt_kite_graph())
    for i, j in G.edges:
        G[i][j]["weight"] = 1.0

    return G, None


def generate_LFR(
    params={
        "n": 1000,
        "tau1": 2,
        "tau2": 2,
        "mu": 0.5,
        "k": 20,
        "minc": 10,
        "maxc": 50,
        "scriptfolder": "./datasets/LFR-Benchmark/lfrbench_udwov",
        "outfolder": "/data/AG/geocluster/LFR/",
    },
    seed=None,
):

    if seed is not None:
        params["seed"] = seed

    command = (
        params["scriptfolder"]
        + " -N "
        + str(params["n"])
        + " -t1 "
        + str(params["tau1"])
        + " -t2 "
        + str(params["tau2"])
        + " -mut "
        + str(params["mu"])
        + " -muw "
        + str(params["mu"])
        + " -maxk "
        + str(params["n"])
        + " -k "
        + str(params["k"])
        + " -name "
        + params["outfolder"]
        + "data"
    )

    os.system(command)

    G = nx.read_weighted_edgelist(
        params["outfolder"] + "data.nse", nodetype=int, encoding="utf-8"
    )

    for e in G.edges:
        G.edges[e]["weight"] = 1

    labels = np.loadtxt(params["outfolder"] + "data.nmc", usecols=1, dtype=int)
    for n in G.nodes:
        G.nodes[n]["block"] = labels[n - 1]

    return G, None


def generate_powerlaw(params={}):

    G = nx.powerlaw_cluster_graph(params["n"], params["m"], params["p"])

    return G, None


def generate_ring_of_cliques(params={"m": 5, "n": 6}):

    G = nx.ring_of_cliques(params["m"], params["n"])

    x1 = np.linspace(-np.pi, np.pi, params["m"])
    x2 = np.linspace(0, 2 * np.pi, params["n"])[::-1]

    posx = np.zeros(params["m"] * params["n"])
    posy = np.zeros(params["m"] * params["n"])
    for i in range(params["m"]):
        for j in range(params["n"]):
            posx[i * params["n"] + j] = np.cos(x1[i]) + 0.5 * np.cos(
                x2[j] + x1[i] + 2 * np.pi * 3 / 5
            )
            posy[i * params["n"] + j] = np.sin(x1[i]) + 0.5 * np.sin(
                x2[j] + x1[i] + 2 * np.pi * 3 / 5
            )

    pos = [[posx[i], posy[i]] for i in range(params["m"] * params["n"])]

    return G, pos


def generate_scale_free(params={"n": 100}, seed=None):

    if seed is not None:
        params["seed"] = seed

    G = nx.scale_free_graph(params["n"])
    G = G.to_undirected()

    return G, None


def generate_swiss_roll(
    params={
        "n": 100,
        "noise": 0.0,
        "elev": 10,
        "azim": 270,
        "similarity": "knn",
        "similarity_par": 10,
    },
    seed=None,
):
    if seed is not None:
        params["seed"] = seed

    G = nx.Graph()

    pos, color = skd.make_swiss_roll(
        n_samples=params["n"], noise=params["noise"], random_state=params["seed"]
    )
    for i, _pos in enumerate(pos):
        G.add_node(i, pos=_pos, color=color[i])

    return G, pos


def generate_S(
    params={
        "n": 300,
        "elev": 10,
        "azim": 290,
        "s": 1.2,
        "similarity": "knn",
        "similarity_par": 10,
    },
    seed=None,
):

    if seed is not None:
        params["seed"] = seed

    G = nx.Graph()

    pos, color = skd.samples_generator.make_s_curve(
        params["n"], random_state=params["seed"]
    )

    for i, _pos in enumerate(pos):
        G.add_node(i, pos=_pos, color=color[i])

    return G, pos


def generate_NWS(params={"n": 50, "k": 2, "p": 0.3}):

    G = nx.newman_watts_strogatz_graph(params["n"], params["k"], params["p"])

    return G, None


def generate_star(params={"n": 10}):

    G = nx.star_graph(params["n"])

    return G, None


def generate_star_of_circles(params={"n": 5, "m": 10}):

    G = nx.Graph()
    G.add_star(np.arange(params["m"]))
    for i in range(1, params["m"]):
        G.add_cycle(
            [i]
            + list(
                range(
                    params["m"] + (i - 1) * params["n"], params["m"] + i * params["n"]
                )
            )
        )

    return G, None


def generate_torus(params={"m": 5, "n": 4}):

    G = nx.grid_2d_graph(params["n"], params["m"], periodic=True)

    pos = {}
    for i in G:
        pos[i] = np.array(G.nodes[i]["old_label"])

    return G, pos


def generate_tree(params={"r": 2, "h": 5}):

    G = nx.balanced_tree(params["r"], params["h"])

    return G, None


def generate_tutte(params={}):

    G = nx.tutte_graph()

    return G, None


def generate_triangle_of_triangles(params={"m": 1, "N": 3}):

    m = params["m"]
    N = params["N"]
    A = np.ones([N, N]) - np.eye(N)
    A = np.kron(np.eye(N ** m), A)
    A[2, 3] = 1
    A[3, 2] = 1
    A[1, 6] = 1
    A[6, 1] = 1
    A[4, 8] = 1
    A[8, 4] = 1
    A = np.vstack((np.hstack((A, np.zeros([9, 9]))), np.hstack((np.zeros([9, 9]), A))))
    A[0, 9] = 1
    A[9, 0] = 1

    G = nx.Graph(A)

    return G, None
