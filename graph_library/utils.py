"""utils functions"""
import networkx as nx


def set_force_position(graph, method="spring_layout"):
    """set positions"""
    if method == "spring_layout":
        pos = nx.spring_layout(graph, weight="weight")
        for u in graph:
            graph.nodes[u]["pos"] = pos[u]

    if method == "fa2":
        # TODO: implement the fa2 faster algorithm
        pass


def set_constant_weights(graph, weight=1.0):
    """set constant edge weights"""
    for u, v in graph.edges:
        graph[u][v]["weight"] = weight


def similarity_matrix(G, sim=None, par=None, symmetric=True):
    """compute similarity graph from points"""
    # WIP
    if sim == None:
        raise ValueError("Specify similarity measure!")
    if par == None:
        raise ValueError("Specify parameter(s) of similarity measure!")

    n = G.number_of_nodes()
    pos = nx.get_node_attributes(G, "pos")
    pos = np.reshape([pos[i] for i in range(n)], (n, len(pos[0])))

    if sim == "euclidean" or sim == "minkowski":
        A = squareform(pdist(pos, sim))

    elif sim == "knn":
        A = skn.kneighbors_graph(
            pos,
            par,
            mode="connectivity",
            metric="minkowski",
            p=2,
            metric_params=None,
            n_jobs=-1,
        )
        A = A.todense()

    elif sim == "radius":
        A = skn.radius_neighbors_graph(
            pos,
            par,
            mode="connectivity",
            metric="minkowski",
            p=2,
            metric_params=None,
            n_jobs=-1,
        )
        A = A.todense()

    elif sim == "rbf":
        gamma_ = par
        A = rbf_kernel(pos, gamma=gamma_)

    if symmetric == True:
        A = check_symmetric(A)

    for i in range(n):
        for j in range(n):
            if np.abs(A[i, j]) > 0:
                G.add_edge(i, j, weight=A[i, j])

    return G
