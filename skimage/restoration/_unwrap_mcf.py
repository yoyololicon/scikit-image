import numpy as np
import networkx as nx

def W(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def mcf(x: np.ndarray, out: np.ndarray, weights=None, capacity=None):
    # construct index for each node
    N, M = x.shape
    index = np.arange(N * M).reshape(N, M)

    if capacity is None:
        cap_args = dict()
    else:
        cap_args = {'capacity': capacity}

    # get pseudo estimation of gradients along x and y axis
    psi1 = W(np.diff(x, axis=0))
    psi2 = W(np.diff(x, axis=1))

    G = nx.Graph()

    demands = np.round(-(psi1[:, 1:] - psi1[:, :-1] - psi2[1:, :] + psi2[:-1, :]) * 0.5 / np.pi).astype(np.int)
    # for convenience let's pad the demands so it match the shape of image
    # this add N + M - 1 dummy nodes with 0 demand
    demands = np.pad(demands, ((0, 1),) * 2, 'constant', constant_values=0)
    G.add_nodes_from(zip(index.ravel(), [{'demand': d} for d in demands.ravel()]))

    # set earth node index to -1, and its demand is the negative of the sum of all demands,
    # so the total demands is zero
    G.add_node(-1, demand=-demands.sum())

    # edges along x and y axis
    edges = np.vstack((
        np.vstack((index[:, :-1].ravel(), index[:, 1:].ravel())).T,
        np.vstack((index[:-1].ravel(), index[1:].ravel())).T)
    )

    if weights is None:
        weights = np.concatenate(((demands[:, :-1] == 0).ravel(), (demands[:-1] == 0).ravel()))
    else:
        weights = np.concatenate((weights[:(N-1), :(M-1)].ravel(), weights[:(N-1)].ravel())
        if not np.issubdtype(weight.dtype, np.integer):
            weights = (weights * 1000).astype(np.int)
    G.add_weighted_edges_from(zip(edges[:, 0], edges[:, 1], weights), **cap_args)

    # add the remaining edges that connected to earth node
    G.add_edges_from(zip([-1] * M, range(M)), **cap_args)
    G.add_edges_from(zip([-1] * M, range((N - 1) * M, N * M)), **cap_args)
    G.add_edges_from(zip([-1] * N, range(0, N * M, M)), **cap_args)
    G.add_edges_from(zip([-1] * N, range(M - 1, N * M, M)), **cap_args)

    # make graph to directed graph, so we can distinguish positive and negative flow
    G = G.to_directed()

    # perform MCF
    cost, flowdict = nx.network_simplex(G)

    # construct K matrix with the same shape as the gradients
    K2 = np.empty((N, M - 1))
    K1 = np.empty((N - 1, M))

    # add the flow to their orthogonal edge
    # the sign of the flow depends on those 4 vectors direction (clockwise or counter-clockwise)
    # when calculating the demands
    for i in range(N - 1):
        for j in range(M):
            if j == 0:
                K1[i][0] = -flowdict[-1][i * M] + flowdict[i * M][-1]
            else:
                K1[i][j] = -flowdict[i * M + j - 1][i * M + j] + flowdict[i * M + j][i * M + j - 1]

    for i in range(N):
        for j in range(M - 1):
            if i == 0:
                K2[i][j] = flowdict[-1][j] - flowdict[j][-1]
            else:
                K2[i][j] = flowdict[(i - 1) * M + j][i * M + j] - flowdict[i * M + j][(i - 1) * M + j]

    # the boundary node with index = 0 have only one edge to earth node,
    # so set one of its edge's K to zero
    K2[0, 0] = 0

    # derive correct gradients
    psi1 += K1 * 2 * np.pi
    psi2 += K2 * 2 * np.pi

    # integrate the gradients
    out[1:, 0] += np.cumsum(psi1[:, 0]) + x[0, 0]
    out[:, 1:] = np.cumsum(psi2, axis=1) + out[:, :1]
    return
