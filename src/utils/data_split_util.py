import numpy as np
import networkx as nx
from scipy.sparse import triu, tril
from scipy.sparse.csgraph import depth_first_tree


# TODO: some optimisations may be possible now that nodes are always integer-indexed.


def leave_fraction_out(data, test_set_frac=0.2, directed=False, **_kwargs):
    if test_set_frac == 0:
        # If no test set edges are asked, then every data edge is seen as a test edge. ALmost every negative edge is
        # also in the test set.
        train_data = data
        test_data_pos = data
        test_data_neg, test_neg_corr_factor = generate_negative_edges(data, nb_neg_per_data_edge=1, directed=directed)

        test_labels_pos = np.ones(test_data_pos.shape[0], np.bool)
        test_labels_neg = np.zeros(test_data_neg.shape[0], np.bool)
        test_data = np.concatenate((test_data_pos, test_data_neg))
        test_labels = np.concatenate((test_labels_pos, test_labels_neg))
        return train_data, (test_data, test_labels, test_neg_corr_factor)

    # Split the data.
    train_data, test_data_pos = quick_split(data, test_set_frac)

    # Filter out positive test edges that contain nodes that do not occur in the training set.
    train_ids = np.unique(train_data[:, :2])
    test_ids = np.unique(test_data_pos[:, :2])
    ids_to_filter = np.setdiff1d(test_ids, train_ids)
    rows_to_filter = np.any(np.isin(test_data_pos[:, :2], ids_to_filter), axis=1)
    test_data_pos = test_data_pos[np.logical_not(rows_to_filter), :2]
    if test_data_pos.shape[0] == 0:
        raise ValueError("In the positive test set, all edges featured an edge that was not in the train set. There are"
                         " therefore no positive set edges left!")
    data = np.concatenate((train_data, test_data_pos), axis=0)

    # Generate negative edges of the same proportion as the test set.
    test_data_neg, test_neg_corr_factor = generate_negative_edges(data, nb_neg_per_data_edge=test_set_frac,
                                                                  directed=directed)

    # Create labels for the edges and concatenate them.
    test_labels_pos = np.ones(test_data_pos.shape[0], np.bool)
    test_labels_neg = np.zeros(test_data_neg.shape[0], np.bool)
    test_data = np.concatenate((test_data_pos, test_data_neg))
    test_labels = np.concatenate((test_labels_pos, test_labels_neg))

    # For good measure, shuffle everything one more time (probably not needed though).
    randomized_idx = np.random.permutation(test_data.shape[0])
    test_data = test_data[randomized_idx]
    test_labels = test_labels[randomized_idx]

    return train_data, (test_data, test_labels, test_neg_corr_factor)


# def simple_split(data, test_set_fraction=0.2):
#     np.random.shuffle(data)
#     nb_train_data_edges = int((1-test_set_fraction) * data.shape[0])
#     train_data = data[:nb_train_data_edges]
#     test_data_pos = data[nb_train_data_edges:]
#     return train_data, test_data_pos


# From https://github.com/Dru-Mara/EvalNE
def quick_split(data, test_set_fraction=0.51, directed=False):
    train_frac = 1 - test_set_fraction
    if train_frac <= 0.0 or train_frac > 1.0:
        raise ValueError('The test_frac parameter needs to be in range: (0.0, 1.0]')
    if train_frac == 1.0:
        return data

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_edges_from(data[:, :2])

    prepped_graph = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted')
    index_to_label = dict(list(zip(sorted(prepped_graph.nodes), sorted(G.nodes))))
    G = prepped_graph

    # Restrict input graph to its main cc
    a = triu(nx.adj_matrix(G, nodelist=sorted(prepped_graph.nodes())))

    # Compute initial statistics and linear indx of nonzeros
    n = a.shape[0]
    num_tr_e = int(a.nnz * train_frac)
    nz_lin_ind = np.ravel_multi_index(a.nonzero(), (n, n))

    # Build a dft starting at a random node. If dir false returns only upper triang
    dft = depth_first_tree(a, np.random.randint(0, a.shape[0]), directed=nx.is_directed(G))
    if nx.is_directed(G):
        dft_lin_ind = np.ravel_multi_index(dft.nonzero(), (n, n))
    else:
        dft_lin_ind = np.ravel_multi_index(triu(tril(dft).T + dft, k=1).nonzero(), (n, n))

    # From all nonzero indx remove those in dft. From the rest take enough to fill train quota. Rest are test
    rest_lin_ind = np.setdiff1d(nz_lin_ind, dft_lin_ind)
    aux = np.random.choice(rest_lin_ind, num_tr_e-len(dft_lin_ind), replace=False)
    lin_tr_e = np.union1d(dft_lin_ind, aux)
    lin_te_e = np.setdiff1d(rest_lin_ind, aux)

    # Unravel the linear indices to obtain src, dst pairs
    tr_e = np.array(np.unravel_index(np.array(lin_tr_e), (n, n))).T
    te_e = np.array(np.unravel_index(np.array(lin_te_e), (n, n))).T

    train_data = []
    for e in tr_e:
        train_data.append([index_to_label[e[0]], index_to_label[e[1]]])
    test_data_pos = []
    for e in te_e:
        test_data_pos.append([index_to_label[e[0]], index_to_label[e[1]]])
    return np.array(train_data), np.array(test_data_pos)


def generate_negative_edges(data, nb_neg_per_data_edge=None, directed=True):
    if directed:
        lhs_ids = np.unique(data[:, 0])
        rhs_ids = np.unique(data[:, 1])
    else:
        ids = np.unique(data[:, :2])
        lhs_ids = ids
        rhs_ids = ids
    nb_lhs_ids = lhs_ids.shape[0]
    nb_rhs_ids = rhs_ids.shape[0]

    if directed:
        total_nb_edges = nb_lhs_ids * nb_rhs_ids
    else:
        total_nb_edges = (nb_lhs_ids * nb_rhs_ids - nb_lhs_ids) / 2
    total_nb_neg = total_nb_edges - data.shape[0]
    nb_negative_samples = int(data.shape[0] * nb_neg_per_data_edge)
    # Calculate how many negative edges we sample, out of the total possible number we could have sampled from.
    neg_corr_factor = total_nb_neg / nb_negative_samples

    # Find linear indexes for the data in a simplified, 0-indexed matrix.
    lhs_id_to_idx = dict(zip(lhs_ids, np.arange(nb_lhs_ids)))
    rhs_id_to_idx = dict(zip(rhs_ids, np.arange(nb_rhs_ids)))
    simplified_data = np.array([(lhs_id_to_idx[edge[0]], rhs_id_to_idx[edge[1]]) for edge in data])
    data_lin_idx = np.ravel_multi_index((simplified_data[:, 0], simplified_data[:, 1]), dims=(nb_lhs_ids, nb_rhs_ids))

    negative_samples = []
    current_nb_negative_samples = 0
    while current_nb_negative_samples < nb_negative_samples:
        # Sample a bunch of edges.
        nb_left_to_sample = nb_negative_samples - current_nb_negative_samples
        lhs_samples = np.random.randint(low=0, high=nb_lhs_ids, size=nb_left_to_sample)
        rhs_samples = np.random.randint(low=0, high=nb_rhs_ids, size=nb_left_to_sample)

        # Check if they are negative by comparing their linear indices.
        candidate_lin_idx = np.ravel_multi_index((lhs_samples, rhs_samples), dims=(nb_lhs_ids, nb_rhs_ids))
        actual_negative_lin_idx = np.setdiff1d(candidate_lin_idx, data_lin_idx)

        # Remove edges of which the mirrored version is in the positive set.
        if not directed:
            lhs_samples, rhs_samples = np.unravel_index(actual_negative_lin_idx, dims=(nb_lhs_ids, nb_rhs_ids))
            reversed_can_lin_idx = np.ravel_multi_index((rhs_samples, lhs_samples), dims=(nb_rhs_ids, nb_lhs_ids))
            actual_negative_lin_idx = np.setdiff1d(reversed_can_lin_idx, data_lin_idx)

        # Avoid self-loops.
        if not directed:
            self_loop_idx = actual_negative_lin_idx % nb_lhs_ids == actual_negative_lin_idx // nb_rhs_ids
            actual_negative_lin_idx = actual_negative_lin_idx[np.logical_not(self_loop_idx)]

        # Keep the actually negative samples.
        actual_negative_samples = np.unravel_index(actual_negative_lin_idx, dims=(nb_lhs_ids, nb_rhs_ids))

        # Before storing them, convert the indices to ids.
        sampled_lhs_ids = lhs_ids[actual_negative_samples[0]]
        sampled_rhs_ids = rhs_ids[actual_negative_samples[1]]
        actual_negative_samples = np.vstack((sampled_lhs_ids, sampled_rhs_ids)).T
        negative_samples.append(actual_negative_samples)
        current_nb_negative_samples += actual_negative_samples.shape[0]

    return np.vstack(negative_samples), neg_corr_factor
