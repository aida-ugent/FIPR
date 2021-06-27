import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import IterableDataset


def build_adjacency_matrix(edges, nb_nodes=None):
    if nb_nodes is None:
        nb_nodes = np.max(edges) + 1
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    data = np.ones(rows.shape[0], dtype=np.int)
    A = sp.csr_matrix((data, (rows, cols)), shape=(nb_nodes, nb_nodes))

    assert(A.data.max() == 1)
    return A

def map_edges(edges, nodes_map):
    edge_src = nodes_map[edges[:, 0]].to_numpy()
    edge_dst = nodes_map[edges[:, 1]].to_numpy()
    edges = np.stack((edge_src, edge_dst), axis=1)
    return edges


class AdjacencySampler(IterableDataset):
    def __init__(self, A, subsample_neg_cols=None, batch_nb_rows=None, row_idx=None, col_idx=None):
        # Try to detect whether A corresponds with a full adjacency matrix (where we should avoid self-loops).
        self._avoid_self_loops = (A.shape[0] == A.shape[1]) and np.all(A.diagonal() == 0)

        self._A = A
        if row_idx is None:
            self._possible_row_idx = np.arange(A.shape[0])
        else:
            self._possible_row_idx = row_idx

        if col_idx is None:
            self._possible_col_idx = np.arange(A.shape[1])
        else:
            self._possible_col_idx = col_idx

        self._n = self._possible_row_idx.shape[0]
        self._m = self._possible_col_idx.shape[0]
        # self._n, self._m = A.shape
        # if self._avoid_self_loops:
        #     self._m -= 1
        self._density = A.nnz / (self._n * self._m)
        self._batch_nb_rows = batch_nb_rows

        if subsample_neg_cols is not None and self._m >= subsample_neg_cols:
            self._subsample_neg_cols = subsample_neg_cols
        else:
            self._subsample_neg_cols = None

        total_len = len(self)
        if total_len > 10**8:
            raise MemoryError(f"The adjacency matrix is sampled from in batches with a total size of {total_len}."
                              f"The current implementation cannot support this size.")

        self._idx_pointer = None

    def __iter__(self):
        self._idx_pointer = 0
        return self

    def __getitem__(self, item):
        raise NotImplementedError

    def __next__(self):
        if self._idx_pointer < self._n:
            if self._batch_nb_rows is None:
                nb_rows = self._n
            else:
                nb_rows = min(self._batch_nb_rows, self._n - self._idx_pointer)
            samples = self._generate_samples(self._idx_pointer, nb_rows)
            self._idx_pointer += nb_rows
            return samples
        else:
            raise StopIteration

    def __len__(self):
        if self._batch_nb_rows is not None:
            rows_per_batch = self._batch_nb_rows
        else:
            rows_per_batch = self._n
        if self._subsample_neg_cols is not None:
            cols_per_batch = self._subsample_neg_cols + self._density * self._m
        else:
            cols_per_batch = self._m
        return int(rows_per_batch * cols_per_batch)

    def _generate_samples(self, start_idx, nb_rows):
        # Gather the random indices for this batch.
        if self._subsample_neg_cols is None:
            if not self._avoid_self_loops:
                row_idx = np.repeat(self._possible_row_idx[start_idx: start_idx + nb_rows], self._m)
                col_idx = np.tile(self._possible_col_idx, nb_rows)
            else:
                row_idx = np.repeat(self._possible_row_idx[start_idx: start_idx + nb_rows], self._m - 1)
                col_idx = np.tile(self._possible_col_idx[:-1], nb_rows)
                # For indices that are in the upper triangle of A, add 1 to account for the lack of self-loops.
                where_beyond_diag = col_idx >= row_idx
                col_idx[where_beyond_diag] += 1
            ground_truth = torch.from_numpy(np.squeeze(self._A[row_idx, col_idx].A)).float()
        else:
            sub_A = self._A[self._possible_row_idx[start_idx: start_idx + nb_rows]][:, self._possible_col_idx]
            pos_vals = sub_A.nonzero()
            pos_row_idx = self._possible_row_idx[pos_vals[0] + start_idx]
            pos_col_idx = self._possible_col_idx[pos_vals[1]]

            random_row_idx = np.repeat(self._possible_row_idx[start_idx: start_idx + nb_rows], self._subsample_neg_cols)
            random_col_idx = np.random.choice(self._possible_col_idx, (nb_rows * self._subsample_neg_cols))
            random_val_labels = self._A[random_row_idx, random_col_idx].A.squeeze()

            row_idx = np.concatenate([pos_row_idx, random_row_idx])
            col_idx = np.concatenate([pos_col_idx, random_col_idx])
            ground_truth = np.concatenate([np.ones(pos_row_idx.shape[0]), random_val_labels])
            ground_truth = torch.from_numpy(ground_truth).float()
        edges = torch.from_numpy(np.stack((row_idx, col_idx), axis=1)).long()
        return edges, ground_truth
