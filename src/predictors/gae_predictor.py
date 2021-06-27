# Adapted from https://github.com/zfjsail/gae-pytorch.


import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from predictors.dot_product_decoder import DotProductDecoder
from utils.adjacency_data import build_adjacency_matrix


class GAEPredictor(DotProductDecoder):
    def __init__(self,
                 fip_strength=1,
                 fip_type='',
                 fip_sample_size=1e5,
                 subsample_neg=None,
                 nb_epochs=1,
                 learning_rate=None,
                 batch_size=None,
                 dimension=1,
                 nb_layers=3,
                 dropout_pct=0,
                 **kwargs):
        super().__init__(**kwargs)

        self.fip_strength = fip_strength
        self.fip_type = fip_type
        self.fip_sample_size = fip_sample_size
        self.subsample_neg = subsample_neg
        self.nb_epochs = nb_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dimension = dimension
        self.nb_layers = nb_layers
        self.dropout_pct = dropout_pct

    def fit(self, train_data, attributes, **kwargs):
        super().fit(train_data, attributes, **kwargs)

    def forward(self, edges, as_logits=False):
        edges = edges.to(self._device)
        embs = self._encoder()
        src_embs = embs[edges[:, 0]]
        dest_embs = embs[edges[:, 1]]

        prods = (src_embs * dest_embs).sum(axis=1)
        if as_logits:
            return prods

        probs = torch.sigmoid(prods)
        return probs

    def _init_encoder(self, train_data, attributes):
        A = build_adjacency_matrix(train_data, nb_nodes=len(attributes))
        A = A.tocoo()
        A = A + sp.eye(A.shape[0])
        degrees = np.squeeze(A.sum(1).A)
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5))
        A_norm = degree_mat_inv_sqrt.dot(A).dot(degree_mat_inv_sqrt).tocoo()
        # A_norm_upper = sp.triu(A_norm)

        indices = torch.from_numpy(np.vstack((A_norm.row, A_norm.col))).long()
        values = torch.from_numpy(A_norm.data)
        shape = torch.Size(A_norm.shape)
        A = torch.sparse_coo_tensor(indices, values, size=shape, dtype=torch.float).to(self._device)
        if len(attributes.columns) > 0:
            feature_matrix = torch.from_numpy(attributes.values).to(self._device)
        else:
            feature_matrix = None
        return GraphAutoEncoder(A, node_features=feature_matrix, dimension=self.dimension,
                                dropout_pct=self.dropout_pct, nb_layers=self.nb_layers)


class GraphAutoEncoder(torch.nn.Module):
    def __init__(self, A, node_features=None, dimension=128, nb_layers=3, dropout_pct=0):
        super().__init__()

        self._A = A
        self._node_features = node_features
        if node_features is not None:
            input_shape = self._node_features.shape[1]
        else:
            input_shape = A.shape[1]
        self._dimension = dimension
        self._dropout_pct = dropout_pct

        weight_matrices = []
        for layer_i in range(nb_layers):
            if layer_i == 0:
                W = torch.nn.Linear(input_shape, self._dimension, bias=True)
            else:
                W = torch.nn.Linear(self._dimension, self._dimension, bias=True)
            torch.nn.init.xavier_uniform_(W.weight)
            weight_matrices.append(W)
        self.W = torch.nn.ModuleList(weight_matrices)

    def forward(self, node_idx=None):
        if self._node_features is not None:
            initial_input = self._A.mm(self._node_features)
        else:
            initial_input = self._A
        output = initial_input.mm(self.W[0].weight.T)

        for layer_i in range(1, len(self.W)):
            output = F.relu(output)
            output = F.dropout(output, self._dropout_pct, training=self.training)
            output = self._A.mm(output).mm(self.W[layer_i].weight.T)

        if node_idx is not None:
            return output[node_idx]
        return output
