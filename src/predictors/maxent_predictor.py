import numpy as np
import pandas as pd
import torch
import torch.utils.data as torch_data
from tqdm import tqdm

from fip.fairness_loss import FairnessLoss
from .predictor import Predictor
from utils.adjacency_data import build_adjacency_matrix, AdjacencySampler, map_edges


class MaxEntPredictor(Predictor):
    def __init__(self,
                 fip_strength=0,
                 fip_type='',
                 nb_epochs=100,
                 learning_rate=None,
                 batch_size=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.fip_strength = fip_strength
        self.fip_type = fip_type
        self.nb_epochs = nb_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self._lambdas = None

    def fit(self, train_data, attributes, **kwargs):
        if self.batch_size is not None:
            raise ValueError("batch_size was not None, but batching does not work properly for fip loss in LBFGS!")

        try:
            partition_mask = attributes['partition']
            attributes = attributes.drop(columns=['partition'])
        except KeyError:
            partition_mask = None

        # Use the predictor's parameters as parameters for the lambdas.
        lambdas_kwargs = self.get_params()

        if partition_mask is not None:
            self._lambdas = LambdasPerPartPair(partition_mask=partition_mask, **lambdas_kwargs)
        else:
            self._lambdas = Lambdas(**lambdas_kwargs)
        self._lambdas.fit(train_data, attributes)

    def get_embeddings(self, ids):
        return np.ones((ids.shape[0], 1), dtype=np.float)

    def predict(self, edges):
        return self._lambdas(edges).detach().numpy()

    def predict_logits(self, edges):
        return self._lambdas(edges, as_logits=True)


class Lambdas(torch.nn.Module):
    def __init__(self,
                 fip_strength=1,
                 fip_type='',
                 nb_epochs=100,
                 learning_rate=1e-2,
                 batch_size=10 ** 7
                 ):
        super().__init__()

        self.fip_strength = fip_strength
        self.fip_type = fip_type
        self.nb_epochs = nb_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self._id_to_idx = None
        self._F_matrix = None
        self._la = None

    def fit(self, train_edges, attributes):
        self._id_to_idx = pd.Series(index=attributes.index, data=np.arange(len(attributes)))
        train_edges = map_edges(train_edges, self._id_to_idx)
        A = build_adjacency_matrix(train_edges, nb_nodes=len(attributes))
        A_sampler = AdjacencySampler(A, batch_nb_rows=None)
        data_loader = torch_data.DataLoader(dataset=A_sampler, num_workers=0, batch_size=None)

        # Construct FeatureMatrix object that returns a sparse tensor of features for the datapoints.
        nb_rows, nb_cols = A.shape
        self._F_matrix = FeatureMatrix(nb_rows=nb_rows, nb_cols=nb_cols)
        nb_features = self._F_matrix.get_nb_features()

        self._la = torch.nn.Linear(in_features=nb_features, out_features=1, bias=False).float()

        optimizer = torch.optim.LBFGS(self._la.parameters(), lr=1, max_iter=self.nb_epochs, history_size=50,
                                      line_search_fn='strong_wolfe')
        # optimizer = torch.optim.Adam(self._la.parameters(), lr=1e-3)
        pred_loss_f = torch.nn.BCEWithLogitsLoss()

        if self.fip_strength > 0:
            fairness_loss_f = FairnessLoss(self.fip_type)

        print("Learning MaxEnt distribution...")
        with tqdm() as pbar:
            # Important: I could not get LBFGS to work properly with minibatching. Therefore, I aggregate the losses
            # of the entire dataset before backpropagation. Note that if the length of the dataset is not a multiple of
            # the batch size, our last batch will be smaller. This is accounted for in this implementation.
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                bce_loss = torch.zeros(1, dtype=torch.float)
                fairness_loss = torch.zeros(1, dtype=torch.float)
                for batch_data in data_loader:
                    edges, labels = batch_data
                    logits = self(edges, as_logits=True, idx_in_known_format=True)
                    bce_loss += pred_loss_f(logits, labels.float()) * (edges.shape[0] / len(data_loader))

                    if self.fip_strength > 0:
                        fairness_loss += fairness_loss_f(torch.sigmoid(logits), edges, labels, attributes) \
                                         * (edges.shape[0] / len(data_loader))

                total_loss = (bce_loss + self.fip_strength * fairness_loss)
                if total_loss.requires_grad:
                    total_loss.backward()
                loss_val = total_loss.item()

                progress_text = f"BCE loss: {bce_loss.item():.8f}"
                if self.fip_strength > 0:
                    progress_text += f", fip loss: {fairness_loss.item() * self.fip_strength:.8f}"
                pbar.set_description(progress_text)
                pbar.update()

                self._last_loss_vals = bce_loss.item(), self.fip_strength * fairness_loss.item()
                return loss_val

            # LBFGS already finds the `optimal' parameters, so there should only be one update.
            optimizer.step(closure)

    def forward(self, edges, as_logits=False, idx_in_known_format=False):
        if self._la is None:
            raise ValueError("MaxEnt lambdas not fitted!")
        if not idx_in_known_format:
            if isinstance(edges, torch.Tensor):
                edges = edges.numpy()
            edges = torch.from_numpy(map_edges(edges, self._id_to_idx))

        features = self._F_matrix.features_of_row_col(edges)
        lambdas_sum = torch.squeeze(self._la(features))
        if as_logits:
            return lambdas_sum

        probs = torch.sigmoid(lambdas_sum)
        return probs


class ZerosLambdas(Lambdas):
    def fit(self, train_edges, attributes):
        pass

    def forward(self, edges, as_logits=False, idx_in_known_format=False):
        if as_logits:
            return torch.ones(edges.shape[0]).float() * (-np.inf)
        return torch.zeros(edges.shape[0]).float()


class LambdasPerPartPair(torch.nn.Module):
    def __init__(self,
                 partition_mask,
                 undirected=True,
                 **kwargs):
        super().__init__()

        self._undirected = undirected
        self._lambdas_params = kwargs

        partitions = np.unique(partition_mask)
        nb_partitions = partitions.shape[0]
        if not np.all(partitions == np.arange(nb_partitions)):
            raise ValueError("Block mask did not contain 0-indexed ordinal values!")
        self._partition_mask = partition_mask
        self._lambdas_per_partpair = np.empty((nb_partitions, nb_partitions), dtype=np.object)

    def fit(self, train_edges, attributes):
        partpairs = map_edges(train_edges, self._partition_mask)
        nb_partitions = self._lambdas_per_partpair.shape[0]
        for src_part in range(nb_partitions):
            if self._undirected:
                first_dst_part = src_part
            else:
                first_dst_part = 0

            src_part_matches = partpairs[:, 0] == src_part
            for dst_part in range(first_dst_part, nb_partitions):
                dst_part_matches = partpairs[:, 1] == dst_part
                partpair_match = np.logical_and(src_part_matches, dst_part_matches)
                partpair_edges = train_edges[partpair_match]

                if partpair_edges.shape[0] > 0:
                    partpair_lambdas = Lambdas(**self._lambdas_params)
                    partpair_attrs = attributes[self._partition_mask.isin([src_part, dst_part])]
                    partpair_lambdas.fit(partpair_edges, partpair_attrs)
                else:
                    partpair_lambdas = ZerosLambdas()
                self._lambdas_per_partpair[src_part, dst_part] = partpair_lambdas

    def forward(self, edges, as_logits=False):
        if not isinstance(self._lambdas_per_partpair[0, 0], Lambdas):
            raise ValueError("MaxEnt lambdas not fitted!")
        if isinstance(edges, torch.Tensor):
            edges = edges.numpy()

        partpairs = map_edges(edges, self._partition_mask)
        uniq_partpairs = np.unique(partpairs, axis=0)

        predictions = torch.empty(edges.shape[0], dtype=torch.float)
        for partpair in uniq_partpairs:
            partpair_match = (partpairs == partpair).all(axis=1)
            partpair_edges = edges[partpair_match]

            # If needed, use the symmetrical lambdas.
            src_part, dst_part = partpair[0], partpair[1]
            if self._undirected and src_part > dst_part:
                src_part, dst_part = dst_part, src_part
                partpair_edges[:, [1, 0]] = partpair_edges[:, [0, 1]]

            partpair_lambdas = self._lambdas_per_partpair[src_part, dst_part]
            predictions[partpair_match] = partpair_lambdas(partpair_edges, as_logits=as_logits)
        return predictions


class FeatureMatrix:
    def __init__(self,
                 nb_rows=None,
                 nb_cols=None):
        self._features = []
        if nb_rows is not None:
            self._features.append(EdgeSource(dim=nb_rows))
        if nb_cols is not None:
            self._features.append(EdgeDest(dim=nb_cols))

        # Precompute the number of features.
        self._nb_features = 0
        for feature in self._features:
            self._nb_features += feature.nb_features()

    def features_of_row_col(self, edges):
        nb_fm_rows = edges.shape[0]

        all_fm_row_coords = []
        all_fm_col_coords = []
        all_fm_data = []

        feature_pointer = 0
        for feature in self._features:
            fm_row_coords, fm_col_coords, fm_data = feature.feature_matrix(feature_pointer, edges)
            all_fm_row_coords.append(fm_row_coords)
            all_fm_col_coords.append(fm_col_coords)
            all_fm_data.append(fm_data)
            feature_pointer += feature.nb_features()

        all_fm_row_coords = np.concatenate(all_fm_row_coords)
        all_fm_col_coords = np.concatenate(all_fm_col_coords)
        all_fm_data = np.concatenate(all_fm_data)

        indices = torch.from_numpy(np.vstack((all_fm_row_coords, all_fm_col_coords)))
        values = torch.from_numpy(all_fm_data)
        feature_matrix = torch.sparse_coo_tensor(indices, values, size=(nb_fm_rows, self._nb_features),
                                                 dtype=torch.float)
        return feature_matrix

    def get_nb_features(self):
        return self._nb_features


class Feature:
    def __init__(self, dim=None):
        self._dim = dim

    def feature_matrix(self, feature_pointer, edges):
        raise NotImplementedError

    def nb_features(self):
        raise NotImplementedError


# Corresponds with row constraint.
class EdgeSource(Feature):
    def feature_matrix(self, feature_pointer, edges):
        nb_fm_rows = edges.shape[0]
        fm_row_coords = np.arange(nb_fm_rows, dtype=np.int)
        fm_col_coords = edges[:, 0] + feature_pointer
        fm_data = np.ones(nb_fm_rows, dtype=np.float)
        return fm_row_coords, fm_col_coords, fm_data

    def nb_features(self):
        return self._dim


# Corresponds with col constraint.
class EdgeDest(Feature):
    def feature_matrix(self, feature_pointer, edges):
        nb_fm_rows = edges.shape[0]
        fm_row_coords = np.arange(nb_fm_rows, dtype=np.int)
        fm_col_coords = edges[:, 1] + feature_pointer
        fm_data = np.ones(nb_fm_rows, dtype=np.float)
        return fm_row_coords, fm_col_coords, fm_data

    def nb_features(self):
        return self._dim
