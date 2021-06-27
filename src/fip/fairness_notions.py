import torch
import pandas as pd
import numpy as np


class FairnessNotion:
    def __new__(cls, notion_name, **kwargs):
        if notion_name == "DP":
            return super().__new__(DemographicParity)

        elif notion_name == "EO":
            return super().__new__(EqualizedOpportunity)

        else:
            raise ValueError(f"{notion_name=} unknown.")

    def __init__(self, notion_name='', with_edge_datapoints=False, undirected=False, **_kwargs):
        self._notion_name = notion_name
        self.with_edge_datapoints = with_edge_datapoints
        self.undirected = undirected

        self._attributes = None
        self._uniq_attr_counts = None
        self._uniq_attr_idx = None
        self._is_fitted = False

    def fit(self, attributes):
        """
        Fit the fairness notion to the given attributes.
        :param attributes: pandas Series of attributes, with the index possible values of x.
        """
        self._attributes = attributes
        self._uniq_attr_counts = attributes.value_counts()
        self._uniq_attr_idx = pd.Series(index=self._uniq_attr_counts.index,
                                        data=np.arange(len(self._uniq_attr_counts)))
        self._is_fitted = True

    def stat_func(self, x, y, labels):
        raise NotImplementedError

    def stat_emp(self, h, x, labels):
        raise NotImplementedError

    def nb_constraints(self):
        nb_sens_groups = len(self._uniq_attr_counts)
        if not self.with_edge_datapoints:
            return nb_sens_groups
        else:
            if self.undirected:
                return int(nb_sens_groups * (nb_sens_groups + 1) / 2)
            else:
                return nb_sens_groups ** 2

    def is_fitted(self):
        return self._is_fitted

    @staticmethod
    def loss_correction(labels):
        return 1

    def _x_to_attr_idx(self, x):
        x = x.numpy()
        nb_sens_groups = len(self._uniq_attr_counts)

        # For every data point x, get the index of the sensitive group that it belongs to.
        if not self.with_edge_datapoints:
            attr_idx = self._uniq_attr_idx[self._attributes[x]]
        else:
            src_attr_idx = self._uniq_attr_idx[self._attributes[x[:, 0]]].to_numpy()
            dest_attr_idx = self._uniq_attr_idx[self._attributes[x[:, 1]]].to_numpy()
            if self.undirected:
                sorted_idx = np.sort(np.stack((src_attr_idx, dest_attr_idx), axis=1), axis=1)
                attr_idx = sorted_idx[:, 0] * (nb_sens_groups - sorted_idx[:, 1]) + sorted_idx[:, 1]
            else:
                attr_idx = src_attr_idx * nb_sens_groups + dest_attr_idx
        return torch.from_numpy(attr_idx).long()


class DemographicParity(FairnessNotion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def stat_func(self, x, y, labels):
        size = torch.Size((x.shape[0], self.nb_constraints()))

        if isinstance(y, (int, float)):
            if y == 0:
                return torch.sparse_coo_tensor(torch.empty((2, 0)), torch.empty(0), size)
            elif y == 1:
                y = torch.ones(size[0])

        # Use the sensitive group indexes to insert 'y's at the correct indices.
        attr_idx = self._x_to_attr_idx(x)
        stats = torch.sparse_coo_tensor(torch.stack((torch.arange(size[0]), attr_idx)), y, size)
        return stats

    def stat_emp(self, h, x, labels):
        global_mean = h.mean()

        # For the sensitive groups that are actually among x, compute their counts. The other counts are zero.
        all_attr_group_counts = torch.zeros(self.nb_constraints(), dtype=torch.long)
        attr_idx = self._x_to_attr_idx(x)
        present_attr_groups, present_attr_group_counts = torch.unique(attr_idx, return_counts=True)
        all_attr_group_counts[present_attr_groups] = present_attr_group_counts

        scaled_means = all_attr_group_counts * global_mean
        return scaled_means


class EqualizedOpportunity(DemographicParity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def stat_func(self, x, y, labels):
        size = torch.Size((x.shape[0], self.nb_constraints()))

        labels_ones = labels == 1
        x = x[labels_ones]

        if isinstance(y, (int, float)):
            if y == 0:
                return torch.sparse_coo_tensor(torch.empty((2, 0)), torch.empty(0), size)
            elif y == 1:
                y = torch.ones(x.shape[0])
        else:
            y = y[labels_ones]

        # Use the sensitive group indexes to insert 'y's at the correct indices.
        attr_idx = self._x_to_attr_idx(x)
        stats = torch.sparse_coo_tensor(torch.stack((torch.nonzero(labels_ones).squeeze(), attr_idx)), y, size)
        return stats

    def stat_emp(self, h, x, labels):
        labels_ones = labels == 1
        h = h[labels_ones]
        x = x[labels_ones]
        emps = super().stat_emp(h, x, labels=None)
        return emps

    @staticmethod
    def loss_correction(labels):
        nb_ones = (labels == 1).sum()
        return labels.shape[0] / nb_ones
