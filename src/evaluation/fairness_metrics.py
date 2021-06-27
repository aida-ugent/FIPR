import numpy as np
from sklearn.metrics import roc_auc_score

from evaluation.metrics import Metrics, MetricAbstract


class FairnessMetricAbstract(MetricAbstract):
    def __init__(self, attr_name, **kwargs):
        super().__init__(**kwargs)
        self.__attr_name = attr_name

    def __call__(self, predictions, labels, attributes):
        raise NotImplementedError

    def metric_name(self):
        return self.__class__.__name__ + "_" + self.__attr_name

    @staticmethod
    def _iter_over_groups(attributes, group_iter_fn):
        vals = []
        for group, group_attrs in attributes.groupby(attributes):
            group_idx = group_attrs.index
            val = group_iter_fn(group_idx)
            vals.append(val)
        return np.array(vals)


class DemographicParity(FairnessMetricAbstract):
    def __init__(self, neg_corr_factor=1, **kwargs):
        super().__init__(**kwargs)
        self.neg_corr_factor = neg_corr_factor

    def __call__(self, predictions, labels, attributes):
        def mean(group_idx):
            group_scores = predictions[group_idx]
            if labels is not None:
                group_labels = labels[group_idx]
                nb_pos = group_labels.sum()
                nb_neg = group_labels.shape[0] - nb_pos
                group_size = nb_pos + nb_neg * self.neg_corr_factor
                group_scores[np.logical_not(group_labels)] *= self.neg_corr_factor
            else:
                group_size = group_scores.shape[0]
            group_sum = np.sum(group_scores)
            return group_sum / group_size
        means = self._iter_over_groups(attributes, mean)

        print(f"{self.__class__.__name__} means: {means}.")
        abs_differences = np.abs(means[:, np.newaxis] - means[np.newaxis, :])
        max_diff = np.max(abs_differences)
        return max_diff


class EqualizedOpportunity(DemographicParity):
    def __call__(self, predictions, labels, attributes):
        predictions = predictions[labels]
        attributes = attributes[labels].reset_index(drop=True)
        max_diff = super().__call__(predictions, labels=None, attributes=attributes)
        return max_diff


class RelativeDP(DemographicParity):
    def __call__(self, predictions, labels, attributes):
        def ovr_auc(group_idx):
            ovr_labels = np.zeros_like(predictions, dtype=np.int)
            ovr_labels[group_idx] = 1
            return roc_auc_score(ovr_labels, predictions)
        aucs = self._iter_over_groups(attributes, ovr_auc)

        print(f"{self.__class__.__name__} scores: {aucs}.")
        max_auc = np.max(aucs)
        return max_auc


class FairnessMetrics(Metrics):
    all_metrics = [DemographicParity, EqualizedOpportunity, RelativeDP]
