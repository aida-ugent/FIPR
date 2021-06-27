from sklearn.metrics import roc_auc_score


class MetricAbstract:
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def metric_name(self):
        return self.__class__.__name__


class AUC(MetricAbstract):
    def __call__(self, predictions, labels):
        return roc_auc_score(labels, predictions)


class Metrics:
    all_metrics = [AUC]

    def __init__(self, **kwargs):
        self._metrics = [metric_class(**kwargs) for metric_class in self.all_metrics]

    def __call__(self, *args):
        res = []
        for metric in self._metrics:
            res.append((metric.metric_name(), metric(*args)))
        return res
