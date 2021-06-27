import torch.nn as nn
from .fip import FairIProjection
from .fairness_notions import FairnessNotion


class FairnessLoss(nn.Module):
    def __init__(self,
                 fip_type='',
                 **kwargs):
        super().__init__()

        self._fairness_notion = FairnessNotion(notion_name=fip_type, with_edge_datapoints=True)
        self._fip = FairIProjection(self._fairness_notion, **kwargs)

    def forward(self, h, x, labels, attributes=None):
        if not self._fairness_notion.is_fitted():
            assert attributes is not None
            self._initialize(attributes)

        # Learn the I-Projection without backprop to h.
        self._fip.fit(h.detach(), x, labels=labels)

        # Faster way to compute loss function.
        loss = -self._fip.detached_loss(h, x, labels=labels)

        # Some fairness notions, like EO, only affect a portion of the data points. Scale by the correction factor that
        # prevents this loss from being larger in general.
        loss *= self._fairness_notion.loss_correction(labels)
        return loss

    def _initialize(self, attributes):
        attributes = attributes.fillna("N/A")
        self._fairness_notion.fit(attributes)
