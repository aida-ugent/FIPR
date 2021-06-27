import numpy as np
import torch

from utils.adjacency_data import build_adjacency_matrix
from .debayes_code.bg_dist import BgDistBuilder
from .dot_product_decoder import DotProductDecoder


class ConditionalNetworkEmbeddingPredictor(DotProductDecoder):
    def __init__(self,
                 fip_strength=0,
                 fip_type='',
                 prior_dist=None,
                 s1=1,
                 s2=16,
                 subsample_neg=100,
                 nb_epochs=1,
                 learning_rate=1e-3,
                 batch_size=10000,
                 dimension=8,
                 **kwargs):
        super().__init__(**kwargs)

        self._prior_dist = prior_dist
        self.fip_strength = fip_strength
        self.fip_type = fip_type
        self.s1 = s1
        self.s2 = s2
        self.subsample_neg = subsample_neg
        self.nb_epochs = nb_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dimension = dimension

        self._device = None
        self._optimizer = None
        self._loss_fn = None

    def fit(self, train_data, attributes, **_kwargs):
        if self._prior_dist is None:
            prior_dist = BgDistBuilder().build('degree')
            A = build_adjacency_matrix(train_data, nb_nodes=len(attributes))
            try:
                block_mask = attributes['partition']
            except KeyError:
                block_mask = None
            prior_dist.fit(A, block_mask)
            self._prior_dist = prior_dist

        super().fit(train_data, attributes)

    def forward(self, edges, as_logits=False):
        with torch.no_grad():
            lambdas_sum = self._prior_dist.predict_logits(edges)
            if isinstance(lambdas_sum, np.ndarray):
                lambdas_sum = torch.from_numpy(lambdas_sum)

        src_embs = self._encoder(edges[:, 0])
        dest_embs = self._encoder(edges[:, 1])

        distances = torch.sum((src_embs - dest_embs) ** 2, dim=1)
        outputs = distances * (1. / self.s2 ** 2 - 1. / self.s1 ** 2) + np.log(self.s2 / self.s1) + lambdas_sum
        if as_logits:
            return outputs
        probs = torch.sigmoid(outputs)
        return probs

    def _init_encoder(self, train_data, attributes):
        return torch.nn.Embedding(len(attributes), self.dimension)
