import numpy as np
import torch as torch
# noinspection PyProtectedMember
from torch.utils.data import DataLoader
from tqdm import tqdm

from fip.fairness_loss import FairnessLoss
from .predictor import Predictor
from utils.adjacency_data import AdjacencySampler, build_adjacency_matrix


class DotProductDecoder(Predictor, torch.nn.Module):
    def __init__(self,
                 fip_strength=1,
                 fip_type='',
                 fip_sample_size=int(1e5),
                 subsample_neg=100,
                 nb_epochs=1,
                 learning_rate=None,
                 batch_size=None,
                 dimension=1,
                 device_name='cpu',
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
        self.device_name = device_name

        self._encoder = None
        self._device = None

    def fit(self, train_data, attributes, sens_attr_name=None, **_kwargs):
        self._device = torch.device(self.device_name)

        if sens_attr_name is None:
            sens_attr_name = attributes.drop(columns=['partition'], errors='ignore').columns[0]
        unsens_attributes = attributes.drop(columns=[sens_attr_name])
        sens_attributes = attributes[sens_attr_name]

        self._encoder = self._init_encoder(train_data, unsens_attributes).to(self._device)
        self._encoder.train()
        data_loader = self._build_dataloader(train_data, unsens_attributes)
        optimizer = torch.optim.Adam(self._encoder.parameters(), lr=self.learning_rate)
        pred_loss_f = torch.nn.BCEWithLogitsLoss()
        if self.fip_strength > 0:
            fairness_loss_f = FairnessLoss(self.fip_type, fip_sample_size=self.fip_sample_size)

        with tqdm(total=self.nb_epochs) as pbar:
            for epoch in range(self.nb_epochs):
                epoch_bce_loss = 0
                epoch_fip_loss = 0
                for data, labels in data_loader:
                    data = data.to(self._device)
                    labels = labels.to(self._device)

                    optimizer.zero_grad()
                    logits = self.forward(data, as_logits=True)

                    # When any logit is positive or negative infinity, the data point.
                    logits_inf = torch.isinf(logits)
                    if torch.any(logits_inf):
                        logits = logits[~logits_inf]
                        data = data[~logits_inf]
                        labels = labels[~logits_inf]

                    bce_loss = pred_loss_f(logits, labels)
                    epoch_bce_loss += bce_loss.item() * (data.shape[0] / len(data_loader.dataset))

                    if self.fip_strength > 0:
                        fairness_loss = fairness_loss_f(torch.sigmoid(logits), data, labels, sens_attributes)
                        epoch_fip_loss += fairness_loss.item() * (data.shape[0] / len(data_loader.dataset))
                        loss = bce_loss + self.fip_strength * fairness_loss
                    else:
                        loss = bce_loss
                    loss.backward()
                    optimizer.step()

                progress_text = f"BCE loss: {epoch_bce_loss:.8f}"
                if self.fip_strength > 0:
                    progress_text += f", fip loss: {epoch_fip_loss * self.fip_strength:.8f}"
                pbar.set_description(progress_text)
                pbar.update()
            self._last_loss_vals = epoch_bce_loss, self.fip_strength * epoch_fip_loss

    def get_embeddings(self, idx):
        self._encoder.eval()
        idx = torch.from_numpy(idx).long()
        with torch.no_grad():
            return self._encoder(idx).cpu().numpy()

    def forward(self, edges, as_logits=False):
        edges = edges.to(self._device)
        src_embs = self._encoder(edges[:, 0])
        dest_embs = self._encoder(edges[:, 1])

        prods = (src_embs * dest_embs).sum(axis=1)
        if as_logits:
            return prods

        probs = torch.sigmoid(prods)
        return probs

    def predict(self, edges):
        self._encoder.eval()
        edges = torch.from_numpy(edges).long()
        with torch.no_grad():
            probs = self.forward(edges).detach().cpu().numpy()
        return probs

    def _build_dataloader(self, train_data, attributes):
        A = build_adjacency_matrix(train_data, nb_nodes=len(attributes))
        try:
            lhs_idx = np.where(attributes['partition'] == 0)[0]
            rhs_idx = np.where(attributes['partition'] == 1)[0]
        except KeyError:
            lhs_idx = None
            rhs_idx = None

        A_matrix = AdjacencySampler(A, subsample_neg_cols=self.subsample_neg, batch_nb_rows=self.batch_size,
                                    row_idx=lhs_idx, col_idx=rhs_idx)
        data_loader = torch.utils.data.DataLoader(dataset=A_matrix, num_workers=0, batch_size=None)
        return data_loader

    def _init_encoder(self, train_data, attributes):
        n = len(attributes)
        return torch.nn.Embedding(num_embeddings=n, embedding_dim=self.dimension, max_norm=1, norm_type=2)
