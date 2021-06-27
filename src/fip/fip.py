import torch
from torch import nn


class FairIProjection(nn.Module):
    def __init__(self,
                 fairness_notion,
                 fip_sample_size=int(1e5),
                 max_iters=100):
        """
        Initialize the I-projection model.
        :param fairness_notion: a FairnessNotion instance.
        :param fip_sample_size: when the forward() function is called on a batch, take a random sample of the batch
        instead. The fip_sample_size (int) corresponds with the number of samples that are used.
        :param max_iters: max number of epochs to run LBFGS.
        """
        super().__init__()
        self.fn = fairness_notion
        self.fip_sample_size = fip_sample_size
        self.max_iters = max_iters

        self._la = None
        self._optimizer = None

    def fit(self, h, x, labels):
        """
        Fit the I-projection.
        :param h: the h(y = 1 | x) values of the model that should be projected.
        :param x: the indices or data points that h attempts to predict things for.
        :param labels: the binary labels for data points x.
        """
        if self._la is None:
            self._initialize()
        self._optimizer = torch.optim.LBFGS(self._la.parameters(), lr=1, max_iter=self.max_iters, history_size=10,
                                            line_search_fn='strong_wolfe')

        # Subsample the data if necessary.
        # ALSO: if this random sample does not contain one of every sensitive group, then that lambda wont be trained!!!
        if self.fip_sample_size is not None:
            if x.shape[0] > self.fip_sample_size:
                rand_idx = torch.randperm(x.shape[0])
                x = x[rand_idx[:self.fip_sample_size]]
                h = h[rand_idx[:self.fip_sample_size]]
                labels = labels[rand_idx[:self.fip_sample_size]]
        h = h.clamp(1e-6, 1 - 1e-6)

        # There is currently no batching, the entire dataset is used every iteration.
        def closure():
            if torch.is_grad_enabled():
                self._optimizer.zero_grad()
            loss = self._loss(h, x, labels)
            if loss.requires_grad:
                loss.backward()
            return loss.item()

        # LBFGS already finds the `optimal' parameters, so there should only be one update.
        self._optimizer.step(closure)

    def forward(self, h, x, labels):
        nom = h * torch.exp(self._la(self.fn.stat_func(x, y=1, labels=labels)))
        denom = self._compute_Z(h, x, labels)
        probs = nom / denom
        return probs

    def detached_loss(self, h, x, labels):
        Z = self._compute_Z(h, x, labels, with_la_grad=False)

        # Note this is the loss that should be minimised ACCORDING TO THE FIP. h should instead maximize this loss.
        with torch.no_grad():
            constants_term = self._la(self.fn.stat_emp(h, x, labels)) / x.shape[0]
        loss = torch.mean(torch.log(Z)) - constants_term
        return loss

    def _compute_Z(self, h, x, labels, with_la_grad=True):
        if with_la_grad:
            lambda_sum_y_0 = torch.exp(self._la(self.fn.stat_func(x, y=0, labels=labels)))
            lambda_sum_y_1 = torch.exp(self._la(self.fn.stat_func(x, y=1, labels=labels)))
        else:
            with torch.no_grad():
                lambda_sum_y_0 = torch.exp(self._la(self.fn.stat_func(x, y=0, labels=labels)))
                lambda_sum_y_1 = torch.exp(self._la(self.fn.stat_func(x, y=1, labels=labels)))
        Z = (1 - h) * lambda_sum_y_0 + h * lambda_sum_y_1
        return Z

    def _initialize(self):
        self._la = Lambdas(self.fn.nb_constraints())

    def _loss(self, h, x, labels):
        Z = self._compute_Z(h, x, labels)

        # Note this is the loss that should be minimised.
        loss = torch.sum(torch.log(Z)) - self._la(self.fn.stat_emp(h, x, labels))
        return loss

    def get_lambdas(self):
        return self._la.la.weight.data.detach().squeeze().numpy()


class Lambdas(torch.nn.Module):
    def __init__(self, nb):
        super().__init__()
        self.la = torch.nn.Linear(in_features=nb, out_features=1, bias=False).float()

    def forward(self, val):
        if val.is_sparse and val.coalesce().values().shape[0] == 0:
            return torch.zeros(1, dtype=torch.float)
        val = val.float()
        return torch.squeeze(self.la(val))
