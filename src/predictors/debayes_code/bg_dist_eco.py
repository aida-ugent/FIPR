# Author: Maarten Buyl
# Contact: maarten.buyl@ugent.be
# Date: 17/07/2020
# From https://github.com/aida-ugent/DeBayes


import numpy as np
from .bg_dist import BgDist, newton_optimization, Lambdas, RowDegreeLambdas, ColumnDegreeLambdas, LambdasAggregator


class BgDistDegreeEco(BgDist):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._lambdas = None

    def _fit(self, A, **kwargs):
        # Lamdas aggregator keeps track of several 'lambdas' objects.
        lambdas = LambdasAggregatorEco(A.shape)

        # Compute the row sum for the matrix. The expected row sum will have to match the actual sum.
        row_sums = A.sum(axis=1).A.squeeze()

        # Construct Lambdas object for the row degree prior.
        row_lambdas = RowDegreeLambdasEco(row_sums)

        # The col_sums are computed in a similar way.
        col_sums = A.sum(axis=0).A.squeeze()

        # Construct Lambdas object for the column degree prior.
        col_lambdas = ColumnDegreeLambdasEco(col_sums)

        lambdas.add_lambdas_object(row_lambdas)
        lambdas.add_lambdas_object(col_lambdas)
        lambdas.compile()

        print("Using economical lambdas calculation, condensed a " + str(A.shape) + " matrix to " +
              str((row_lambdas.la.shape[0], col_lambdas.la.shape[0])))

        newton_optimization(lambdas, nit=500)

        self._lambdas = lambdas

    def _get_probability(self, row_ids, col_ids):
        return self._lambdas.probability(*self._lambdas.map_idx(row_ids, col_ids), inner_idx=True)

    def _get_lambdas_sum(self, row_ids, col_ids):
        return self._lambdas.lambdas_sum(*self._lambdas.map_idx(row_ids, col_ids), inner_idx=True)

    def _arrange_lambdas(self):
        return self._lambdas.arrange_all_lambdas()

    def string_code(self):
        return "deg"


class LambdasEco(Lambdas):
    """
    General class for Lagrange multipliers or 'lambdas'.
    """
    def __init__(self, constraints):
        # Find unique constraints.
        uni_cs, idx_to_uni, cs_counts = np.unique(constraints, return_inverse=True, return_counts=True)

        super().__init__(uni_cs)
        self.idx_to_uni = idx_to_uni
        self.multip = cs_counts

        self._row_multip = None
        self._col_multip = None

    def set_multiplicators(self, row_multip, col_multip):
        self._row_multip = row_multip
        self._col_multip = col_multip

    def lambdas_sum(self, row_idx=None, col_idx=None, inner_idx=False):
        raise NotImplementedError

    def compute_grad(self, P, E_div_Z_sqr, row_idx):
        raise NotImplementedError


class RowDegreeLambdasEco(RowDegreeLambdas, LambdasEco):
    def __init__(self, row_sums):
        super().__init__(row_sums, col_mask=None)

    def set_multiplicators(self, row_multip, col_multip):
        super().set_multiplicators(row_multip, col_multip)
        self._col_mask = np.ones_like(col_multip, dtype=np.bool)

    def lagrangian_term(self):
        return np.sum(self.la * self.constraints * self._row_multip)

    def compute_grad(self, P, E_div_Z_sqr, _row_idx):
        # Ignore _row_idx, since we do not expect batching for this class.
        self.grad = P.dot(self._col_multip) - self.constraints
        self.grad_second_order = E_div_Z_sqr.dot(self._col_multip)


class ColumnDegreeLambdasEco(ColumnDegreeLambdas, LambdasEco):
    def __init__(self, col_sums):
        super().__init__(col_sums, row_mask=None)

    def set_multiplicators(self, row_multip, col_multip):
        super().set_multiplicators(row_multip, col_multip)
        self._row_mask = np.ones_like(row_multip, dtype=np.bool)

    def lagrangian_term(self):
        return np.sum(self.la * self.constraints * self._col_multip)

    def compute_grad(self, P, E_div_Z_sqr, _row_idx):
        self.grad = P.T.dot(self._row_multip) - self.constraints
        self.grad_second_order = E_div_Z_sqr.T.dot(self._row_multip)


class LambdasAggregatorEco(LambdasAggregator):
    """
    Perform aggregation operations on the lambdas objects. It is assumed that every element follows an independent
    Bernoulli distribution.
    """

    def __init__(self, A_shape, _batch_size=None):
        super().__init__(A_shape, batch_size=None)

        self._row_multip = None
        self._col_multip = None

        self._row_idx_to_uni = None
        self._col_idx_to_uni = None

    def compile(self):
        self._row_multip = self._lambdas_list[0].multip
        self._col_multip = self._lambdas_list[1].multip
        for lambdas in self._lambdas_list:
            lambdas.set_multiplicators(self._row_multip, self._col_multip)

        self._row_idx_to_uni = self._lambdas_list[0].idx_to_uni
        self._col_idx_to_uni = self._lambdas_list[1].idx_to_uni

    def forward(self, with_grad=False):
        if with_grad:
            if not self._grad_is_zero:
                raise ValueError("Call 'zero_grad()' before calling forward with 'with_grad=True'!")
            else:
                self._grad_is_zero = False

        E = np.exp(self.lambdas_sum())
        Z = self._Z(E)

        lagrangian = self._row_multip.dot(np.log(Z)).dot(self._col_multip)
        for lambdas in self._lambdas_list:
            lagrangian -= lambdas.lagrangian_term()

        if with_grad:
            P = E / Z
            E_div_Z_sqr = P / Z
            self.grad(P, E_div_Z_sqr)
            self.prepare_delta_la()

        return lagrangian

    def map_idx(self, row_idx, col_idx):
        mapped_row_idx = self._row_idx_to_uni[row_idx]
        mapped_col_idx = self._col_idx_to_uni[col_idx]
        return mapped_row_idx, mapped_col_idx

    def arrange_all_lambdas(self):
        all_lambda_vals = []
        for lambdas in self._lambdas_list:
            for i, uni_idx in enumerate(lambdas.idx_to_uni):
                all_lambda_vals.append(lambdas.la[uni_idx])
        return np.array(all_lambda_vals)
