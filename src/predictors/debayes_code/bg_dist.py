# Author: Maarten Buyl
# Contact: maarten.buyl@ugent.be
# Date: 17/07/2020
# From https://github.com/aida-ugent/DeBayes


import time
import numpy as np
from tqdm import tqdm


class BgDistBuilder:
    @staticmethod
    def build(prior_type):
        """
        :param prior_type: prior type.
        :return: the desired background distribution object (unfitted).
        """
        if prior_type == 'degree':
            from .bg_dist_eco import BgDistDegreeEco
            return BgDistDegreeEco()

        if prior_type == 'biased_degree':
            return BgDistBiasedDegree()

        raise ValueError('Prior {:s} is not implemented'.format(prior_type))


class BgDist:
    def __init__(self, **kwargs):
        self.__nb_nodes = None
        self.__block_mask = None
        self.__dist_per_block = None
        self.__row_id_to_block_idx = None
        self.__col_id_to_block_idx = None
        self.__undirected = None

    def fit(self, A, block_mask=None, undirected=True, **kwargs):
        self.__nb_nodes = A.shape[0]
        self.__undirected = undirected

        # If no block structure is given, then simply treat this class as the only distribution.
        if block_mask is None:
            self._fit(A, **kwargs)
            return

        block_types = np.unique(block_mask)
        nb_blocks = block_types.shape[0]
        if not np.all(block_types == np.arange(nb_blocks)):
            raise ValueError("Block mask did not contain 0-indexed ordinal values!")

        # If there is only one block, treat the problem as if there are no blocks.
        if nb_blocks == 1:
            self._fit(A, **kwargs)
            return

        self.__block_mask = block_mask
        self.__row_id_to_block_idx = np.empty_like(block_mask, dtype=np.int)
        self.__col_id_to_block_idx = np.empty_like(block_mask, dtype=np.int)

        # Block types were detected. Construct a distribution for every block. Note that if graph is undirected, the
        # prior is symmetric. We then only compute the upper triangle of the blocked matrix.
        self.__dist_per_block = np.empty((nb_blocks, nb_blocks), dtype=np.object)
        for type_i in range(nb_blocks):
            row_mask = block_mask == type_i

            if undirected:
                j_range = range(type_i, nb_blocks)
            else:
                j_range = range(nb_blocks)

            for type_j in j_range:
                col_mask = block_mask == type_j
                self.__row_id_to_block_idx[row_mask] = np.arange(np.sum(row_mask), dtype=np.int)
                self.__col_id_to_block_idx[col_mask] = np.arange(np.sum(col_mask), dtype=np.int)
                sub_A = A[row_mask][:, col_mask]

                if sub_A.count_nonzero() == 0:
                    sub_dist = BgDistZero()
                else:
                    # Fit this distribution over the submatrices defined by row_mask and col_mask.
                    sub_dist: BgDist = self.__class__(row_mask=row_mask, col_mask=col_mask)

                sub_dist._fit(sub_A, **kwargs)
                self.__dist_per_block[type_i, type_j] = sub_dist

    def _fit(self, A, **kwargs):
        """
        Fit function of the subclass.
        """
        raise NotImplementedError

    def _compute_per_block(self, row_ids, col_ids, func_name):
        """
        Compute any function per block for the elements specified by row_ids and col_ids.
        :param row_ids: The row indices of the requested probabilities.
        :param col_ids: The column indices of the requested probabilities.
        :param func_name: 'prob' to compute probabilities, 'lam' to compute lambda sums.
        """
        row_ids = np.array(row_ids)
        col_ids = np.array(col_ids)

        if self.__block_mask is None:
            if func_name == 'prob':
                return self._get_probability(row_ids, col_ids)
            elif func_name == 'lam':
                return self._get_lambdas_sum(row_ids, col_ids)
            else:
                raise ValueError
        else:
            results = np.empty_like(row_ids, dtype=np.float)

            row_blocks = self.__block_mask[row_ids]
            col_blocks = self.__block_mask[col_ids]

            block_combos = np.stack((row_blocks, col_blocks), axis=-1)
            uniq_block_combos = np.unique(block_combos, axis=0)

            for block_combo in uniq_block_combos:
                where_block_combo = np.all(block_combos == block_combo, axis=1)
                row_block, col_block = block_combo[0], block_combo[1]

                sub_row_ids = self.__row_id_to_block_idx[row_ids[where_block_combo]]
                sub_col_ids = self.__col_id_to_block_idx[col_ids[where_block_combo]]

                use_symmetric_dist = self.__undirected and row_block > col_block
                if use_symmetric_dist:
                    # Use symmetry in probability matrix.
                    row_block, col_block = col_block, row_block
                    sub_row_ids, sub_col_ids = sub_col_ids, sub_row_ids

                bg_dist: BgDist = self.__dist_per_block[row_block, col_block]
                if func_name == 'prob':
                    results[where_block_combo] = bg_dist._get_probability(sub_row_ids, sub_col_ids)
                elif func_name == 'lam':
                    results[where_block_combo] = bg_dist._get_lambdas_sum(sub_row_ids, sub_col_ids)
                else:
                    raise ValueError
        return results

    def get_probability(self, row_ids, col_ids):
        if isinstance(row_ids, int) or len(row_ids.shape) < 1:
            row_ids = np.ones_like(col_ids) * row_ids
            return np.squeeze(self._compute_per_block(row_ids, col_ids, 'prob'))
        return self._compute_per_block(row_ids, col_ids, 'prob')

    def get_full_P_matrix(self):
        """
        Mainly for debugging purposes: compute the entire probability matrix.
        """
        n = self.__nb_nodes
        prob_rows = np.empty((n, n), dtype=np.float)
        for row_id in range(n):
            row_idx = np.empty(n, dtype=np.int)
            row_idx.fill(row_id)
            prob_rows[row_id, :] = self.get_probability(row_idx, np.arange(n))
        return prob_rows

    def predict_logits(self, edges):
        return self._compute_per_block(edges[:, 0], edges[:, 1], 'lam')

    def _get_probability(self, row_ids, col_ids):
        """
        get_probability for the subclass.
        """
        raise NotImplementedError

    def _get_lambdas_sum(self, row_ids, col_ids):
        raise NotImplementedError

    def string_code(self):
        raise NotImplementedError


class BgDistZero(BgDist):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__shape = None

    def _fit(self, A, **kwargs):
        self.__shape = A.shape

    def _get_probability(self, row_ids, col_ids):
        nb_rows = row_ids.shape[0]
        assert row_ids.shape[0] == col_ids.shape[0]
        return np.zeros(nb_rows, dtype=np.int)

    def _get_lambdas_sum(self, row_ids, col_ids):
        return np.ones_like(row_ids) * (-np.inf)

    def string_code(self):
        return "zero"


class BgDistBiasedDegree(BgDist):
    def __init__(self, row_mask=None, col_mask=None, **kwargs):
        """
        Implements the degree constraint. If attributes are given, it is the biased degree constraint.
        :param row_mask: the rows over which constraints may be taken. If None, all rows are used.
        :param col_mask: the columns over which constraints may be taken. If None, all cols are used.
        """
        super().__init__(**kwargs)

        self.__row_mask = row_mask
        self.__col_mask = col_mask

        self.__lambdas = None

    def _fit(self, A, attributes=None, **kwargs):
        # Lamdas aggregator keeps track of several 'lambdas' objects.
        lambdas = LambdasAggregator(A.shape, batch_size=None)

        # Define some functions to easily construct row and column lambdas.
        def construct_row_lambdas(attributes_col_mask):
            sub_A = A[:, attributes_col_mask]

            # Compute the row sum for the given submatrix. The expected row sum will have to match the actual sum.
            row_sums = sub_A.sum(axis=1).A.squeeze()

            row_lambdas = RowDegreeLambdas(row_sums, attributes_col_mask)
            lambdas.add_lambdas_object(row_lambdas)

        def construct_col_lambdas(attributes_row_mask):
            sub_A = A[attributes_row_mask, :]

            # The col_sums are computed in a similar way.
            col_sums = sub_A.sum(axis=0).A.squeeze()

            col_lambdas = ColumnDegreeLambdas(col_sums, attributes_row_mask)
            lambdas.add_lambdas_object(col_lambdas)

        if attributes is None:
            # If there are no attributes, simply construct one row and one column constraint over all of A.
            construct_row_lambdas(np.ones(A.shape[0], dtype=np.bool))
            construct_col_lambdas(np.ones(A.shape[1], dtype=np.bool))
        else:
            already_full_row_constraint, already_full_col_constraint = False, False

            col_attributes = attributes[self.__col_mask] if self.__col_mask is not None else attributes
            for attribute_val in np.unique(col_attributes):
                # Find which columns have the given attribute value and construct a row constraint over them.
                attr_col_mask = col_attributes == attribute_val
                if np.all(attr_col_mask):
                    if already_full_col_constraint:
                        continue
                    else:
                        already_full_col_constraint = True
                construct_row_lambdas(attr_col_mask)

            row_attributes = attributes[self.__row_mask] if self.__row_mask is not None else attributes
            for attribute_val in np.unique(row_attributes):
                attr_row_mask = row_attributes == attribute_val
                if np.all(attr_row_mask):
                    if already_full_row_constraint:
                        continue
                    else:
                        already_full_row_constraint = True
                construct_col_lambdas(attr_row_mask)

        # Find lambda values using Newton optimization.
        newton_optimization(lambdas, nit=100)

        self.__lambdas = lambdas

    def _get_probability(self, row_ids, col_ids):
        return self.__lambdas.probability(row_ids, col_ids, inner_idx=True)

    def _get_lambdas_sum(self, row_ids, col_ids):
        return self.__lambdas.lambdas_sum(row_ids, col_ids, inner_idx=True)

    def string_code(self):
        return "bias-deg"


def newton_optimization(lambdas, nit=100, tol=1e-8):
    alpha = 1.0
    prev_alpha = alpha
    start_time = time.time()
    for k in tqdm(range(nit)):
        # For the current lambdas, evaluate the lagrangian for the lambdas and compute the gradient.
        lagrangian = lambdas.forward(with_grad=True)
        grad = lambdas.get_grad()
        delta_la = lambdas.get_delta_la()

        # Find the largest alpha that satisfies the first Wolfe condition.
        # This is done by halving alpha until it happens.
        while True:
            # Step in direction of gradient.
            lambdas.try_step(alpha)

            # Compute lagrangian with this alpha. The gradient should not be recomputed.
            lagrangian_try = lambdas.forward(with_grad=False)

            # Check first Wolfe condition.
            if lagrangian_try <= lagrangian + 1e-4 * alpha * (delta_la.dot(grad)):
                # Condition is satisfied. Wipe the previous gradient.
                lambdas.zero_grad()

                # Note: the lagrangian of the next iteration is actually equal to lagrangian_try. However, recomputing
                # this is fairly inexpensive, since we need to calculate almost all necessary intermediate values
                # anyway. One further optimization could be to maintain these intermediate values, but this would ruin
                # the interface with the LambdasAggregator class.
                break
            else:
                alpha /= 2
                if alpha < 1e-8:
                    print("Low alpha reached before optimizing step. We probably entered an infinite loop!")
                    break

        # Some stop conditions.
        if np.linalg.norm(grad) / grad.shape[0] < tol or k >= nit - 1 or alpha < 1e-8:
            time_diff = time.time() - start_time
            print("Computed the prior in " + str(k + 1) +
                  " iterations (" + str(int(time_diff / 60)) + "m " + str(int(time_diff % 60)) + "s) " +
                  "for a matrix with size " + str(lambdas.shape))
            break

        # If the previous best alpha was the same as the current best alpha, then increase alpha.
        if prev_alpha == alpha:
            prev_alpha = alpha
            alpha = min(1.0, alpha * 2)
        else:
            prev_alpha = alpha


class Lambdas:
    """
    General class for Lagrange multipliers or 'lambdas'.
    """

    def __init__(self, constraints):
        self.constraints = constraints
        self.la = np.zeros_like(self.constraints, dtype=np.float)

        self.grad = None
        self.grad_second_order = None
        self.delta_la = None
        self.__backup_la = None

        self.zero_grad()

    def lambdas_sum(self, row_idx=None, col_idx=None, inner_idx=False):
        raise NotImplementedError

    def compute_grad(self, P, E_div_Z_sqr, row_idx):
        raise NotImplementedError

    def prepare_delta_la(self):
        if self.grad is None or self.grad_second_order is None:
            raise ValueError("Trying to prepare delta_la while no gradient is stored!")
        self.delta_la = -self.grad / (self.grad_second_order + 1e-10 * self.grad_second_order.shape[0])

    def lagrangian_term(self):
        return np.sum(self.la * self.constraints)

    def try_step(self, alpha):
        if self.delta_la is None:
            raise ValueError("Cannot try a step if delta_la is not finalized first by calling prepare_delta_la()!")
        if self.__backup_la is None:
            self.__backup_la = self.la
        self.la = self.__backup_la + alpha * self.delta_la

    def zero_grad(self):
        self.grad = np.zeros_like(self.la)
        self.grad_second_order = np.zeros_like(self.grad)
        self.delta_la = None
        self.__backup_la = None


class RowDegreeLambdas(Lambdas):
    """
    For the constraint where the expected row sum (for the submatrix specified by row_mask and col_mask) is equal to the
    actual sum.
    """

    def __init__(self, row_sums, col_mask):
        super().__init__(constraints=row_sums)
        self._col_mask = col_mask

        if self._col_mask is not None:
            # Initialization based on heuristics.
            P_estimate = (row_sums + 1) / (np.sum(col_mask) + 1)
            self.la = np.log(P_estimate / (1.01 - P_estimate)) / 2

    def lambdas_sum(self, row_idx=None, col_idx=None, inner_idx=False):
        las = self.la
        if row_idx is not None:
            las = las[row_idx]

        cols = self._col_mask
        if col_idx is not None:
            cols = cols[col_idx]

        if inner_idx:
            return las * cols
        else:
            return np.outer(las, cols)

    def compute_grad(self, P, E_div_Z_sqr, row_idx):
        # Check that the size of the given intermediate vars matches the given row index bounds.
        assert P.shape[0] == E_div_Z_sqr.shape[0] == row_idx.shape[0]
        self.grad[row_idx] = (P.dot(self._col_mask)) - self.constraints[row_idx]
        self.grad_second_order[row_idx] = (E_div_Z_sqr.dot(self._col_mask))


class ColumnDegreeLambdas(Lambdas):
    """
    For the constraint where the expected column sum (for the submatrix specified by row_mask and col_mask) is equal to
    the actual sum.
    """

    def __init__(self, col_sums, row_mask):
        super().__init__(constraints=col_sums)
        self._row_mask = row_mask

        if self._row_mask is not None:
            # Initialization based on heuristics.
            P_estimate = (col_sums + 1) / (np.sum(row_mask) + 1)
            self.la = np.log(P_estimate / (1.01 - P_estimate)) / 2

    def lambdas_sum(self, row_idx=None, col_idx=None, inner_idx=False):
        las = self.la
        if col_idx is not None:
            las = self.la[col_idx]

        rows = self._row_mask
        if row_idx is not None:
            rows = rows[row_idx]

        if inner_idx:
            return rows * las
        else:
            return np.outer(rows, las)

    def compute_grad(self, P, E_div_Z_sqr, row_idx):
        # Check that the size of the given intermediate vars matches the given row index bounds.
        assert P.shape[0] == E_div_Z_sqr.shape[0] == row_idx.shape[0]

        # To compute this gradient, we only require the current P matrix col sums to be a fraction of the total,
        # because we only sum over (i_e - i_s) rows in this iteration.
        self.grad += P.T.dot(self._row_mask[row_idx]) - self.constraints * (row_idx.shape[0] / self._row_mask.shape[0])
        self.grad_second_order += (E_div_Z_sqr.T.dot(self._row_mask[row_idx]))


class LambdasAggregator:
    """
    Perform aggregation operations on the lambdas objects. It is assumed that every element follows an independent
    Bernoulli distribution.
    """

    def __init__(self, A_shape, batch_size=None):
        # Shape of the adjacency matrix.
        self.shape = A_shape

        # The number of rows that are evaluated per batch. Decrease if memory runs out!
        # If batch_size is 'None', then there is no batching.
        self.batch_size = batch_size

        self._lambdas_list = []
        self._grad_is_zero = True

    def add_lambdas_object(self, lambdas):
        assert isinstance(lambdas, Lambdas)
        self._lambdas_list.append(lambdas)

    def lambdas_sum(self, row_idx=None, col_idx=None, inner_idx=False):
        """
        Compute the 'lambdas' sum of the row indices and col indices.
        :param row_idx: the row indices.
        :param col_idx: the column indices.
        :param inner_idx: if True, then we only compute lambdas for positions with the same index in the row_idx and
        col_idx arrays. If False, then we compute lambdas for the entire submatrix specified by row_idx and col_idx.
        """
        lambdas_sum = 0
        for lambdas in self._lambdas_list:
            lambda_sub_sum = lambdas.lambdas_sum(row_idx, col_idx, inner_idx)
            lambdas_sum = lambdas_sum + lambda_sub_sum
        return lambdas_sum

    @staticmethod
    def _Z(E):
        return 1 + E

    def grad(self, P, E_div_Z_sqr, row_idx=None):
        [lambdas.compute_grad(P, E_div_Z_sqr, row_idx) for lambdas in self._lambdas_list]

    def prepare_delta_la(self):
        [lambdas.prepare_delta_la() for lambdas in self._lambdas_list]

    def forward(self, with_grad=False):
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = self.shape[0] + 1

        if with_grad:
            if not self._grad_is_zero:
                raise ValueError("Call 'zero_grad()' before calling forward with 'with_grad=True'!")
            else:
                self._grad_is_zero = False

        # Initialize a part of the lagrangian.
        log_partition = 0
        for batch_i in range((self.shape[0] // batch_size) + 1):
            batch_start = batch_i * batch_size
            batch_end = min((batch_i + 1) * batch_size, self.shape[0])
            batch_range = np.arange(batch_start, batch_end)

            # Compute E: the exponential of the sum of lambdas for indices (i,j).
            E = np.exp(self.lambdas_sum(row_idx=batch_range))

            # Compute Z: the partition function.
            Z = self._Z(E)

            # Compute the contribution of this batch's log-partition function to the lagrangian.
            log_partition_batch = np.sum(np.log(Z))
            log_partition += log_partition_batch

            if with_grad:
                # For computing the gradient and curvature information, we use two intermediate variables:
                P = E / Z
                # Sidenote: this specific step is slightly faster by computing (E/(1+E)) as (1/(1+E^1))

                E_div_Z_sqr = P / Z
                self.grad(P, E_div_Z_sqr, row_idx=batch_range)

        if with_grad:
            # Now that gradients are computed for all rows, set the delta_la for all lambdas.
            if with_grad:
                self.prepare_delta_la()

        # Add the contribution of the constraints individually to the lagrangian.
        lagrangian = log_partition
        for lambdas in self._lambdas_list:
            lagrangian -= lambdas.lagrangian_term()
        return lagrangian

    def try_step(self, alpha):
        for lambdas in self._lambdas_list:
            lambdas.try_step(alpha)

    def zero_grad(self):
        for lambdas in self._lambdas_list:
            lambdas.zero_grad()
        self._grad_is_zero = True

    def get_grad(self):
        return np.concatenate([lambdas.grad for lambdas in self._lambdas_list])

    def get_delta_la(self):
        return np.concatenate([lambdas.delta_la for lambdas in self._lambdas_list])

    def probability(self, row_idx, col_idx, inner_idx=False):
        """
        Compute the probability for rows row_idx and columns col_idx.
        """
        # Send the lambda sum through a sigmoid function (in the binary case).
        E = np.exp(self.lambdas_sum(row_idx=row_idx, col_idx=col_idx, inner_idx=inner_idx))
        return E / self._Z(E)

    def arrange_all_lambdas(self):
        all_lambda_vals = []
        for lambdas in self._lambdas_list:
            all_lambda_vals.extend(list(lambdas.la))
        return np.array(all_lambda_vals)
