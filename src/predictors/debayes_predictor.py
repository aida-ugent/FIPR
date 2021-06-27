# Author: Maarten Buyl
# Contact: maarten.buyl@ugent.be
# Date: 17/07/2020
# From https://github.com/aida-ugent/DeBayes


import numpy as np
from .predictor import Predictor
from .cne_predictor import ConditionalNetworkEmbeddingPredictor
from .debayes_code.bg_dist import BgDistBuilder

from utils.adjacency_data import build_adjacency_matrix


class DeBayes(Predictor):
    def __init__(self,
                 s1=1,
                 s2=16,
                 subsample=100,
                 nb_epochs=1000,
                 learning_rate=1e-1,
                 dimension=8,
                 training_prior_type='biased_degree',
                 eval_prior_type='degree',
                 **kwargs):
        super().__init__(**kwargs)

        self.dimension = dimension
        self.s1 = s1
        self.s2 = s2
        self.subsample = subsample
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
        self.training_prior_type = training_prior_type
        self.eval_prior_type = eval_prior_type

        self.__training_bg_dist = None
        self.__eval_bg_dist = None
        self.cne = None

    def fit(self, train_data, attributes, **_kwargs):
        # Using the integer-indexed graph, build an adjacency matrix with the same indices.
        A = build_adjacency_matrix(train_data, nb_nodes=len(attributes))

        # Fit the training and evaluation prior.
        self.__training_bg_dist = self.__fit_prior(A, attributes, self.training_prior_type)

        # training_P = self.__training_bg_dist.get_full_P_matrix()
        self.__eval_bg_dist = self.__fit_prior(A, attributes, self.eval_prior_type)

        # CNE arguments:
        self.cne = ConditionalNetworkEmbeddingPredictor(
            prior_dist=self.__training_bg_dist, dimension=self.dimension, s1=self.s1, s2=self.s2,
            nb_epochs=self.nb_epochs, learning_rate=self.learning_rate, subsample_neg=self.subsample)
        self.cne.fit(train_data, attributes)
        # self.cne = ConditionalNetworkEmbedding(
        #     prior=self.__training_bg_dist, d=self.dimension, s1=self.s1, s2=self.s2, nb_epochs=self.nb_epochs,
        #     learning_rate=self.learning_rate, k_subsample=self.subsample, sampling_correction=False)
        # self.cne.fit(A, attributes)

        # Switch the prior for CNE to the evaluation prior.
        self.cne.prior_dist = self.__eval_bg_dist

    @staticmethod
    def __fit_prior(A, attributes, prior_type):
        bg_dist = BgDistBuilder.build(prior_type)

        print("Computing a background distribution of type: " + prior_type + ".")
        try:
            block_mask = attributes['partition'].to_numpy()
            attributes = attributes.drop(columns=['partition'])
        except KeyError:
            block_mask = None
        if 'biased' in prior_type:
            attr_name = attributes.columns[0]
            attributes = attributes[attr_name].fillna("N/A")
            bg_dist.fit(A, block_mask=block_mask, attributes=attributes.to_numpy())
        else:
            bg_dist.fit(A, block_mask=block_mask)
        return bg_dist

    def get_embeddings(self, node_idx):
        return self.cne.get_embeddings(node_idx)

    def predict(self, edges):
        scores = self.cne.predict(edges)
        return np.array(scores)

    def filename(self):
        return self.__class__.__name__ + "_trp_" + self.training_prior_type + "_evp_" + \
               self.eval_prior_type
