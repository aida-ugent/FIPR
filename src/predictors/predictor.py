from sklearn.base import BaseEstimator


class Predictor(BaseEstimator):
    # Factory method to get Predictor classes by their short string name.
    def __new__(cls, predictor_name=None, **kwargs):
        if predictor_name is None:
            if cls == Predictor:
                raise ValueError("No predictor name was provided in the Predictor config!")
            return super().__new__(cls)

        if predictor_name == "maxent":
            from .maxent_predictor import MaxEntPredictor
            return super().__new__(MaxEntPredictor)

        if predictor_name == "dotproduct":
            from .dot_product_decoder import DotProductDecoder
            return super().__new__(DotProductDecoder)

        if predictor_name == "cne":
            from .cne_predictor import ConditionalNetworkEmbeddingPredictor
            return super().__new__(ConditionalNetworkEmbeddingPredictor)

        if predictor_name == "gae":
            from .gae_predictor import GAEPredictor
            return super().__new__(GAEPredictor)

        raise ValueError(f"Got {predictor_name=} in config, but no predictor is known by that code!")

    def __init__(self,
                 predictor_name='',
                 **_):
        super().__init__()

        self._predictor_name = predictor_name
        self._last_loss_vals = None

    def fit(self, train_data, attributes, **kwargs):
        """
        Fit the predictor on the train data.
        :param train_data: an (N,2) numpy array with N the number of training edges.
        :param attributes: a Pandas DataFrame with a row for every node. Columns include the sensitive attribute and
        optionally a 'partition' column that indicates which partition a node belongs to if the graph is k-partite.
        The 'partition number' of a node should be an integer value in [0, k-1]
        :param kwargs: additional keyword arguments, not always used by all predictors.
        Currently used:
        'dataset_name' = name of the dataset
        'random_seed' = identifier for the used random seed
        'sens_attr_name' = name of the sensitive attribute, corresponds with its column name in the attributes param.
        """
        raise NotImplementedError

    def predict(self, data):
        """
        Predict the likelihood of edges in 'data'. No gradient computations are performed.
        :param data: an (N,2) numpy array with N the number of edges to be predicted.
        :return: a size N numpy vector of likelihood scores.
        """
        raise NotImplementedError

    def get_embeddings(self, nodes):
        """
        Get embeddings for evaluation purposes.
        :param nodes: a size N numpy vector of integer indices of the nodes for which we should get embeddings.
        :return: a size (N,dim) numpy vector of embeddings.
        """
        raise NotImplementedError

    def as_string(self):
        string = self.__class__.__name__ + ":\n"
        for name, value in self.get_params().items():
            string += f"{name}: {value}\n"
        return string

    def last_loss_vals(self):
        return self._last_loss_vals

    def get_filename(self):
        try:
            if self.fip_strength != 0:
                return f"{self._predictor_name}_{self.fip_strength}_{self.fip_type}"
            else:
                return f"{self._predictor_name}_0"
        except AttributeError:
            return f"{self._predictor_name}_0"

    def __eq__(self, other):
        return self.get_params() == other.get_params()
