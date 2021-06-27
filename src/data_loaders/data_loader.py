import os
import numpy as np
import pandas as pd

from utils.adjacency_data import map_edges
from utils.data_split_util import leave_fraction_out


class DataLoader:
    def __new__(cls, dataset_name='', **kwargs):
        if dataset_name == "ml-100k":
            from .ml_loader import Movielens100kLoader
            return super().__new__(Movielens100kLoader)
        elif dataset_name == "polblogs":
            from .polblogs_loader import PolBlogsLoader
            return super().__new__(PolBlogsLoader)
        elif dataset_name == "karate":
            from .karate_loader import KarateLoader
            return super().__new__(KarateLoader)
        elif dataset_name == "facebook":
            from .facebook_loader import FacebookLoader
            return super().__new__(FacebookLoader)
        else:
            raise ValueError(f"{dataset_name=} not recognized!")

    def __init__(self, dataset_path, **kwargs):
        self.__data_path = dataset_path

        self.__train_edges = None
        self.__test_set = None
        self.__attributes = None

    def get_dataset_name(self):
        raise NotImplementedError

    def get_train_data(self):
        return self.__train_edges[:, :2]

    def get_test_set(self):
        return self.__test_set

    def get_attributes(self):
        return self.__attributes

    def get_sens_attr_name(self):
        raise NotImplementedError

    def load(self):
        print("Loading and preparing " + self.get_dataset_name() + " data.")

        # Load all positive edges and node attributes.
        positive_edges, attributes = self._load()

        if not isinstance(attributes, pd.DataFrame):
            raise ValueError("Attributes is not a pandas DataFrame object!")

        if not (nb_positive_edges := np.unique(positive_edges).shape[0]) == \
               (nb_attributed_nodes := len(attributes)):
            raise ValueError(f"There are {nb_positive_edges=} in the dataset and {nb_attributed_nodes=}!")

        attributes['int_idx'] = np.arange(len(attributes))
        positive_edges = map_edges(positive_edges, nodes_map=attributes['int_idx'])
        attributes = attributes.set_index('int_idx', drop=True)

        train_edges, test_set = leave_fraction_out(positive_edges, **self._get_data_split_kwargs())
        self.__train_edges, self.__test_set = train_edges, test_set
        self.__attributes = attributes

    def _load(self):
        raise NotImplementedError

    def _get_data_split_kwargs(self):
        raise NotImplementedError

    def _load_file(self, file_name, delimiter, encoding='utf-8', skip_header=0, deletechars='"', dtype=str,
                   max_rows=None):
        data_file_path = os.path.join(self._get_folder_path(), file_name)
        with open(data_file_path, mode='r', encoding=encoding) as file:
            data = np.genfromtxt(file, delimiter=delimiter, comments=None, skip_header=skip_header,
                                 deletechars=deletechars, autostrip=True, dtype=dtype, max_rows=max_rows)
        return data

    def _get_folder_path(self):
        return os.path.join(self.__data_path, self.get_dataset_name())

    def as_string(self):
        string = f"dataset_name: {self.get_dataset_name()}\n"
        return string
