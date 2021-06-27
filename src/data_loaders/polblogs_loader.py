import numpy as np
import pandas as pd
import networkx as nx
from os.path import join

from .data_loader import DataLoader


class PolBlogsLoader(DataLoader):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def get_dataset_name(self):
        return "polblogs"

    def get_sens_attr_name(self):
        return "party"

    def _load(self):
        G = nx.read_gml(join(self._get_folder_path(), "polblogs.gml"))
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Only keep largest connected component.
        G = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])

        blog_parties = pd.DataFrame.from_dict(G.nodes, orient='index')
        blog_parties = blog_parties.rename(columns={'value': 'party'}).drop(columns=['source'])
        blog_parties.party = blog_parties.party.map({0: 'left', 1: 'right'})

        positive_edges = np.array(list(G.edges()))
        return positive_edges, blog_parties

    def _get_data_split_kwargs(self):
        return {
            'test_set_frac': 0.2,
            'directed': False
        }
