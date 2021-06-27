import numpy as np
import pandas as pd
import networkx as nx

from .data_loader import DataLoader


class KarateLoader(DataLoader):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def get_dataset_name(self):
        return "karate"

    def get_sens_attr_name(self):
        return "club"

    def _load(self):
        G = nx.karate_club_graph()
        G.remove_edges_from(nx.selfloop_edges(G))

        # Only keep largest connected component.
        G = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
        positive_edges = np.array(list(G.edges()))

        club_membership = pd.DataFrame.from_dict(G.nodes, orient='index')
        return positive_edges, club_membership

    def _get_data_split_kwargs(self):
        return {
            'test_set_frac': 0,
            'directed': False
        }
