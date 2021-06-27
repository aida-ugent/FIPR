import numpy as np
import pandas as pd

from data_loaders.data_loader import DataLoader


class FacebookLoader(DataLoader):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def get_dataset_name(self):
        return "facebook"

    def get_sens_attr_name(self):
        return "gender"

    def _load(self):
        positive_edges = super()._load_file("facebook_combined.txt", delimiter=" ", dtype=np.int)

        attributes = {}
        ego_ids = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
        for ego_id in ego_ids:
            feat_names = super()._load_file(f"{ego_id}.featnames", delimiter=" ")
            gender_rows = feat_names[:, 1] == "gender;anonymized"
            gender_feat_idx = feat_names[gender_rows][:, 0].astype(np.int)

            ego_feats = super()._load_file(f"{ego_id}.egofeat", delimiter=" ", dtype=np.int)
            ego_gender_feats = ego_feats[gender_feat_idx]
            if np.sum(ego_gender_feats) == 1:
                attributes[ego_id] = ego_gender_feats[0]

            other_feats = super()._load_file(f"{ego_id}.feat", delimiter=" ", dtype=np.int)
            exactly_one_gender_given = other_feats[:, 1 + gender_feat_idx].sum(axis=1) == 1
            other_feats = other_feats[exactly_one_gender_given]
            other_attrs = dict(zip(other_feats[:, 0], other_feats[:, 1 + gender_feat_idx[0]]))
            attributes.update(other_attrs)

        attributes = pd.Series(attributes).astype('str').to_frame('gender')
        edges_with_gendered_nodes = np.isin(positive_edges, attributes.index.to_numpy()).all(axis=1)
        positive_edges = positive_edges[edges_with_gendered_nodes]
        return positive_edges, attributes

    def _get_data_split_kwargs(self):
        return {
            'test_set_frac': 0.2,
            'directed': False
        }
