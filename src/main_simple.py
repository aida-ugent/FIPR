# Author: Maarten Buyl
# Contact: maarten.buyl@ugent.be
# Date: 25/06/2021

import random
import os
import numpy as np
import torch.random
import time

from data_loaders import DataLoader
from evaluation import GraphEmbeddingEvaluator
from predictors.dot_product_decoder import DotProductDecoder


def polblogs_example():
    # Initialize with example options.
    dataset_name = 'polblogs'
    data = DataLoader(dataset_name=dataset_name,
                      dataset_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    evaluator = GraphEmbeddingEvaluator()
    predictor = DotProductDecoder(
        fip_strength=100,
        fip_type='DP',
        nb_epochs=100,
        learning_rate=1e-2,
        batch_size=10000,
        dimension=128)
    seed = 0

    # Set random seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load and format dataset.
    data.load()

    # Get the attributes for every entity in the data.
    # Note: it is expected that these attributes are not used as features, so there is no train/test split.
    attributes = data.get_attributes()
    sens_attr_name = data.get_sens_attr_name()

    # Fit the predictor.
    start_time = time.perf_counter()
    predictor.fit(train_data=data.get_train_data(), attributes=attributes.copy(),
                  dataset_name=dataset_name, random_seed=seed, sens_attr_name=sens_attr_name)
    fitting_time = time.perf_counter() - start_time

    # Evaluate the predictor, based on its 'predict' function.
    evaluator.set_predictor(predictor)
    results = evaluator.evaluate(data.get_test_set(), attributes, sens_attr_name)
    results.append(['time', fitting_time])
    print(results)

if __name__ == '__main__':
    polblogs_example()
