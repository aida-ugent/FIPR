import random
import os
from os.path import dirname, abspath, join
import numpy as np
import torch.random
import time
import config_all as cfg

from data_loaders import DataLoader
from evaluation import GraphEmbeddingEvaluator
from predictors import Predictor


def __run_all_predictors():
    data = DataLoader(**cfg.data_cfg)
    evaluator = GraphEmbeddingEvaluator(**cfg.eval_cfg)

    # Run the entire pipeline for all predictors.
    for predictor_cfg in cfg.predictors:
        predictor = Predictor(**predictor_cfg)
        print(f"Running seeds for predictor {predictor_cfg['predictor_name']}")
        if not issubclass(type(predictor), Predictor):
            raise ValueError(f"The 'predictor' has class {type(predictor)}, but that is not a subclass of Predictor!")
        __run_all_seeds(data, evaluator, predictor)


def __run_all_seeds(data, evaluator, predictor):
    # For this predictor, run all seeds.
    all_results = []
    for seed in cfg.rand_seeds:
        results = __run_experiment(seed, data, evaluator, predictor)

        # Print temporary results.
        print('\n'.join(str(result) for result in results))
        all_results.append(results)

    # Aggregate all metrics over the splits.
    example_results = all_results[0]
    aggregated_results = []
    full_results = []
    if len(all_results) > 1:
        print("\n\nAggregated results:")
    for metric_i, metric in enumerate(example_results):
        metric_name = metric[0]
        aggregated_med = [metric_name, np.median([seed_result[metric_i][1] for seed_result in all_results])]
        aggregated_results.append(aggregated_med)
        if len(all_results) > 1:
            print(aggregated_med)
        full_results.append(["all_" + metric_name, [seed_result[metric_i][1] for seed_result in all_results]])
    all_results = aggregated_results + full_results

    # Save full results to file.
    __save_final_results(all_results, data, evaluator, predictor)


def __run_experiment(random_seed, data, evaluator, predictor):
    print("")
    print("-----------------------------------------------------")
    print("Running experiment with seed {}.".format(random_seed))
    print("-----------------------------------------------------")

    # Set random seeds.
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load and format dataset.
    data.load()

    # Get the attributes for every entity in the data.
    # Note: it is expected that these attributes are not used as features, so there is no train/test split.
    attributes = data.get_attributes()
    sens_attr_name = data.get_sens_attr_name()

    # Fit predictor to the training data.
    start_time = time.perf_counter()
    predictor.fit(train_data=data.get_train_data(), attributes=attributes.copy(),
                  dataset_name=cfg.data_cfg['dataset_name'], random_seed=random_seed, sens_attr_name=sens_attr_name)
    fitting_time = time.perf_counter() - start_time

    # Evaluate the predictor, based on its 'predict' function.
    evaluator.set_predictor(predictor)
    results = evaluator.evaluate(data.get_test_set(), attributes, sens_attr_name)
    results.append(['time', fitting_time])
    return results


def __get_results_file(extension, predictor):
    # Setup file path and make directory if needed.
    results_folder_name = cfg.data_cfg['dataset_name']
    results_file_name = predictor.get_filename() + extension
    results_path = join(dirname(abspath(__file__)), "..", "results", results_folder_name, results_file_name)
    os.makedirs(dirname(results_path), exist_ok=True)

    return results_path


def __save_final_results(results, data, evaluator, predictor):
    string = "Results:\n"
    for result in results:
        string += f"{result[0]}: {result[1]}\n"
    string += "===========================================\n"

    string += f"rand_seeds: {cfg.rand_seeds}\n"

    for obj in [data, predictor, evaluator]:
        string += obj.as_string()
        string += "===========================================\n"

    results_file = __get_results_file(".txt", predictor)
    with open(results_file, "w") as file:
        file.write(string)


if __name__ == '__main__':
    __run_all_predictors()
