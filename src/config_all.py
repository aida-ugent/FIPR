import os

# Rand seeds. The entire pipeline will be run for every seed and the results are averaged.
rand_seeds = list(range(10))
save_model = True
load_model = True

data_cfg = {
    # Uncomment the desired dataset.
    'dataset_name':
        # "ml-100k",
        # "facebook",
        "polblogs",
        # "karate",

    # Path to the datasets folder. Should automatically follow the folder structure in the git repo.
    'dataset_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
}

eval_cfg = {
    'binarize_probs': False
}

######
# Configuration of predictor(s).
# At the bottom, list the predictor(s) that should be used in the experiments.

FIP_STRENGTH_GLOBAL = 100

# MaxEnt predictor.
maxent_normal = {
    'predictor_name': "maxent",

    'fip_strength': 0,
    'fip_type': '',
    'learning_rate': 1e-3,
    'nb_epochs': 100
}

maxent_fip_DP = maxent_normal.copy()
maxent_fip_DP['fip_strength'] = FIP_STRENGTH_GLOBAL
maxent_fip_DP['fip_type'] = 'DP'

maxent_fip_EO = maxent_fip_DP.copy()
maxent_fip_EO['fip_type'] = 'EO'

dotproduct_normal = {
    'predictor_name': "dotproduct",

    'fip_strength': 0,
    'fip_type': '',
    'nb_epochs': 100,
    'learning_rate': 1e-2,
    'batch_size': 10000,
    'dimension': 128
}

dotproduct_fip_EO = dotproduct_normal.copy()
dotproduct_fip_EO['fip_strength'] = FIP_STRENGTH_GLOBAL
dotproduct_fip_EO['fip_type'] = 'EO'

dotproduct_fip_DP = dotproduct_fip_EO.copy()
dotproduct_fip_DP['fip_type'] = 'DP'

cne_normal = {
    'predictor_name': 'cne',

    's2': 16,
    'subsample_neg': 100,
    'fip_strength': 0,
    'fip_type': '',
    'nb_epochs': 200,
    'learning_rate': 1e-1,
    'batch_size': 10000,
    'dimension': 8
}

cne_fip_EO = cne_normal.copy()
cne_fip_EO['fip_type'] = 'EO'
cne_fip_EO['fip_strength'] = FIP_STRENGTH_GLOBAL

cne_fip_DP = cne_fip_EO.copy()
cne_fip_DP['fip_type'] = 'DP'

gae_normal = {
    'predictor_name': "gae",
    'fip_strength': 0,
    'fip_sample_size': int(1e5),
    'batch_size': 10000,
    'subsample_neg': 100,
    'fip_type': '',
    'nb_epochs': 100,
    'learning_rate': 1e-2,
    'dimension': 16,
    'nb_layers': 2,
    'dropout_pct': 0.5,
    'device_name': 'cpu'
}

gae_fip_EO = gae_normal.copy()
gae_fip_EO['fip_strength'] = FIP_STRENGTH_GLOBAL
gae_fip_EO['fip_type'] = 'EO'

gae_fip_DP = gae_fip_EO.copy()
gae_fip_DP['fip_type'] = 'DP'

# Configs of the predictor(s) that are actually used in the experiment.
predictors = [
    dotproduct_normal,
    dotproduct_fip_EO,
    dotproduct_fip_DP,
    maxent_normal,
    maxent_fip_EO,
    maxent_fip_DP,
    gae_normal,
    gae_fip_DP,
    gae_fip_EO,
    cne_normal,
    cne_fip_DP,
    cne_fip_EO,
]
