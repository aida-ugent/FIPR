# Fair I-Projection Regularizer
Experiment code for the paper "The KL-Divergence between a Graph Model and its Fair I-Projection as a Fairness Regularizer", published at ECML-PKDD 2021.

## Running the code
Running *main_all.py* executes the pipeline as configured by *config.py*.
NOTE: this will overwrite any results in the results/ folder. 

Alternatively, run *main_simple.py* for a simple example usage. 

## Setup
1) Download the desired datasets. See the data/ folder for their links.

2) Install the required packages as indicated by requirements.txt.

## Fair I-Projection as a regularizer in your projects
The expected interface of each (link) *Predictor* is documented in *predictor.py*.

The distance from a model to its I-projection can be computed using the *FairnessLoss* PyTorch module. A forward call on this module consists of two steps. First, the fair I-projection (code in *fip.py*) is fit to the given model values *h* and data points *x*. Second, the gradient of the fair I-projection's loss is computed with respect to the model values *h*. 

Two fairness notions are implemented in *fairness_notions.py*: 'DP' for Demographic Parity and 'EO' for Equalised Opportunity.

## Citation
If you found our work useful in your own project, please cite our paper:

    @inproceedings{buyl2021kl,
        author={Buyl, Maarten and De Bie, Tijl},
        title={The KL-Divergence Between a Graph Model and its Fair I-Projection as a Fairness Regularizer},
        booktitle={Machine Learning and Knowledge Discovery in Databases},
        year={2021},
        publisher={Springer International Publishing},
        pages={351--366}
    }

## Maintenance
Further development may be done in the future and bugs will be fixed. If you have any questions or concerns, feel free to report it here or send an email to 'maarten.buyl@ugent.be'.
