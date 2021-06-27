import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.model_selection as sk_ms
import sklearn.linear_model as sk_lm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from evaluation.fairness_metrics import FairnessMetrics
from evaluation.metrics import Metrics


class GraphEmbeddingEvaluator:
    def __init__(self,
                 binarize_probs=False,
                 undirected=True):
        self.binarize_probs = binarize_probs
        self.undirected = undirected

        self.__predictor = None
        self.__normal_metrics = None
        self.__fairness_metrics = None

    def set_predictor(self, predictor):
        self.__predictor = predictor

    def evaluate(self, test_set, attributes, sens_attr_name):
        if not self.undirected:
            raise Warning("Evaluating on directed edges was not tested yet!")

        print("Evaluating on test data...")
        test_data, test_labels, test_neg_corr_factor = test_set

        normal_metrics = Metrics()

        # The partition info is not used in the evaluation.
        attributes = attributes[[sens_attr_name]].fillna("N/A")

        # Construct a fairness metric per attribute type.
        fairness_metrics = {}
        for attr_name in attributes.columns:
            fairness_metrics[attr_name] = FairnessMetrics(
                attr_name=attr_name,
                neg_corr_factor=test_neg_corr_factor
            )

        # Predict the edges using the 'predict' function.
        predictions = self.__predictor.predict(test_data)

        if self.binarize_probs:
            frac_positive = test_labels.sum() / test_labels.shape[0]
            threshold = np.percentile(predictions, (1 - frac_positive) * 100)
            predictions = (predictions >= threshold).astype(np.float)

        # Process predictions for normal metrics.
        results = normal_metrics(predictions, test_labels)

        # Retrieve attribute values.
        for attr_name in attributes.columns:
            attr_col = attributes[attr_name]
            src_attr = attr_col[test_data[:, 0]].reset_index(drop=True)
            dst_attr = attr_col[test_data[:, 1]].reset_index(drop=True)
            edge_attrs = pd.concat({'src': src_attr, 'dst': dst_attr}, axis=1, ignore_index=True)
            if self.undirected:
                edge_attrs = pd.DataFrame(np.sort(edge_attrs, axis=1),
                                          index=edge_attrs.index, columns=edge_attrs.columns)
            edge_attrs = (edge_attrs[0] + "_" + edge_attrs[1])
            if "N/A_N/A" in edge_attrs:
                raise ValueError("Found edges without any attributes, this may cause errors later on!")

            # Compute fairness metrics.
            fairness_results = fairness_metrics[attr_name](predictions, test_labels, edge_attrs)
            results.extend(fairness_results)

            # Evaluate bias in embeddings.
            results.extend(self.__test_bias_embedding(attr_name, attr_col))

            # Add the loss for the last batch of training as a measure.
            last_loss_vals = self.__predictor.last_loss_vals()
            if last_loss_vals is not None:
                pred_loss, fair_loss = last_loss_vals
                results.append(['LP', pred_loss])
                results.append(['LF', fair_loss])

        return results

    def __test_bias_embedding(self, attr_name, attr_col, cv_count=3, test_size_frac=0.2):
        """
        Test bias by training a logistic regression on the embeddings of the predictor, trying to predict the sensitive
        attribute.
        :return:
        """
        notna_attrs = attr_col[attr_col != "N/A"]
        X = self.__predictor.get_embeddings(notna_attrs.index.to_numpy())
        Y = notna_attrs.to_numpy()

        # Filter out labels that are very infrequent (less than what is required for a proper logistic regression).
        all_labels, label_counts = np.unique(Y, return_counts=True)

        frequent_labels = all_labels[(label_counts * (1 - test_size_frac)) >= cv_count + 1]
        frequent_label_mask = np.isin(Y, frequent_labels)
        Y = Y[frequent_label_mask]
        X = X[frequent_label_mask]

        if np.unique(Y).shape[0] > 2:
            scorer = "roc_auc_ovr_weighted"
        else:
            scorer = "roc_auc"

        clf = GridSearchCV(
            Pipeline([('scaler', sk.preprocessing.StandardScaler()),
                      ('logi', sk_lm.LogisticRegression(multi_class='multinomial', solver='saga', max_iter=100))
                      ]), param_grid={'logi__C': 100. ** np.arange(-2, 3), 'logi__penalty': ['l1', 'l2']}, cv=cv_count,
            scoring=scorer)

        # Split the embeddings into train and test examples.
        X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(X, Y, test_size=test_size_frac, stratify=Y)

        clf.fit(X_train, Y_train)
        try:
            print(clf.best_params_)
        except AttributeError:
            pass
        clf = clf.best_estimator_

        clf = CalibratedClassifierCV(clf, cv='prefit')
        clf.fit(X_train, Y_train)

        train_score = sk.metrics.get_scorer(scorer)(clf, X_train, Y_train)
        print(f"Using a logistic regression classifier, "
              f"the TRAIN {scorer} score for predicting {attr_name} is {train_score}.")
        test_score = sk.metrics.get_scorer(scorer)(clf, X_test, Y_test)
        print(f"Using a logistic regression classifier, "
              f"the TEST {scorer} score for predicting {attr_name} is {test_score}.")

        results = [["ReprBias_{}_{}".format(attr_name, "logi"), test_score]]
        return results

    def as_string(self):
        string = "Evaluator:\n"
        string += f"binarize_probs: {self.binarize_probs}\n"
        return string
