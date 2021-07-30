import numpy as np
import pandas as pd
from .metrics import ks_stat
from IPython.display import  clear_output
from scipy.stats import t, sem
from sklearn.metrics import max_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, log_loss
from sklearn.utils import resample


def evaluate_classifier(y, X, classifier, verbose=True, digits=3):

    """

    Evaluate a scikit-learn classifier with 'predict' and 'predict_proba' methods on it.
    Current metrics: roc_auc_score, precision_score, recall_score, accuracy_score, f1_score log_loss and KS-Statistic.


    Parameters
    ----------

    :param y: 1d array-like, or label indicator array / sparse matrix, Required.
        Ground truth (correct) target values.

    :param X: {array-like, sparse matrix} of shape (n_samples, n_features), Required.
        The input samples. Internally, it will be converted to dtype=np.float32 and if a
        sparse matrix is provided to a sparse csr_matrix.

    :param classifier: estimator object fitted, Required.
        The object to predict classes and probabilities.

    :param verbose: boolean, Optional.
        if print the values of the calculated metrics. Default is True.

    :param digits: int/float, Optional.
        If verbose is True. The number of decimals to use when printing the numbers. Default is 3.

    :return dict
        A dictionary with all the metrics calculated.

    """

    y_pred = classifier.predict(X)
    y_proba = classifier.predict_proba(X)

    roc_auc = roc_auc_score(y, y_pred, average='macro', sample_weight=None)
    accuracy = accuracy_score(y, y_pred, sample_weight=None)
    precision = precision_score(y, y_pred, average='binary', sample_weight=None)
    recall = recall_score(y, y_pred, average='binary', sample_weight=None)
    f1 = f1_score(y, y_pred, average='binary', sample_weight=None)
    ks = ks_stat(y, y_proba[:, 1])
    logloss = log_loss(y, y_proba[:, 1], sample_weight=None)

    if verbose:
        print("ROC-AUC Score:", round(roc_auc, digits))
        print("Accuracy:", round(accuracy, digits))
        print("Precision:", round(precision, digits))
        print("Recall:", round(recall, digits))
        print("F1-Score:", round(f1, digits))
        print("KS-Score:", round(ks, digits))
        print("Log-Loss:", round(logloss, digits))

    return {"ROC-AUC Score": roc_auc,
            "Acurracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "KS-Score": ks,
            "Log-Loss": logloss}


def evaluate_regressor(y_true, y_pred, verbose=True, digits=3):

    """

    Calculates the following metrics.

    "R-Squared": (coefficient of determination) regression score function.
    "25% Abs. Error": Return values at quantile 25 over requested axis.
    "Median Abs. Error": Median absolute error regression loss.
    "Mean Squared Error": Mean squared error regression loss.
    "Mean Abs. Error": Mean absolute error regression loss.
    "75% Abs. Error": Return values at quantile 75 over requested axis.
    "Max. Abs. Error": calculates the maximum residual error.



    Parameters
    ----------

    :param y_true: 1d array-like, or label indicator array / sparse matrix, Required.
        Ground truth (correct) target values.

    :param y_pred: 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a regressor.

    :param verbose: boolean, Optional.
        if print the values of the calculated metrics. Default is True.

    :param digits: int/float, Optional.
        If verbose is True. The number of decimals to use when printing the numbers. Default is 3.

    :return dict
        A dictionary with all the metrics calculated.

    """

    error = y_pred - y_true

    r2 = min(1, max(0, r2_score(y_true, y_pred)))
    abs_error_q25 = error.abs().quantile(0.25)
    abs_error_q50 = median_absolute_error(y_true, y_pred)
    mean_sqrt_error = mean_squared_error(y_true, y_pred, squared=False)
    mean_abs_error = mean_absolute_error(y_true, y_pred)
    abs_error_q75 = error.abs().quantile(0.75)
    error_max = max_error(y_true, y_pred)

    if verbose:
        print("R-Squared:", round(r2, digits))
        print("25% Abs. Error:", round(abs_error_q25, digits))
        print("Median Abs. Error:", round(abs_error_q50, digits))
        print("Mean Squared Error:", round(mean_sqrt_error, digits))
        print("Mean Abs. Error:", round(mean_abs_error, digits))
        print("75% Abs. Error:", round(abs_error_q75, digits))
        print("Max. Abs. Error:", round(error_max, digits))

    return {"R-Squared": r2,
            "25% Abs. Error": abs_error_q25,
            "Median Abs. Error": abs_error_q50,
            "Mean Squared Error": mean_sqrt_error,
            "Mean Abs. Error": mean_abs_error,
            "75% Abs. Error": abs_error_q75,
            "Max. Abs. Error": error_max}


def confidence_interval(sample, degrees_freedom, confidence_level=0.95, return_score='both'):

    sample_mean = np.mean(sample)
    sse = sem(sample)

    if return_score == 'lower':
        return t.interval(alpha=confidence_level, df=degrees_freedom, loc=sample_mean, scale=sse)[0]

    elif return_score == 'upper':
        return t.interval(alpha=confidence_level, df=degrees_freedom, loc=sample_mean, scale=sse)[1]
    else:
        return t.interval(alpha=confidence_level, df=degrees_freedom, loc=sample_mean, scale=sse)

class BootstrapValidation:

    stats = []
    seeds = {}
    results = {}

    def __init__(self, clf=True, test_size=0.30, n_iterations=100, stratify=None, random_state=0, confidence=0.95):

        self.clf = clf
        self.test_size = test_size
        self.n_iterations = n_iterations
        self.stratify = stratify
        self.random_state = random_state
        self.confidence = confidence

        if self.clf:
            metrics = ["ROC-AUC Score", "Accuracy", "Recall (TPR)", "Precision", "F1-score", "KS-Statistic"]

        else:
            metrics = ["R-Squared", "25% Abs. Error", "Median Abs. Error", "Mean Squared Error",
                       "Mean Abs. Error", "75% Abs. Error", "Max. Abs. Error"]

        self.dmetrics = {metric: [] for metric in metrics}

    def validate(self, X, y, estimator):

        self.seeds = dict(pd.Series(pd.Series(range(0, 10000)).sample(self.n_iterations, random_state=2020).values,
                                    index=range(self.n_iterations)))
        boom = 0
        Xy = pd.concat([X, y], axis=1)
        n_size = int(len(Xy) * (1 - self.test_size))
        for i in range(self.n_iterations):
            perc = round(i * 100 / self.n_iterations, 1)
            print(str(perc) + "% complete...")
            clear_output(wait=True)
            # prepare train and test sets
            train = resample(Xy, n_samples=n_size, stratify=self.stratify, random_state=self.seeds[i])
            test = Xy.loc[[x for x in Xy.index if x not in train.index]]
            # fit model
            estimator.fit(train.iloc[:, :-1], train.iloc[:, -1])
            # evaluate model
            predictions = estimator.predict(test.iloc[:, :-1])

            if self.clf:
                probabilities = estimator.predict_proba(test.iloc[:, :-1])[:, 1]
                metrics_dict = evaluate_classifier(test.iloc[:, -1], predictions, probabilities)
                for metric in metrics_dict:
                    self.dmetrics[metric].append(metrics_dict[metric])
            else:
                metrics_dict = evaluate_regressor(test.iloc[:, -1], predictions)
                if metrics_dict['R-Squared'] == 0:
                    boom += 1
                    continue
                else:
                    for metric in metrics_dict:
                        self.dmetrics[metric].append(metrics_dict[metric])

        for metric in self.dmetrics.keys():

            mean = np.mean(self.dmetrics[metric])
            p = ((1.0 - self.confidence) / 2.0) * 100

            if self.clf:
                lower = max(0.0, np.percentile(self.dmetrics[metric], p))

            else:
                lower = np.percentile(self.dmetrics[metric], p)

            p = (self.confidence + ((1.0 - self.confidence) / 2.0)) * 100

            if self.clf:
                upper = min(1.0, np.percentile(self.dmetrics[metric], p))

            else:
                upper = np.percentile(self.dmetrics[metric], p)

            interval = str(int(self.confidence * 100)) + '%'

            self.results.update({metric: {"Mean": mean,
                                          "Min.": np.min(self.dmetrics[metric]),
                                          "Max.": np.max(self.dmetrics[metric]),
                                          "Lower " + interval: lower,
                                          "Upper " + interval: upper}})

        print()
        print("--------------------Validation Results--------------------")
        print(pd.DataFrame.from_dict(self.results, orient='index'))
        print("----------------------------------------------------------")
        print('Exploded:', boom)

        return self
