import pandas as pd
import numpy as np
from IPython.display import clear_output
from sklearn.metrics import (max_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_error,
                             roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, log_loss)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.utils import resample
from scipy.stats import t, sem, ks_2samp
import time
pd.set_option("display.precision", 3)


def ks_stat(y, yhat):

    """

    This function calculates the Kolmogorov KS-Statistic

    Params
    ------

    y: list-array like
       A list or an array of a  binary or continous variable.
    y_hat: list-array-like



    """
    return ks_2samp(yhat[y == 1], yhat[y != 1]).statistic


def evaluate_classifier(data, target, estimator):

    # Predicting on validation data
    y_pred = estimator.predict(data)
    roc_auc = roc_auc_score(target, y_pred, average='macro')
    accuracy = accuracy_score(target, y_pred, sample_weight=None)
    precision = precision_score(target, y_pred, average='binary')
    recall = recall_score(target, y_pred, average='binary')
    f1 = f1_score(target, y_pred, average='binary')
    ks = ks_stat(target, estimator.predict_proba(data)[:, 1])
    logloss = log_loss(target, estimator.predict_proba(data)[:, 1])

    print("ROC-AUC Score:", round(roc_auc, 3))
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1-Score:", round(f1, 3))
    print("KS-Score:", round(ks, 3))
    print("Log-Loss:", round(logloss,3))


def confidence_interval(sample, degrees_freedom, confidence_level=0.95, return_score='both'):
    sample_mean = np.mean(sample)
    sse = sem(sample)

    if return_score == 'lower':
        return t.interval(alpha=confidence_level, df=degrees_freedom, loc=sample_mean, scale=sse)[0]

    elif return_score == 'upper':
        return t.interval(alpha=confidence_level, df=degrees_freedom, loc=sample_mean, scale=sse)[1]
    else:
        return t.interval(alpha=confidence_level, df=degrees_freedom, loc=sample_mean, scale=sse)


class CrossValidator:

    def __init__(self, clf=True,  cv=5, scoring=None, fit_params=None, groups=None, stratify=False, confidence=0.95,
                 shuffle=True, return_train_score=True, verbose=0, n_jobs=None, random_state=0):

        self.clf = clf
        self.cv = cv
        self.confidence = confidence
        self.return_train_score = return_train_score
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify
        self.groups = groups
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.fit_params = fit_params
        self.verbose = verbose
        self.scores = np.array([])
        self.kfold = {}

    def fit(self):

        if self.stratify:
            self.kfold = StratifiedKFold(n_splits=self.cv,
                                         shuffle=self.shuffle,
                                         random_state=self.random_state)

        else:
            self.kfold = KFold(n_splits=self.cv,
                               shuffle=self.shuffle,
                               random_state=self.random_state)

        return self

    def validate(self, X, y, estimator):

        if (self.scoring is None) & self.clf:
            self.scoring = ('accuracy',
                            'balanced_accuracy',
                            'f1',
                            'precision',
                            'recall',
                            'roc_auc')
        else:
            if (self.scoring is None) & (self.clf is False):
                self.scoring = ('r2',
                                'neg_mean_absolute_error',
                                'neg_median_absolute_error',
                                'neg_mean_squared_error',
                                'neg_root_mean_squared_error',
                                'max_error')

        self.scores = cross_validate(estimator, X, y=y,
                                     groups=self.groups,
                                     scoring=self.scoring,
                                     cv=self.kfold,
                                     n_jobs=self.n_jobs,
                                     verbose=self.verbose,
                                     fit_params=self.fit_params,
                                     pre_dispatch='2*n_jobs',
                                     return_train_score=self.return_train_score,
                                     return_estimator=False)

        scores_df = pd.DataFrame(self.scores)
        if self.return_train_score:
            train_col = [x for x in scores_df.columns if 'train' in x]
            train_report = pd.DataFrame(index=train_col)
            train_report['Mean'] = scores_df[train_col].abs().mean()
            train_report['Std.'] = scores_df[train_col].abs().std()
            train_report['Min.'] = scores_df[train_col].abs().min()
            train_report['Max.'] = scores_df[train_col].abs().max()

            temp_name = 'Lower {:.0f}%'.format(self.confidence * 100)
            train_report[temp_name] = scores_df[train_col].abs().apply(lambda x: confidence_interval(
                x, degrees_freedom=self.cv - 1,  confidence_level=self.confidence, return_score='lower'), axis=0)

            temp_name = 'Upper {:.0f}%'.format(self.confidence * 100)
            train_report[temp_name] = scores_df[train_col].abs().apply(lambda x: confidence_interval(
                x, degrees_freedom=self.cv - 1, confidence_level=self.confidence, return_score='upper'), axis=0)

            print('---------------------------Results on Training Set---------------------------')
            print(train_report)
            print()

        test_col = [x for x in scores_df.columns if 'test' in x]
        test_report = pd.DataFrame(index=test_col)
        test_report['Mean'] = scores_df[test_col].abs().mean()
        test_report['Std.'] = scores_df[test_col].std()
        test_report['Min.'] = scores_df[test_col].abs().min()
        test_report['Max.'] = scores_df[test_col].abs().max()

        temp_name = 'Lower {:.0f}%'.format(self.confidence * 100)
        test_report[temp_name] = scores_df[test_col].abs().apply(lambda x: confidence_interval(
                x, degrees_freedom=self.cv - 1, confidence_level=self.confidence, return_score='lower'), axis=0)

        temp_name = 'Upper {:.0f}%'.format(self.confidence * 100)
        test_report[temp_name] = scores_df[test_col].abs().apply(lambda x: confidence_interval(
                x, degrees_freedom=self.cv - 1, confidence_level=self.confidence, return_score='upper'), axis=0)

        print('--------------------------Results on Validation Set--------------------------')
        print(test_report)

        return self


def search(pipeline, parameters, X_train, y_train, X_test, y_test, optimizer='grid_search', n_iter=None):
    start = time.time()

    if optimizer == 'grid_search':
        grid_obj = GridSearchCV(estimator=pipeline,
                                param_grid=parameters,
                                cv=5,
                                refit=True,
                                return_train_score=False,
                                scoring='accuracy',
                                )
        grid_obj.fit(X_train, y_train, )

    elif optimizer == 'random_search':
        grid_obj = RandomizedSearchCV(estimator=pipeline,
                                      param_distributions=parameters,
                                      cv=5,
                                      n_iter=n_iter,
                                      refit=True,
                                      return_train_score=False,
                                      scoring='accuracy',
                                      random_state=1)
        grid_obj.fit(X_train, y_train, )

    else:
        print('enter search method')
        return

    estimator = grid_obj.best_estimator_
    cvs = cross_val_score(estimator, X_train, y_train, cv=5)
    results = pd.DataFrame(grid_obj.cv_results_)

    print("##### Results")
    print("Score best parameters: ", grid_obj.best_score_)
    print("Best parameters: ", grid_obj.best_params_)
    print("Cross-validation Score: ", cvs.mean())
    print("Test Score: ", estimator.score(X_test, y_test))
    print("Time elapsed: ", time.time() - start)
    print("Parameter combinations evaluated: ", results.shape[0])

    return results, estimator


def evaluate_regressor(target, predictions):

    error = predictions - target

    metrics = {"R-Squared": min(1, max(0, r2_score(target, predictions))),
               "25% Abs. Error": error.abs().quantile(0.25),
               "Median Abs. Error": median_absolute_error(target, predictions),
               "Mean Squared Error": mean_squared_error(target, predictions, squared=False),
               "Mean Abs. Error": mean_absolute_error(target, predictions),
               "75% Abs. Error": error.abs().quantile(0.75),
               "Max. Abs. Error": max_error(target, predictions)}

    return metrics


def evaluate_clf(target, predictions, probabilities):

    metrics = {"ROC-AUC Score": roc_auc_score(target, predictions, average='macro'),
               "Accuracy": accuracy_score(target, predictions, sample_weight=None),
               "Recall (TPR)": recall_score(target, predictions, sample_weight=None),
               "Precision": precision_score(target, predictions, sample_weight=None),
               "F1-score": f1_score(target, predictions, sample_weight=None),
               "KS-Statistic": KS(target, probabilities)}

    return metrics


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
                metrics_dict = evaluate_clf(test.iloc[:, -1], predictions, probabilities)
                for metric in metrics_dict:
                    self.dmetrics[metric].append(metrics_dict[metric])
            else:
                metrics_dict = evaluate_reg(test.iloc[:, -1], predictions)
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
