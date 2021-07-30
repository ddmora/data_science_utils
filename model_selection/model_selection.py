import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import time


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
