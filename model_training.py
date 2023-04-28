"""
Contains all neccessary utilities for model training.
"""
import sys

import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from exception import ProjectException
from logger import logging


# Dictionary of models to be evaluated.
models = {"Logistic Regression": LogisticRegression(),
          "Gradient Boost Forest": XGBClassifier(),
          #"Support Vector": SVC()
         }

# Parameter grid of models to be evaluated.
param_grid = {"Logistic Regression": {

    "C": np.logspace(-5, 5, 10),
    "max_iter": [10000]
},

    "Gradient Boost Forest": {

        "learning_rate": (0.01, 0.05, 0.10),
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [5, 10, 15],
        "gamma": [0.0, 0.1, 0.2],

    },

#     "Support Vector": {

#         'kernel': ['linear', 'poly', 'rbf'],
#         'C': np.logspace(-5, 5, 10)

#     }

}


def model_evaluation(X, y, models, param_grid, cv=5, test_size=0.1):
    """
    Evaluates a dictionary of models using GridSearchCV using ROC AUC as its criteria.
    X: Feature data
    y: target class vector
    models: dictionary of model classes
    param_grid: grid of hyper-parameters being searched for each model.
    cv: Number of cross validation folds
    """

    try:
        best_model = None
        best_params = None
        best_roc_test = 0
        results = {}

        # Splitting data into train/test sets
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= test_size,
                                                                            random_state=42, stratify=y)
        y_train = np.array(y_train).ravel()
        y_test = np.array(y_test).ravel()

        # Iterating through all models inside dictionary
        for i in range(0, len(list(models))):

            # Splitting model names and class constructors
            model_keys, model_classes = list(models.keys()), list(models.values())
            model = model_classes[i]
            params = param_grid[model_keys[i]]

            # Performing GridSearchCV using current model and parameter grid.
            grid_search = model_selection.GridSearchCV(model, params, cv=cv)
            grid_search.fit(X, y)

            # Build the best performing model
            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            # Test and Train metrics using ROC AUC
            roc_auc_train = metrics.roc_auc_score(y_train, model.predict(X_train))
            roc_auc_test = metrics.roc_auc_score(y_test, model.predict(X_test))

            # Saving train and test results to a dictionary
            results[model] = (roc_auc_train,roc_auc_test)

            # Tracking best model
            if roc_auc_test > best_roc_test:
                best_roc_test = roc_auc_test
                best_model = model
                best_params = grid_search.best_params_

        return results, best_model, best_params

    except Exception as e:
        logging.info(str(ProjectException(e,sys)))
