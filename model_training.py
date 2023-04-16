import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import traceback

models = {"Logistic Regression": LogisticRegression(),
          "Gradient Boost Forest": XGBClassifier(),
          "Support Vector": SVC()}

param_grid = {"Logistic Regression": {

    "C": np.logspace(-5, 5, 10),
    "max_iter": [10000]
},

    "Gradient Boost Forest": {

        "learning_rate": (0.01, 0.05, 0.10, 0.15),
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 5, 15],
        "gamma": [0.0, 0.1, 0.2],

    },

    "Support Vector": {

        'kernel': ['linear', 'poly', 'rbf'],
        'C': np.logspace(-5, 5, 10)

    }

}


def model_evaluation(X, y, models, param_grid, cv=5):
    try:

        results = {}

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1,
                                                                            random_state=42, stratify=y)
        y_train = np.array(y_train).ravel()
        y_test = np.array(y_test).ravel()

        for i in range(0, len(list(models))):
            model_keys, model_classes = list(models.keys()), list(models.values())
            model = model_classes[i]
            params = param_grid[model_keys[i]]

            grid_search = model_selection.GridSearchCV(model, params, cv=5)
            grid_search.fit(X, y)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            roc_auc_train = metrics.roc_auc_score(y_train, model.predict(X_train))
            roc_auc_test = metrics.roc_auc_score(y_test, model.predict(X_test))

            results[model_keys[i]] = (roc_auc_train,roc_auc_test)

        return results

    except Exception as e:
        print(traceback.format_exc())
