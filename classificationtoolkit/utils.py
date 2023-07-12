
from sklearn import tree
from hyperopt import hp
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression




opts = [
    {
        'name': 'DecisionTree',
        'model': tree.DecisionTreeClassifier,
        'evals': 10,
        'search_space': {
            'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
            'max_depth': hp.choice('max_depth', [None, 2, 3, 4, 5]),
            'criterion': hp.choice('criterion', ['gini', 'entropy', 'log_loss']),
        },

    },
    {
        'name': 'LogisticRegression',
        'model': LogisticRegression,
        'evals': 10,
        'search_space': {
            'penalty': hp.choice('penalty', ['l2', None]),
            'C': hp.choice('C', [x/10 for x in range(1,11)]),
            'fit_intercept':hp.choice('fit_intercept', [True, False])
        }
    },
    {
        'name': 'GradientBoostingClassifier',
        'model': GradientBoostingClassifier,
        'evals': 10,
        'search_space': {
            'learning_rate':hp.choice('learning_rate', [0.1, 0.25, 0.5, 0.75]),
            'n_estimators': hp.choice('n_estimators', [750,  800, 850, 900, 950]),
            'subsample': hp.choice('subsample', [0.1, 0/25, 0.5, 0.75])
        }
    },
    {
        'name': 'XGBoost',
        'model': XGBClassifier,
        'evals': 10,
        'search_space': {
            'min_child_weight': hp.choice('min_child_weight', [1, 5, 10]),
            'gamma': hp.choice('gamma', [0.5, 1, 1.5, 2, 5]),
            'subsample': hp.choice('subsample', [0.6, 0.8, 1.0]),
            'max_depth': hp.choice('max_depth', [3, 4, 5]),
        }
    },
    {
        'name': 'KNN',
        'model': KNeighborsClassifier,
        'evals': 10,
        'search_space': {
            'n_neighbors': hp.choice('n_neighbors', [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        }
    }
]