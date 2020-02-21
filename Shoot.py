# https://scikit-learn.org/stable/modules/clustering.html#
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
#Create classification/regression/clustering/dimentional red methods https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC

import pandas as pd

import GlobalVars as gv


workers = gv.workers
folds = gv.folds
verbose = gv.verbose
global_data_scale = gv.global_data_scale
random_state = gv.random_state

# for future use:
# shot_gun_choke = .5  # how widely do you want to spread from the local default. from 0 to 1 with 0 being the widest spread and 1 being the tightest, the wider the spread the more pellets should be added. the higher the spread the least like you are to fall into a local minina
# shell_pellet_count = 2  # how many values are going to be in each numeric param. ie gamma, C, alpha
# number_of_shells = 1  # how many times you want to try a different fold of the data to. more is better at cost of performance


def classify(X_train, y_train):
    # comment out any classifier that should not be used
    classifiers = [
        (KNeighborsClassifier(), "KNeighborsClassifier", 1 * global_data_scale),
        (SVC(), "SVC", 1 * global_data_scale),
        (SVC(), "RBFSVC", 1 * global_data_scale),
        (LinearSVC(), "LinearSVC", 1 * global_data_scale),
        (GaussianProcessClassifier(), "GaussianProcessClassifier", 1 * global_data_scale),
        (DecisionTreeClassifier(), "DecisionTreeClassifier", 1 * global_data_scale),
        (RandomForestClassifier(),  "RandomForestClassifier", 1 * global_data_scale),
        (MLPClassifier(), "MLPClassifier", 1 * global_data_scale),
        (AdaBoostClassifier(), "AdaBoostClassifier", 1 * global_data_scale),
        (GaussianNB(),  "GaussianNB", 1 * global_data_scale),
        (QuadraticDiscriminantAnalysis(), "QuadraticDiscriminantAnalysis", 1 * global_data_scale)
    ]

    # set the list of the values that should be used in grid search
    params_dict = {
        "KNeighborsClassifier": {
            "n_neighbors": [30, 20, 10, 5],
            "weights": ["uniform", "distance"],
            "algorithm": ["ball_tree", "kd_tree"],
            "leaf_size": [50, 40, 30],
            "p": [1, 2]
        },
        "SVC": {
            "C": [1],
            "kernel": ["poly", "sigmoid"],
            "degree": [2, 3, 4],
            "gamma": ["scale"],
            "coef0": [0.0],
            "shrinking": [True],
            "max_iter": [5000000],
            "class_weight": [None, "balanced"]
        },
        "RBFSVC": {
            "C": [1000000, 10000, 100, 1],
            "kernel": ["rbf"],
            "gamma": ["scale", "auto", 10, 1, .1, .01, .001, .0001, .000001],
            "class_weight": [None, "balanced"]
            # "max_iter": [200000],
        },
        "LinearSVC": {
            "C": [1.2, 1, .5],
            "penalty": ["l1", "l2"],
            "loss": ["hinge", "squared_hinge"],
            "tol": [.001, .0001, .00001],
            "multi_class": ["ovr", "crammer_singer"],
            "class_weight": [None, "balanced"]
        },
        "GaussianProcessClassifier": {
            "max_iter_predict": [200, 100, 50],
            "warm_start": [True, False],
        },
        "DecisionTreeClassifier": {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_features": ["sqrt", "log2", None],
            "max_depth": [None, 100, 50, 5],
        },
        "RandomForestClassifier": {
            "n_estimators": [5, 10, 100, 200],
            "max_depth": [None, 100, 50, 5],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2", None],
        },
        "MLPClassifier": {
            "max_iter": [5000],
            "hidden_layer_sizes": [(10,), (10, 2)],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            # "alpha": [.001, .0001, .00001],
            # "learning_rate": ["constant", "invscaling", "adaptive"],
            # "learning_rate_init": [.01, .001, .0001],
            # "power_t": [.25, .5, .75],
            # "tol": [.001, .0001, .00001],
            # "momentum": [.7, .9, 1.],
            # "nesterovs_momentum": [True, False],
            # "beta_1": [.7, .8, .9],
            # "beta_2": [.9, .99],
            # "epsilon": [.0000001, .00000001, .000000001],
            # "n_iter_no_change": [5, 10, 20],
        },
        "AdaBoostClassifier": {
            "n_estimators": [10, 50, 100],
            "learning_rate": [.5, 1., 1.5],
        },
        "GaussianNB": {
            "var_smoothing": [.000001, .0000001, .00000001, .000000001, .0000000001, .00000000001, .000000000001],
        },
        "QuadraticDiscriminantAnalysis": {
            "tol": [.1, .01, .001, .0001, .00001, .000001],
        }
    }

    for classifier, params, frac in classifiers:
        full = pd.DataFrame(X_train).join(pd.DataFrame(y_train))
        loan_data = full.sample(frac=frac, random_state=random_state)
        X = loan_data.drop("loan_status", axis=1)
        y = loan_data["loan_status"]
        grid = GridSearchCV(classifier, params_dict[params], verbose=verbose, cv=folds, n_jobs=workers)
        grid.fit(X, y)
        yield grid, params
