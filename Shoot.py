# https://scikit-learn.org/stable/modules/clustering.html#
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
#Create classification/regression/clustering/dimentional red methods https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, HuberRegressor, TheilSenRegressor, \
    RANSACRegressor, PassiveAggressiveRegressor, ARDRegression, BayesianRidge, OrthogonalMatchingPursuit, Lars, \
    ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC, NuSVR, SVR, LinearSVR

import pandas as pd
import numpy as np

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


def regress(X_train, y_train):
    # comment out any classifier that should not be used
    classifiers = [
        (SGDRegressor(), "SGDRegressor", 1 * global_data_scale),
        (LinearRegression(), "LinearRegression", 1 * global_data_scale),
        (Ridge(), "Ridge", 1 * global_data_scale),
        (Lasso(), "Lasso", 1 * global_data_scale),
        (ElasticNet(), "ElasticNet", 1 * global_data_scale),
        (Lars(), "Lars", 1 * global_data_scale),
        (OrthogonalMatchingPursuit(), "OrthogonalMatchingPursuit", 1 * global_data_scale),
        (BayesianRidge(), "BayesianRidge", 1 * global_data_scale),
        (ARDRegression(), "ARDRegression", 1 * global_data_scale),
        ### NOTE the scoring might be different of PassiveAggressiveRegressor
        (PassiveAggressiveRegressor(), "PassiveAggressiveRegressor", 1 * global_data_scale),
        ### NOTE the scoring might be different of RANSACRegressor
        (RANSACRegressor(), "RANSACRegressor", 1 * global_data_scale),
        (TheilSenRegressor(), "TheilSenRegressor", 1 * global_data_scale),
        (HuberRegressor(), "HuberRegressor", 1 * global_data_scale),
        (DecisionTreeRegressor(), "DecisionTreeRegressor", 1 * global_data_scale),
        (GaussianProcessRegressor(), "GaussianProcessRegressor", 1 * global_data_scale),
        (MLPRegressor(), "MLPRegressor", 1 * global_data_scale),
        (KNeighborsRegressor(), "KNeighborsRegressor", 1 * global_data_scale),
        (RadiusNeighborsRegressor(), "RadiusNeighborsRegressor", 1 * global_data_scale),
        (SVR(), "SVR", 1 * global_data_scale),
        (NuSVR(), "NuSVR", 1 * global_data_scale),
        (LinearSVR(), "LinearSVR", 1 * global_data_scale),
        (KernelRidge(), "KernalRidge", 1 * global_data_scale),
        (IsotonicRegression(), "IsotonicRegression", 1 * global_data_scale)
    ]

    # set the list of the values that should be used in grid search
    params_dict = {
        "SGDRegressor": {
            "penalty": ["l2", "l1"],
            "alpha": [.001, .0001, .00001],
            "l1_ratio": [.15, .2, .25],
            "fit_intercept": [True, False],
            "max_iter": [1000],
            "shuffle": [True, False],
            "epsilon": [.05, .1, .2],
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "eta0": [.005, .01, .02],
            "power_t": [.2, .25, .3]
        },
        "LinearRegression": {
            "fit_intercept": [True, False],
            "normalize": [True, False]
        },
        "Ridge": {
            "alpha": [.8, 1., 1.2],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "tol": [.01, .001, .0001],
            "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
        },
        "Lasso": {
            "alpha": [.8, 1., 1.2],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "positive": [True, False],
            "precompute": [True, False]
        },
        "ElasticNet": {
            "alpha": [.8, 1., 1.2],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "precompute": [True, False],
            "positive": [True, False],
            "selection": ["cyclic", "random"]
        },
        "Lars": {
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "precompute": [True, False],
            "n_nonzero_coefs": [np.inf]
        },
        "OrthogonalMatchingPursuit": {
            "n_nonzero_coefs": [np.inf, None],
            "precompute": [True, False],
            "fit_intercept": [True, False],
            "normalize": [True, False]
        },
        "BayesianRidge": {
            "tol": [.01, .001, .0001],
            "alpha_1": [1e-5, 1e-6, 1e-7],
            "alpha_2": [1e-5, 1e-6, 1e-7],
            "lambda_1": [1e-5, 1e-6, 1e-7],
            "lambda_2": [1e-5, 1e-6, 1e-7],
            "fit_intercept": [True, False],
            "normalize": [True, False]
        },
        "ARDRegression": {
            "tol": [.01, .001, .0001],
            "alpha_1": [1e-5, 1e-6, 1e-7],
            "alpha_2": [1e-5, 1e-6, 1e-7],
            "lambda_1": [1e-5, 1e-6, 1e-7],
            "lambda_2": [1e-5, 1e-6, 1e-7],
            "threshold_lambda": [1000, 10000, 100000],
            "fit_intercept": [True, False],
            "normalize": [True, False]
        },
        "PassiveAggressiveRegressor": {
            "C": [.8, 1., 1.2 ],
            "tol": [1e-2, 1e-3, 1e-4],
            "n_iter_no_change": [3, 5, 8],
            "shuffle": [True, False],
            "average": [True, False]
        },
        "RANSACRegressor": {
            "base_estimator": [LinearRegression()]
        },
        "TheilSenRegressor": {
            "max_subpopulation": [1e3, 1e4, 1e5],
            "tol": [1e-2, 1e-3, 1e-4]
        },
        "HuberRegressor": {
            "epsilon": [1.1, 1.35,  1.5],
            "alpha": [1e-3, 1e-4, 1e-5],
            "warm_start": [True, False],
            "fit_intercept": [True, False],
            "": [1e-4, 1e-5, 1e-6]
        },
        "DecisionTreeRegressor": {
            "criterion": ["mse", "friedman_mse", "mae"],
            "splitter": ["best", "random"],
            "min_samples_split": [2, 3],
            "min_samples_leaf": [1, 2],
            "min_weight_fraction_leaf": [.0],
            "max_features": ["auto", "sqrt", "log2"],
            "min_impurity_split": [1e-6, 1e-7, 1e-8]
        },
        "GaussianProcessRegressor": {
            "alpha": [1e-8, 1e-10, 1e-12],
            "optimizer": ["fmin_l_bfgs_b"],
            "normalize_y": [True, False]
        },
        "MLPRegressor": {
            "hidden_layer_sizes": [(100,)],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": [1e-3, 1e-4, 1e-5],
            # "learning_rate": ["constant", "invscaling", "adaptive"],
            # "learning_rate_init": [1e-2, 1e-3, 1e-4],
            # "power_t": [.3, .5, .8],
            # "shuffle": [True, False],
            # "tol": [1e-3, 1e-4, 1e-5],
            # "momentum": [.8, .9, .99],
            # "beta_1": [.8, .9, .99],
            # "beta_2": [.999],
            # "epsilon": [1e-7, 1e-8, 1e-9],
            # "n_iter_no_change": [10],
            # "max_fun": [15000]
        },
        "KNeighborsRegressor": {
            "n_neighbors": [20, 10, 5, 3],
            "weights": ["uniform", "distance"],
            "algorithm": ["ball_tree", "kd_tree", "brute"],
            "leaf_size": [20, 30, 40],
            "p": [1, 2]
        },
        "RadiusNeighborsRegressor": {
            "radius": [.8, 1, 1.2],
            "n_neighbors": [20, 10, 5, 3],
            "weights": ["uniform", "distance"],
            "algorithm": ["ball_tree", "kd_tree", "brute"],
            "leaf_size": [20, 30, 40],
            "p": [1, 2]
        },
        "SVR": {
            "kernel": ["poly", "rbf", "sigmoid"],
            "degree": [2, 3, 5],
            "gamma": ["scale", "auto"],
            "coef0": [.0],
            "tol": [1e-2, 1e-3, 1e-4],
            "C": [.8, .1, 1.2],
            "epsilon": [.08, .1, .12],
            "shrinking": [True, False],
            "max_iter": [-1]
        },
        "NuSVR": {
            "nu": [.2, .5, .8],
            "C": [.8, .1, 1.2],
            "kernel": ["poly", "rbf", "sigmoid"],
            "degree": [2, 3, 5],
            "gamma": ["scale", "auto"],
            "coef0": [.0],
            "shrinking": [True, False],
            "tol": [1e-2, 1e-3, 1e-4],
            "max_iter": [-1]
        },
        "LinearSVR": {
            "epsilon": [.0],
            "tol": [1e-3, 1e-4, 1e-5],
            "C": [.8, .1, 1.2],
            "fit_intercept": [True, False],
            "dual": [True, False],
            "intercept_scaling": [.8, 1., 1.2]
        },
        "KernelRidge": {
            "coef0": [.8, 1, 1.2],
            "degree": [2, 3, 5],
        },
        "IsotonicRegression": {
            "increasing": [True, False],
        }
    }

    for model, params, frac in classifiers:
        full = pd.DataFrame(X_train).join(pd.DataFrame(y_train))
        loan_data = full.sample(frac=frac, random_state=random_state)
        X = loan_data.drop("loan_status", axis=1)
        y = loan_data["loan_status"]
        grid = GridSearchCV(model, params_dict[params], verbose=verbose, cv=folds, n_jobs=workers)
        grid.fit(X, y)
        yield grid, params