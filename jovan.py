# This script produces the final submission as shown on
# the kaggle leaderboard.
#
# It runs correctly if placed next to the folders /src
# and /data. The folder /src contains whatever other
# scripts you need (provided by you). The folder /data
# can be assumed to contain targets.csv and two folders /set_train and
# /set_test which again contain the training and test
# samples respectively (provided by user, i.e. us).
#
# Its output is "final_sub.csv"

import pickle
import numpy as np
import nilearn.image
import nilearn.plotting
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.metrics import make_scorer
from sklearn.pipeline import  Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
# from sklearn.neural_network import multilayer_perceptron, MLPClassifier
# from sklearn.preprocessing import MultiLabelBinarizer

def fit_predict(x, y, t):
    y_gender = y[:, 0]
    y_age = y[:, 1]
    y_health = y[:, 2]

    feature_selector_gender = SelectKBest(score_func=f_classif, k=8000)
    feature_selector_gender.fit(x, y_gender)
    x_gender = feature_selector_gender.transform(x)
    t_gender = feature_selector_gender.transform(t)

    feature_selector_age = SelectKBest(score_func=f_classif, k=8000)
    feature_selector_age.fit(x, y_age)
    x_age = feature_selector_age.transform(x)
    t_age = feature_selector_age.transform(t)

    feature_selector_health = SelectKBest(score_func=f_classif, k=8000)
    feature_selector_health.fit(x, y_health)
    x_health = feature_selector_health.transform(x)
    t_health = feature_selector_health.transform(t)


    clf_gender = SVC(kernel='linear', probability=True)
    # clf_gender = RandomForestClassifier(n_estimators=1000, max_depth=10)
    clf_gender.fit(x_gender, y_gender)
    results_gender = clf_gender.predict(t_gender)
    # clf_ada_gender = AdaBoostClassifier(base_estimator=clf_gender, n_estimators=1000)
    # clf_ada_gender.fit(x_gender, y_gender)
    # results_gender = clf_ada_gender.predict(t_gender)
    #results_gender = results_gender[:, 1]

    # clf_age = SVC(kernel='linear', probability=True)
    clf_age = RandomForestClassifier(n_estimators=1000, max_depth=10)
    clf_age.fit(x_age, y_age)
    results_age = clf_age.predict(t_age)
    # clf_ada_age = AdaBoostClassifier(base_estimator=clf_age, n_estimators=1000)
    # clf_ada_age.fit(x_age, y_age)
    # results_age = clf_ada_age.predict(t_age)
    #results_age = results_age[:, 1]

    # clf_health = SVC(kernel='linear', probability=True)
    clf_health = RandomForestClassifier(n_estimators=1000, max_depth=10)
    clf_health.fit(x_health, y_health)
    results_health = clf_health.predict(t_health)
    #clf_ada_health = AdaBoostClassifier(base_estimator=clf_rf_health, n_estimators=1000)
    #clf_ada_health.fit(x_health, y_health)
    #results_health = clf_ada_health.predict(t_health)
    #results_health = results_health[:, 1]

    results = []
    results.append(results_gender)
    results.append(results_age)
    results.append(results_health)


    return np.array(results).transpose()