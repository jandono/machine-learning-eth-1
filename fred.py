import numpy as np
import nilearn.image
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from multilabel import *

def output_preds(name, results):
    labels = ['gender', 'age', 'health']
    f = open('/cluster/home/flafranc/mlp3/code/sub' + name + '.csv', 'w+')
    n = len(results)
    f.write('ID,Sample,Label,Predicted\n')
    i = 0
    while i < n:
        for ii in range(3):
            f.write(str(i) + ',' + str(int(i/3)) + ',' + labels[i % 3] + ',' + str(bool(results[i])) + '\n')
            i += 1


BEST = 8000

def fit_predict(x, y, t):

    # ---- Intrinsic classifiers ----
    intrinsics = [
        # Gender
        Pipeline([('f_classif', SelectKBest(f_classif, k=BEST)),
                    ('svm', SVC(kernel='linear', probability=True))]),

        # Age
        Pipeline([('f_classif', SelectKBest(f_classif, k=BEST)),
                    ('rfor', RandomForestClassifier(n_estimators=1000, max_depth=10))]),

        # Health
        Pipeline([('f_classif', SelectKBest(f_classif, k=BEST)),
                    ('rfor', RandomForestClassifier(n_estimators=1000, max_depth=10))])
    ]


    # ---- Relational classifiers ----
    relationals = []

    gender_selector = SelectKBest(f_classif, k=BEST)
    gender_selector.fit(x, y[:, 0])
    gender_classif = Relational(gender_selector, SVC(kernel='linear', probability=True))
    relationals.append(gender_classif)

    age_selector = SelectKBest(f_classif, k=BEST)
    age_selector.fit(x, y[:, 1])
    age_classif = Relational(age_selector, RandomForestClassifier(n_estimators=1000, max_depth=10))
    relationals.append(age_classif)

    health_selector = SelectKBest(f_classif, k=BEST)
    health_selector.fit(x, y[:, 2])
    health_classif = Relational(health_selector, RandomForestClassifier(n_estimators=1000, max_depth=10))
    relationals.append(health_classif)


    clf = MultilabelPredictor()
    print('==== Fitting... ====')
    clf.fit(x, y, intrinsics, relationals)

    print('==== Predicting... ====')
    all_results = clf.predict(t)
    #print(results)

    return all_results

    '''
    for i, results in enumerate(all_results):
        results = np.ravel(results).round().astype(int)
        output_preds(str(i), results)
    '''