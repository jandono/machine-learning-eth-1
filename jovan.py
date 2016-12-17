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
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.metrics import make_scorer
from sklearn.pipeline import  Pipeline
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

path_train = 'data/set_train/mri/'#4mm/'
path_test = 'data/set_test/mri/'#4mm/'

train_size = 278
test_size = 138

cut = {}
cut['cut_x'] = (16, 160) # original 18 158
cut['cut_y'] = (19, 190) # original 19 189
cut['cut_z'] = (11, 155) # original 13 153


x = []
t = []

for i in xrange(0, train_size):
    img = nilearn.image.load_img(path_train + 'mwp1train_' + str(i + 1) + '.nii')
    d = img.get_data()
    d = d[cut['cut_x'][0]:cut['cut_x'][1],  cut['cut_y'][0]:cut['cut_y'][1],  cut['cut_z'][0]:cut['cut_z'][1]]
    x.append(np.ravel(d))

for i in xrange(0, test_size):
    img = nilearn.image.load_img(path_test + 'mwp1test_' + str(i + 1) + '.nii')
    d = img.get_data()
    d = d[cut['cut_x'][0]:cut['cut_x'][1],  cut['cut_y'][0]:cut['cut_y'][1],  cut['cut_z'][0]:cut['cut_z'][1]]
    t.append(np.ravel(d))

t = np.array(t)

y = np.genfromtxt('data/targets.csv', delimiter=',')
y = y[0:train_size]

clf = Pipeline([('f_classif', SelectKBest(f_classif, k=8000)), ('svm', LinearSVC())])
multi_clf = OneVsRestClassifier(clf)
multi_clf.fit(x, y)

results = multi_clf.predict(t)
print results
results = np.ravel(results)
print results

labels = ['gender', 'age', 'health']
f = open('final_sub.csv', 'w+')
n = len(results)
f.write('ID,Sample,Label,Predicted\n')
i = 0
while i < n:
    for ii in xrange(3):
        f.write(str(i) + ',' + str(i/3) + ',' + labels[i % 3] + ',' + str(bool(results[i])) + '\n')
        i += 1

'''y_gender = y[:, 0]
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


clf_rf_gender = RandomForestClassifier(n_estimators=1000, max_depth=10)
clf_ada_gender = AdaBoostClassifier(base_estimator=clf_rf_gender, n_estimators=1000)
clf_ada_gender.fit(x_gender, y_gender)
results_gender = clf_ada_gender.predict(t_gender)
#results_gender = results_gender[:, 1]

clf_rf_age = RandomForestClassifier(n_estimators=1000, max_depth=10)
clf_ada_age = AdaBoostClassifier(base_estimator=clf_rf_age, n_estimators=1000)
clf_ada_age.fit(x_age, y_age)
results_age = clf_ada_age.predict(t_age)
#results_age = results_age[:, 1]

clf_rf_health = RandomForestClassifier(n_estimators=1000, max_depth=10)
clf_ada_health = AdaBoostClassifier(base_estimator=clf_rf_health, n_estimators=1000)
clf_ada_health.fit(x_health, y_health)
results_health = clf_ada_health.predict(t_health)
#results_health = results_health[:, 1]

results = []
results.append(results_gender)
results.append(results_age)
results.append(results_health)
labels = ['gender', 'age', 'health']


f = open('final_sub.csv', 'w+')
n = len(results_gender) * 3
f.write('ID,Sample,Label,Predicted\n')
i = 0
while i < n:
    for ii in xrange(3):
        f.write(str(i) + ',' + str(i/3) + ',' + labels[i % 3] + ',' + str(bool(results[i % 3][i/3])) + '\n')
        i += 1'''

