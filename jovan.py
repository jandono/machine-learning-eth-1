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

path_train = 'data/set_train/mri/'#4mm/'
path_test = 'data/set_test/mri/'#4mm/'

train_size = 278
test_size = 138

cut_health = {}
cut_health['cut_x'] = (30, 140)    #original (16, 160)
cut_health['cut_y'] = (40, 160)    #original (19, 190)
cut_health['cut_z'] = (5, 110)     #original (11, 155)

cut_others = {}
cut_others['cut_x'] = (16, 160)
cut_others['cut_y'] = (19, 190)
cut_others['cut_z'] = (11, 155)

x = []
x_health = []
t = []
t_health = []

for i in xrange(0, train_size):
    img = nilearn.image.load_img(path_train + 'mwp1train_' + str(i + 1) + '.nii')
    d = img.get_data()
    d1 = d[cut_others['cut_x'][0]:cut_others['cut_x'][1],  cut_others['cut_y'][0]:cut_others['cut_y'][1],  cut_others['cut_z'][0]:cut_others['cut_z'][1]]
    # d2 = d[cut_health['cut_x'][0]:cut_health['cut_x'][1],  cut_health['cut_y'][0]:cut_health['cut_y'][1],  cut_health['cut_z'][0]:cut_health['cut_z'][1]]
    x.append(np.ravel(d1))
    # x_health.append(np.ravel(d2))

for i in xrange(0, test_size):
    img = nilearn.image.load_img(path_test + 'mwp1test_' + str(i + 1) + '.nii')
    d = img.get_data()
    d1 = d[cut_others['cut_x'][0]:cut_others['cut_x'][1],  cut_others['cut_y'][0]:cut_others['cut_y'][1],  cut_others['cut_z'][0]:cut_others['cut_z'][1]]
    # d2 = d[cut_health['cut_x'][0]:cut_health['cut_x'][1],  cut_health['cut_y'][0]:cut_health['cut_y'][1],  cut_health['cut_z'][0]:cut_health['cut_z'][1]]
    t.append(np.ravel(d1))
    # t_health.append(np.ravel(d2))

t = np.array(t)
# t_health = np.array(t_health)

y = np.genfromtxt('data/targets.csv', delimiter=',')
y = y[0:train_size]
# y = y.tolist()
#
# outliers = [[0., 1., 0.], [1., 1., 0.]]
# single_to_multilabel = [[0., 0., 0.], [0., 0., 1.], [0., 1., 1.], [1., 0., 0.], [1., 0., 1.], [1., 1., 1.]]
# single_to_multilabel.append(outliers[0])
# single_to_multilabel.append(outliers[1])
# #print single_to_multilabel
#
# multilabel_to_single = {}
# for i in xrange(8):
#     multilabel_to_single[str(single_to_multilabel[i])] = i
#
# for i in xrange(train_size):
#     y[i] = multilabel_to_single[str(y[i])]
# print y

# feature_selector = SelectKBest(score_func=f_classif, k=300)
# feature_selector.fit(x, y[:, 1])
# x = feature_selector.transform(x)
# t = feature_selector.transform(t)
# nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
# nn.fit(x, y)
# results = nn.predict(t)
# print results
# # print results.shape

# svm = SVC(kernel='linear', probability=True) #0.12077
# clf = Pipeline([('f_classif', SelectKBest(f_classif, k=300)), ('svm', svm)])
# multi_clf = OneVsRestClassifier(clf)
# multi_clf.fit(x, y)
# results = multi_clf.predict(t)
#
# labels = ['gender', 'age', 'health']
# f = open('final_sub.csv', 'w+')
# n = len(results) * 3
# f.write('ID,Sample,Label,Predicted\n')
#
# i = 0
# while i < n:
#     # result = single_to_multilabel[results[i/3]]
#     for ii in xrange(3):
#         f.write(str(i) + ',' + str(i/3) + ',' + labels[i % 3] + ',' + str(bool(results[i / 3][i % 3])) + '\n')
#         # f.write(str(i) + ',' + str(i/3) + ',' + labels[ii] + ',' + str(bool(result[ii])) + '\n')
#         i += 1

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
labels = ['gender', 'age', 'health']


f = open('final_sub.csv', 'w+')
n = len(results_gender) * 3
f.write('ID,Sample,Label,Predicted\n')
i = 0
while i < n:
    for ii in xrange(3):
        f.write(str(i) + ',' + str(i/3) + ',' + labels[i % 3] + ',' + str(bool(results[i % 3][i/3])) + '\n')
        i += 1