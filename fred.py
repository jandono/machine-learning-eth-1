import numpy as np
import nilearn.image
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

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


path_train = '../data/set_train/mri/'#4mm/'
path_test = '../data/set_test/mri/'#4mm/'

train_size = 278
test_size = 138

cut = {}
cut['cut_x'] = (16, 160) # original 18 158
cut['cut_y'] = (19, 190) # original 19 189
cut['cut_z'] = (11, 155) # original 13 153


x = []
t = []

for i in range(0, train_size):
    img = nilearn.image.load_img(path_train + 'mwp1train_' + str(i + 1) + '.nii')
    d = img.get_data()
    d = d[cut['cut_x'][0]:cut['cut_x'][1],  cut['cut_y'][0]:cut['cut_y'][1],  cut['cut_z'][0]:cut['cut_z'][1]]
    x.append(np.ravel(d))

for i in range(0, test_size):
    img = nilearn.image.load_img(path_test + 'mwp1test_' + str(i + 1) + '.nii')
    d = img.get_data()
    d = d[cut['cut_x'][0]:cut['cut_x'][1],  cut['cut_y'][0]:cut['cut_y'][1],  cut['cut_z'][0]:cut['cut_z'][1]]
    t.append(np.ravel(d))

x = np.array(x)
t = np.array(t)

y = np.genfromtxt('../data/targets.csv', delimiter=',')
y = y[0:train_size]

# Would like:
# Create 3x selectKbest, fit for every label
# Intrinsic just use that
# Relational classifs transform only the original part

# Every label has its own transformer
BEST = 300
# Intrinsic classifiers
intrinsics = []
for lbl in range(0, 3):
    intrinsics.append(Pipeline([('f_classif', SelectKBest(f_classif, k=BEST)), ('svm', LinearSVC())]))

# Fit every label to its own transformer and add to relational classifiers
relationals = []
for lbl in range(0, 3):
    this_lbl_transformer = SelectKBest(f_classif, k=BEST)
    this_lbl_transformer.fit(x, y[:, lbl])
    relationals.append(Relational(this_lbl_transformer, LinearSVC()))


clf = MultilabelPredictor()
print('==== Fitting... ====')
clf.fit(x, y, intrinsics, relationals)

print('==== Predicting... ====')
all_results = clf.predict(t)
#print(results)

for i, results in enumerate(all_results):
    results = np.ravel(results)
    output_preds(str(i), results)
