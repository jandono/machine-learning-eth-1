import numpy as np
import nilearn.image
from sklearn.cross_validation import KFold
from sklearn.metrics import hamming_loss, make_scorer
import fred
import jovan

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

#train_size = 40
#test_size = 20

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

SPLITS = 5

cvor = KFold(len(y), 5)
scores = []
for trains, tests in cvor:
    train_x = x[trains]
    train_y = y[trains]
    test_x = x[tests]

    actual = y[tests]
    #predicted = jovan.fit_predict(train_x, train_y, test_x)
    predicted = fred.fit_predict(train_x, train_y, test_x)[-1].round().astype(int) # take the LAST prediction

    scores.append(hamming_loss(actual, predicted))

print('Scores:')
print(scores)
print('Mean: ' + str(np.mean(scores)) + ' -- Stddev: ' + str(np.std(scores)))
