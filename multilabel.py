import numpy as np
import itertools
from sklearn import cross_validation
from sklearn.metrics import hamming_loss, make_scorer

hamming_scorer = make_scorer(hamming_loss, greater_is_better=False)

class MultilabelPredictor():

    def __init__(self):
        self.intrinsic = []
        self.relational = []
        self.order = None

    '''
        X: data instances (shape n_samples x n_dimensions)
        y: multilabel data (shape n_samples x n_labels)
        intrinsic_classifiers: array of classifiers (len: n_labels)
        relational_classifiers: also n_labels classifiers, but those should also
        implement their own transform function. The data will first be transformed
        by that method, then the predict method will be called with the additional
        features.
    '''
    def fit(self, X, y, intrinsic_classifiers, relational_classifiers):

        mean_intrinsic_scores = []

        for label, clas in enumerate(intrinsic_classifiers):
            # Train intrinsic classifiers: they are normal binary classifiers`
            # that just look at the data and classify one label.
            clas.fit(X, y[:, label])
            self.intrinsic.append(clas)

            cv = cross_validation.cross_val_score(clas, X, y[:, label], cv=5, scoring=hamming_scorer)
            mean_intrinsic_scores.append(np.mean(cv))

        print("=== Mean intrinsic scores: ", mean_intrinsic_scores)

        mean_relational_scores = []

        for label, clas in enumerate(relational_classifiers):
            # Train relational classifiers: in addition to the data, they look
            # at the other labels (they still classify one label).

            XT = clas.transform(X)
            X_relational = np.hstack(itertools.chain((XT,), self.all_but_one(y, label)))
            clas.fit(X_relational, y[:, label])
            self.relational.append(clas)

            cv = cross_validation.cross_val_score(clas, X_relational, y[:, label], cv=5, scoring=hamming_scorer)
            mean_relational_scores.append(np.mean(cv))

        print("=== Mean relational scores: ", mean_relational_scores)

        improv = np.array(mean_relational_scores) - np.array(mean_intrinsic_scores)
        self.order = list(np.argsort(improv))
        self.order.reverse()

    def predict(self, Z):
        pred = np.zeros(shape=(len(Z), len(self.intrinsic)))

        # How many entries changed in the last iteration
        change_amount = len(Z) * len(self.intrinsic)

        print('Predicting intrinsically')
        for label in range(len(self.intrinsic)):
            pred[:, label] = self.intrinsic[label].predict_proba(Z)[:, 1]

        print('Predicting with relations')
        ITER = 0

        # Pre-bake the transformed data for every relational classifier
        all_Z_transformed = []
        for rel_clf in self.relational:
            ZT = rel_clf.transform(Z)
            all_Z_transformed.append(ZT)

        all_preds = [pred.copy()]

        while change_amount > 0: # Totally arbitrary
            print('Iteration ' + str(ITER))
            ITER += 1
            change_amount = 0

            print (self.order)
            for label in self.order:
                print('Relationally predicting', label)
                # Obtain data for this classifier
                thisZ = all_Z_transformed[label]
                Z_relational = np.hstack(itertools.chain((thisZ,), self.all_but_one(pred, label)))
                pred_label = self.relational[label].predict_proba(Z_relational)[:, 1]

                # Compute amount of change for this label
                equal = pred_label.round() == pred[:, label].round()
                change_amount += len(equal) - len(equal[equal])

                # Update the predictions
                pred[:, label] = pred_label

            print('Change: ' + str(change_amount))
            all_preds.append(pred.copy())

        return all_preds

    '''
    Returns the opposite of y[:, col] (i.e. all columns except col)
    as a generator.
    '''
    def all_but_one(self, y, col):
        for i in range(y.shape[1]):
            if (i == col):
                continue
            yield y[:, i].reshape(len(y), 1)


# A convenience class for the relational classifiers
class Relational():

    '''
    Note: the transformer here should have already been fitted
    '''
    def __init__(self, transformer, predictor):
        self.transformer = transformer
        self.predictor = predictor

    def transform(self, X):
        return self.transformer.transform(X)

    '''
    Fit on the already-transformed data X and the labels y
    '''
    def fit(self, X, y):
        return self.predictor.fit(X, y)

    '''
    Predict with proba, but round all of them.
    '''
    def predict(self, Z):
        return self.predict_proba(Z)[:, 1].round().astype(int)

    '''
    Predict from the already-transformed data Z
    '''
    def predict_proba(self, Z):
        return self.predictor.predict_proba(Z)

    def get_params(self, deep=True):
         if deep:
             return self.predictor.get_params(deep) | self.transformer.get_params(deep)
         else:
             return { 'transformer': self.transformer, 'predictor': self.predictor }