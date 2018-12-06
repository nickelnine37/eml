import numpy as np
from assertions import is_integer, is_numpy, has_dims
from regression import Logistic

class BaseClassifier:

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def accuracy(self, X, y):
        raise NotImplementedError


class KNNClassifier(BaseClassifier):

    def __init__(self, k=5, weights='uniform'):

        assert is_integer(k), 'K must be an integer'
        assert weights in ['uniform', 'distance'], "weights must be one of ['uniform', 'distance']"

        self.k = k
        self.weights = weights
        self.fitted = False

    def pairwise_distances(self, X1, X2):
        """
        Return an (N1, N2) matrix where the element (i, j) is the euclidean
        distance between the vectors represented by the ith row of
        X1 and the jth row of X2.

        Inputs:
            X1 (N1, M): A matrix with N1 M-dimensional vectors as rows
            X2 (N2, M): A matrix with N2 M-dimensional vectors as rows

        Returns:
            D  (N1, N2): The pairwise euclidean distances
        """
        assert is_numpy(X1), 'X1 must be a numpy array'
        assert is_numpy(X2), 'X2 must be a numpy array'
        assert has_dims(X1, 2), 'X1 must be 2-dimensional'
        assert has_dims(X2, 2), 'X2 must be 2-dimensional'

        return (- 2 * X1 @ X2.T + (X1 ** 2).sum(1)[:, None] + (X2 ** 2).sum(1)) ** 0.5

    def fit(self, X, y):
        """
        Fit data

        Inputs:
            X (N, M)        : A design matrix with N M-dimensional vectors as rows
            y (N,) OR (N, D): The output labels. Can be binary vector of 0's and 1's
                              or True's and False's or a one-hot encoding of D classes
        """

        assert is_numpy(X), 'X must be a numpy array'
        assert is_numpy(y), 'y must be a numpy array'
        assert has_dims(X, 2), 'X must be 2-dimensional'

        if has_dims(y, 1):
            self.one_class = True
            self.y = y.reshape(-1, 1)
        elif has_dims(y, 2):
            self.one_class = y.shape[1] == 1
            self.y = y
        else:
            raise TypeError('y must be 1 or 2 dimensional')

        self.X = X
        self.fitted = True

    def predict(self, X):
        """
        Predict on new design matrix X

        Inputs:
            X (N, M)        : A design matrix with N M-dimensional vectors as rows
        """

        assert self.fitted, 'Fit data before predicting'
        assert is_numpy(X), 'X must be a numpy array'
        assert has_dims(X, 2), 'X must be 2-dimensional'

        if self.one_class:

            distances = self.pairwise_distances(self.X, X)
            sortd = np.argsort(distances, axis=0)[:self.k, :]
            labels = self.y[sortd.T]

            if self.weights == 'uniform':
                guess = np.mean(labels, axis=1) >= 0.5
                return guess

            elif self.weights == 'distance':
                ws = np.nan_to_num(1 / np.take_along_axis(distances, sortd, axis=0).T)
                ws = ws / ws.sum(1)[:, None]
                guess = (labels * ws).sum(1) >= 0.5
                return guess

        else:
            raise NotImplementedError('this hasnt been done yet... ') #TODO implement one-hot encoding for KNN

        raise ValueError('Weights set incorrectly')

    def accuracy(self, X, y):
        return 1 - ((self.predict(X).astype(int) - y.astype(int)) ** 2).sum() / len(y)

    def print_accuracy(self, X, y):
        acc = self.accuracy(X, y)
        print('Accuracy: {}%'.format(100 * acc))


class LogisticClassifier(Logistic):

    def __init__(self):
        super().__init__()

    def predict(self, X):
        return (self.regress(X) > 0.5).astype(int)

    def accuracy(self, X, y):
        return 1 - ((self.predict(X).astype(int) - y.astype(int)) ** 2).sum() / len(y)

    def print_accuracy(self, X, y):
        acc = self.accuracy(X, y)
        print('Accuracy: {}%'.format(100 * acc))
