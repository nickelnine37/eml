import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from assertions import is_integer, is_float, is_numpy, has_dims


def sigmoid(x):
    """
    Simple sigmoid

    Inputs:
        x: float or int or numpy array
    Returns:
        σ(x)
    """
    assert any([is_integer(x), is_float(x), is_numpy(x)]), 'x must be an integer or a float or numpy array'

    return (1 + np.exp(-x)) ** -1

class BaseRegressor:

    def fit(self, X, y):
        raise NotImplementedError

    def grad(self, w):
        raise NotImplementedError

    def numerical_grad(self, epsilon=None):
        raise NotImplementedError

    def loss(self, w):
        raise NotImplementedError

    def regress(self, X):
        raise NotImplementedError



class Logistic(BaseRegressor):

    def __init__(self):
        pass

    def fit(self, X, y, method='L-BFGS-B', display_opt=False):
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
        assert y.shape[0] == X.shape[0], 'X and y must have the same 1st dimension length'

        if has_dims(y, 1):
            self.one_class = True
            self.y = y.reshape(-1, 1)
        elif has_dims(y, 2):
            self.one_class = y.shape[1] == 1
            self.y = y
        else:
            raise TypeError('y must be 1 or 2 dimensional')

        self.N, self.M = X.shape

        self.z = 2 * self.y - 1
        self.X = X

        options = {'maxiter': 500, 'disp': display_opt}
        result = minimize(self.loss, np.random.normal(size=self.M), method=method, jac=self.grad, options=options)

        self.w = result.x.reshape(-1, 1)

    def grad(self, w):
        """
        Calculate grad of the loss function wrt the weights

        Returns:
            grad (M, 1): grad wrt each weight dimension
        """
        return -  ( (1 - sigmoid(self.z * self.X @ w.reshape(-1, 1))) * self.z * self.X).sum(0).reshape(-1, 1)

    def numerical_grad(self, epsilon=0.0001):
        """
        Calculate grad of the loss function wrt the weights numerically
        as a check on the grad function

        Returns:
            grad (M, 1): grad wrt each weight dimension
        """

        grads = np.zeros((self.M, 1))

        for i in range(self.M):
            dw = np.zeros_like(self.w)
            dw[i] = epsilon / 2
            grads[i] = (self.loss(self.w + dw) - self.loss(self.w - dw)) / epsilon

        return grads

    def loss(self, w):
        return - np.log(sigmoid(self.z * self.X @ w.reshape(-1, 1))).sum(0)

    def regress(self, X):
        return sigmoid(X @ self.w)
