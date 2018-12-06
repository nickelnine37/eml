import numpy as np
import matplotlib.pyplot as plt
from classifiers import KNNClassifier, LogisticClassifier
from regression import Logistic

def binary_data(N: int, D: int):
    """
    Generate N D-dimensional test points from two multivariate gaussian
    distributions.

    Inputs:
        N int:  The number of points to generate
        D int: The dimensionality of those points
    Returns:
        X (N, D): Design matrix
        y (N, 1): binary vector
    """

    n1 = N // 2; n2 = N - n1
    Xs = []; ys = []

    for i, ni in zip([0, 1], [n1, n2]):
        mean = np.random.normal(size=D)
        cov_seed = np.random.normal(size=(D, D))
        cov = cov_seed.T @ cov_seed
        Xs.append(np.random.multivariate_normal(mean, cov, size=ni))
        ys.append(np.ones((ni, 1)) * i)

    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    data = np.concatenate([Xs, ys], axis=1)

    np.random.shuffle(data)

    return data[:, :-1], data[:, -1].reshape(-1, 1)

def one_hot_data(N: int, D: int, K: int):
    """
    Generate N D-dimensional test points from two multivariate gaussian
    distributions, with corresponding one-hot matrix

    Inputs:
        N int: The number of points to generate
        D int: The dimensionality of those points
        K int: The number of classes
    Returns:
        X (N, D): Design matrix
        y (N, K): one hot matrix
    """

    n1 = N // 2; n2 = N - n1
    Xs = []; ys = []
    Ns = [N // K for _ in range(K - 1)] + [N - (K - 1) * (N // K)]

    for i, ni in enumerate(Ns):
        mean = np.random.normal(size=D)
        cov_seed = np.random.normal(size=(D, D))
        cov = cov_seed.T @ cov_seed
        Xs.append(np.random.multivariate_normal(mean, cov, size=ni))
        y = np.zeros((ni, K))
        y[:, i] = 1
        ys.append(y)

    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    data = np.concatenate([Xs, ys], axis=1)

    np.random.shuffle(data)

    return data[:, :D], data[:, D:]

if __name__ == '__main__':

    # KNNClassifier Test

    N = 1000
    D = 5

    tts = 0.8
    n_train = int(tts * N)
    n_test = N - n_train

    X, y = binary_data(N, D)
    train_X = X[:n_train, :]; test_X = X[n_train:, :]
    train_y = y[:n_train, :]; test_y = y[n_train:, :]

    rgr = LogisticClassifier()
    rgr.fit(train_X, train_y)
    rgr.print_accuracy(test_X, test_y)







