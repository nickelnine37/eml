import numpy as np

def is_integer(n):
    """
    Determine whether n is an integer

    Intputs:
        n: An object of any type
    Returns
        truth: Boolean inicating whether n is an integer
    """
    try:
        int(n)
    except TypeError:
        return False

    return int(n) == n

def is_float(n):
    """
    Determine whether n is a float

    Intputs:
        n: An object of any type
    Returns
        truth: Boolean inicating whether n is a float
    """
    try:
        float(n)
    except TypeError:
        return False

    return type(n) is float

def is_numpy(A):
    """
    Determine whether A is a numpy array

    Intputs:
        A: An object of any type
    Returns
        truth: Boolean inicating whether A is a numpy array
    """

    return type(A) is np.ndarray

def has_dims(A: np.ndarray, n: int):
    """
    Determine whether the numpy array A has the expected dimensions

    Intputs:
        A: A numpy array
        n: the expected number of dimensions
    Returns
        truth: Boolean inicating whether A has the expexcted dimensions
    """
    assert is_integer(n), 'n must be an integer'
    assert is_numpy(A), 'A must be a numpy array'

    return len(A.shape) == n

