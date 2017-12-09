import numpy as np


class Kernel(object):
    """Check kernels here https://en.wikipedia.org/wiki/Support_vector_machine"""
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)
        #return lambda X: np.dot(X.T, X)

    @staticmethod
    def gaussian(sigma):
        return lambda x, y: np.exp(-np.dot(x-y, (x-y).T)/(2*sigma**2))
        #return lambda X: np.exp(np.array([X-x[:,np.newaxis] for x in X])/(2*sigma**2))
    
    