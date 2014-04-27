import util
from numpy import *

from sklearn.decomposition import PCA

class ZCA(object):
    def __init__(self, regularization=0.1):
        self.regularization = regularization

    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=0)
        X -= self.mean_
        sigma = cov(X, rowvar=0)
        evals, evecs = linalg.eigh(sigma)
        ievals = sqrt(1.0/evals + self.regularization)
        self.components_ = dot(evecs*ievals, evecs.T)
        return self

    def transform(self, X):
        X -= self.mean_
        return dot(X, self.components_)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        X += self.mean_
        return X

class Preprocessor(object):
    def __init__(self, n_components=None, whiten=True, copy=True):
        self.whiten = whiten
        self.n_components = None
        self.copy = copy
        self.zca = None
        self.pca = PCA(n_components = n_components, whiten=False)
        return

    def fit(self, X, y=None):
        if self.copy and copy: X = X.copy()
        X -= X.mean(axis=0)
        X /= sqrt(X.std(axis=0) + 10)
        X = self.pca.fit_transform(X)
        self.fit_(X)
        return self

    def fit_(self, X):
        if self.whiten:
            self.zca = ZCA()
            self.zca.fit(X)

    def transform(self, X):
        if self.copy: X = X.copy()
        X -= X.mean(axis=0)
        X /= sqrt(X.std(axis=0) + 10)
        X = self.pca.transform(X)
        return self.transform_(X)

    def transform_(self, X):
        if self.whiten:
            return self.zca.transform(X)
        else:
            return X

    def fit_transform(self, X, y=None):
        N_fit = 30000
        if self.copy: X = X.copy()
        X -= X.mean(axis=0)
        X /= sqrt(X.std(axis=0) + 10)
        X = self.pca.fit(X[:N_fit]).transform(X)
        self.fit_(X[:N_fit])
        return self.transform_(X)

    def inverse_transform(self, X):
        X = self.pca.inverse_transform(X)
        return X
