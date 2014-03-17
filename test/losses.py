# -*- coding: utf-8 -*-


import numpy as np
from scipy.optimize import rosen, rosen_der


def sigmoid(x):
    x = x.clip(-50, 50)
    return 1 / 1 + np.exp(-x)


class LogisticRegression(object):

    def __init__(self, n_inpt=5, n_classes=3, n_samples=10, seed=12345):
        self.n_inpt = n_inpt
        self.n_classes = n_classes
        self.n_samples = n_samples

        np.random.seed(seed)
        self.pars = np.random.standard_normal(n_inpt * n_classes + n_classes)
        self.X, self.Z = self.make_data()

    def make_data(self):
        xs = []
        zs = []
        for i in range(self.n_classes):
            x = np.random.standard_normal((self.n_samples, self.n_inpt))
            # Make somehow sure that they are far away from each other.
            x += 5 * i
            z = np.zeros((self.n_samples, self.n_classes))
            z[:, i] = 1
            xs.append(x)
            zs.append(z)
        X = np.vstack(xs)
        Z = np.vstack(zs)
        return X, Z

    def predict(self, wrt, inpt):
        n_weights = self.n_inpt * self.n_classes
        W = wrt[:n_weights].reshape((self.n_inpt, self.n_classes))
        b = wrt[n_weights:]
        sWXb = sigmoid(np.dot(inpt, W) + b)
        return sWXb / sWXb.sum(axis=1)[:, np.newaxis]

    def f(self, wrt, inpt, target):
        prediction = self.predict(wrt, inpt)
        log_pred = np.log(prediction)
        return -(log_pred * target).mean()

    def fprime(self, wrt, inpt, target):
        prediction = self.predict(wrt, inpt)
        d_f_d_W = np.dot(inpt.T, target - prediction)
        d_f_d_b = (target - prediction).sum(axis=0)

        d_f_d_all = np.concatenate((d_f_d_W.flatten(), d_f_d_b))

        return d_f_d_all / prediction.shape[0]

    def f_Hp(self, pars, p, inpt, target):
        eps = 1E-6
        deriv = self.fprime(pars, inpt, target)
        offseted = self.fprime(pars + p * eps, inpt, target)
        return (offseted - deriv) / eps

    def score(self):
        return self.f(self.pars, self.X, self.Z)

    def solved(self, tolerance=0.1):
        return self.score() - tolerance < 0


class Quadratic(object):

    H = np.array([[1, 0],[0, 100]])
    b = np.zeros(2)

    def __init__(self):
        self.pars = np.random.standard_normal(2) + 5

    def f(self, x):
        return 0.5 * np.dot(x, np.dot(self.H, x)) - np.inner(x, self.b)

    def fprime(self, x):
        return np.dot(self.H, x) - self.b

    def f_Hp(self, pars, p):
        return np.dot(self.H, p)

    def solved(self, tolerance=0.01):
        return (abs(self.fprime(self.pars)) < tolerance).all()


class BigQuadratic(object):


    def __init__(self, dim, seed=101010):
        np.random.seed(seed)
        self.pars = np.random.standard_normal(dim) + 5

        L = np.random.standard_normal((dim, dim))
        self.H = np.dot(L, L.T)
        self.b = np.zeros(dim)

    def f(self, x):
        return 0.5 * np.dot(x, np.dot(self.H, x)) - np.inner(x, self.b)

    def fprime(self, x):
        return np.dot(self.H, x) - self.b

    def f_Hp(self, pars, p):
        return np.dot(self.H, p)

    def solved(self, tolerance=0.01):
        return (abs(self.fprime(self.pars)) < tolerance).all()


class Rosenbrock(object):

    def __init__(self):
        self.pars = np.zeros(2)

    def f(self, x):
        return rosen(x)

    def fprime(self, x):
        return rosen_der(x)

    def f_Hp(self, pars, p):
        eps = 1E-6
        return (self.fprime(pars + p * eps) - self.fprime(pars)) / eps

    def solved(self, tolerance=0.1):
        return abs(self.pars - [1, 1]).mean() < tolerance
