import itertools

import numpy as np

from climin import Ista


def test_murphy_example():
    np.random.seed(1010)
    w = np.array([[0, -1.67, .13, 0, 0, 1.19, 0, -.04, .33, 0]]).T
    X = np.random.random((20, 10))
    Z = np.dot(X, w)

    f_residual = lambda wrt, x, z: (np.dot(x, w) - z)
    f_prime = lambda wrt, x, z: np.dot(x.T, np.dot(x, wrt) - z) / z.shape[0]
    f_loss = lambda wrt, x, z: ((np.dot(x, w) - z)**2).mean(axis=0).sum()

    wrt = np.zeros(w.shape)
    args = itertools.repeat(((X, Z), {}))
    opt = Ista(wrt, f_loss, f_residual, f_prime, c_l1=1e-9, step_rate=0.1,
               args=args)
    for info in opt:
        if info['n_iter'] == 50:
            break

    assert ((abs(wrt) > 1e-2) == (abs(w) > 1e-2)).all()
