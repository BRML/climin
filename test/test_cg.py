import scipy
import numpy as np

from climin import ConjugateGradient


A = np.array([[0.1, 0],[0, 100]])
b = np.array([25, 3])
quadratic = lambda x: scipy.dot(x, scipy.dot(A,x)) - scipy.dot(b,x)
quadraticprime = lambda x: scipy.dot(A,x) - b
quadraticandprime = lambda x: (quadratic(x), quadraticprime(x))


def test_cg():
    dim = 2
    wrt = scipy.random.standard_normal((dim,)) * 10 + 5

    opt = ConjugateGradient(wrt, quadratic, quadraticprime)
    for i, info in enumerate(opt):
        if i > 10:
            break
    assert (abs(quadraticprime(wrt)) < 0.01).all(), 'did not find solution'
