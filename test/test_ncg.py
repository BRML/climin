import scipy
from scipy.optimize import rosen, rosen_der
import numpy as np

from climin import NonlinearConjugateGradient


A = np.array([[1, 0],[0,100]])
b = np.array([25, 3])
quadratic = lambda x: 0.5 * scipy.dot(x, scipy.dot(A,x)) - scipy.dot(b,x)
quadraticprime = lambda x: scipy.dot(A,x) - b
quadraticandprime = lambda x: (quadratic(x), quadraticprime(x))


def test_ncg():
    dim = 2
    wrt = scipy.random.standard_normal((dim,)) * 10 + 5

    opt = NonlinearConjugateGradient(wrt, quadratic, quadraticprime)
    for i, info in enumerate(opt):      
        if i > 20:
            break
    assert (abs(quadraticprime(wrt)) < 0.01).all(), 'did not find solution'


def test_ncg_rosen():
    dim = 2
    wrt = scipy.zeros((dim,))

    opt = NonlinearConjugateGradient(wrt, rosen, rosen_der)
    for i, info in enumerate(opt):  
        if i > 14:            
            break
    assert (abs(wrt - [1, 1]) < 0.01).all(), 'did not find solution'
