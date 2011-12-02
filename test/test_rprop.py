import scipy

from climin import Rprop


quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x
quadraticandprime = lambda x: (quadratic(x), quadraticprime(x))


def test_rprop():
    dim = 10
    wrt = scipy.random.standard_normal((dim,)) * 10 + 5

    fandprime = lambda: quadraticandprime(wrt)

    opt = Rprop(wrt, fandprime, stepshrink=0.1, stepgrow=1.2, 
                minstep=1E-6, maxstep=0.1)
    for i, info in enumerate(opt):
        if i > 1000:
            break
    assert (abs(wrt) < 0.01).all(), 'did not find solution'
