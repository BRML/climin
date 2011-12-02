import scipy

from climin import GradientDescent


quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x
quadraticandprime = lambda x: (quadratic(x), quadraticprime(x))


def test_gradient_descent_constant_step():
    dim = 10
    wrt = scipy.random.standard_normal((dim,)) * 10 + 5

    fandprime = lambda: quadraticandprime(wrt)

    opt = GradientDescent(wrt, fandprime, steprate=0.01)
    for i, info in enumerate(opt):
        if i > 1000:
            break
    assert (abs(wrt) < 0.01).all(), 'did not find solution'
