import numpy as np

from climin import linesearch


def test_polyinterp():
    """
    polyinterp is used for interpolating
    functions based on function/derviative values.
    """
    #
    points = np.array([
        [0, 13675.4102446153, -51949952485403.5],
        [5.13158867008399e-10, 60596.6916564044, 408884281904298]
    ])
    t, fm = linesearch.polyinterp(points)
    assert abs(t - 1.94786036422603e-10) < 10**-12, 'polyinterp failed.'
    #
    points = np.array([[0, 3455.1645406324, -459.680191241405],
                       [1, 3459.20341983103, 837.841741413213]])
    t, fm = linesearch.polyinterp(points)
    assert abs(t - 0.564620052389446) < 10**-12, 'polyinterp failed.'
    #
    points = np.array([[0, 5685.6017238964, -121.263963883048],
                       [1, 5569.21178006813, -111.62733563021]])
    t, fm = linesearch.polyinterp(points, xminBound=1.01, xmaxBound=10)
    assert t == 10, 'polyinterp failed.'


def test_WolfeLineSearch_x2():
    f = lambda x: x[0]**2
    fprime = lambda x: 2 * np.asarray(x)
    for d in (-1, 0.1, 1, 10):
        wrt = np.array([-2.])
        d = np.array([d])
        ls = linesearch.WolfeLineSearch(wrt, f, fprime)
        t = ls.search(d)
        assert abs(f(wrt + t * d) == 0) < 0.01
