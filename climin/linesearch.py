# -*- coding: utf-8 -*-


import itertools

import scipy.optimize


class LineSearch(object):

    def __init__(self, wrt):
        self.wrt = wrt


class BackTrack(LineSearch):
    """Class implementing a back tracking line search.
 
    The idea is to try out jumps along the search direction until
    one satisfies a condition. Jumps are done by multiplying the search
    direction with a scalar. The field `schedule` holds an iterator which
    successively yields those scalars.

    To not possibly iterate forever, the field `tolerance` holds a very small
    value (1E-20 per default). As soon as the absolute value of every component
    of the step (direction multiplied with the scalar from `schedule`) is less
    than `tolerance`.
    """

    def __init__(self, wrt, f, schedule=None, tolerance=1E-20):
        super(BackTrack, self).__init__(wrt)
        self.f = f
        if schedule is not None:
            self.schedule = schedule
        else:
            self.schedule = (0.95**i for i in itertools.count())
        self.tolerance = tolerance

    def search(self, direction, args, kwargs, loss0=None):
        # Recalculate the current loss if it has not been given.
        if loss0 is None:
            loss0 = self.f(self.wrt, *args, **kwargs)

        # Try out every point in the schedule until a reduction has been found.
        for s in self.schedule:
            step = s * direction
            if abs(step.max()) < self.tolerance:
                # If the step is too short, just return 0.
                return 0
            candidate = self.wrt + step
            loss = self.f(candidate, *args, **kwargs)
            if loss0 - loss > 0:
                return s


class StrongWolfeBackTrack(BackTrack):

    def __init__(self, wrt, f, fprime, schedule=None, c1=1E-4, c2=.9,
                 tolerance=1E-20):
        super(StrongWolfeBackTrack, self).__init__(wrt, f, schedule, tolerance)
        self.fprime = fprime
        self.c1 = c1
        self.c2 = c2

    def search(self, direction, args, kwargs, loss0=None):
        # Recalculate the current loss if it has not been given.
        if loss0 is None:
            loss0 = self.f(self.wrt, *args, **kwargs)

        # Try out every point in the schedule until one satisfying strong Wolfe
        # conditions has been found.
        for s in self.schedule:
            step = s * direction
            if abs(step.max()) < self.tolerance:
                # If the step is too short, just return 0.
                return 0
            candidate = self.wrt + step
            loss = self.f(candidate, *args, **kwargs)
            dir_dot_grad0 = scipy.inner(direction, self.fprime(self.wrt))
            # Wolfe 1
            if loss <= loss0 + self.c1 * s * dir_dot_grad0:
                grad = self.fprime(candidate, *args, **kwargs)
                dir_dot_grad = scipy.inner(direction, grad)
                # Wolfe 2
                if abs(dir_dot_grad) >= self.c2 * abs(dir_dot_grad0):
                    return s


class ScipyLineSearch(LineSearch):

    def __init__(self, wrt, f, fprime):
        super(ScipyLineSearch, self).__init__(wrt)
        self.f = f
        self.fprime = fprime

    def search(self, direction, args, kwargs):
        if kwargs:
            raise ValueError('keyword arguments not supported')
        gfk = self.fprime(self.wrt, *args)
        return scipy.optimize.line_search(
            self.f, self.fprime, self.wrt, direction, gfk, args=args)[0]
