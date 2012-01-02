# -*- coding: utf-8 -*-


import itertools

import scipy.optimize
import numpy as np
import scipy as sp


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
                return 0.0
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
                # If the step is too short, stop trying.
                break
            candidate = self.wrt + step
            loss = self.f(candidate, *args, **kwargs)
            grad0 = self.fprime(self.wrt, *args, **kwargs)
            dir_dot_grad0 = scipy.inner(direction, grad0)
            # Wolfe 1
            if loss <= loss0 + self.c1 * s * dir_dot_grad0:
                grad = self.fprime(candidate, *args, **kwargs)
                dir_dot_grad = scipy.inner(direction, grad)
                # Wolfe 2
                if abs(dir_dot_grad) <= self.c2 * abs(dir_dot_grad0):
                    return s

        return 0.0


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


def polyinterp(points, xminBound=None, xmaxBound=None):
    """
    Minimum of interpolating polynomial based
    on function and derivative values.

    Note: doPlot from minFunc missing!!!
    """
    nPoints = points.shape[0]
    order = np.sum(np.isreal(points[:, 1:3])) - 1

    # Code for most common case:
    # - cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # - no xminBound/xmaxBound

    if (nPoints == 2) and (order == 3) and xminBound is None and xmaxBound is None:
        # Solution in this case (where x2 is the farthest point):
        #    d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
        #    d2 = sqrt(d1^2 - g1*g2);
        #    minPos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
        #    t_new = min(max(minPos,x1),x2);
        minVal = np.min(points[:, 0])
        minPos = np.argmin(points[:, 0])
        notMinPos = 1 - minPos  
        d1 = points[minPos, 2] + points[notMinPos,2] - \
                3*(points[minPos, 1] - points[notMinPos, 1])/ \
                (points[minPos, 0] - points[notMinPos, 0])
        d2 = sp.sqrt(d1**2 - points[minPos, 2] * points[notMinPos, 2])
        if np.isreal(d2):
            t = points[notMinPos, 0] -\
                    ( points[notMinPos, 0] - points[minPos, 0] ) * \
                    (\
                      (points[notMinPos, 2] + d2 - d1)/ \
                      (points[notMinPos, 2] - points[minPos, 2] + 2*d2)\
                    )
            minPos = np.minimum(np.maximum(t, points[minPos,0]), points[notMinPos, 0])
            
        else:
            minPos = np.mean(points[:,0])
        # fmin is not returned here
        return minPos, None 
    #
    #
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    # Compute Bouns of Interpolation Area
    if xminBound is None:
        xminBound = xmin
    if xmaxBound is None:
        xmaxBound = xmax
    #
    #
    # Collect constraints for parameter estimation 
    A = np.zeros((2*nPoints, order+1))
    b = np.zeros((2*nPoints, 1))
    # Constraints based on available function values
    for i, p in enumerate(points[:, 1]):
        if np.isreal(p):
            A[i] = [points[i, 0]**(order - j) for j in xrange(order+1)]
            b[i] = p
    # Constraints based on available derivatives
    for i, p in enumerate(points[:, 2]):
        if np.isreal(p):
            A[nPoints + i] = [(order-j+1)*points[i, 0]**(order-j) for j in xrange(1, order+1)] + [0]
            b[nPoints + i] = points[i, 2]
    #
    # Find interpolating polynomial
    params = np.linalg.lstsq(A, b)[0].flatten()

    # Compute critical points
    dParams =  [(order-j)*params[j] for j in xrange(order)]

    cp = [xminBound, xmaxBound] + list(points[:, 0])
    if not np.any(np.isinf(dParams)):
        cp += list(np.roots(dParams))

    # Test critical points
    fmin = np.inf
    # Default to bisection if no critical points are valid
    minPos = (xminBound + xmaxBound)/2.
    for x in cp:
        if np.isreal(x) and x >= xminBound and x <= xmaxBound:
            fx = np.polyval(params, x)
            if np.isreal(fx) and fx <= fmin:
                minPos = x
                fmin = fx
    return minPos, fmin


def mixedExtrap(x0, f0, g0, x1, f1, g1,
        minStep, maxStep):
    """
    From minFunc, without switches doPlot and debug.
    """
    alpha_c, _ = polyinterp(points=np.array([[x0, f0, g0],[x1, f1, g1]]), 
            xminBound=minStep, xmaxBound=maxStep)
    #
    alpha_s, _ = polyinterp(points=np.array([[x0, f0, g0],[x1, 1j, g1]]), 
            xminBound=minStep, xmaxBound=maxStep)
    if alpha_c > minStep and abs(alpha_c - x1) < abs(alpha_s - x1):
        # Cubic Extrapolation
        t = alpha_c;
    else:
        # Secant Extrapolation
        t = alpha_s
    return t


#def mixedInterp(bracket, bracketFval, bracketGval, d, Tpos,
#        oldLOval,oldLOFval,oldLOGval):
#    """
#    From minFunc, without switches for doPlot and debug
#    """
#    # This needs a check!!
#    #
#    # In particular this next line!
#    nonTpos = 1 - Tpos
#
#    # And these three lines ....
#    gtdT = np.dot(bracketGval(:,Tpos), d)
#    gtdNonT = np.dot(bracketGval(:,nonTpos), d)
#    oldLOgtd = np.dot(oldLOGval, d)
#    #
#    if bracketFval(Tpos) > oldLOFval:
#        # A comment here would be nice ...
#        alpha_c, _ = polyinterp(np.array([[oldLOval, oldLOFval, oldLOgtd],\
#                [bracket(Tpos), bracketFval(Tpos), gtdT]]))
#        #
#        alpha_q, _ = polyinterp(np.array([[oldLOval, oldLOFval, oldLOgtd],\
#                [bracket(Tpos), bracketFval(Tpos), 1j]]))
#        if abs(alpha_c - oldLOval) < abs(alpha_q - oldLOval):
#            # Cubic Interpolation
#            t = alpha_c
#        else:
#            # Mixed Quad/Cubic Interpolation
#            t = (alpha_q + alpha_c)/2.
#    elif np.dot(gtdT, oldLOgtd) < 0:
#        # A comment here would be nice ...
#        alpha_c, _ = polyinterp(np.array([[oldLOval, oldLOFval, oldLOgtd],\
#                [bracket(Tpos), bracketFval(Tpos), gtdT]]))
#        #
#        alpha_s, _ = polyinterp(np.array([[oldLOval, oldLOFval, oldLOgtd],\
#                [bracket(Tpos), 1j, gtdT]]))
#        if abs(alpha_c - bracket(Tpos)) >= abs(alpha_s - bracket(Tpos))
#            # Cubic Interpolation
#            t = alpha_c
#        else:
#            # Quad Interpolation
#            t = alpha_s
#    elif abs(gtdT) <= abs(oldLOgtd):
#        alpha_c, _ = polyinterp(np.array([[oldLOval, oldLOFval, oldLOgtd],\
#                [bracket(Tpos), bracketFval(Tpos), gtdT]]),\
#                np.min(bracket), np.max(bracket))
#        #
#        alpha_s, _ = polyinterp(np.array([[oldLOval, 1j, oldLOgtd],\
#                [bracket(Tpos), bracketFval(Tpos), gtdT]]),\
#                np.min(bracket), np.max(bracket))
#        #
#        if (alpha_c > min(bracket)) and (alpha_c < max(bracket)):
#            if abs(alpha_c - bracket(Tpos)) < abs(alpha_s - bracket(Tpos)):
#                # Bounded Cubic Extrapolation
#                t = alpha_c
#            else:
#                # Bounded Secant Extrapolation
#                t = alpha_s
#        else:
#            # Bounded Secant Extrapolation
#            t = alpha_s
#
#        if bracket(Tpos) > oldLOval:
#            t = min(bracket(Tpos) + 0.66*(bracket(nonTpos) - bracket(Tpos)), t)
#        else:
#            t = max(bracket(Tpos) + 0.66*(bracket(nonTpos) - bracket(Tpos)), t)
#    else:
#        t = polyinterp(np.array([[bracket(nonTpos), bracketFval(nonTpos), gtdNonT],\
#                [bracket(Tpos), bracketFval(Tpos), gtdT]]))
#    return t
