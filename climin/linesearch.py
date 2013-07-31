# -*- coding: utf-8 -*-

"""Module containing various line searches.

Line searches are at the heart of many optimizers. After finding a suitable
search direction (e.g. the steepest descent direction) we are left with a
one-dimensional optimization problem, which can then be solved by a line search.
"""

# TODO: this module needs lots of pep8 love.


import itertools

import scipy.optimize
import numpy as np
import scipy as sp


class LineSearch(object):

    def __init__(self, wrt):
        self.wrt = wrt

    def search(self, direction, initialization, args=None, kwargs=None):
        raise NotImplemented()


class BackTrack(LineSearch):
    """Class implementing a back tracking line search.

    The idea is to jump to a starting step length :math:`t` and then shrink that
    step length by multiplying it with :math:`\\gamma` until we improve upon
    the loss.

    At most ``max_iter`` attempts will be done. If the largest absolut value of
    a component of the step falls below ``tolerance``, we stop as well. In both
    cases, a step length of 0 is returned.


    To not possibly iterate forever, the field `tolerance` holds a small
    value (1E-20 per default). As soon as the absolute value of every component
    of the step (direction multiplied with the scalar from `schedule`) is less
    than `tolerance`, we stop.


    Attributes
    ----------

    wrt : array_like
        Parameters over which the optimization is done.

    f : Callable
        Objective function.

    decay : float
        Factor to multiply trials for the step length with.

    tolerance : float
        Minimum absolute value of a component of the step without stopping the
        line search.
    """

    def __init__(self, wrt, f, decay=0.9, max_iter=float('inf'),
                 tolerance=1E-20):
        """Create BackTrack object.

        Parameters
        ----------

        wrt : array_like
            Parameters over which the optimization is done.

        f : Callable
            Objective function.

        decay : float
            Factor to multiply trials for the step length with.

        max_iter : int, optional, default infinity
            Number of step lengths to try.

        tolerance : float
            Minimum absolute value of a component of the step without stopping the
            line search.
        """

        super(BackTrack, self).__init__(wrt)
        self.f = f
        self.max_iter = max_iter
        self.decay = decay

        self.tolerance = tolerance

    def search(self, direction, initialization=1, args=None, kwargs=None,
               loss0=None):
        """Return a step length ``t`` given a search direction.

        Perform the line search along a direction. Search will start at
        ``initialization`` and assume that the loss is ``loss0`` at ``t == 0``.

        Parameters
        ----------

        direction : array_like
            Has to be of the same size as .wrt. Points along that direction
            will tried out to reduce the loss.

        initialization : float
            First attempt for a step size. Will be reduced by a factor of
            ``.decay`` afterwards.

        args : list, optional, default: None
            list of optional arguments for ``.f``.

        kwargs : dictionary, optional, default: None
            list of optional keyword arguments for ``.f``.

        loss0 : float, optional
            Loss at the current parameters. Will be calculated of not given.
        """
        args = [] if args is None else args
        kwargs = {} if kwargs is None else kwargs

        # Recalculate the current loss if it has not been given.
        if loss0 is None:
            loss0 = self.f(self.wrt, *args, **kwargs)

        # Try out every point in the schedule until a reduction has been found.
        schedule = (self.decay ** i * initialization for i in itertools.count())
        for i, s in enumerate(schedule):
            if i + 1 >= self.max_iter:
                break
            step = initialization * s * direction
            if abs(step).max() < self.tolerance:
                # If the step is too short, just return 0.
                break
            candidate = self.wrt + step
            loss = self.f(candidate, *args, **kwargs)
            if -(loss0 - loss) < 0:
                # We check here for negative improvement to also not continue in
                # the case of NaNs.
                return s

        return 0


class StrongWolfeBackTrack(BackTrack):
    """Class implementing a back tracking line search that finds points
    satisfying the Strong Wolfe conditions.

    The idea is to jump to a starting step length :math:`t` and then shrink that
    step length by multiplying it with :math:`\\gamma` until the strong Wolfe
    conditions are satisfied. That is the Armijo rule

    .. math::
       f(\\theta_t+ \\alpha_t d_t) & \\leq f(\\theta)+ c_1 \\alpha_t d_t^T f'(\\theta),

    and the curvature condition

    .. math::
       \\big|d_k^TTf('\\theta_t+\\alpha_t d_t)\\big| & \\leq c_2 \\big|d_t^T f'(\\theta_t)\\big|.

    At most ``max_iter`` attempts will be done. If the largest absolut value of
    a component of the step falls below ``tolerance``, we stop as well. In both
    cases, a step length of 0 is returned.

    To not possibly iterate forever, the field `tolerance` holds a small
    value (1E-20 per default). As soon as the absolute value of every component
    of the step (direction multiplied with the scalar from `schedule`) is less
    than `tolerance`, we stop.


    Attributes
    ----------

    wrt : array_like
        Parameters over which the optimization is done.

    f : Callable
        Objective function.

    decay : float
        Factor to multiply trials for the step length with.

    tolerance : float
        Minimum absolute value of a component of the step without stopping the
        line search.

    c1 : float
        Constant in the strong Wolfe conditions.

    c2 : float
        Constant in the strong Wolfe conditions.
    """

    def __init__(self, wrt, f, fprime, decay=None, c1=1E-4, c2=.9,
                 tolerance=1E-20):
        """Create StrongWolfeBackTrack object.

        Parameters
        ----------

        wrt : array_like
            Parameters over which the optimization is done.

        f : Callable
            Objective function.

        decay : float
            Factor to multiply trials for the step length with.

        tolerance : float
            Minimum absolute value of a component of the step without stopping
            the line search.
        """
        super(StrongWolfeBackTrack, self).__init__(wrt, f, decay, tolerance)
        self.fprime = fprime
        self.c1 = c1
        self.c2 = c2

    def search(self, direction, args, kwargs, loss0=None):
        # TODO: respect initialization
        # Recalculate the current loss if it has not been given.
        if loss0 is None:
            loss0 = self.f(self.wrt, *args, **kwargs)

        self.grad = grad0 = self.fprime(self.wrt, *args, **kwargs)
        dir_dot_grad0 = scipy.inner(direction, grad0)
        # Try out every point in the schedule until one satisfying strong Wolfe
        # conditions has been found.
        for s in self.schedule:
            step = s * direction
            if abs(step.max()) < self.tolerance:
                # If the step is too short, stop trying.
                break
            candidate = self.wrt + step
            loss = self.f(candidate, *args, **kwargs)
            # Wolfe 1
            if loss <= loss0 + self.c1 * s * dir_dot_grad0:
                grad = self.fprime(candidate, *args, **kwargs)
                dir_dot_grad = scipy.inner(direction, grad)
                # Wolfe 2
                if abs(dir_dot_grad) <= self.c2 * abs(dir_dot_grad0):
                    self.grad = grad
                    return s
        return 0.0


class ScipyLineSearch(LineSearch):
    """Wrapper around the scipy line search."""

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


class WolfeLineSearch(LineSearch):
    """Port of Mark Schmidt's line search."""

    def __init__(self, wrt, f, fprime, c1=1E-4, c2=0.9, maxiter=25,
                 min_step_length=1E-9, typ=4):
        super(WolfeLineSearch, self).__init__(wrt)
        self.f = f
        self.fprime = fprime
        self.c1 = c1
        self.c2 = c2
        self.maxiter = maxiter
        self.min_step_length = min_step_length
        self.typ = typ

        # TODO: find better API for this
        self.first_try = True

    def search(self, direction, initialization=None, args=None, kwargs=None,
               loss0=None):
        args = args if args is not None else ()
        kwargs = kwargs if kwargs is not None else {}
        loss0 = self.f(self.wrt, *args, **kwargs) if loss0 is None else loss0
        grad0 = self.fprime(self.wrt, *args, **kwargs)
        direct_deriv0 = scipy.inner(grad0, direction)
        f = lambda x: (self.f(x, *args, **kwargs),
                       self.fprime(x, *args, **kwargs))

        if self.first_try:
            self.first_try = False
            t = min(1, 1 / sum(abs(grad0)))
        else:
            t = initialization if initialization is not None else 1

        step, fstep, fprimestep, n_evals = wolfe_line_search(
            self.wrt, t, direction, loss0, grad0, direct_deriv0,
            self.c1, self.c2, self.typ, self.maxiter, self.min_step_length,
            f)

        self.val = fstep
        self.grad = fprimestep

        return step


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

    if nPoints == 2 and order == 3 and xminBound is None and xmaxBound is None:
        # Solution in this case (where x2 is the farthest point):
        #    d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
        #    d2 = sqrt(d1^2 - g1*g2);
        #    minPos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
        #    t_new = min(max(minPos,x1),x2);
        minVal = np.min(points[:, 0])
        minPos = np.argmin(points[:, 0])
        notMinPos = 1 - minPos

        x1 = points[minPos, 0]
        x2 = points[notMinPos, 0]
        g1 = points[minPos, 2]
        g2 = points[notMinPos, 2]
        f1 = points[minPos, 1]
        f2 = points[notMinPos, 1]

        d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
        d2 = sp.sqrt(d1 ** 2 - g1 * g2)
        if np.isreal(d2):
            t = points[notMinPos, 0] -\
                    (points[notMinPos, 0] - points[minPos, 0]) * \
                    (
                      (points[notMinPos, 2] + d2 - d1) /
                      (points[notMinPos, 2] - points[minPos, 2] + 2 * d2)
                    )
            minPos = np.minimum(
                np.maximum(t, points[minPos, 0]), points[notMinPos, 0])

        else:
            minPos = np.mean(points[:, 0])
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
    A = np.zeros((2 * nPoints, order + 1))
    b = np.zeros((2 * nPoints, 1))
    # Constraints based on available function values
    for i in range(points.shape[0]):
        if np.isreal(points[i, 1]):
            A[i] = [points[i, 0] ** (order - j) for j in xrange(order + 1)]
            b[i] = points[i, 1]
            points[i, 0], points[i, 1]
    # Constraints based on available derivatives
    for i, p in enumerate(points[:, 2]):
        if np.isreal(p):
            A[nPoints + i] = [(order - j + 1) * points[i, 0] ** (order - j)
                              for j in xrange(1, order + 1)] + [0]
            b[nPoints + i] = points[i, 2]
    #
    # Find interpolating polynomial
    params = np.linalg.lstsq(A, b)[0].flatten()

    # Compute critical points
    dParams = [(order - j) * params[j] for j in xrange(order)]

    cp = [xminBound, xmaxBound] + list(points[:, 0])
    if not np.any(np.isinf(dParams)):
        cp += list(np.roots(dParams))

    # Test critical points
    fmin = np.inf
    # Default to bisection if no critical points are valid
    minPos = (xminBound + xmaxBound) / 2.
    for x in cp:
        if np.isreal(x) and x >= xminBound and x <= xmaxBound:
            fx = np.polyval(params, x)
            if np.isreal(fx) and fx <= fmin:
                minPos = x
                fmin = fx
    return minPos, fmin


def mixedExtrap(x0, f0, g0, x1, f1, g1, minStep, maxStep):
    """
    From minFunc, without switches doPlot and debug.
    """
    alpha_c, _ = polyinterp(
        points=np.array([[x0, f0, g0], [x1, f1, g1]]),
        xminBound=minStep, xmaxBound=maxStep)
    #
    alpha_s, _ = polyinterp(
        points=np.array([[x0, f0, g0], [x1, 1j, g1]]),
        xminBound=minStep, xmaxBound=maxStep)
    if alpha_c > minStep and abs(alpha_c - x1) < abs(alpha_s - x1):
        # Cubic Extrapolation
        t = alpha_c
    else:
        # Secant Extrapolation
        t = alpha_s
    return t


def isLegal(v):
    """
    Do exactly that.
    """
    return not (np.any(np.iscomplex(v)) or
                np.any(np.isnan(v)) or
                np.any(np.isinf(v)))


def armijobacktrack(x, t, d, f, fr, g, gtd, c1, LS, tolX, funObj):
    """
    Backtracking linesearch satisfying Armijo condition.

    From minFunc. Missing: doPlot, saveHessianComp, varargin
    -> varargin are passed to funObj as parameters, need to
    be put in here!!!! Hessian at initial guess is _not_ returned

    Check again with minFunc!!!

    x: starting location
    t: initial step size
    d: descent direction
    f: function value at starting location
    fr: reference function value (usually funObj(x))
    gtd: directional derivative at starting location
    c1: sufficient decrease parameter
    debug: display debugging information
    LS: type of interpolation
    tolX: minimum allowable step length
    funObj: objective function
    varargin: parameters of objective function

    Outputs:
    t: step length
    f_new: function value at x+t*d
    g_new: gradient value at x+t*d
    funEvals: number function evaluations performed by line search
    """

    # Evaluate objective and gradient at initial step
    # Hessian part missing here!
    f_new, g_new = funObj(x + t * d)
    funEvals = 1

    while f_new > fr + c1 * t * gtd or not isLegal(f_new):
        # A comment here will be nice!
        temp = t
        # this could be nicer, if idea how to work out 'LS'
        if LS == 0 or not isLegal(f_new):
            # Backtrack with fixed backtracing rate
            t = 0.5 * t
        elif LS == 2 and isLegal(g_new):
            # Backtrack with cubic interpolation with derivative
            t, _ = polyinterp(
                np.array([[0, f, gtd], [t, f_new, np.dot(g_new, d)]]))
        elif funEvals < 2 or not isLegal(f_prev):
            # Backtracking with quadratic interpolation
            # (no derivatives at new point available)
            t, _ = polyinterp(np.array([[0, f, gtd], [t, f_new, 1j]]))
        else:
            # Backtracking with cubin interpolation
            # (no derviatives at new point available)
            t, _ = polyinterp(
                np.array([[0, f, gtd], [t, f_new, 1j], [t_prev, f_prev, 1j]]))
        #
        # Adjust if change in t is too small ...
        if t < 1e-3 * temp:
            t = 1e-3 * temp
        # or too large
        elif t > 0.6 * temp:
            t = 0.6 * temp

        f_prev = f_new
        t_prev = temp
        # Missing part: call return Hessian

        f_new, g_new = funObj(x + t*d)
        #
        funEvals += 1

        # Check if step size has become too small
        if np.sum(np.abs(t*d)) <= tolX:
            # Backtracking line search failed -> maybe some print out?
            t = 0
            f_new = f
            g_new = g
            break

    # Missing: evaluate at new point
    #
    x_new = x + t*d
    # Hessian is missing here!
    return t, x_new, f_new, g_new, funEvals


def mixedInterp(bracket, bracketFval, bracketGval, d, Tpos,
                oldLOval, oldLOFval, oldLOGval):
    """
    From minFunc, without switches for doPlot and debug
    """
    # This needs a check!!
    #
    # In particular this next line!
    nonTpos = 1 - Tpos

    # And these three lines ....
    gtdT = np.dot(bracketGval[Tpos], d)
    gtdNonT = np.dot(bracketGval[nonTpos], d)
    oldLOgtd = np.dot(oldLOGval, d)
    #
    if bracketFval[Tpos] > oldLOFval:
        # A comment here would be nice ...
        alpha_c, _ = polyinterp(np.array([[oldLOval, oldLOFval, oldLOgtd],\
                [bracket[Tpos], bracketFval[Tpos], gtdT]]))
        #
        alpha_q, _ = polyinterp(np.array([[oldLOval, oldLOFval, oldLOgtd],\
                [bracket[Tpos], bracketFval[Tpos], 1j]]))
        if abs(alpha_c - oldLOval) < abs(alpha_q - oldLOval):
            # Cubic Interpolation
            t = alpha_c
        else:
            # Mixed Quad/Cubic Interpolation
            t = (alpha_q + alpha_c)/2.
    elif np.dot(gtdT, oldLOgtd) < 0:
        # A comment here would be nice ...
        alpha_c, _ = polyinterp(np.array([[oldLOval, oldLOFval, oldLOgtd],\
                [bracket[Tpos], bracketFval[Tpos], gtdT]]))
        #
        alpha_s, _ = polyinterp(np.array([[oldLOval, oldLOFval, oldLOgtd],\
                [bracket[Tpos], 1j, gtdT]]))
        if abs(alpha_c - bracket[Tpos]) >= abs(alpha_s - bracket[Tpos]):
            # Cubic Interpolation
            t = alpha_c
        else:
            # Quad Interpolation
            t = alpha_s
    elif abs(gtdT) <= abs(oldLOgtd):
        alpha_c, _ = polyinterp(np.array([[oldLOval, oldLOFval, oldLOgtd],\
                [bracket[Tpos], bracketFval[Tpos], gtdT]]),\
                np.min(bracket), np.max(bracket))
        #
        alpha_s, _ = polyinterp(np.array([[oldLOval, 1j, oldLOgtd],\
                [bracket[Tpos], bracketFval[Tpos], gtdT]]),\
                np.min(bracket), np.max(bracket))
        #
        if (alpha_c > min(bracket)) and (alpha_c < max(bracket)):
            if abs(alpha_c - bracket[Tpos]) < abs(alpha_s - bracket[Tpos]):
                # Bounded Cubic Extrapolation
                t = alpha_c
            else:
                # Bounded Secant Extrapolation
                t = alpha_s
        else:
            # Bounded Secant Extrapolation
            t = alpha_s

        if bracket[Tpos] > oldLOval:
            t = min(bracket[Tpos] + 0.66*(bracket[nonTpos] - bracket[Tpos]), t)
        else:
            t = max(bracket[Tpos] + 0.66*(bracket[nonTpos] - bracket[Tpos]), t)
    else:
        t, _ = polyinterp(
            np.array([[bracket[nonTpos], bracketFval[nonTpos], gtdNonT],
                     [bracket[Tpos], bracketFval[Tpos], gtdT]]))
    return t


def wolfe_line_search(x, t, d, f, g, gtd,
        c1, c2, LS, maxLS, tolX, funObj):
        """
        Bracketing Line Search to Satisfy Wolfe Conditions

        From minFunc. Missing!!! debug, doPlot, saveHessian, varargin
         Inputs:
           x: starting location
           t: initial step size
           d: descent direction
           f: function value at starting location
           g: gradient at starting location
           gtd: directional derivative at starting location
           c1: sufficient decrease parameter
           c2: curvature parameter
           debug: display debugging information
           LS: type of interpolation
           maxLS: maximum number of iterations
           tolX: minimum allowable step length
           doPlot: do a graphical display of interpolation
           funObj: objective function
           varargin: parameters of objective function

         Outputs:
           t: step length
           f_new: function value at x+t*d
           g_new: gradient value at x+t*d
           funEvals: number function evaluations performed by line search
           NOT:
           H: Hessian at initial guess (only computed if requested
        """
        #Evaluate the Objective and Gradient at the Initial Step
        f_new, g_new = funObj(x+t*d)
        funEvals = 1

        gtd_new = np.dot(g_new, d)

        # Bracket an intervail containing a point
        # satisfying the wolfe criteria
        LSiter = 0
        t_prev = 0
        f_prev = f
        g_prev = g
        gtd_prev = gtd
        done = False

        while LSiter < maxLS:

            # Bracketing phase
            if not isLegal(f_new) or not isLegal(g_new):
                t = (t + t_prev)/2.
                # missing: if 0 in minFunc!!
                #
                # Extrapolated into illegal region, switching
                # to Armijo line search
                # no Hessian is computed!!
                t, x_new, f_new, g_new, _fevals = armijobacktrack(
                    x, t, d, f, f, g, gtd, c1, max(0, min(LS-2, 2)), tolX,
                    funObj)
                funEvals += _fevals
                return t, f_new, g_new, funEvals
            #
            if (f_new > f + c1*t*gtd) or (LSiter > 1 and f_new >= f_prev):
                bracket = [t_prev, t]
                bracketFval = [f_prev, f_new]
                # check here: two gradients next to each other, in columns
                bracketGval = np.array([g_prev, g_new])
                break
            elif abs(gtd_new) <= -c2*gtd:
                bracket = np.array([t])
                bracketFval = np.array([f_new])
                bracketGval = np.array([g_new])
                done = True
                break
            elif gtd_new >= 0:
                bracket = [t_prev, t]
                bracketFval = [f_prev, f_new]
                # check here (again), see above
                bracketGval = np.array([g_prev, g_new])
                break
            temp = t_prev
            t_prev = t
            minStep = t + 0.01*(t-temp)
            maxStep = t*10
            #
            if LS == 3:
                # Extending Braket
                t = maxStep
            elif LS == 4:
                # Cubic Extrapolation
                t, _ = polyinterp(np.array([[temp, f_prev, gtd_prev],\
                        [t, f_new, gtd_new]]), minStep, maxStep)
            else:
                t = mixedExtrap(temp, f_prev, gtd_prev, t, f_new, gtd_new,
                        minStep, maxStep)
            #
            f_prev = f_new
            g_prev = g_new
            gtd_prev = gtd_new
            #
            # no saveHessian stuff!!!
            f_new, g_new = funObj(x + t*d)
            funEvals += 1
            gtd_new = np.inner(g_new, d)
            LSiter += 1
        # while ....
        #
        if LSiter == maxLS:
            bracket = [0, t]
            bracketFval = [f, f_new]
            # check here, same again!
            bracketGval = np.array([g, g_new])

        # Zoom Phase:
        # We now either have point satisfying the criteria
        # or a bracket surrounding a point satisfying the criteria.
        # Refine the bracket until we find a point satifying the criteria.
        #
        insufProgress = False
        # Next line needs a check!!!!!
        Tpos = 1
        LOposRemoved = False
        while not done and LSiter < maxLS:
            # Find high and low points in the bracket
            # check here, axees needed??
            f_LO = np.min(bracketFval)
            LOpos = np.argmin(bracketFval)
            HIpos = 1 - LOpos
            #
            # Compute new trial value
            if LS == 3 or not isLegal(bracketFval) or not isLegal(bracketGval):
                # Bisecting
                t = np.mean(bracket)
            elif LS == 4:
                # Grad cubic interpolation
                t, _ = polyinterp(
                    np.array(
                        [[bracket[0], bracketFval[0], np.dot(bracketGval[0], d)],
                         [bracket[1], bracketFval[1], np.dot(bracketGval[1], d)]]))
            else:
                # Mixed case
                # Is this correct ???????
                nonTpos = 1 - Tpos
                if not LOposRemoved:
                    oldLOval = bracket[nonTpos]
                    oldLOFval = bracketFval[nonTpos]
                    oldLOGval = bracketGval[nonTpos]
                t = mixedInterp(
                        bracket, bracketFval, bracketGval, d, Tpos, oldLOval,
                        oldLOFval, oldLOGval)

            #
            # Test that we are making sufficient progress
            bracket_min = min(bracket)
            bracket_max = max(bracket)

            if min(bracket_max - t, t - bracket_min) / (bracket_max - bracket_min) < 0.1:
                # Interpolation close to boundary
                if insufProgress or (t >= np.max(bracket)) or (t <= np.min(bracket)):
                    # Evaluating at 0.1 away from boundary
                    if np.abs(t - np.max(bracket)) < np.abs(t - np.min(bracket)):
                        t = np.max(bracket) - 0.1 * (np.max(bracket) - np.min(bracket))
                    else:
                        t = np.min(bracket) + 0.1 * (np.max(bracket) - np.min(bracket))
                    insufProgress = False
                #
                else:
                    insufProgress = True
            #
            else:
                insufProgress = False

            # Evaluate new point
            # no Hessian!
            t = scipy.real(t)
            f_new, g_new = funObj(x + t * d)
            funEvals += 1
            gtd_new = np.dot(g_new, d)
            LSiter += 1

            if f_new > f + c1 * t * gtd or f_new >= f_LO:
                # Armijo condition not satisfied or
                # not lower than lowest point
                bracket[HIpos] = t
                bracketFval[HIpos] = f_new
                bracketGval[HIpos] = g_new
                Tpos = HIpos
            else:
                if np.abs(gtd_new) <= -c2 * gtd:
                    # Wolfe conditions satisfied
                    done = True
                elif gtd_new * (bracket[HIpos] - bracket[LOpos]) >= 0:
                    # old HI becomes new LO
                    bracket[HIpos] = bracket[LOpos]
                    bracketFval[HIpos] = bracketFval[LOpos]
                    bracketGval[HIpos] = bracketGval[LOpos]
                    if LS == 5:
                        # LO Pos is being removed
                        LOposRemoved = True
                        oldLOval = bracket[LOpos]
                        oldLOFval = bracketFval[LOpos]
                        oldLOGval = bracketGval[LOpos]
                    #

                # New point becomes new LO
                bracket[LOpos] = t
                bracketFval[LOpos] = f_new
                bracketGval[LOpos] = g_new
                Tpos = LOpos

            if not done and np.abs(bracket[0] - bracket[1]) * gtd_new < tolX:
                # Line search can not make further progress
                break
        # while ...

        # TODO a comment here maybe nice
        if LSiter == maxLS:
            # could give info:
            # Line Search exceeded maximum line search iterations
            # TODO: what to do here?
            pass
        #
        # check if axes necessary?
        f_LO = np.min(bracketFval)
        LOpos = np.argmin(bracketFval)
        t = bracket[LOpos]
        f_new = bracketFval[LOpos]
        g_new = bracketGval[LOpos]

        # missing Hessain evaluation
        return t, f_new, g_new, funEvals
