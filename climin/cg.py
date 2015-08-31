# -*- coding: utf-8 -*-

"""Module containing functionality for conjugate gradients.

Conjugate gradients is motivated from a first order Taylor expansion of the
objective:

.. math::
   f(\\theta_t + \\alpha_t d_t) \\approx f(\\theta_t) + \\alpha_td_t^Tf'(\\theta_t).

To locally decrease the objective, it is optimal to set
:math:`d_t \propto -f'(\\theta_t)` and find :math:`\\alpha_t` with a line search
algorithm, which is known as steepest descent. Yet, a well known disadvantage
of this approach is that directions found at :math:`t` will often interfere with
directions found for :math:`t' < t`.

The solution to this problem is to chose :math:`d_t` in a way that it does not
interfere with previous updates. If the dimensions of our problem were
independent, we could just move along these dimensions. If they were independent
up to rotation, we would have to chose directions which are orthogonal to each
other. This is exactly the case when the Hessian of the problem, :math:`A` is
diagonal. If it is not diagonal, we have to move along directions which are
called *conjugate* to each other with respect to the matrix :math:`A`.

The conjugate gradients algorithms provide methods to do so efficiently. The
linear conjugate gradients algorithm assumes that the objective is a quadratic
and can thus determine :math:`\\alpha` exactly. Nonlinear conjugate gradients
works on arbitrary functions (yet, the Taylor expansion assumption above has to
be reasonable). Since the Hessian :math:`A` is not constant in this case, the
previous directions (to which a new direction has to be conjugate) have to be
reset from time to time. Additionally, we need to perform a line search to solve
for :math:`\\alpha_t`.
"""

from __future__ import absolute_import

import warnings

import numpy as np
import scipy
import scipy.linalg
import scipy.optimize

from .base import Minimizer, is_nonzerofinite
from .linesearch import WolfeLineSearch


class ConjugateGradient(Minimizer):
    """ConjugateGradient class.

    Minimize a quadratic objective of the form

    .. math::
       f(\\theta) = {1 \over 2} \\theta^TA\\theta + \\theta^Tb + c.

    The minimization will take place by moving along conjugate directions of
    steepest decrease in the error. This will take at most as many steps as
    the dimensionality of the problem.

    .. note::
       In most cases it is better to use ``scipy.optimize.solve``. Only use
       this function if you want to monitor intermediate quantities and are
       not entirely interested in optimization of a quadratic objective, but in
       a different error measure. E.g. as in Hessian free optimization.


    Attributes
    ----------

    wrt : array_like
        Parameters of the problem.

    H : array_like, 2 dimensional, square
        Curvature term of the quadratic, the Hessian.

    b : array_like
        Linear term of the quadratic.

    f_Hp : callable
        Function to calculcate the dot product of a Hessian with an
        arbitrary vector.

    min_grad : float, optional, default: 1e-14
        If all components of the gradient fall below this threshold,
        stop optimization.

    precond : array_like
        Matrix to precondition the problem. If a vector, is taken to
        represent a diagonal matrix.

    """

    def __init__(self, wrt, H=None, b=None, f_Hp=None, min_grad=1e-14,
                 precond=None):
        """Create a ConjugateGradient object.

        Parameters
        ----------

        wrt : array_like
            Parameters of the problem.

        H : array_like, 2 dimensional, square
            Curvature term of the quadratic, the Hessian.

        b : array_like
            Linear term of the quadratic.

        f_Hp : callable
            Function to calculcate the dot product of a Hessian with an
            arbitrary vector.

        min_grad : float, optional, default: 1e-14
            If all components of the gradient fall below this threshold,
            stop optimization.

        precond : array_like
            Matrix to precondition the problem. If a vector, is taken to
            represent a diagonal matrix.
        """

        super(ConjugateGradient, self).__init__(
            wrt, args=None)
        self.f_Hp = f_Hp if f_Hp is not None else lambda p: np.dot(H, p)
        self.b = b
        self.min_grad = min_grad
        self.precond = precond

    def set_from_info(self, info):
        raise NotImplemented('nobody has found the time to implement this yet')

    def extended_info(self, **kwargs):
        raise NotImplemented('nobody has found the time to implement this yet')

    def solve(self, r):
        if self.precond is None:
            return r
        elif self.precond.ndim == 1:
        #if the preconditioning matrix is diagonal,
        #then it is supposedly given as a vector
            return r / self.precond
        else:
            return scipy.linalg.solve(self.precond, r)

    def __iter__(self):
        grad = self.f_Hp(self.wrt) - self.b
        y = self.solve(grad)
        direction = -y

        # If the gradient is exactly zero, we stop. Otherwise, the
        # updates will lead to NaN errors because the direction will
        # be zero.
        if (grad == 0).all():
            warnings.warn('gradient is 0')
            return

        for i in range(self.wrt.size):
            Hp = self.f_Hp(direction)
            ry = np.dot(grad, y)
            pHp = np.inner(direction, Hp)
            step_length = ry / pHp
            self.wrt += step_length * direction

            # We do this every few iterations to compensate for possible
            # numerical errors due to additions.
            if i % 10 == 0:
                grad = self.f_Hp(self.wrt) - self.b
            else:
                grad += step_length * Hp

            y = self.solve(grad)
            beta = np.dot(grad, y) / ry

            direction = - y + beta * direction

            # If we don't bail out here, we will enter regions of numerical
            # instability.
            if (abs(grad) < self.min_grad).all():
                warnings.warn('gradient is below threshold')
                break

            yield {
                'ry': ry,
                'Hp': Hp,
                'pHp': pHp,
                'step_length': step_length,
                'n_iter': i,
            }


class NonlinearConjugateGradient(Minimizer):
    """
    NonlinearConjugateGradient optimizer.

    NCG minimizes functions by following directions which are conjugate to each
    other with respect to the Hessian. Since the curvature changes if the
    objective is nonquadratic, the Hessian will not be accurate and thus the
    conjugacy of successive search directions as well. Furthermore, the optimal
    step length cannot be found in closed form and has to be obtained with a
    line search.

    During optimization, we sometimes perform a restart. That means we give up
    on trying to find conjugate directions and use the gradient as a new search
    direction. This is done whenever two successive directions are far from
    orthogonal, an indicator that the quadratic assumption is either inaccurate
    or the Hessian has changed too much lately.

    Attributes
    ----------

    wrt : array_like
        Array of parameters of the problem.

    f : callable
        Objective function.

    fprime : callable
        First derivative of the objective function.

    min_grad : float
        If all components of the gradient fall below this value, stop
        minimization.

    line_search : LineSearch object.
        Line search object to perform line searches with.

    args : iterable
        Iterable of arguments passed on to the objective function and its
        derivatives.
    """

    def __init__(self, wrt, f, fprime, min_grad=1e-6, args=None):
        """Create a NonlinearConjugateGradient object.

        Parameters
        ----------

        wrt : array_like
            Array of parameters of the problem.

        f : callable
            Objective function.

        fprime : callable
            First derivative of the objective function.

        min_grad : float
            If all components of the gradient fall below this value, stop
            minimization.

        args : iterable, optional
            Iterable of arguments passed on to the objective function and its
            derivatives.
        """
        super(NonlinearConjugateGradient, self).__init__(wrt, args=args)
        self.f = f
        self.fprime = fprime

        self.line_search = WolfeLineSearch(wrt, self.f, self.fprime, c2=0.2)
        self.min_grad = min_grad

    def set_from_info(self, info):
        raise NotImplemented('nobody has found the time to implement this yet')

    def extended_info(self, **kwargs):
        raise NotImplemented('nobody has found the time to implement this yet')

    def find_direction(self, grad_m1, grad, direction_m1):
        # Computation of beta as a compromise between Fletcher-Reeves
        # and Polak-Ribiere.
        grad_norm_m1 = np.dot(grad_m1, grad_m1)
        grad_diff = grad - grad_m1
        betaFR = np.dot(grad, grad) / grad_norm_m1
        betaPR = np.dot(grad, grad_diff) / grad_norm_m1
        betaHS = np.dot(grad, grad_diff) / np.dot(direction_m1, grad_diff)
        beta = max(-betaFR, min(betaPR, betaFR))

        # Restart if not a direction of sufficient descent, ie if two
        # consecutive gradients are far from orthogonal.
        if np.dot(grad, grad_m1) / grad_norm_m1 > 0.1:
            beta = 0

        direction = -grad + beta * direction_m1
        return direction, {}

    def __iter__(self):
        args, kwargs = next(self.args)
        grad = self.fprime(self.wrt, *args, **kwargs)
        grad_m1 = np.zeros(grad.shape)
        loss = self.f(self.wrt, *args, **kwargs)
        loss_m1 = 0

        for i, (next_args, next_kwargs) in enumerate(self.args):
            if i == 0:
                direction, info = -grad, {}
            else:
                direction, info = self.find_direction(grad_m1, grad, direction)

            if not is_nonzerofinite(direction):
                warnings.warn('gradient is either zero, nan or inf')
                break

            # Line search minimization.
            initialization = 2 * (loss - loss_m1) / np.dot(grad, direction)
            initialization = min(1, initialization)
            step_length = self.line_search.search(
                direction, initialization,  args, kwargs)
            self.wrt += step_length * direction

            # If we don't bail out here, we will enter regions of numerical
            # instability.
            if (abs(grad) < self.min_grad).all():
                warnings.warn('gradient is too small')
                break

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs
            grad_m1[:], grad[:] = grad, self.line_search.grad
            loss_m1, loss = loss, self.line_search.val

            info.update({
                'n_iter': i,
                'args': args,
                'kwargs': kwargs,

                'loss': loss,
                'gradient': grad,
                'gradient_m1': grad_m1,
                'step_length': step_length,
            })
            yield info
