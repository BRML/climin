# -*- coding: utf-8 -*-

import warnings

import scipy
import numpy as np
import scipy.linalg
import scipy.optimize

from base import Minimizer, is_nonzerofinite
from linesearch import WolfeLineSearch


class ConjugateGradient(Minimizer):
    """
    Linear Conjugate Gradient Method for quadratic function :
    f = 1/2 * xTAx -bx
    fprime = Ax - b
    """

    def __init__(self, wrt, H=None, b=None, f_Hp=None, epsilon=1e-14,
                 precond=None):
        # TODO rename epsilon to sth sensible
        super(ConjugateGradient, self).__init__(
            wrt, args=None)
        self.f_Hp = f_Hp if f_Hp is not None else lambda p: np.dot(H, p)
        self.b = b
        self.epsilon = epsilon
        self.precond = precond

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
            if (abs(grad) < self.epsilon).all():
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
    Nonlinear Conjugate Gradient Method
    """
    # TODO: document class

    def __init__(self, wrt, f, fprime, epsilon=1e-6, args=None):
        # TODO: document method
        super(NonlinearConjugateGradient, self).__init__(wrt, args=args)
        self.f = f
        self.fprime = fprime

        self.line_search = WolfeLineSearch(wrt, self.f, self.fprime, c2=0.2)
        self.epsilon = epsilon

    def find_direction(self, grad_m1, grad, direction_m1):
        # TODO: document method
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
        args, kwargs = self.args.next()
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
            if (abs(grad) < self.epsilon).all():
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
