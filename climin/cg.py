# -*- coding: utf-8 -*-

import itertools

import scipy
import numpy as np
import scipy.linalg
import scipy.optimize

from base import Minimizer


class ConjugateGradient(Minimizer):
    """
    Linear Conjugate Gradient Method for quadratic function :
    f = 1/2 * xTAx -bx
    fprime = Ax - b
    """

    def __init__(self, wrt, H=None, b=None, f_Hp=None, epsilon=1e-14,
                 logfunc=None):
        super(ConjugateGradient, self).__init__(wrt, args=None, logfunc=logfunc)
        self.f_Hp = f_Hp if f_Hp is not None else lambda p: np.dot(H, p)
        self.b = b
        self.epsilon = epsilon

    def __iter__(self):
        grad = self.f_Hp(self.wrt) - self.b
        direction = -grad
        
        for i in range(self.wrt.size):
            # If the gradient is exactly zero, we stop. Otherwise, the
            # updates will lead to NaN errors because the direction will
            # be zero.
            if (grad == 0).all():
                self.logfunc({'message': 'gradient is 0'})
                break

            Hp = self.f_Hp(direction)
            rr = np.dot(grad, grad)
            step_length = rr / np.dot(direction, Hp)
            self.wrt += step_length * direction
            grad = grad + step_length * Hp
            beta = np.dot(grad, grad)/ rr
            direction = - grad + beta * direction
            
            # If we don't bail out here, we will enter regions of numerical
            # instability.
            if (abs(grad) < self.epsilon).all():
                self.logfunc(
                    {'message': 'converged - gradient smaller than epsilon'})
                break

            yield {
                'rr': rr,
                'Hp': Hp,
                'step_length': step_length,
                'n_iter': i,
            }
