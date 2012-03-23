# -*- coding: utf-8 -*-

import itertools


from base import Minimizer, repeat_or_iter


class Nesterov(Minimizer):

    def __init__(self, wrt, fprime, steprate, args=None, logfunc=None):
        super(Nesterov, self).__init__(
            wrt, args=args, logfunc=logfunc)

        self.fprime = fprime
        self.steprates = repeat_or_iter(steprate)

    def __iter__(self):
        """
        http://mlg.eng.cam.ac.uk/mlss09/mlss_slides/vandenberghe_1_2.pdf
        page 101
        """
        step_m1 = 0
        periterargs = itertools.izip(self.steprates, self.args)
        for i, j in enumerate(periterargs):
            steprate, (args, kwargs) = j
            if i > 0:
                y = self.wrt + (i-1)/(i+2)*step_m1
                grad_y = self.fprime(y, *args, **kwargs)

                step = (i-1)/(i+2)*step_m1 - steprate*grad_y
                self.wrt += step
            else:
                grad = self.fprime(self.wrt, *args, **kwargs)
                
                step = -steprate*grad
                self.wrt += step_m1
            yield dict(step=step, steprate=steprate, 
                       args=args, kwargs=kwargs, n_iter=i)
            step_m1 = step
