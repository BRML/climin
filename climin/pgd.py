# -*- coding: utf-8 -*-

import itertools


from base import Minimizer, repeat_or_iter


class PartitionedGradientDescent(Minimizer):
    """Use different learning rates for different parts of the gradient.

    Some models (e.g. mcRBM's) need to have different
    learning rate scales for different parts of their
    parameter vector. This external knowledge can be brought
    into the standard gradient descent via a partition
    (indices into the parameter vector) and a set of
    learning rates (steprate is a set of generators,
    one for every partition).
    """
    def __init__(self, wrt, fprime, steprate, partition, momentum=0.0, 
                 args=None, logfunc=None, **kwargs):
        super(PartitionedGradientDescent, self).__init__(
            wrt, args=args, logfunc=logfunc)

        self.fprime = fprime
        self.steprates = repeat_or_iter(steprate)
        self.momentums = repeat_or_iter(momentum)
        self.partition = partition

    def __iter__(self):
        step_m1 = 0
        periterargs = itertools.izip(self.steprates, self.momentums, self.args)
        for i, j in enumerate(periterargs):
            steprate, momentum, (args, kwargs) = j
            gradient = self.fprime(self.wrt, *args, **kwargs)
            # get correct size for gradient (and allow gpu);
            # there should be a better way.
            step = 0*gradient
            k = 0
            for l, m in enumerate(self.partition):
                step[k:m] = -gradient[k:m]*steprate[l]
                k += m
            step[k:] = -gradient[k:] * steprate[l+1] 
            step += momentum * step_m1

            self.wrt += step

            yield dict(gradient=gradient, steprate=steprate, 
                       args=args, kwargs=kwargs, n_iter=i,
                       momentum=momentum, step=step)

            step_m1 = step
