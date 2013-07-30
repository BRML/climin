# -*- coding: utf-8 -*-


"""This module provides an implementation of gradient descent."""


import itertools


from base import Minimizer, repeat_or_iter


class GradientDescent(Minimizer):
    """Classic gradient descent optimizer.

    Gradient descent works by iteratively performing updates solely based on
    the first derivative of a problem. The gradient is calculated and multiplied
    with a scalar (or component wise with a vector) to do a step in the problem
    space. For speed ups, a technique called "momentum" is often used, which
    averages search steps over iterations.

    Even though gradient descent is pretty simple it can be very effective if
    well tuned (in terms of its hyper parameters step rate and momentum).
    Sometimes the use of schedules for both parameters is necessary. See
    ``climin.schedule`` for basic schedules.

    Gradient descent is also very robust to stochasticity in the objective
    function. This might result from noise injected into it (e.g. in the case
    of denoising auto encoders) or because it is based on data samples (e.g. in
    the case of stochastic mini batches.)

    Given a step rate :math:`\\alpha` and a function :math:`f'` to evaluate the
    search direction the current paramters :math:`\\theta_t` the following
    update is performed:

    .. math::
        v_{t+1} &= \\alpha f'(\\theta_t) \\\\
        \\theta_{t+1} &= \\theta_t + v_{t+1}.

    If we also have a momentum :math:`\\beta` and are using standard momentum,
    we update the parameters according to:

    .. math::
        v_{t+1} &= \\alpha f'(\\theta_t) + \\beta v_{t} \\\\
        \\theta_{t+1} &= \\theta_t + v_{t+1}

    In some cases (e.g. learning the parameters of deep networks), using
    Nesterov momentum can be beneficial. In this case, we first make a momentum
    step and then evaluate the gradient at the location in between. Thus,
    there is an additional cost of an addition of the parameters.

    .. math::
        \\theta_{t+{1 \\over 2}} &= \\theta_t + \\beta v_t \\\\
        v_{t+1} &= \\alpha f'(\\theta_{t + {1 \\over 2}}) \\\\
        \\theta_{t+1} &= \\theta_t + v_{t+1}

    which can be specified additionally by the initialization argument
    ``momentum_type``.

    .. note::
       Works with gnumpy.


    Attributes
    ----------
    wrt : array_like
        Current solution to the problem. Can be given as a first argument to \
        ``.fprime``.

    fprime : Callable
        First derivative of the objective function. Returns an array of the \
        same shape as ``.wrt``.

    steprates : Iterable
        Iterable of step rates. Will be consumed over optimization.

    momentums : Iterable
        Iterable of momentums. Will be consumed over optimization.

    momentum_type : string (either "standard" or "nesterov")
        When to add the momentum term to the parameter vector; in the first \
        case it will be done after the calculation of the gradient, in the\
        latter before.

    """

    def __init__(self, wrt, fprime, steprate=0.1, momentum=0.0,
                 momentum_type='standard',
                 args=None):
        """Create a GradientDescent object.

        Parameters
        ----------

        wrt : array_like
            Array that represents the solution. Will be operated upon in
            place.  ``fprime`` should accept this array as a first argument.

        fprime : callable
            Callable that given a solution vector as first parameter and *args
            and **kwargs drawn from the iterations ``args`` returns a
            search direction, such as a gradient.

        steprate : float or array_like, or iterable of that
            Step rate to use during optimization. Can be given as a single
            scalar value or as an array for a different step rate of each
            parameter of the problem.

            Can also be given as an iterator; in that case, every iteration
            of the optimization takes a new element as a step rate from that
            iterator.

        momentum : float or array_like, or iterable of that
          Momentum to use during optimization. Can be specified analoguously
          (but independent of) step rate.

        momentum_type : string (either "standard" or "nesterov")
            When to add the momentum term to the paramter vector; in the first
            case it will be done after the calculation of the gradient, in the
            latter before.

        args : iterable
            Iterator of arguments which ``fprime`` will be called
            with.
        """
        super(GradientDescent, self).__init__(wrt, args=args)

        self.fprime = fprime
        self.steprates = repeat_or_iter(steprate)
        self.momentums = repeat_or_iter(momentum)

        if momentum_type not in ('nesterov', 'standard'):
            raise ValueError('unknown momentum type')
        self.momentum_type = momentum_type

    def __iter__(self):
        step_m1 = 0
        periterargs = itertools.izip(self.steprates, self.momentums, self.args)
        for i, j in enumerate(periterargs):
            steprate, momentum, (args, kwargs) = j

            if self.momentum_type == 'standard':
                gradient = self.fprime(self.wrt, *args, **kwargs)
                step = gradient * steprate + momentum * step_m1
                self.wrt -= step
            elif self.momentum_type == 'nesterov':
                big_jump = momentum * step_m1
                self.wrt -= big_jump

                gradient = self.fprime(self.wrt, *args, **kwargs)
                correction = steprate * gradient
                self.wrt -= correction

                step = big_jump + correction

            yield dict(gradient=gradient, steprate=steprate,
                       args=args, kwargs=kwargs, n_iter=i,
                       momentum=momentum, step=step)

            step_m1 = step
