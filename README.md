Optimization package for Python
===============================

We present prior thoughts on writing an for an optimization package aimed at
machine learning researchers using the Python programming language.


Requirements
------------

The most commonly used optimization functions in Python come from scipy's
optimization package. The API is not really consistent, however most
optimizers behave somewhat like this:

    fmin(f, x0, maxiter, verbose)

There are several downsides with this approach. While it works fine as a
black box, requirements in machine learning are typically more complicated.
E.g. it is often necessary to monitor certain values during optimization, e.g.
for early stopping to avoid overfitting.

We list a couple of requirements for optimization which we want to fulfill.


### Use scipy ###

Scipy is the defacto standard for numerical computation with Python. Use
scipy arrays for the parameters. As a special case, gnumpy might be needed
at some point for ML computations on the GPU. We need to find out how to do
this best transparently.


### Always minimize ###

Any maximization problem can be turned into a minimization problem by
putting a minus in front. No need to build an API for that.


### Optimize over a single consecutive array and save memory ###

In big optimization problems, the number of parameters might exceed millions.
It is thus essential, that as few copies of the parameter vector are made
as possible.

We argue that the best way to achieve this is to let an array be specified
from the outside be specified which holds the parameters. The api should
therefore look something like this:

    fmin(f, pars, ...)

where pars is a scipy.array of dimensionality 1. The latter should hold
because several operations required during optimization should be
performed efficiently. E.g., if the covariance matrix of the parameters
has to be calculated, a single call of ``scipy.cov`` should suffice.


### Different function calls might require different parameters ###

Evaluating the objective function might need different arguments at each
iteration. Thus, it is essential to make it possible for the optimizer to
pass different arguments to the objective function on each iteration.

This is a shortcoming of the scipy API: while args can be specified, it is
assumed that they do not vary.

Several optimizers (even some tuned towards batch processing, like RPROP)
benefit from using mini batches. This is an example where such behaviour
is needed.

We propose to make it possible to pass an iterable to the optimizer, which
holds the successive arguments to the objective function.

    def fmin(f, pars, args)

During optimization, the optimizer can chose to consume the iterable at its
will.


### Yield control back to the user ###

Another shortcoming of the scipy API is that control is never yielded back
to the user during optimization. This is however essential in machine learning
applications as well as research. A researcher might need to inspect certain
values (like the magnitude of the gradient) during optimization, while a
practitioner wants to implement a specific stopping criterion in this way.

We propose that each optimizer is implemented as a generator function. After
each iteration step, control is handed back to the user who can then inspect
certain statistics of the optimization process.

    for info in fmin(f, pars):
        print 'gradient magnitude: %.5f' % (info['gradient']**2).sum()
        print 'step magnitude: %.5f' % (info['step']**2).sum()
        if (info['loss'] - previous_loss) < 0.001:
            break

An optimizer would then look something like this:

    def fmin(f, pars):
        while True:
           loss = f(pars)
           gradient = ...
           step = ...
           pars += step
           yield dict(loss=loss, gradient=gradient, step=step)


### Specify number of iterations ###

Since control is given back to the user, the number of iterations can
be specified by breaking the optimization loop. To perform a fixed
number of iterations, library functions can be written. E.g.

    opt = fmin(...)
    do_steps(opt, 500)

for 500 optimization steps.


### Don't yield always ###

In some cases it is more efficient to not yield after every iteration. For
example, an inner loop might be implemented using a lower level language
like cython or C. In that case, it should be possible to specify from the
outside to only yield every N weight updates:

   fmin(f, pars, ..., stop=10)

Here, the generator yields only all 10 iterations.


Optimization algorithms used in machine learning
------------------------------------------------

- Stochastic gradient descent with/without Quasi-Newton diagonal approximation
- nonlinear conjugate gradient
- Hessian free
- RPROP
- (online) LBFGS
- Averaging gradient
- TONGA
- SMD
