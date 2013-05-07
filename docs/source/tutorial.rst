Tutorial
========

In the following we will explain the basic features of climin with a simple
example.


Defining a Loss Function
------------------------

At the heart of optimizitation lies the objective function we wish to optimize.
In the case of climin, we will *always* be concernced with minimization. Even
though algorithms are sometimes defined with respect to maximization (e.g.i
Bayesian optimization or evolution strategies) in the literature. Thus, we will
also be talking about loss functions.

A loss function in climin follows a simple protocol: a callable (e.g. a
function) which takes a numpy vector as input and returns a scalar. An example
would be a simple polynomial of degree 2::

  def loss(x):
      return (x ** 2).sum()

For derivative free optimizers, this is actually enough. Yet, gradient-based
optimization is powerful and one often uses the directional information as well.
We will thus also define the derivative of that quadratic::

    def loss_wrt_x(x):
        return 2 * x

The simplest optimizer in climin is plain gradient descent. Let's use that to
find a minimum of our loss.


An Array for the Parameters
---------------------------

First we will need to allocate a region of memory where our parameters live,
though. Climin will work inplace most of the time to let the user control as
much as possible. First we import numpy for that and create an empty array for
our solution::

    >>> import numpy as np
    >>> wrt = np.empty(1)
    >>> wrt[0] = 2

where the ``1`` refers to the dimensionality of our problem. We also start the 
optimization at the value ``2`` for no particular reason apart that it is not
too close and not too far from the minimizer.


Creating an Optimizer
---------------------

We then import climin and initialize our first optimizer, a ``GradientDescent``
object::

    >>> import climin
    >>> opt = climin.GradientDescent(wrt, loss_wrt_x)

We created a new object called ``opt``. For initialization, we passed it two
parameters. For all optimizers, the first parameter is `always` the parameter
numpy array. This one will be worked upon in place and will always contain the
latest parameters found. 

Another point is that we only supply the derivative to the optimizer. The reason
is that plain gradient descent does not need to know the loss, it just moves
along the gradient.


Optimization as Iteration
-------------------------

Many optimization algorithms are iterative. To transfer this metaphor into
programming code, optimization with climin is as simple as iterating over 
our optimizer object::

    >>> for i in opt:   # Infinite loop!
    ...   pass

This will result in an infinite loop. Climin does not handle stopping from
within optimizer objects; instead, you will have to do it manually. Let's
iterate for a fixed number of iterations, say 100:

    >>> counter = 0
    >>> for i in opt:
    ...   counter += 1
    ...   if counter >= 100:
    ...     break

This is not very pythonic, though. Instead, it is good practice to do it like
that:

    >>> for i, j in enumerate(opt):
    ...   if i >= 100:
    ...     break

We can now look at our variable what the solution is::

    >>> x
    ... array([  3.25925756e-10])

Pretty close to the optimum!


Useful things to do during Iteration
------------------------------------

What is the benefit of performing optimization as iteration? In machine
learning, we are frequently interested in the results during optimization:

 - We rarely optimize a loss directly; instead, we optimize a proxy of our real
   loss. In classification, our real loss is the number of examples classified
   correctly of all possible examples; yet, we minimize the negative
   log-likelihood in case of logistic regression on a subset of the real world,
   the training data.
 - When optimizing, we not only want to monitor the training error, but also an
   error on a validation set.
 - We want to stop optimization due to several heuristics: if the gradient is
   zero, if the adaptable parameters don't change to much, early stopping, 
   time used for training, ...

We can implement all of these efficiently in the block of the for loop.





