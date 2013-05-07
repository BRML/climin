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

We then import climin and initialize our first optimizer, a ``GradientDescent``
object::

    >>> import climin
    >>> opt = climin.GradientDescent(wrt, loss_wrt)

We created a new object called ``opt``. For initialization, we passed it two
parameters. For all optimizers, the first parameter is `always` the parameter
numpy array. This one will be worked upon in place and will always contain the
latest parameters found. 

