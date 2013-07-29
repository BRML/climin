Tutorial
========

In the following we will explain the basic features of climin with a simple
example. For that we will first use a simple quadratic polynomial and then 
multinomial logistic regression, which suffices to show much of climin's
functionality.

Although we will use numpy in this example, we have spent quite some effort to
make most optimizers work with gnumpy as well. This makes the use of GPUs
possible.


Defining a Loss Function
------------------------

At the heart of optimizitation lies the objective function we wish to optimize.
In the case of climin, we will *always* be concernced with minimization. Even
though algorithms are sometimes defined with respect to maximization (e.g.i
Bayesian optimization or evolution strategies) in the literature. Thus, we will
also be talking about loss functions.

A loss function in climin follows a simple protocol: a callable (e.g. a
function) which takes a numpy array with a single dimension as input and
returns a scalar. An example would be a simple polynomial of degree 2::

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

First we will need to allocate a region of memory where our parameters live.
Climin will work inplace most of the time to let the user control as
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
array. This one will be worked upon in place and will always contain the
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

We can now look at our variable what the solution is::

    >>> x
    ... array([  3.25925756e-10])

Pretty close to the optimum!


Logistic Regression
-------------------

Optimizing a scalar quadratic with an iterative technique is all nice, but
we want to do more complicated things. We will thus move on to use logistic
regression. First we need a function to extract the weight matrix and bias
values from a flat numpy array::

    import numpy as np

    def unpack_parameters(pars):
        w = pars[:n_inpt * n_output].reshape((n_inpt, n_output))
        b = pars[n_inpt * n_output:].reshape((1, n_output))
        return w, b

Given some input We can then do predictions with the following function::

    def predict(parameters, inpt):
        w, b = unpack_parameters(parameters)
        before_softmax = np.dot(inpt, w) + b
        softmaxed = np.exp(before_softmax - before_softmax.max(axis=1)[:, np.newaxis])
        return softmaxed / softmaxed.sum(axis=1)[:, np.newaxis]

For optimization, we are interested in the loss and the gradient of that loss
with respect to the parameters::

    def f_loss(parameters, inpt, targets):
        predictions = predict(parameters, inpt)
        loss = -np.log(predictions) * targets
        return loss.sum(axis=1).mean()

    def f_d_loss_wrt_pars(parameters, inpt, targets):
        p = predict(parameters, inpt)
        g_w = np.dot(inpt.T, p - targets) / inpt.shape[0]
        g_b = (p - targets).mean(axis=0)
        return np.concatenate([g_w.flatten(), g_b])

Although this implementation can be optimized with no doubt, it suffices for this
documentation.


Using data
----------

So far we have optimized a function that did not work on any data. Yet, this
is always the case in machine learning, which is why we will show how to 
incorporate it now.

In climin, we will always look at streams of data. Even if we do batch
learning, the recommended way of doing so is a repeating stream of the same
data. How does that stream look? In Python, we have a convenient data structure
which is the iterator. It can be thought of as an infinite lazy list.

The climin API expects that the loss function (and the gradient function) will
accept the parameter array as the first argument. All further arguments can be
as the user wants. When we initialize an optimizer, a keyword argument ``args``
can be specified. This is expected to be an iterator which yields pairs of
``(a, kw)`` which are then passed to the loss as 
``f(parameters, *a, **kw)`` and ``fprime(parameters, *a, *kw)`` in case of the
derivative.

In this example, we will be using the MNIST data set (surprise!), which can be
downloaded from here http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz.
We will first load it and convert the target variables to a one-of-k representation,
which is what our loss functions expect::

    # You can get this at http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    datafile = 'mnist.pkl.gz'
    # Load data. 
    with gzip.open(datafile,'rb') as f:
        train_set, val_set, test_set = cPickle.load(f)

    X, Z = train_set
    VX, VZ = val_set
    TX, TZ = test_set

    def one_hot(array, n_classes=None):
        """Return one of k vectors for an array of class indices.

        :param array: 1D array containing N integers from 0 to k-1.
        :param classes: Amount of classes, k. If None, this will be inferred
            from array (and might take longer).
        :returns: A 2D array of shape (N, k).
        """
        if n_classes is None:
            n_classes = len(set(array.tolist()))
        n = array.shape[0]
        arr = np.zeros((n, n_classes), dtype=np.float32)
        arr[xrange(n), array] = 1.
        return arr

    Z = one_hot(Z, 10)
    VZ = one_hot(VZ, 10)
    TZ = one_hot(TZ, 10)

To create our data stream, we will just repeat the training data ``(X, Z)``::

    import itertools
    args = itertools.repeat(([X, Z], {}))


Learning Logistic Regression
----------------------------

The MNIST data set has an input dimensionality of 784. Together with the 10 
possible classes, we get 7850 parameters in total::

    pars = np.random.normal(0, 0.1, 7850)

Since the loss functions, the data and the parameter array are ready we can
proceed to optimization. We will create a basic gradient descent optimizer::

    import climin
    opt = climin.GradientDescent(parameters, f_d_loss_wrt_pars, steprate=0.1, momentum=.95, args=args)

We want to stress that we do not actually need the loss ``f_loss``, because 
gradient descent does not care about the loss; it just follows the gradient.

We run the optimizer and stop after 100 iterations::

    print f_loss(paramters, VX, VZ)   # prints something like 2.49771627484

    for info in opt:
        if info['n_iter'] == 100:
            break

    print f_loss(paramters, VX, VZ)   # prints something like 0.324243334583

When we iteratore over the optimizer, we iterate over dictionaries. Each
of these contains various information about the current state of the
optimizer. Here, we check the number of iterations that have already been
performed.


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





