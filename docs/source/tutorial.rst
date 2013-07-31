Tutorial
========

In the following we will explain the basic features of climin with a simple
example. For that we will first use a multinomial logistic regression, which
suffices to show much of climin's functionality.

Although we will use numpy in this example, we have spent quite some effort to
make most optimizers work with gnumpy as well. This makes the use of GPUs
possible. Check the reference documentation for specific optimizers whether
the usage of GPU is supported.


Defining a Loss Function
------------------------

At the heart of optimizitation lies the objective function we wish to optimize.
In the case of climin, we will *always* be concernced with minimization. Even
though algorithms are sometimes defined with respect to maximization (e.g.i
Bayesian optimization or evolution strategies) in the literature. Thus, we will
also be talking about loss functions.

A loss function in climin follows a simple protocol: a callable (e.g. a
function) which takes an array with a single dimension as input and
returns a scalar. An example would be a simple polynomial of degree 2::

  def loss(x):
      return (x ** 2).sum()

In machine learning, this will mostly be the parameters of our model.
Additionally, we will often have further arguments to the loss, the most
important being the data that our model works on.


Logistic Regression
-------------------

Optimizing a scalar quadratic with an iterative technique is all nice, but
we want to do more complicated things. We will thus move on to use logistic
regression. 

In climin, the parameter vector will always be one dimensional. Your loss
function will have to unpack that vector into the various parameters of
different shapes. While this might seem tedious at first, it makes some
calculations much easier and also more efficient.

Logistic regression has commonly two different parameter sets, the
weight matrix (or coefficients) and the bias (or intercept). To unpack
the parameters we define the following function::

    import numpy as np

    def unpack_parameters(pars):
        w = pars[:n_inpt * n_output].reshape((n_inpt, n_output))
        b = pars[n_inpt * n_output:].reshape((1, n_output))
        return w, b

Given some input We can then make predictions with the following function::

    def predict(parameters, inpt):
        w, b = unpack_parameters(parameters)
        before_softmax = np.dot(inpt, w) + b
        softmaxed = np.exp(before_softmax - before_softmax.max(axis=1)[:, np.newaxis])
        return softmaxed / softmaxed.sum(axis=1)[:, np.newaxis]

For multiclass classification, we use the cross entropy loss::

    def loss(parameters, inpt, targets):
        predictions = predict(parameters, inpt)
        loss = -np.log(predictions) * targets
        return loss.sum(axis=1).mean()

Gradient-based optimization requires not only the loss but also the
first derivative with respect to the parameters.
That gradient function has to return the gradients aligned with the parameters,
which is why we concatenate them into a big array after we flattened out the
weight matrix::

    def f_d_loss_wrt_pars(parameters, inpt, targets):
        p = predict(parameters, inpt)
        g_w = np.dot(inpt.T, p - targets) / inpt.shape[0]
        g_b = (p - targets).mean(axis=0)
        return np.concatenate([g_w.flatten(), g_b])

Although this implementation can be optimized with no doubt, it suffices for this
tutorial.


An Array for the Parameters
---------------------------

First we will need to allocate a region of memory where our parameters live.
Climin tries to allocate as little additional memory as possible and will thus 
work inplace most of the time. After each optimization iteration, the current
solution will always be in the array we created. This lets the user control as
much as possible. We create an empty array for our solution::

    wrt = np.empty(7850)

where the ``7850`` refers to the dimensionality of our problem. We picked this
number because we will be tackling the MNIST data set. It makes sense to
initialize the parameters randomly (depending on the problem), even though the
convexity of logistic regressions guarantees that we will always find the
minimum. Climin offers convenience functions in its ``initialize`` module::

    import climin.initialize
    climin.initialize.randomize_normal(wrt, 0, 1)

This will populated the parameters with values drawn from
:math:`\mathcal{N}(0, 1)`.


Using data
----------

Now that we have set up our model and loss and initialized the parameters,
we need to manage the data.

In climin, we will always look at streams of data. Even if we do batch
learning, the recommended way of doing so is a repeating stream of the same
data. How does that stream look? In Python, we have a convenient data structure
which is the iterator. It can be thought of as a lazy list of infinite length.

The climin API expects that the loss function (and the gradient function) will
accept the parameter array as the first argument. All further arguments can be
as the user wants. When we initialize an optimizer, a keyword argument ``args``
can be specified. This is expected to be an iterator which yields pairs of
``(a, kw)`` which are then passed to the loss as 
``f(parameters, *a, **kw)`` and ``fprime(parameters, *a, *kw)`` in case of the
derivative.

We will be using the MNIST data set , which can be downloaded from
`here http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz.`_.
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

    def one_hot(arr):
        result = np.zeros((arr.shape[0], 10))
        result[xrange(n), array] = 1.
        return result

    Z = one_hot(Z, 10)
    VZ = one_hot(VZ, 10)
    TZ = one_hot(TZ, 10)

To create our data stream, we will just repeat the training data ``(X, Z)``::

    import itertools
    args = itertools.repeat(([X, Z], {}))

This certainly seems like overkill for logistic regression. Yet, even this
simple model can often be sped up by estimating the gradients on "mini
batches". Going even further, you might want to have a continuous stream that
is read from the network,  a data set that does not fit into RAM or which you
want to transform on the fly. All these things can be elegantly implemented
with iterators.


Creating an Optimizer
---------------------

Now that we have set everything up, we are ready to create our first
optimizer, a ``GradientDescent`` object::

    import climin
    opt = climin.GradientDescent(parameters, d_loss_wrt_pars, step_rate=0.1, momentum=.95, args=args)

We created a new object called ``opt``. For initialization, we passed it
several parameters:

 - The parameters ``wrt``. This will *always* be the first argument to any
   optimizer in climin.
 - The derivative ``d_loss_wrt_pars``; we do not need ``loss`` itself for
   gradient descent.
 - A scalar to multiply the negative gradient with for the next search step,
   ``step_rate``. This parameter is often referred to as learning rate in the
   literature.
 - A momentum term ``momentum`` to speed up learning.
 - Our data stream ``args``.

The parameters ``wrt`` and ``args`` are consistent over optimizers. All others
may vary wildly, according to what an optimizer expects.


Optimization as Iteration
-------------------------

Many optimization algorithms are iterative and so are all in climin. To
transfer this metaphor into programming code, optimization with climin is as
simple as iterating over our optimizer object::

    for i in opt:   # Infinite loop!
        pass

This will result in an infinite loop. Climin does not handle stopping from
within optimizer objects; instead, you will have to do it manually, since you
know it much better. Let's iterate for a fixed number of iterations, say 100::

    print loss(paramters, VX, VZ)   # prints something like 2.49771627484
    for info in opt:
        if info['n_iter'] >= 100:
            break
    print loss(paramters, VX, VZ)   # prints something like 0.324243334583

When we iteratore over the optimizer, we iterate over dictionaries. Each
of these contains various information about the current state of the
optimizer. The exact contents depend on the optimizer, but might contain
the last step, gradient, etc. Here, we check the number of iterations that
have already been performed. 


Conclusion and Next Steps
-------------------------

This tutorial explained the basic functionality of climin. There is a lot
more to explore to fully leverage the functionality of this library:

 - Different optimizers,
 - Schedules of step rates and momentum for gradient descent,
 - Specialized initializations,
 - Advanced data streams,
 - Criteria to check for convergence.

We hope to hear from you!
