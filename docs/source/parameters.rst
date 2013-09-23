Dealing with parameters
=======================

Parameters in climin are a long and one dimensional array. This might seem as
a restriction at first, yet it makes things easier in other places. Consider a
model involving complicated array dimensionalities; now consider how higher
order derivatives of those might look like. Yes that's right, a pretty messy
thing. Furthermore, letting paramters occupy consecutive regions of memory has
further advantages from an implementation point of view. We can easier write it
to disk, randomize its contents or similar things.


Creating parameter sets
-----------------------

Creating of parameter arrays need not be tedious, though. Climin comes with a 
nice convenience function, ``climin.util.empty_with_views`` which does most of the
work. You just need to feed it the shapes of parameters you are interested in.

Let us use logistic regressiom from the :doc:`tutorial` and see where it comes in
handy. First, we will create a parameter array and the various views according
to a template:

.. testcode:: [parameters]

   import numpy as np
   import climin.util

   tmpl = [(784, 10), 10]          # w is matrix and b a vector
   flat, (w, b) = climin.util.empty_with_views(tmpl)

Now, ``flat`` is a one dimensional array. ``w`` and ``b`` are a two dimensional
and a one dimensional array respectively. They share memory with ``flat``, so 
any change we will do in ``w`` or ``b`` will be reflected in ``flat`` and vice
versa. In order for a predict function to get the parameters out of the flat
array, there is ``climin.util.shaped_from_flat`` which does the same job as
``empty_with_views``, except that it receives ``flat`` and does not create it.
In fact, the latter uses the former internally.

Let's adapt the ``predict`` function to use ``w`` and ``b`` instead:

.. testcode:: [parameters]

   def predict(parameters, inpt):
       w, b = climin.util.shaped_from_flat(parameters, tmpl)
       before_softmax = np.dot(inpt, w) + b
       softmaxed = np.exp(before_softmax - before_softmax.max(axis=1)[:, np.newaxis])
       return softmaxed / softmaxed.sum(axis=1)[:, np.newaxis] 

This might seem like overkill for logistic regression, but becomes invaluable
when complicated models with many different parameters are used.


Calculating derivatives in place
--------------------------------

When calculating derivatives, you can make use of this as well--which is
important because climin expects derivatives to be flat as well, nicely aligned
with the parameter array:

.. testcode:: [parameters]

   def f_d_loss_wrt_pars(parameters, inpt, targets):
       p = predict(parameters, inpt)
       d_flat, d_w, d_b = climin.util.empty_with_views(tmpl)
       d_w[...] = np.dot(inpt.T, p - targets) / inpt.shape[0]
       d_b[...] = (p - targets).mean(axis=0)
       return d_flat

What are we doing here? First, we get ourselves a new array and preshaped views
on it in the same way as the parameters. Then we overwrite the views in place
with the derivatives and finally return the flat array as a result.
The in place assignment is important.  If we did it using ``d_w = ...``, Python
would just reassign the name and the changes would not turn up in ``d_flat``.

As a further note, ``np.dot`` supports an extra argument ``out`` which specifies
where to write the result. To safe memory, we could perform the following
instead::

        np.dot(inpt.T, p - targets, out=d_w)
        d_w  /= inpt.shape[0]


Initializing parameters
-----------------------

Initializing parameters with empty values is asking for trouble. You probably
want to populate an array with random numbers or zeros. Of course it is easy to
do so by hand:

.. testcode:: [parameters]

   flat[...] = np.random.normal(0, 0.1, flat.shape)

We found this quite tedious to write though; especially as soon as flat becomes
the field of a nested object. Thus, we have a short hand in the initialize
module which does exaclty that:
 
.. testcode:: [parameters]

   import climin.initialize
   climin.initialize.randomize_normal(flat, 0, 0.1)

There are more functions to do similar things. Check out :doc:`initialize`.
