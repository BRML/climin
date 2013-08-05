Advanced data handling
======================

We have already seen how to use batch learning in the :doc:`tutorial`. The use
of ``itertools.repeat`` to handle simple batch learning seemed rather over
expressive; it makes more sense if we consider more advanced use cases.


Mini batches
------------

Consider the case where we do not want to calculcate the full gradient in each
iteration but estimate it given a mini batch. The contract of climin is that
each item of the ``args`` iterator will be used in a single iteration and thus
for one parameter update. The way to enable mini batch learning is thus to
provide an ``args`` argument which is an infinite list of mini batches.

Let's revisit our example of logistic regression. Here, we created the args
iterator using ``itertools.repeat`` on the same array again and again::

    import itertools
    args = itertools.repeat(([X, Z], {}))

What we want to do now is to have an infinite stream of slices of ``X`` and
``Z``.  How do we access the n'th batch of ``X`` and ``Z``? We offer you a
convenience function that will give you random (with or without replacement)
slices from a container::

    batch_size = 100
    args = climin.util.iter_minibatches([X, Z], batch_size, [0, 0])

The last argument, ``[0, 0]`` gives the axes along which to slice ``[X, Z]``.
In some cases, samples might be aligned along axis ``0`` for the input, but
along axis ``1`` in the target data.


External memory
---------------


Further usages
--------------

This architecture can be exploited in many different ways. E.g., a stream over 
a network can be directly used. A single pass over a file without keeping more
than necessary is another option.
