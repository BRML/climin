Manifest
========

Climin was started in the beginning with the observation that plain
stochastic gradient descent was not able to find good solutions for
sparse filtering [sparsefiltering]_. The original article mentions
the use of LBFGS as an optimization method, yet the implementation
offered by scipy did not solve the problem for us immediately. Since
matlab has a very powerful optimization library, we decided that
it was time for Python to catch up in this respect.

We found several requirements for a good optimization library. 

 - Optimization in machine learning is mostly accompanied by online 
   evaluation code: live plotting of error curves, parameters or
   sending you an email once your model has beaten the current state of
   the art. Also, you might have your own stopping criterions.
   We call this the "side logic" of an optimization.
   Every user has his own way of dealing with this side logic, and a lot of
   throw away code is being written for this. We wanted to make this part as
   easy as possible for the user.
 - Do one thing, and do that right: climin is independent of your models
   and the way you work with optimization. We have a simple protocol: loss
   functions (and their derivatives) and a parameter array. Also, we do not
   force any framework on you on and come up with things that try to solve
   everything.
 - Most of the optimizers, i.e. those that do not rely on too much linear
   algebra such as matrix inversions, should not only work on the CPU via
   numpy but also on the GPU via gnumpy.
 - Optimizers should be easily switchable; if we have a model and a loss
   we wanted to be able to quickly experiment with different methods.
 - Optimizers should be reasonably fast. Most of the computational work
   in machine learning is done within the models anyway. Yet, we want a
   clean python code base without C extensions. We also found that speeding
   up everything with Cython would be a good way to go where necessary.
   Since numba is around the corner, we wanted to decide this in a later
   version.
 - Make development of new optimizers straight forward. The implementation 
   of every optimizer has very little overhead, the most of it being assigning hyper parameters to class values.

The main idea of climin is to treat optimizers as iterators. This allows
to have the logic surrounding it right in the same scope and written code
block as the optimization. Also, callbacks are really ugly! Python has better
tools for that.

.. [sparsefiltering] Ngiam, Jiquan, et al. "Sparse filtering." Advances in
   Neural Information Processing Systems. 2011.
