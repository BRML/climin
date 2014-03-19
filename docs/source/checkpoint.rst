Interrupting and Resuming via Checkpoints
=========================================

It is important to be able to interrupt optimization and continue right from
where you left off. Reasons include scheduling on shared resources, branching 
the optimization with different settings or securing yourself against crashes
in long-running processes.

.. note:: Currently, this is not supported by all optimizers. It is the case for gradient
          descent, rmsprop, adadelta and rprop.

Climin makes this in parts possible and leaves the responsibility to the user in
other parts. More specifically, the user has to take over the serialization of
the parameter vector (i.e. ``wrt``), the objective function and its derivatives
(e.g. ``fprime``) and the data (i.e. ``args``). The reason for this is that
one cannot build a generic procedure for this. The data might be depending
on an open file descriptor and only a subset of Python functions can be
serialized, which is those that are defined at the top level.

Saving the state to disk (or somewhere else)
--------------------------------------------

The idea is that the ``info`` dictionary which is the result of each
optimization step carries all the information necesseray to resume. Thus a
recipe to write your state to disk is as follows.

.. code-block:: python

   import numpy as np
   import cPickle
   from climin.gd import Gradient Descent

   pars = make_pars()
   fprime = make_fprime()
   data = make_data()
   opt = GradientDescent(pars, fprime, args=data)
   for info in opt:
       with open('state.pkl', 'w') as fp:
           cPickle.dump(info, fp)
       np.savetxt('parameters.csv', pars)

This snippet first generates the necessery quantities from library functions
which we assume given. We then create a ``GradientDescent`` object over which we
iterate to optimize. In each iteration, we pickle the info dictionary to disk.

.. note:: Pickling an info dictionary directly to disk might be a bad idea in
          many cases. E.g. it will contain the current data element or a gnumpy
          array, which is not picklable. It is the users's responsibility to
          take care of that.


Loading the state from disk
---------------------------

We will now load the info dictionary from file, create an optimizer object an
initialize it with values from the info dictionary.

.. code-block:: python

   import numpy as np
   import cPickle
   from climin.gd import Gradient Descent

   pars = np.loadtxt('parameters.csv')
   fprime = make_fprime()
   data = make_data()
   with open('state.pkl') as fp:
       info = cPickle.load(fp)

   opt = GradientDescent(pars, fprime, args=data)
   opt.set_from_info(info)

We can continue optimization right from where we left off.
