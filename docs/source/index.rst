.. climin documentation master file, created by
   sphinx-quickstart on Tue May  7 13:56:19 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

climin: optimization, straight-forward
======================================

introduction
------------

Optimization is a key ingredient in modern machine learning. While many models
can be optimized in specific ways, several need very general gradient based
techniques--e.g. neural networks. What's even worse is that you never know
whether your model does not work or you just have not found the right optimizer.

This is where climin comes in. We offer you many different off-the-shelf
optimizers, such as LBFGS, stochastic gradient descent with Nesterov momentum,
nonlinear conjugate gradients, resilient propagation, rmsprop and more.

But what is best, is that we know that optimization is a process that needs to
be analyzed. This is why climin does not offer you a black box function call
that takes ages to run and might give you a good minimum of your training loss.
Since this is not what you care about, climin takes you with you on its travel
through the error landscape... in a classic for loop::

   import climin

   network = make_network()                 # your neural network
   training_data = load_train_data()        # your training data
   validation_data = load_validation_data() # your validation data
   test_data = load_test_data()             # your testing data

   opt = climin.Lbfgs(network.parameters, 
                      network.loss, 
                      network.d_loss_d_parameters, 
                      args=itertools.repeat((training_data, {})))

   for info in opt:
       validation_loss = network.loss(network.parameters, validation_data)
       print info['loss'], validation_loss

   print network.loss(test_data)

Climin works on the CPU (via numpy and scipy) and in parts on the GPU (via
gnumpy).


Starting points
^^^^^^^^^^^^^^^

If you want to see how climin works and use climin asap, check out the 
:doc:`tutorial`. A list of the optimizers implemented can be found in the 
overview below. If you want to understand the design decisions of climin,
read the :doc:`manifest`.



Basics
------

.. toctree::
   tutorial
   :maxdepth: 2

Optimizer overview
------------------

.. toctree::
   gd
   rmsprop
   rprop
   cg
   :maxdepth: 1


Convenience, Utilities
----------------------

.. toctree::
   schedule
   initialize
   linesearch
   :maxdepth: 1




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

