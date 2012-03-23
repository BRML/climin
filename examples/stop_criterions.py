# Demonstrates the usage of existing stop criterions and
# creation of new stop criterions.

# Stop criterions are a way to stop (or pause) the minimizer
# under certain conditions. Maybe you want the minimizer to
# stop when a certain error is converged, or after a certain
# time, or a number of iterations. For all these common cases
# climin offers existing stop criterions in its `stops` module.
# For other situations, you can easily create your own stop
# criterion with the building blocks that climin includes.

# In the following examples, we will always minimize a quadratic 
# function in 10 dimensions with a simple GradientDescent minimizer.
# Note that each example initializes the parameters `wrt` randomly
# and creates a new GradientDescent optimizer `opt`.

import scipy

from climin import GradientDescent
from climin.stops import *

# defining the loss and the derivative
quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x
dim = 10

### stop criterions in general, stops and pauses

# for this example, we use a very simple stop criterion,
# which is called `after_1000_iterations`, to illustrate
# how stop criterions work. There are two ways in which
# criterions can be used:

# The first is to check the stop criterions manually, 
# which we will call a "pause".

wrt = scipy.random.standard_normal((dim,)) * 10 + 5
opt = GradientDescent(wrt, quadraticprime, steprate=0.01)

for info in opt:
    print "iteration %3i loss=%g" % (info['n_iter'], quadratic(wrt))
    if after_1000_iterations(info):
        print "1000 iterations done."
        break

# as you can see above, a stop criterion always takes the info 
# dictionary as argument, and returns either True or False.
# Here, we break if the criterion returns True, but this is not 
# necessary. We could just react in some other way and afterwards
# continue minimization. That's why it is called a "pause".

# The second option is to run the optimizer until a stop condition
# occurs, without requiring control in between steps:

wrt = scipy.random.standard_normal((dim,)) * 10 + 5
opt = GradientDescent(wrt, quadraticprime, steprate=0.01)
info = opt.minimize_until(after_1000_iterations)
print "1000 iterations done."

# the optimizer's minimize_until() method takes either a single
# stop criterion or a list of criterions and iterates until
# at least 1 stop criterion is True (logical OR). The last info
# dictionary is then passed out as a return value.
# We consider this way of using criterions a "stop" rather than 
# a "pause", because there is no way to continue minimizing
# after the stop criterion was fulfilled.

# minimize_until() is a convenience function that should only
# be used if the user does not require any interaction during 
# the minimization process and only wants the end result.

### stop criterion factories

# TODO

### stop after some time

# TODO

### stop after number of iterations

# TODO 

### stop on convergence 

# the `converged` stop criterion allows to stop when a certain
# value has converged. the first argument of the converged
# stop criterion factory is a function `func` that takes no parameters,
# but returns the value of which we seek convergence. There
# are two optional parameters with default values, n=10 and 
# epsilon=1e-5, which can be changed if needed.
# The converged stop criterion contains a ring buffer of length n,
# which it fills with consecutive calls to func(). The stopping
# condition is fulfilled, if the ringbuffer is completely filled
# (at least n iterations), and the difference between the
# maximum value and the minimum value of the buffer is less than
# epsilon. This means that all n values lie within an epsilon
# envelope.

# In this example we want to stop when the loss is converged.
# we pass an anonymous lambda function into the factory, that
# returns the loss based on the parameters wrt. 

wrt = scipy.random.standard_normal((dim,)) * 10 + 5
opt = GradientDescent(wrt, quadraticprime, steprate=0.01)
loss_converged = converged(lambda: quadratic(wrt))    
       
for info in opt:
    print "iteration %3i loss=%g" % (info['n_iter'], quadratic(wrt))
    if loss_converged(info):
        print "loss converged."
        break
    

### create your own stop criterions

# TODO

