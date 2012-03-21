import itertools
import random

from climin import tonga, GradientDescent

import pylab
import scipy
import theano
import theano.tensor as T
import zeitgeist.data
import zeitgeist.util as util
from zeitgeist.model import transfermap
import numpy as np


# Hyper parameters.

n_inpt = 784
n_hidden = 300
n_output = 10


#Expressions for a one-layer network
def mlp(insize, hiddensize, outsize, transferfunc='tanh', outfunc='id'):
    P = util.ParameterSet(
        inweights=(insize, hiddensize),
        hiddenbias=hiddensize,
        outweights=(hiddensize, outsize),
        outbias=outsize)

    P.randomize(1e-4)

    inpt = T.matrix('inpt')
    hidden_in = T.dot(inpt, P.inweights)
    hidden_in += P.hiddenbias

    nonlinear = transfermap[transferfunc]
    hidden = nonlinear(hidden_in)
    output_in = T.dot(hidden, P.outweights)
    output_in += P.outbias
    output = output_in
    output = transfermap[outfunc](output_in)

    exprs = {'inpt': inpt,
             'hidden-in': hidden_in,
             'hidden': hidden,
             'output-in': output_in,
             'output': output}
    return exprs, P




exprs, P = mlp(n_inpt, n_hidden, n_output, transferfunc='sig', outfunc='softmax')
# To make the passing of the parameters explicit, we need to substitute it
# later with the givens parameter.
par_sub = T.vector()

# Some tensor for constructing the loss. the loss will only be defined on
# the end of the sequences.
inpt = exprs['inpt']
target = T.matrix('target')
output = exprs['output']
output_in = exprs['output-in']

# Shorthand to create a cross entropy expression
def cross_entropy(a, b):
    epsilon = 0
    return -(a * T.log(b+epsilon)).sum(axis=1)

# Vector for the expression of the Hessian vector product, where this will
# be the vector.
p = T.vector('p')


# The loss and its gradient.
loss = cross_entropy(target, output)
lossmean = T.mean(loss)
lossgrad = T.grad(lossmean, P.flat)
lossjacobian = theano.gradient.jacobian(loss, P.flat)

# Functions.
givens = {P.flat: par_sub}
f = theano.function([par_sub, inpt, target], lossmean, givens=givens)
fraw = theano.function([par_sub, inpt, target], loss, givens=givens)
fjacobian = theano.function([par_sub, inpt, target], lossjacobian, givens=givens)
fgrad = theano.function([par_sub, inpt, target], lossgrad, givens=givens)

# Build a dataset.
#MNIST
import cPickle, gzip

# Load the dataset
with gzip.open('mnist.pkl.gz','rb') as MNISTfile:
    train_set, valid_set, test_set = cPickle.load(MNISTfile)

X, labels = train_set
N, _ = X.shape
Y = np.zeros((N, 10))
for i in range(N):
    Y[i][labels[i]] = 1

print "data loaded"
#initialization
P['hiddenbias'][:] = np.random.randn(n_hidden)*1e-4
P['outbias'][:] = np.random.randn(n_output)*1e-4

P['inweights'][:,:] = np.random.randn(n_inpt, n_hidden)*1e-4
P['outweights'][:,:] = np.random.randn(n_hidden, n_output)*1e-4




#minibatches 
batchsize = 500
n_batches = N/batchsize
random_numbers = (random.randint(0, n_batches - 1) for _ in itertools.count())
idxs = ((r * batchsize, (r + 1) * batchsize) for r in random_numbers)
minibatches = ((X[lower:upper], Y[lower:upper]) for lower, upper in idxs)
args = ((m, {}) for m in minibatches)

print '#pars:', P.data.size

## import numericalGradientChecker

## checker = numericalGradientChecker.numericalGradientChecker(fraw, fprime, inputDim=P.data.size, outputDim=batchsize, args=args, bounds=None)
## for i, info in enumerate(checker):
##     print "errors in gradient", info['errors']
##     if i>0:
##         break

import chopmunk

ignore = ['args', 'kwargs', 'gradient', 'Hp']
console_sink = chopmunk.prettyprint_sink()
console_sink = chopmunk.dontkeep(console_sink, ignore)

file_sink = chopmunk.file_sink('mnist.log')
file_sink = chopmunk.jsonify(file_sink)
file_sink = chopmunk.dontkeep(file_sink, ignore)

logger = chopmunk.broadcast(console_sink, file_sink)
logfunc = logger.send


blocksizes = np.ones(n_hidden + n_output)
blocksizes[:n_hidden] *= (n_inpt+1)
blocksizes[n_hidden:] *= (n_hidden+1)

opt = tonga(P.data, fgrad, fjacobian
            , damping=1e-4, blocksizes=blocksizes, args=args, logfunc=logfunc)
#opt = GradientDescent(P.data, fgrad, steprate = 1e-4, args=args, logfunc=logfunc)

print "initialization done"

lossTab = scipy.empty(503)

for i, info in enumerate(opt):
    x, y = info['args']
    loss = f(P.data, x, y)
    print 'iteration', i
    print 'loss', loss
    lossTab[i] = loss
    
    
    logfunc(info)
    if i > 500:
        break

fileLog = open('logTonga', 'w')
fileLog.write('')
fileLog.close()
fileLog = open('logTonga', 'a')
for i in range(501):
    fileLog.write(str(lossTab[i]))
    fileLog.write('\n')
fileLog.close()

pylab.plot(lossTab)
pylab.show()

