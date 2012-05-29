import itertools
import random
import time
import sys

from climin import tonga, NaturalNewton, Lbfgs, NaturalNewtonLR

import scipy
import theano
import theano.tensor as T
import zeitgeist.data
import zeitgeist.util as util
from zeitgeist.model import transfermap
import numpy as np

## seed = random.randint(0, sys.maxint)
## random.seed(seed)
## print seed

#random.seed(8561093027007434982)

# Hyper parameters.

n_inpt = 784
n_hidden = 300
n_output = 10

#optimizer = 'tonga'
#optimizer = 'lbfgs'
optimizer = 'nn'



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
    return -(a * T.log(b+epsilon)).mean()

# Vector for the expression of the Hessian vector product, where this will
# be the vector.
p = T.vector('p')


# The loss and its gradient.
loss = cross_entropy(target, output)
lossgrad = T.grad(loss, P.flat)
#misclassifications = 50 * T.mean(T.sum(T.abs_(target-T.round(output)), axis=1))
misclassifications = (1-T.mean(T.eq(T.argmax(target, axis = 1), T.argmax(output, axis = 1))))*100



# Functions.
givens = {P.flat: par_sub}
f = theano.function([par_sub, inpt, target], loss, givens=givens)
fprime = theano.function([par_sub, inpt, target], lossgrad, givens=givens)
fmisclass = theano.function([par_sub, inpt, target], misclassifications, givens=givens)

# Build a dataset.
#MNIST
import cPickle, gzip

# Load the dataset
MNISTfile = gzip.open('mnist.pkl.gz','rb') 
train_set, valid_set, test_set = cPickle.load(MNISTfile)
MNISTfile.close()


X, labels = train_set
Xval, labelsVal = valid_set
Xtest, labelsTest = test_set

N, _ = X.shape
Nval, _ = Xval.shape
Ntest, _ = Xtest.shape

Y = np.zeros((N, 10))
Yval = np.zeros((Nval, 10))
Ytest = np.zeros((Ntest, 10))

for i in range(N):
    Y[i][labels[i]] = 1
for i in range(Nval):
    Yval[i][labelsVal[i]] = 1
for i in range(Ntest):
    Ytest[i][labelsTest[i]] = 1

    
print "data loaded"


#initialization
factor1 = 4*np.sqrt(6./float(n_inpt + n_hidden))
factor2 = 4*np.sqrt(6./float(n_output + n_hidden))
P['hiddenbias'][:] = np.random.randn(n_hidden) * factor1
P['outbias'][:] = np.random.randn(n_output) * factor1

P['inweights'][:,:] = np.random.randn(n_inpt, n_hidden)* factor2
P['outweights'][:,:] = np.random.randn(n_hidden, n_output)* factor2




#minibatches 
batchsize = 5000
n_batches = N/batchsize
random_numbers = (random.randint(0, n_batches - 1) for _ in itertools.count())
idxs = ((r * batchsize, (r + 1) * batchsize) for r in random_numbers)
minibatches = ((X[lower:upper], Y[lower:upper]) for lower, upper in idxs)
args = ((m, {}) for m in minibatches)

cov_batchsize = 150
cov_n_batches = N/cov_batchsize
cov_random_numbers = (random.randint(0, cov_n_batches - 1) for _ in itertools.count())
cov_idxs = ((r * cov_batchsize, (r + 1) * cov_batchsize) for r in cov_random_numbers)
cov_minibatches = ((X[lower:upper], Y[lower:upper]) for lower, upper in cov_idxs)
cov_args = ((m, {}) for m in cov_minibatches)

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


#size of blocks used for the diagonal approximation

blocksizes = np.ones(n_hidden + n_output)
blocksizes[:n_hidden] *= (n_inpt+1)
blocksizes[n_hidden:] *= (n_hidden+1)

if optimizer == 'tonga':
    opt = tonga(P.data, fprime, damping=1e-2, blocksizes=blocksizes, nb_estimates=200, args=args, cov_args=cov_args, logfunc=logfunc)
    fileName = 'logTonga'
elif optimizer == 'nn':
    opt = NaturalNewtonLR(P.data, f, fprime, blocksizes=blocksizes,  N=batchsize, args=args, logfunc=logfunc)
    fileName = 'logNNlr'
elif optimizer == 'lbfgs':
    opt = Lbfgs(P.data, f, fprime, args=args, logfunc=logfunc)
    fileName = 'logLbfgs'

fileLog = open(fileName, 'w')
fileLog.write('')
fileLog.close()

print "initialization done"


N_ITER_MAX = 1000
STEP = 5


lossTab = scipy.empty(N_ITER_MAX +3)
scheduleTab = scipy.empty(N_ITER_MAX +3)
lossTot =  scipy.empty(N_ITER_MAX/STEP+1)
lossVal = scipy.empty(N_ITER_MAX/STEP+1)
lossTest = scipy.empty(N_ITER_MAX/STEP+1)
misclassif = scipy.empty(N_ITER_MAX/STEP+1)
misclassifVal = scipy.empty(N_ITER_MAX/STEP+1)
misclassifTest = scipy.empty(N_ITER_MAX/STEP+1)

start = time.clock()
bestLoss = 25
bestPars = np.zeros(P.data.size)

for i, info in enumerate(opt):
    x, y = info['args']
    loss = f(P.data, x, y)
    print 'iteration', i
    print 'loss', loss
    lossTab[i] = loss
    scheduleTab[i] = time.clock()-start

    if i%STEP == 0:
        delay = time.clock()
        loss = f(P.data, *(X,Y), **{})
        lossval = f(P.data, *(Xval,Yval), **{})
        lossTot[i/STEP] = loss
        lossVal[i/STEP] = lossval
        lossTest[i/STEP] = f(P.data, *(Xtest,Ytest), **{})

        misclassif[i/STEP] = fmisclass(P.data, *(X,Y), **{})
        misclassifVal[i/STEP] = fmisclass(P.data, *(Xval,Yval), **{})
        misclassifTest[i/STEP] = fmisclass(P.data, *(Xtest,Ytest), **{}) 

        print '---'
        print 'loss on the whole dataset', loss
        print 'loss on the validation set', lossval
        print 'loss on the test set', lossTest[i/STEP]

        print 'misclassification on the training set: ', misclassif[i/STEP], '%'
        print 'misclassification on the validation set: ', misclassifVal[i/STEP], '%'
        print 'misclassification on the test set: ', misclassifTest[i/STEP], '%'
        print '---'

        if (lossval < bestLoss):
            bestLoss = lossval
            bestPars[:] = P.data[:]


        fileLog = open(fileName, 'a')
        for j in range(min(STEP,i),0,-1):
            fileLog.write(str(lossTab[i-j]))
            fileLog.write('\n')
            fileLog.write(str(scheduleTab[i-j]))
            fileLog.write('\n')        
        fileLog.write(str(lossTot[i/STEP]))
        fileLog.write('\n')     
        fileLog.write(str(lossVal[i/STEP]))
        fileLog.write('\n')
        fileLog.write(str(lossTest[i/STEP]))
        fileLog.write('\n')
        fileLog.write(str(misclassif[i/STEP]))
        fileLog.write('\n')
        fileLog.write(str(misclassifVal[i/STEP]))
        fileLog.write('\n')
        fileLog.write(str(misclassifTest[i/STEP]))
        fileLog.write('\n')
        fileLog.close()

        start += (time.clock()-delay)

    logfunc(info)
    if (i >= N_ITER_MAX):
        fileLog = open(fileName, 'a')
        fileLog.write(str(lossTab[i]))
        fileLog.write('\n')
        fileLog.write(str(scheduleTab[i]))
        fileLog.write('\n')
        fileLog.close()
        break


print '---'
print 'best loss on the whole dataset', f(bestPars, *(X,Y), **{})
print 'best loss on the validation set', f(bestPars, *(Xval, Yval), **{})
print 'best loss on the test set', f(bestPars, *(Xtest, Ytest), **{})

print 'best misclassification on the training set: ', fmisclass(bestPars, *(X,Y), **{}) , '%'
print 'best misclassification on the validation set: ',fmisclass(bestPars, *(Xval,Yval), **{}) , '%'
print 'best misclassification on the test set: ',fmisclass(bestPars, *(Xtest,Ytest), **{})  , '%'
print '---'






