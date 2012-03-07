import itertools
import random

from climin import KrylovSubspaceDescent, HessianFree, Rprop

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
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 2000
n_output = 10

initial_damping = 15




# Expressions for the deep network.
def deep_mlp(insize, hiddensize1, hiddensize2, hiddensize3, outsize, transferfunc='tanh', outfunc='id'):
    P = util.ParameterSet(
        inweights=(insize, hiddensize1),
        hiddenbias1=hiddensize1,
        hiddenweights1 = (hiddensize1, hiddensize2),
        hiddenbias2=hiddensize2,
        hiddenweights2 = (hiddensize2, hiddensize3),
        hiddenbias3=hiddensize3,
        outweights=(hiddensize3, outsize),
        outbias=outsize)

    P.randomize(0.1)

    inpt = T.matrix('inpt')
    hidden1_in = T.dot(inpt, P.inweights)
    hidden1_in += P.hiddenbias1

    nonlinear = transfermap[transferfunc]
    hidden1 = nonlinear(hidden1_in)

    hidden2_in = T.dot(hidden1, P.hiddenweights1)
    hidden2_in += P.hiddenbias2
    hidden2 = nonlinear(hidden2_in)

    hidden3_in = T.dot(hidden2, P.hiddenweights2)
    hidden3_in += P.hiddenbias3
    hidden3 = nonlinear(hidden3_in)

    
    output_in = T.dot(hidden3, P.outweights)
    output_in += P.outbias
    output = output_in
    output = transfermap[outfunc](output_in)

    exprs = {'inpt': inpt,
             'hidden1-in': hidden1_in,
             'hidden1': hidden1,
             'hidden2-in': hidden2_in,
             'hidden2': hidden2,
             'hidden3-in': hidden3_in,
             'hidden3': hidden3,
             'output-in': output_in,
             'output': output}
    return exprs, P


exprs, P = deep_mlp(n_inpt, n_hidden1, n_hidden2, n_hidden3, n_output, transferfunc='sig', outfunc='softmax')
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


# Expression for the Gauss-Newton matrix.
Jp = T.Rop(output_in, P.flat, p)
HJp = T.grad(T.sum(T.grad(loss, output_in) * Jp),
             output_in, consider_constant=[Jp])
Hp = T.grad(T.sum(HJp * output_in), P.flat, consider_constant=[HJp, Jp])


# Functions.
givens = {P.flat: par_sub}
f = theano.function([par_sub, inpt, target], loss, givens=givens)
fprime = theano.function([par_sub, inpt, target], lossgrad, givens=givens)
f_Hp = theano.function([par_sub, p, inpt, target], Hp, givens=givens)
f_predict = theano.function([par_sub, inpt], exprs['output'], givens=givens)

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

#sparse initialization
P['hiddenbias1'][:] = scipy.zeros(n_hidden1)
P['hiddenbias2'][:] = scipy.zeros(n_hidden2)
P['hiddenbias3'][:] = scipy.zeros(n_hidden3)
P['outbias'][:] = scipy.zeros(n_output)


P['inweights'][:,:] = np.random.randn(n_inpt, n_hidden1)
P['hiddenweights1'][:,:] = np.random.randn(n_hidden1, n_hidden2)
P['hiddenweights2'][:,:] = np.random.randn(n_hidden2, n_hidden3)
P['outweights'][:,:] = np.random.randn(n_hidden3, n_output)

def sparse_initialization(a, b, s, MaxNonZeroPerColumn = 15):
    for j in range(b):
        perm = np.random.permutation(a)
        P[s][perm[MaxNonZeroPerColumn:], j] *= 0

sparse_initialization( n_inpt, n_hidden1, 'inweights')
sparse_initialization( n_hidden1, n_hidden2, 'hiddenweights1')  
sparse_initialization( n_hidden2, n_hidden3, 'hiddenweights2')
sparse_initialization( n_hidden3, n_output, 'outweights')



#minibatches for HF: whole dataset
args = (([X, Y], {}) for _ in itertools.repeat(()))


#minibatches for cg: 5000
batchsize = 5000
n_batches = N/batchsize
random_numbers = (random.randint(0, n_batches - 1) for _ in itertools.count())
idxs = ((r * batchsize, (r + 1) * batchsize) for r in random_numbers)
minibatches = ((X[lower:upper], Y[lower:upper]) for lower, upper in idxs)
cg_args = ((m, {}) for m in minibatches)

print '#pars:', P.data.size


import chopmunk

ignore = ['args', 'kwargs', 'gradient', 'Hp']
console_sink = chopmunk.prettyprint_sink()
console_sink = chopmunk.dontkeep(console_sink, ignore)

file_sink = chopmunk.file_sink('mnist.log')
file_sink = chopmunk.jsonify(file_sink)
file_sink = chopmunk.dontkeep(file_sink, ignore)

logger = chopmunk.broadcast(console_sink, file_sink)
logfunc = logger.send

optimizer = 'hf'

if optimizer == 'ksd':
    opt = KrylovSubspaceDescent(
        P.data, f, fprime, f_Hp, n_bases=10,
        args=args, logfunc=logfunc)
elif optimizer == 'rprop':
    opt = Rprop(P.data, f, fprime, args=args, logfunc=logfunc)
elif optimizer == 'hf':
    opt = HessianFree(
        P.data, f, fprime, f_Hp, args=args, cg_args=cg_args,
        initial_damping=initial_damping,
        precond='martens',
        logfunc=logfunc)

for i, info in enumerate(opt):
    logfunc(info)
    if i > 100:
        break
    if info['step_length'] == 0 and (info['direction_m1'] == 0).all():
        break

#pylab.plot(steps)
#pylab.plot(losses)
#pylab.show()
#
#pylab.plot(f_predict(X)[:, 0, 0])
#pylab.plot(Z[:, 0, 0])
#pylab.show()

## P = f_predict(P.data, TX)[-5:, :3, 0]
## print (P > 0.5).astype('uint8')
## print P
