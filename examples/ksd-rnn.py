import itertools
import random

from climin import KrylovSubspaceDescent, HessianFree, Rprop

import pylab
import scipy
import theano
import theano.tensor as T
from zeitgeist.model import rnn, lstmrnn
import zeitgeist.data


# Hyper parameters.

n_inpt = 1
n_hidden = 100
n_output = 1
n_memory = 5
structural_damping_factor = 0.00

# Expressions.

exprs, P = rnn(n_inpt, n_hidden, n_output, transferfunc='sig', outfunc='sig')
par_sub = T.vector()

target = T.tensor3('target')
output = exprs['output']
hidden = exprs['hidden']

subtarget = target[-n_memory:]
suboutput = output[-n_memory:]

def cross_entropy(a, b):
    return -(a * T.log(b + 1e-8) + (1 - a) * T.log(1 - b + 1e-8)).mean()

p = T.vector('p')

changed_hidden = theano.clone(hidden, {P.flat: P.flat + p})
diff_hidden = cross_entropy(hidden, changed_hidden) 

loss = cross_entropy(subtarget, suboutput)
lossgrad = T.grad(loss, P.flat)

damping = structural_damping_factor * diff_hidden
damped_loss = loss + damping

#Hp = T.Rop(lossgrad, P.flat, p)
#Hp = T.grad(T.sum(lossgrad * p), P.flat)

Jp = T.Rop(output, P.flat, p)
HJp = T.grad(T.sum(T.grad(damped_loss, output) * Jp),
             output, consider_constant=[Jp])
Gp = T.grad(T.sum(HJp * output), P.flat, consider_constant=[HJp, Jp])


# Functions.
inpt = exprs['inpt']
givens = {P.flat: par_sub}
f = theano.function([par_sub, inpt, target], loss, givens=givens)
fprime = theano.function([par_sub, inpt, target], lossgrad, givens=givens)
f_Hp = theano.function([par_sub, p, inpt, target], Gp, givens=givens)
f_predict = theano.function([par_sub, inpt], exprs['output'], givens=givens)

# Build a dataset.
n_samples = 500
X = scipy.ones((30, 2 * n_samples, 1)) * 0.2
X[:n_memory] = scipy.random.random(X[:n_memory].shape) > 0.5
X[-n_memory] = scipy.ones(X[0].shape)
Z = scipy.ones(X.shape) * 0.5
Z[-n_memory:] = X[:n_memory]

TX = X[:, n_samples:]
TZ = Z[:, n_samples:]
X = X[:, :n_samples]
Z = Z[:, :n_samples]


#t = scipy.arange(0, 10, 0.05)[:, scipy.newaxis, scipy.newaxis]
#X = scipy.sin(t / 2) * 2 + scipy.sin(t)
#Z = X[:-20].copy()
#X = X[20:]

P.randomize(1E-4)
args = (([X, Z], {}) for _ in itertools.repeat(()))
#X_minibatches = zeitgeist.data.minibatches(X, 100, d=1)
#Z_minibatches = zeitgeist.data.minibatches(Z, 100, d=1)
#minibatches = zip(X_minibatches, Z_minibatches)
#idxs = (random.randint(0, len(minibatches) - 1) for _ in itertools.count())
# cg_args = ((minibatches[i], {}) for i in idxs)
cg_args = args
kargs = (([X, Z], {}) for _ in itertools.repeat(()))
Hargs = (([X, Z], {}) for _ in itertools.repeat(()))

print '#pars:', P.data.size

def logfunc(info): 
    return
    print info
    print

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
        initial_damping=0.5,
        logfunc=logfunc)

for i, info in enumerate(opt):
    X, Z = info['args']
    loss = f(P.data, X, Z)
    vloss = f(P.data, TX, TZ)
    print 'loss', loss, vloss
    if i > 300:
        break

#pylab.plot(steps)
#pylab.plot(losses)
#pylab.show()
#
#pylab.plot(f_predict(X)[:, 0, 0])
#pylab.plot(Z[:, 0, 0])
#pylab.show()

P = f_predict(P.data, TX)[-5:, :3, 0]
print (P > 0.5).astype('uint8')
print P
print TZ[-5:, :3, 0]
