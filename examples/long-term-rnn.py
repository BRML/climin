import itertools
import random

import chopmunk

from climin import KrylovSubspaceDescent, HessianFree, Rprop
from climin.linesearch import WolfeLineSearch
from climin.initialize import sparsify_columns

import pylab
import scipy
import numpy as np
import theano
import theano.tensor as T
from zeitgeist.model import rnn, lstmrnn
import zeitgeist.data


# Hyper parameters.

n_inpt = 2
n_hidden = 100
n_output = 2
n_memory = 5
initial_damping = .1
damping_to_structural_damping = 0.3

# Expressions for the recurrent network.
# TODO: this functionality needs to be include in here to remove zeitgeist 
# dependency.
exprs, P = rnn(n_inpt, n_hidden, n_output, transferfunc='sig')

# To make the passing of the parameters explicit, we need to substitute it
# later with the givens parameter.
par_sub = T.vector()

# Some tensor for constructing the loss. the loss will only be defined on
# the end of the sequences.
inpt = exprs['inpt']
target = T.tensor3('target')
output = exprs['output']
exp_output = T.exp(output)
output = exp_output / exp_output.sum(axis=2).dimshuffle(0, 1, 'x')
output_in = exprs['output-in']
hidden_in_rec = exprs['hidden-in-rec']
hidden = exprs['hidden']
subtarget = target[-n_memory:]
suboutput = output[-n_memory:]

# Shorthand to create a cross entropy expression, which we will need for
# structural damping as well as the overall loss.
def cross_entropy(a, b):
    return -(a * T.log(b)).mean()

# Vector for the expression of the Hessian vector product, where this will
# be the vector.
p = T.vector('p')

# Expression for the change of hidden variables if we move the parameters
# into direction p.
changed_hidden = theano.clone(hidden, {P.flat: P.flat + p})

# Expression for the difference between the hiddens of the moved parameters
# and the previous hiddens.
#diff_hidden = ((hidden - changed_hidden)**2).mean()
def bernoulli_cross_entropy(target, output):
    eps = 1E-8
    return -(T.log(output + eps) * target +
             T.log(1 - output + eps) * (1 - target)).mean() 
diff_hidden = bernoulli_cross_entropy(hidden, changed_hidden)

# The loss and its gradient.
#loss = cross_entropy(target, output)
loss = cross_entropy(target, output)
lossgrad = T.grad(loss, P.flat)
empirical_loss = T.eq(T.gt(suboutput, 0.5), subtarget).mean()

# Expression for the Gauss-Newton matrix for the loss.
Jp = T.Rop(output_in, P.flat, p)
HJp = T.grad(T.sum(T.grad(loss, output_in) * Jp),
             output_in, consider_constant=[Jp])
Hp = T.grad(T.sum(HJp * output_in), P.flat, consider_constant=[HJp, Jp])

# The loss for the damping which will only be included in our Gauss-Newton
# matrix.
damping_factor = T.dscalar('damping-factor')
structural_damping = diff_hidden
structural_damping *= damping_to_structural_damping * damping_factor

d_Jp = T.Rop(hidden_in_rec, P.flat, p)
d_HJp = T.grad(T.sum(T.grad(structural_damping, hidden_in_rec) * d_Jp),
             hidden_in_rec, consider_constant=[d_Jp])
d_Hp = T.grad(T.sum(d_HJp * hidden_in_rec), P.flat, consider_constant=[d_HJp, d_Jp])

Hp += d_Hp


# Functions.
givens = {P.flat: par_sub}
f = theano.function([par_sub, inpt, target], loss, givens=givens)
fprime = theano.function([par_sub, inpt, target], lossgrad, givens=givens)
f_Hp = theano.function([par_sub, p, damping_factor, inpt, target], Hp, 
                       givens=givens)
f_predict = theano.function([par_sub, inpt], exprs['output'], givens=givens)
f_empirical  = theano.function([par_sub, inpt, target], empirical_loss, givens=givens)
f_hidden = theano.function([par_sub, inpt], exprs['hidden'], givens=givens)

# Build a dataset.
n_samples = 128
n_timesteps = 60

X = scipy.zeros((n_timesteps, 2 * n_samples, 2))
X[:, :, 1] = 1

X[:n_memory, :, 0] = scipy.random.random(X[:n_memory, :, 0].shape)
X[:n_memory, :, 1] = 1 - X[:n_memory, :, 0]
X[:n_memory] = X[:n_memory] > 0.5
X[-n_memory, 0, :] = 4, 0

Z = scipy.zeros((n_timesteps, 2 * n_samples, 2))
Z[:, :, 1] = 1
Z[-n_memory:] = X[:n_memory]

TX = X[:, n_samples:]
TZ = Z[:, n_samples:]
X = X[:, :n_samples]
Z = Z[:, :n_samples]

P.randomize(1./15)

# Tune down the values of inputs to hiddens.
P['inweights'] *= 15

sparsify_columns(P['hiddenweights'], 15)

#hiddens = f_hidden(P.data, X[:, 0:1, :])
#import matplotlib.pyplot as plt
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(hiddens[:, 0, :])
#fig.savefig('initial.png')
#del fig
#1/0


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

ignore = ['args', 'kwargs', 'gradient', 'Hp', 'direction', 'step',
          'cg_minimum', 'basis', 'gradient_diff', 'coefficients', 'grad']
console_sink = chopmunk.prettyprint_sink()
console_sink = chopmunk.dontkeep(console_sink, ignore)

file_sink = chopmunk.file_sink('longterm.log')
file_sink = chopmunk.jsonify(file_sink)
file_sink = chopmunk.dontkeep(file_sink, ignore)

logger = chopmunk.broadcast(console_sink, file_sink)
logger = chopmunk.timify(logger)
logfunc = logger.send

optimizer = 'hf'

if optimizer == 'ksd':
    opt = KrylovSubspaceDescent(
        P.data, f, fprime, f_Hp, n_bases=30,
        args=args, logfunc=logfunc)
elif optimizer == 'rprop':
    opt = Rprop(P.data, f, fprime, args=args, logfunc=logfunc)
elif optimizer == 'hf':
    #line_search = WolfeLineSearch(P.data, f, fprime)
    opt = HessianFree(
        P.data, f, fprime, f_Hp, args=args, cg_args=cg_args,
        initial_damping=initial_damping,
        explicit_damping=True,
        logfunc=logfunc)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
for i, info in enumerate(opt):
    info['test-loss'] = f(P.data, TX, TZ)
    info['train-empirical'] = f_empirical(P.data, X, Z)
    info['test-empirical'] = f_empirical(P.data, TX, TZ)

    hiddens = f_hidden(P.data, X[:, 0:1, :])
    ax.plot(hiddens[:, 0, :])
    fig.savefig('%i.png' % i)
    ax.cla()

    logfunc(info)
    if i > 1000:
        break
