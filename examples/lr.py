"""Train logistic regression on the mnist data set."""

import cPickle
import gzip
import itertools
import sys

import numpy as np

import climin
import climin.util
import climin.initialize

tmpl = [(784, 10), 10]          # w is matrix and b a vector


def predict(parameters, inpt):
    w, b = climin.util.shaped_from_flat(parameters, tmpl)
    before_softmax = np.dot(inpt, w) + b
    softmaxed = np.exp(before_softmax - before_softmax.max(axis=1)[:, np.newaxis])
    return softmaxed / softmaxed.sum(axis=1)[:, np.newaxis]


def d_loss_wrt_pars(parameters, inpt, targets):
    p = predict(parameters, inpt)
    d_flat, (d_w, d_b) = climin.util.empty_with_views(tmpl)
    d_w[...] = np.dot(inpt.T, p - targets) / inpt.shape[0]
    d_b[...] = (p - targets).mean(axis=0)
    return d_flat


def loss(parameters, inpt, targets):
    predictions = predict(parameters, inpt)
    loss = -np.log(predictions) * targets
    return loss.sum(axis=1).mean()


def main():
    # Hyper parameters.
    optimizer = 'lbfgs'        # or use: ncg, lbfgs, rmsprop
    batch_size = 10000

    flat, (w, b) = climin.util.empty_with_views(tmpl)
    climin.initialize.randomize_normal(flat, 0, 0.1)

    datafile = 'mnist.pkl.gz'
    # Load data.
    with gzip.open(datafile, 'rb') as f:
        train_set, val_set, test_set = cPickle.load(f)

    X, Z = train_set
    VX, VZ = val_set
    TX, TZ = test_set

    def one_hot(arr):
        result = np.zeros((arr.shape[0], 10))
        result[xrange(arr.shape[0]), arr] = 1.
        return result

    Z = one_hot(Z)
    VZ = one_hot(VZ)
    TZ = one_hot(TZ)

    if batch_size is None:
        args = itertools.repeat(([X, Z], {}))
        batches_per_pass = 1
    else:
        args = climin.util.iter_minibatches([X, Z], batch_size, [0, 0])
        args = ((i, {}) for i in args)
        batches_per_pass = X.shape[0] / batch_size

    if optimizer == 'gd':
        opt = climin.GradientDescent(flat, d_loss_wrt_pars, steprate=0.1,
                                     momentum=.95, args=args)
    elif optimizer == 'lbfgs':
        opt = climin.Lbfgs(flat, loss, d_loss_wrt_pars, args=args)
    elif optimizer == 'ncg':
        opt = climin.NonlinearConjugateGradient(flat, loss, d_loss_wrt_pars,
                                                args=args)
    elif optimizer == 'rmsprop':
        opt = climin.RmsProp(flat, d_loss_wrt_pars, steprate=1e-4, decay=0.9,
                             args=args)
    elif optimizer == 'rprop':
        opt = climin.Rprop(flat, d_loss_wrt_pars, args=args)
    else:
        print 'unknown optimizer'
        return 1

    for info in opt:
        if info['n_iter'] % batches_per_pass == 0:
            print '%i/%i test loss: %g' % (
                info['n_iter'], batches_per_pass * 10, loss(flat, VX, VZ))
        if info['n_iter'] >= 10 * batches_per_pass:
            break


if __name__ == '__main__':
    sys.exit(main())
