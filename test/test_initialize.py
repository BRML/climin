# -*- coding: utf-8 -*-

import numpy as np
import unittest


from climin.initialize import sparsify_columns, orthogonal


def test_sparsify_columns():
    pars = np.ones((8, 10))
    sparsify_columns(pars, 3)
    assert (pars.sum(axis=0) == [3] * 10).all(), 'sparsify_columns did not work'


class OrthoInitTest(unittest.TestCase):

    @classmethod
    def isOrthogonal(cls, arr):
        """Product of orthonormal matrices is an identity"""
        res = arr.dot(arr.T) - np.eye(arr.shape[-2])
        return np.allclose(res, 0)

    def test_orthonormal(self):
        arr = np.empty((3, 3))
        orthogonal(arr)
        self.assertTrue(self.isOrthogonal(arr))
        self.assertAlmostEqual(abs(np.linalg.det(arr)), 1)

    def test_shape(self):
        true_shape = (3, 3)
        for shape in ((9,), (9, 1), (1, 9)):
            arr = np.empty(shape)
            orthogonal(arr, shape=true_shape)
            self.assertTrue(self.isOrthogonal(arr.reshape(true_shape)))

    def test_tensor(self):
        true_shape = (3, 3)
        for shape in ((3, 9,), (2, 9, 1), (4, 1, 9), (1, 2, 3, 3)):
            arr = np.empty(shape)
            orthogonal(arr, shape=true_shape)
            arr = arr.reshape([-1] + list(true_shape))
            for a in arr:
                self.assertTrue(self.isOrthogonal(a))

    def test_vector_init_raises(self):
        for shape in ((9,), (9, 1), (1, 9)):
            self.assertRaises(ValueError, orthogonal, np.empty(shape))

    def test_invalid_shape_raises(self):
        for shape in ((9,), (9, 1), (1, 9)):
            self.assertRaises(ValueError, orthogonal, np.empty(shape), shape=(2, 2))