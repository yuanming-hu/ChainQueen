import tensorflow as tf
import numpy as np
import unittest
import random
from tensorflow.python.framework import ops

scatter_inc_module = tf.load_op_library('Build/libscatter_inc.so')


def scatter_inc(base, indices, delta):
    return scatter_inc_module.scatter_inc(base, indices, delta)


@ops.RegisterGradient("ScatterInc")
def _ScatterIncGradient(op, grad):
    return [grad, None, scatter_inc_module.scatter_inc_grad(grad, op.inputs[1])]


@ops.RegisterGradient("ScatterIncGrad")
def _ScatterIncGradientGradient(op, grad):
    return [scatter_inc_module.scatter_inc_grad_grad(grad, op.inputs[1], op.inputs[0]), None]


class TestScatterAdd(unittest.TestCase):
    def setup(self):
        self.n = 3  # batch size
        self.m = 5  # len of indices/delta
        self.l = 7  # len of input/output

        self.base_ = tf.placeholder(dtype=tf.float32, shape=(self.n, self.l))
        self.base = np.random.rand(self.n, self.l)
        self.delta_ = tf.placeholder(dtype=tf.float32, shape=(self.n, self.m))
        self.delta = np.random.rand(self.n, self.m)

        self.indices_ = tf.placeholder(dtype=tf.int32, shape=(self.n, self.m))
        self.indices = np.zeros(dtype=np.int32, shape=(self.n, self.m))
        for k in range(self.n):
            for i in range(self.m):
                self.indices[k, i] = random.randrange(self.l)

        self.updated_ = scatter_inc(self.base_, self.indices_, self.delta_)

        self.loss = tf.reduce_sum(self.updated_ ** 2) * 0.5
        self.feed_dict = {
            self.base_: self.base,
            self.indices_: self.indices,
            self.delta_: self.delta,
        }
        self.grad = tf.gradients(self.loss, [self.delta_])[0]
        self.grad_loss = tf.reduce_sum(self.grad ** 2 - 1) * 0.5

    def test_forward(self):
        self.setup()

        updated = self.base.copy()
        for i in range(self.n):
            for j in range(self.m):
                updated[i][self.indices[i][j]] += self.delta[i][j]

        c_updated = self.updated_.eval(feed_dict=self.feed_dict)
        np.testing.assert_array_almost_equal(c_updated, updated)

    def test_grad(self):
        self.setup()

        error = tf.test.compute_gradient_error(x=self.delta_, x_shape=self.delta.shape, y=self.loss, y_shape=(),
                                               x_init_value=self.delta, extra_feed_dict={
                self.base_: self.base,
                self.indices_: self.indices
            })

        self.assertAlmostEqual(error, 0, delta=1e-3)

        error = tf.test.compute_gradient_error(x=self.base_, x_shape=self.base.shape, y=self.loss, y_shape=(),
                                               x_init_value=self.base, extra_feed_dict={
                self.delta_: self.delta,
                self.indices_: self.indices
            })

        self.assertAlmostEqual(error, 0, delta=1e-3)

    def test_grad_grad(self):
        self.setup()

        error = tf.test.compute_gradient_error(x=self.delta_, x_shape=self.delta.shape, y=self.grad_loss, y_shape=(),
                                               x_init_value=self.delta, extra_feed_dict={
                self.base_: self.base,
                self.indices_: self.indices
            })

        self.assertAlmostEqual(error, 0, delta=1e-3)

        error = tf.test.compute_gradient_error(x=self.base_, x_shape=self.base.shape, y=self.grad_loss, y_shape=(),
                                               x_init_value=self.base, extra_feed_dict={
                self.delta_: self.delta,
                self.indices_: self.indices
            })

        self.assertAlmostEqual(error, 0, delta=3e-3)


    def test_normal_grad_grad(self):
        x = tf.placeholder(dtype=tf.float32, shape=(8, 5))
        loss = tf.reduce_sum(tf.sin(x))
        grad_loss = tf.reduce_sum(tf.gradients(loss, [x])[0] ** 2)

        error = tf.test.compute_gradient_error(x=x, x_shape=(8, 5), y=grad_loss, y_shape=(),
                                               x_init_value=np.random.randn(8, 5))
        self.assertAlmostEqual(error, 0, delta=2e-3)


if __name__ == '__main__':
    with tf.Session() as sess:
        unittest.main()
