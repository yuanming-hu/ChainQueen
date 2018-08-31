import unittest
import numpy as np
import tensorflow as tf

from scatter_inc import scatter_inc

import tensorflow as tf
from tensorflow.python.framework import ops

N = 32

def img2hist(imgs):
    N = 32
    size = int(imgs.shape[1])
    imgs = tf.reshape(imgs, shape=[-1, size * size, 3])

    imgs *= N - 1
    imgs = tf.clip_by_value(imgs, 0, N - 1 - 1e-4)

    imgs_int = tf.floor(imgs)
    imgs_frac = imgs - imgs_int
    imgs_int = tf.cast(imgs_int, tf.int32)

    hist = tf.zeros(shape=(imgs.shape[0], N * N * N), dtype=tf.float32)

    indices = imgs_int[:, :, 0] * N * N + imgs_int[:, :, 1] * N + imgs_int[:, :, 2]

    for l in range(8):
        i, j, k = (l % 2, l / 2 % 2, l / 4 % 2)
        weights = 1

        if i == 0:
            weights = weights * (1 - imgs_frac[:, :, 0])
        else:
            weights = weights * imgs_frac[:, :, 0]

        if j == 0:
            weights = weights * (1 - imgs_frac[:, :, 1])
        else:
            weights = weights * imgs_frac[:, :, 1]

        if k == 0:
            weights = weights * (1 - imgs_frac[:, :, 2])
        else:
            weights = weights * imgs_frac[:, :, 2]

        offset = i * N * N + j * N + k
        hist = scatter_inc(hist, indices + offset, weights)

    return tf.reshape(hist, shape=(-1, N, N, N, 1))

class Img2HistOpTest(unittest.TestCase):

    def test_zero(self):
        img = np.zeros(shape=(5, 64, 64, 3), dtype=np.float32)
        hist = img2hist(img).eval()
        for i in range(5):
            self.assertEqual(hist[i, 0, 0, 0, 0], 4096)
            hist[i, 0, 0, 0, 0] = 0
        np.testing.assert_array_equal(hist, np.zeros(shape=(5, 32, 32, 32, 1)))

    def test_special(self):
        img_input = np.zeros(shape=(5, 64, 64, 3), dtype=np.float32)
        img_input[2, 6, 8, 0] = 3.7 / (N - 1)
        img_input[2, 6, 8, 1] = 6.2 / (N - 1)
        img_input[2, 6, 8, 2] = 7.1 / (N - 1)
        hist = img2hist(img_input).eval()
        self.assertAlmostEqual(hist[2, 3, 6, 7], 0.3 * 0.8 * 0.9, delta=1e-6)
        self.assertAlmostEqual(hist[2, 3, 6, 8], 0.3 * 0.8 * 0.1, delta=1e-6)
        self.assertAlmostEqual(hist[2, 3, 7, 7], 0.3 * 0.2 * 0.9, delta=1e-6)
        self.assertAlmostEqual(hist[2, 3, 7, 8], 0.3 * 0.2 * 0.1, delta=1e-6)
        self.assertAlmostEqual(hist[2, 4, 6, 7], 0.7 * 0.8 * 0.9, delta=1e-6)
        self.assertAlmostEqual(hist[2, 4, 6, 8], 0.7 * 0.8 * 0.1, delta=1e-6)
        self.assertAlmostEqual(hist[2, 4, 7, 7], 0.7 * 0.2 * 0.9, delta=1e-6)
        self.assertAlmostEqual(hist[2, 4, 7, 8], 0.7 * 0.2 * 0.1, delta=1e-6)

    def test_special_gradient(self):
        img = tf.placeholder(tf.float32, shape=(5, 64, 64, 3))

        img_input = np.zeros(shape=(5, 64, 64, 3))
        mask = np.random.randn(5, 32, 32, 32)
        img_input[2, 6, 8, 0] = 3.7 / (N - 1)
        img_input[2, 6, 8, 1] = 6.2 / (N - 1)
        img_input[2, 6, 8, 2] = 7.1 / (N - 1)

        loss = tf.reduce_sum(img2hist(img) * mask[:, :, :, :, None])
        grad = tf.gradients(loss, [img])[0]

        grad = grad.eval(feed_dict={img: img_input})

        '''
        self.assertAlmostEqual(grad[2, 6, 8, 0] / N,
                               -0.8 * 0.9 * mask[2, 3, 6, 7]
                               - 0.8 * 0.1 * mask[2, 3, 6, 8]
                               - 0.2 * 0.9 * mask[2, 3, 7, 7]
                               - 0.2 * 0.1 * mask[2, 3, 7, 8]
                               + 0.8 * 0.9 * mask[2, 4, 6, 7]
                               + 0.8 * 0.1 * mask[2, 4, 6, 8]
                               + 0.2 * 0.9 * mask[2, 4, 7, 7]
                               + 0.2 * 0.1 * mask[2, 4, 7, 8]
                               , delta=1e-6)

        self.assertAlmostEqual(grad[2, 6, 8, 1] / N,
                               -0.3 * 0.9 * mask[2, 3, 6, 7]
                               - 0.3 * 0.1 * mask[2, 3, 6, 8]
                               - 0.7 * 0.9 * mask[2, 4, 6, 7]
                               - 0.7 * 0.1 * mask[2, 4, 6, 8]
                               + 0.3 * 0.9 * mask[2, 3, 7, 7]
                               + 0.3 * 0.1 * mask[2, 3, 7, 8]
                               + 0.7 * 0.9 * mask[2, 4, 7, 7]
                               + 0.7 * 0.1 * mask[2, 4, 7, 8]
                               , delta=1e-6)
        '''

        old_value = loss.eval(feed_dict={img: img_input})
        img_input[2, 6, 8, 2] += 0.001
        new_value = loss.eval(feed_dict={img: img_input})
        print 'num...', (new_value - old_value) / 0.001, grad[2, 6, 8, 2]


if __name__ == '__main__':
    with tf.Session('') as sess:
        unittest.main()
