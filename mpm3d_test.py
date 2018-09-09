import os
import unittest
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
from IPython import embed
import mpm3d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MPMOpTest(unittest.TestCase):
    
    def test_forward(self):
    
        print('\n==============\ntest_forward start')
        
        with tf.Session('') as sess:
            x = tf.placeholder(tf.float32, shape = (1, 3, 1))
            v = tf.placeholder(tf.float32, shape = (1, 3, 1))
            C = tf.constant(np.zeros([1, 3, 3, 1]).astype(np.float32))
            f = np.zeros([1, 3, 3, 1]).astype(np.float32)
            f[0, 0, 0, 0] = 1
            f[0, 1, 1, 0] = 1
            f[0, 2, 2, 0] = 1
            F = tf.constant(f)
            xx, vv, CC, FF, PP, grid = mpm3d.mpm(x, v, C, F)
            step = mpm3d.mpm(xx, vv, CC, FF)
            feed_dict = {x: np.array([[[0.5], [0.5], [0.5]]]).astype(np.float32),
                v: np.array([[[0.1], [0.1], [0.1]]]).astype(np.float32)}
            o = sess.run(step, feed_dict = feed_dict)
            a, b, c, d, e, f = o
            print(o)
            print(f.max())

    def test_backward(self):
    
        print('\n==============\ntest_backward start')
        
        with tf.Session('') as sess:
            x = tf.placeholder(tf.float32, shape = (1, 3, 1))
            v = tf.placeholder(tf.float32, shape = (1, 3, 1))
            C = tf.constant(np.zeros([1, 3, 3, 1]).astype(np.float32))
            f = np.zeros([1, 3, 3, 1]).astype(np.float32)
            f[0, 0, 0, 0] = 1
            f[0, 1, 1, 0] = 1
            f[0, 2, 2, 0] = 1
            F = tf.constant(f)
            xx, vv, CC, FF, PP, grid = mpm3d.mpm(x, v, C, F)
            feed_dict = {x: np.array([[[0.5], [0.5], [0.5]]]).astype(np.float32),
                v: np.array([[[0.1], [0.1], [0.1]]]).astype(np.float32)}
            dimsum = tf.reduce_sum(xx)
            dydx = tf.gradients(dimsum, x)
            dydv = tf.gradients(dimsum, v)
            print('dydx', dydx)
            print('dydv', dydv)

            y0 = sess.run(dydx, feed_dict = feed_dict)
            y1 = sess.run(dydv, feed_dict = feed_dict)
            print(y0)
            print(y1)
                
if __name__ == '__main__':
    unittest.main()
