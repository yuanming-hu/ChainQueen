import os
import time
import unittest
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
from IPython import embed
import mpm3d
from simulation import Simulation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sess = tf.Session()

def main():
  #shape = (1024, 1024, 256)
  shape = (100, 1024, 256)
  ones = np.ones(shape=shape)

  a = tf.Variable(ones, dtype=tf.float32)
  b = tf.Variable(ones, dtype=tf.float32)

  sess.run(tf.global_variables_initializer())
  #op = tf.reduce_max(tf.assign(a, a + b))
  op = tf.reduce_max(a + b)
  #op = tf.assign(a, a)

  for i in range(1000):
    t = time.time()
    ret = sess.run(op)
    print(ret, time.time() - t)

if __name__ == '__main__':
  main()
