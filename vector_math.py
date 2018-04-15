import tensorflow as tf
import numpy as np

def polar_decomposition(m):
  assert False

def matmatmul(a, b):
  assert len(a.shape) == 4 # Batch, particles, row, column
  assert len(b.shape) == 4 # Batch, particles, row, column
  dim = 2
  c = [[None for i in range(dim)] for j in range(dim)]
  for i in range(dim):
    for j in range(dim):
      for k in range(dim):
        if k == 0:
          c[i][j] = a[:, :, i, k] * b[:, :, k, j]
        else:
          c[i][j] += a[:, :, i, k] * b[:, :, k, j]
  row0 = tf.stack([c[0][0], c[0][1]], axis=2)
  row1 = tf.stack([c[1][0], c[1][1]], axis=2)
  C = tf.stack([row0, row1], axis=2)
  return C

def matvecmul(a, b):
  assert len(a.shape) == 4 # Batch, particles, row, column
  assert len(b.shape) == 3 # Batch, particles, row
  dim = 2
  c = tf.zeros_like(b)
  for i in range(dim):
    for k in range(dim):
      c[:, :, i] += a[:, :, i, k] * b[:, :, k]
  return c


# a is column, b is row
def outer_product(a, b):
  assert len(a.shape) == 3 # Batch, particles, row
  assert len(b.shape) == 3 # Batch, particles, row
  dim = 2
  c = tf.zeros(b.shape + (dim,))
  for i in range(dim):
    for j in range(dim):
      c[:, :, i, j] = a[:, :, j] * b[:, :, i]
  return c

if __name__ ==  '__main__':
  a = np.random.randn(2, 2)
  b = np.random.randn(2, 2)
  with tf.Session() as sess:
    prod1 = np.matmul(a, b)
    prod2 = matmatmul(tf.constant(a[None, None, :, :]), tf.constant(b[None, None, :, :]))
    prod2 = sess.run(prod2)[0, 0]
    np.testing.assert_array_almost_equal(prod1, prod2)
  print("All tests passed.")
