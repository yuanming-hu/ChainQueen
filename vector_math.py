import tensorflow as tf
import numpy as np


def polar_decomposition(m):
  assert False


def matmatmul(a, b):
  assert len(a.shape) == 4  # Batch, particles, row, column
  assert len(b.shape) == 4  # Batch, particles, row, column
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
  assert len(a.shape) == 4  # Batch, particles, row, column
  assert len(b.shape) == 3  # Batch, particles, row
  dim = 2
  c = [None for i in range(dim)]
  for i in range(dim):
    for k in range(dim):
      if k == 0:
        c[i] = a[:, :, i, k] * b[:, :, k]
      else:
        c[i] += a[:, :, i, k] * b[:, :, k]
  return tf.stack(c, axis=2)


# a is column, b is row
def outer_product(a, b):
  assert len(a.shape) == 3  # Batch, particles, row
  assert len(b.shape) == 3  # Batch, particles, row
  dim = 2
  c = [[None for i in range(dim)] for j in range(dim)]
  for i in range(dim):
    for j in range(dim):
      c[i][j] = a[:, :, j] * b[:, :, i]
  row0 = tf.stack([c[0][0], c[0][1]], axis=2)
  row1 = tf.stack([c[1][0], c[1][1]], axis=2)
  C = tf.stack([row0, row1], axis=2)
  return C


if __name__ == '__main__':
  a = np.random.randn(2, 2)
  b = np.random.randn(2, 2)
  c = np.random.randn(2, 1)
  d = np.random.randn(2, 1)
  with tf.Session() as sess:
    # Matmatmul
    prod1 = np.matmul(a, b)
    prod2 = matmatmul(
        tf.constant(a[None, None, :, :]), tf.constant(b[None, None, :, :]))
    prod2 = sess.run(prod2)[0, 0]
    np.testing.assert_array_almost_equal(prod1, prod2)

    # Matvecmul
    prod1 = np.matmul(a, c)
    prod2 = matvecmul(
        tf.constant(a[None, None, :, :]), tf.constant(c[None, None, :, 0]))
    prod2 = sess.run(prod2)[0, 0]
    np.testing.assert_array_almost_equal(prod1[:, 0], prod2)

    # outer_product
    prod2 = outer_product(
        tf.constant(c[None, None, :, 0]), tf.constant(d[None, None, :, 0]))
    prod2 = sess.run(prod2)[0, 0]
    for i in range(2):
      for j in range(2):
        np.testing.assert_array_almost_equal(c[j] * d[i], prod2[i, j])
  print("All tests passed.")
