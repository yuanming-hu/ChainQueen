import tensorflow as tf
import numpy as np


def make_matrix2d(m00, m01, m10, m11):
  assert len(m00.shape) == 2  # Batch, particles
  assert len(m01.shape) == 2  # Batch, particles
  assert len(m10.shape) == 2  # Batch, particles
  assert len(m11.shape) == 2  # Batch, particles
  row0 = tf.stack([m00, m01], axis=2)
  row1 = tf.stack([m10, m11], axis=2)
  return tf.stack([row0, row1], axis=2)


def polar_decomposition(m):
  # Reference: http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
  assert len(m.shape) == 4  # Batch, particles, row, column
  x = m[:, :, 0, 0] + m[:, :, 1, 1]
  y = m[:, :, 1, 0] - m[:, :, 0, 1]
  scale = 1.0 / tf.sqrt(x ** 2 + y ** 2)
  c = x * scale
  s = y * scale
  r = make_matrix2d(c, -s, s, c)
  return r, matmatmul(transpose(r), m)

def inverse(m):
  # Reference: http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
  assert len(m.shape) == 4  # Batch, particles, row, column
  Jinv = 1.0 / determinant(m)
  return Jinv[:, :, None, None] * make_matrix2d(m[:, :,1, 1], -m[:, :, 0, 1], -m[:, :, 1, 0], m[:, :, 0, 0])


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
  return make_matrix2d(c[0][0], c[0][1], c[1][0], c[1][1])


def transpose(a):
  assert len(a.shape) == 4  # Batch, particles, row, column
  dim = 2
  c = [[None for i in range(dim)] for j in range(dim)]
  for i in range(dim):
    for j in range(dim):
      c[i][j] = a[:, :, i, j]
  c[0][1], c[1][0] = c[1][0], c[0][1]
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


def determinant(a):
  assert len(a.shape) == 4  # Batch, particles, row, column
  return a[:, :, 0, 0] * a[:, :, 1, 1] - a[:, :, 1, 0] * a[:, :, 0, 1]


def trace(a):
  assert len(a.shape) == 4  # Batch, particles, row, column
  return a[:, :, 0, 0] + a[:, :, 1, 1]


if __name__ == '__main__':
  a = np.random.randn(2, 2)
  b = np.random.randn(2, 2)
  c = np.random.randn(2, 1)
  d = np.random.randn(2, 1)
  with tf.Session() as sess:
    # Polar decomposition
    R, S = polar_decomposition(tf.constant(a[None, None, :, :]))
    r, s = sess.run([R, S])
    r = r[0, 0]
    s = s[0, 0]
    np.testing.assert_array_almost_equal(np.matmul(r, s), a)
    np.testing.assert_array_almost_equal(np.matmul(r, np.transpose(r)), [[1, 0], [0, 1]])
    np.testing.assert_array_almost_equal(s, np.transpose(s))

    # Inverse
    prod2 = inverse(tf.constant(a[None, None, :, :]))
    prod2 = sess.run(prod2)[0, 0]
    np.testing.assert_array_almost_equal(np.matmul(prod2, a), [[1, 0], [0, 1]])

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

    # transpose
    prod1 = np.transpose(a)
    prod2 = transpose(tf.constant(a[None, None, :, :]))
    prod2 = sess.run(prod2)[0, 0]
    np.testing.assert_array_almost_equal(prod1, prod2)

    # outer_product
    prod2 = outer_product(
        tf.constant(c[None, None, :, 0]), tf.constant(d[None, None, :, 0]))
    prod2 = sess.run(prod2)[0, 0]
    for i in range(2):
      for j in range(2):
        np.testing.assert_array_almost_equal(c[j] * d[i], prod2[i, j])
  print("All tests passed.")
