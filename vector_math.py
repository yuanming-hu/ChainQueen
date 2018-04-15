import tensorflow as tf

def polar_decomposition(m):
  assert False

def matmatmul(a, b):
  assert len(a.shape) == 4 # Batch, particles, row, column
  assert len(b.shape) == 4 # Batch, particles, row, column
  dim = 2
  c = tf.zeros_like(a)
  for i in range(dim):
    for j in range(dim):
      for k in range(dim):
        c[:, :, i, j] += a[:, :, i, k] + b[:, :, k, j]
  return c

def matvecmul(a, b):
  assert len(a.shape) == 4 # Batch, particles, row, column
  assert len(b.shape) == 3 # Batch, particles, row
  dim = 2
  c = tf.zeros_like(b)
  for i in range(dim):
    for k in range(dim):
      c[:, :, i] += a[:, :, i, k] + b[:, :, k]
  return c
