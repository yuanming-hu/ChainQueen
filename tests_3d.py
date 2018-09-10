import os
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

class TestSimulator3D(unittest.TestCase):

  def assertAlmostEqualFloat32(self, a, b, relative_tol=1e-3, clip=1e-3):
    if abs(a - b) > relative_tol * max(max(abs(a), abs(b)), clip):
      self.assertEqual(a, b)

  def test_forward(self):
    # print('\n==============\ntest_forward start')
    x = tf.placeholder(tf.float32, shape=(1, 3, 1))
    v = tf.placeholder(tf.float32, shape=(1, 3, 1))
    C = tf.constant(np.zeros([1, 3, 3, 1]).astype(np.float32))
    f = np.zeros([1, 3, 3, 1]).astype(np.float32)
    f[0, 0, 0, 0] = 1
    f[0, 1, 1, 0] = 1
    f[0, 2, 2, 0] = 1
    F = tf.constant(f)
    xx, vv, CC, FF, PP, grid = mpm3d.mpm(x, v, C, F)
    step = mpm3d.mpm(xx, vv, CC, FF)
    feed_dict = {
        x: np.array([[[0.5], [0.5], [0.5]]]).astype(np.float32),
        v: np.array([[[0.1], [0.1], [0.1]]]).astype(np.float32)
    }
    o = sess.run(step, feed_dict=feed_dict)
    a, b, c, d, e, f = o
    print(o)
    print(d.max())

  def test_backward(self):
    # print('\n==============\ntest_backward start')
    x = tf.placeholder(tf.float32, shape=(1, 3, 1))
    v = tf.placeholder(tf.float32, shape=(1, 3, 1))
    C = tf.constant(np.zeros([1, 3, 3, 1]).astype(np.float32))
    f = np.zeros([1, 3, 3, 1]).astype(np.float32)
    f[0, 0, 0, 0] = 1
    f[0, 1, 1, 0] = 1
    f[0, 2, 2, 0] = 1
    F = tf.constant(f)
    xx, vv, CC, FF, PP, grid = mpm3d.mpm(x, v, C, F)
    feed_dict = {
        x: np.array([[[0.5], [0.5], [0.5]]]).astype(np.float32),
        v: np.array([[[0.1], [0.1], [0.1]]]).astype(np.float32)
    }
    dimsum = tf.reduce_sum(xx)
    dydx = tf.gradients(dimsum, x)
    dydv = tf.gradients(dimsum, v)
    print('dydx', dydx)
    print('dydv', dydv)

    y0 = sess.run(dydx, feed_dict=feed_dict)
    y1 = sess.run(dydv, feed_dict=feed_dict)
    print(y0)
    print(y1)

  def test_bouncing_cube(self):
    gravity = (0, -10, 0)
    batch_size = 1
    dx = 0.03
    N = 10
    num_particles = N ** 3
    steps = 10
    dt = 1e-2
    sim = Simulation(
      grid_res=(30, 30, 30),
      dx=dx,
      num_particles=num_particles,
      gravity=gravity,
      dt=dt,
      batch_size=batch_size,
      sess=sess)
    position = np.zeros(shape=(batch_size, 3, num_particles))
    initial_velocity = tf.placeholder(shape=(3,), dtype=tf.float32)
    velocity = initial_velocity[None, :, None] + tf.zeros(shape=(batch_size, 3, num_particles))
    for b in range(batch_size):
      for i in range(N):
        for j in range(N):
          for k in range(N):
            position[b, :, i * N * N + j * N + k] =\
              (((i + b * 3) * 0.5 + 12.75) * dx, (j * 0.5 + 12.75) * dx,
               (k * 0.5 + 12.75) * dx)
    input_state = sim.get_initial_state(
      position=position,
      velocity=velocity)

    loss = tf.reduce_mean(sim.initial_state.center_of_mass()[:, 0])
    memo = sim.run(steps, input_state, initial_feed_dict={initial_velocity: [1, 0, 2]})

    sim.set_initial_state(input_state)
    sym = sim.gradients_sym(loss=loss, variables=[initial_velocity])

    sim.visualize(memo)
    grad = sim.eval_gradients(sym, memo)
    print(grad)
    self.assertAlmostEqualFloat32(grad[0][0], steps * dt)
    self.assertAlmostEqualFloat32(grad[0][1], 0)
    self.assertAlmostEqualFloat32(grad[0][2], 0)
    
  def test_gradients(self):
    gravity = (0, -10, 0)
    batch_size = 1
    dx = 0.03
    N = 3
    num_particles = N ** 3
    steps = 3
    dt = 1e-2
    sim = Simulation(
      grid_res=(30, 30, 30),
      dx=dx,
      num_particles=num_particles,
      gravity=gravity,
      dt=dt,
      batch_size=batch_size,
      sess=sess)
    
    position_ph = tf.placeholder(shape=(batch_size, 3, num_particles), dtype=tf.float32)
    velocity_ph = tf.placeholder(shape=(batch_size, 3, num_particles), dtype=tf.float32)
    
    position_val = np.zeros(shape=(batch_size, 3, num_particles))
    velocity_val = np.zeros(shape=(batch_size, 3, num_particles))
    
    for b in range(batch_size):
      for i in range(N):
        for j in range(N):
          for k in range(N):
            position_val[b, :, i * N * N + j * N + k] = \
              (((i + b * 3) * 0.5 + 12.75) * dx, (j * 0.5 + 12.75) * dx,
               (k * 0.5 + 12.75) * dx)
            
    input_state = sim.get_initial_state(position=position_ph, velocity=velocity_ph)
  
    loss = sim.initial_state.position[:, 0, 0]
  
    sim.set_initial_state(input_state)
    sym = sim.gradients_sym(loss=loss, variables=[position_ph, velocity_ph])
    
    def forward(pos, vel):
      memo = sim.run(steps, input_state, initial_feed_dict={velocity_ph: vel, position_ph: pos}, loss=loss)
      return memo.loss[0]
  
    #sim.visualize(memo)
    memo = sim.run(steps, input_state, initial_feed_dict={velocity_ph: velocity_val, position_ph: position_val})
    grad = sim.eval_gradients(sym, memo)
    delta = 1e-1
    dim = 3

    for i in range(dim):
      for j in range(num_particles):
        position_val[0, i, j] += delta
        v1 = forward(position_val, velocity_val)

        position_val[0, i, j] -= 2 * delta
        v2 = forward(position_val, velocity_val)

        position_val[0, i, j] += delta
        
        g = (v1 - v2) / (2 * delta)
        self.assertAlmostEqualFloat32(g, grad[0][0, i, j], clip=0, relative_tol=1e-3)
        
    for i in range(dim):
      for j in range(num_particles):
        velocity_val[0, i, j] += delta
        v1 = forward(position_val, velocity_val)
    
        velocity_val[0, i, j] -= 2 * delta
        v2 = forward(position_val, velocity_val)
    
        velocity_val[0, i, j] += delta
    
        g = (v1 - v2) / (2 * delta)
        print(g)
        self.assertAlmostEqualFloat32(g, grad[1][0, i, j], clip=1e-3, relative_tol=1e-3)

if __name__ == '__main__':
  unittest.main()
