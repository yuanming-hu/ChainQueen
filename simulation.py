import tensorflow as tf
import cv2
import os
import numpy as np
import random
import math

from vector_math import *
'''
TODO:
dx
'''

batch_size = 1
particle_count = 100
gravity = (0, -9.8)
dt = 0.05
total_steps = 10
res = 20
dim = 2



class State:

  def __init__(self):
    pass

  def get_evaluated(self):
    return {
      'position': self.position,
      'velocity': self.velocity,
      'mass': self.mass,
      'grid': self.grid,
      'deformation_gradient': self.deformation_gradient,
      'kernels': self.kernels,
    }

  @staticmethod
  def compute_kernels(positions):
    grid_node_coord = [[(i, j) for j in range(-1, 2)] for i in range(-1, 2)]
    grid_node_coord = np.array(grid_node_coord)[None, None, :, :]
    frac = (positions - tf.floor(positions))[:, :, None, None, :]

    x = tf.abs(frac - grid_node_coord)
    #print('x', x.shape)

    mask = tf.cast(x < 0.5, tf.float32)
    y = mask * (0.75 - x * x) + (1 - mask) * (0.5 * (1.5 - x) ** 2)
    #print('y', y.shape)
    y = tf.reduce_prod(y, axis=4, keepdims=True)
    #print('y', y.shape)
    return y

# Initial State
class InitialState(State):

  def __init__(self):
    super().__init__()
    self.position = tf.placeholder(
      tf.float32, [batch_size, particle_count, dim], name='position')
    self.velocity = tf.placeholder(
      tf.float32, [batch_size, particle_count, dim], name='velocity')
    self.deformation_gradient = tf.placeholder(
      tf.float32, [batch_size, particle_count, dim * dim], name='dg')
    self.mass = tf.zeros(shape=(batch_size, res, res, 1))
    self.grid = tf.zeros(shape=(batch_size, res, res, dim))
    self.kernels = tf.zeros(shape=(batch_size, res, res, 3, 3))
    '''
    TODO:
    mass, volume, Lame parameters (Young's modulus and Poisson's ratio)
    '''


# Updated state
class UpdatedState(State):

  def __init__(self, previous_state):
    super().__init__()
    # Rotational velocity field
    self.velocity = (previous_state.position) * 0 + 1

    # Advection

    self.grid = tf.zeros(shape=(batch_size, res, res, dim))

    # Rasterize mass and velocity
    base_indices = tf.cast(tf.floor(previous_state.position - 0.5), tf.int32)
    assert batch_size == 1
    # print('base indices', base_indices.shape)
    # Add the batch size indices
    base_indices = tf.concat(
      [
        tf.zeros(shape=(batch_size, particle_count, 1), dtype=tf.int32),
        base_indices
      ],
      axis=2)
    # print('base indices', base_indices.shape)
    self.mass = tf.zeros(shape=(batch_size, res, res, 1))

    # Momentum and velocity
    self.grid = tf.zeros(shape=(batch_size, res, res, dim))

    self.kernels = self.compute_kernels(previous_state.position)
    assert self.kernels.shape == (batch_size, particle_count, 3, 3, 1)

    for i in range(3):
      for j in range(3):
        assert batch_size == 1
        delta_indices = np.array([0, i, j])[None, None, :]
        #print((base_indices + delta_indices).shape)
        self.mass = self.mass + tf.scatter_nd(
          shape=(batch_size, res, res, 1),
          indices=base_indices + delta_indices,
          updates=self.kernels[:, :, i, j])


        grid_velocity_contributions = self.kernels[:, :, i, j] * self.velocity
        self.grid = self.grid + tf.scatter_nd(
          shape=(batch_size, res, res, dim),
          indices=base_indices + delta_indices,
          updates=grid_velocity_contributions)
    assert self.mass.shape == (batch_size, res, res, 1), 'shape={}'.format(self.mass.shape)

    # Resample

    self.deformation_gradient = previous_state.deformation_gradient

    # Boundary conditions

    self.position = previous_state.position + self.velocity * dt


class Simulation:

  def __init__(self, sess):
    self.sess = sess
    self.initial_state = InitialState()
    self.updated_states = []
    previous_state = self.initial_state

    for i in range(total_steps):
      new_state = UpdatedState(previous_state)
      self.updated_states.append(new_state)
      previous_state = new_state

    self.states = [self.initial_state] + self.updated_states

  def run(self):
    results = [s.get_evaluated() for s in self.states]

    feed_dict = {
      self.initial_state.position: [[[
        random.uniform(0.3, 0.4) * res,
        random.uniform(0.3, 0.5) * res
      ] for i in range(particle_count)]],
      self.initial_state.velocity: [[[0, 0] for i in range(particle_count)]],
      self.initial_state.deformation_gradient:
        np.array([1, 0, 0, 1])[None, None, :] +
        np.zeros(shape=(batch_size, particle_count, 1))
    }

    results = self.sess.run(results, feed_dict=feed_dict)

    while True:
      for i, r in enumerate(results):
        self.visualize(i, r)

  def visualize(self, i, r):
    pos = r['position'][0]
    mass = r['mass'][0]
    grid = r['grid'][0]
    kernel_sum = np.sum(r['kernels'][0], axis=(1, 2))
    if i > 0:
      np.testing.assert_array_almost_equal(kernel_sum, 1, decimal=3)
      np.testing.assert_array_almost_equal(mass.sum(), particle_count, decimal=3)

    scale = 20

    # Pure-white background
    img = np.ones((scale * res, scale * res, 3), dtype=np.float)

    for p in pos:
      x, y = tuple(map(lambda x: math.ceil(x * scale), p))
      #if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
      #  img[x, y] = (0, 0, 1)
      cv2.circle(img, (y, x), radius=scale // 3, color=(0, 0, 1), thickness=-1)

    img = img.swapaxes(0, 1)[::-1, :, ::-1]
    mass = mass.swapaxes(0, 1)[::-1, :, ::-1]
    grid = grid.swapaxes(0, 1)[::-1, :, ::-1]
    grid = np.concatenate([grid, grid[:, :, 0:1] * 0], axis=2)
    mass = cv2.resize(mass, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    cv2.imshow('Particles', img)
    cv2.imshow('Mass', mass / 10)
    cv2.imshow('Velocity', grid)
    cv2.waitKey(1)
