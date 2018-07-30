import tensorflow as tf
import tensorflow.contrib.layers as ly
from functools import partial
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

lr = 1e-2
batch_size = 1
num_groups = 5
sample_density = 10
group_size = sample_density**2
particle_count = group_size * num_groups
group_offsets = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]
gravity = (0, -9.8)
#gravity = (0, 0)
dt = 0.03
actuation_strength = 1
total_steps = 25
res = 25
dim = 2

# Lame parameters
E = 500
nu = 0.3
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
sticky = False
linear = False

identity_matrix = np.array([[1, 0], [0, 1]])[None, None, :, :]


class State:

  def __init__(self, sim):
    self.sim = sim
    self.affine = tf.zeros(shape=(batch_size, particle_count, 2, 2))

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
    grid_node_coord = [[(i, j) for j in range(3)] for i in range(3)]
    grid_node_coord = np.array(grid_node_coord)[None, None, :, :]
    frac = (positions - tf.floor(positions - 0.5))[:, :, None, None, :]

    x = tf.abs(frac - grid_node_coord)
    #print('x', x.shape)

    mask = tf.cast(x < 0.5, tf.float32)
    y = mask * (0.75 - x * x) + (1 - mask) * (0.5 * (1.5 - x)**2)
    #print('y', y.shape)
    y = tf.reduce_prod(y, axis=4, keepdims=True)
    #print('y', y.shape)
    return y


# Initial State
class InitialState(State):

  def __init__(self, sim, initial_velocity):
    super().__init__(sim)
    self.t = 0
    self.position = tf.placeholder(
        tf.float32, [batch_size, particle_count, dim], name='position')

    broadcaster = [int(i > particle_count // 2) for i in range(particle_count)]
    self.velocity = np.array(broadcaster)[None, :, None] * initial_velocity[
        None, None, :]
    # print(self.velocity.shape)
    '''
    self.velocity = tf.placeholder(
        tf.float32, [batch_size, particle_count, dim], name='velocity')
    '''
    self.deformation_gradient = tf.placeholder(
        tf.float32, [batch_size, particle_count, dim, dim], name='dg')
    self.mass = tf.zeros(shape=(batch_size, res, res, 1))
    self.grid = tf.zeros(shape=(batch_size, res, res, dim))
    self.kernels = tf.zeros(shape=(batch_size, res, res, 3, 3))
    '''
    TODO:
    mass, volume, Lame parameters (Young's modulus and Poisson's ratio)
    '''


def particle_mask(start, end):
  r = tf.range(0, particle_count)
  return tf.cast(tf.logical_and(start <= r, r < end), tf.float32)[None, :]

def particle_mask_from_group(g):
  return particle_mask(g * group_size, (g + 1) * group_size)

# hidden_size = 10
W1 = tf.Variable(0.01 * tf.random_normal(shape=(2, 20)), trainable=True)
b1 = tf.Variable([[0.0, 0.0]], trainable=True)


# Updated state
class UpdatedState(State):

  def get_centroids(self, previous_state):
    # return centroid positions and velocities
    states = []
    for i in range(num_groups):
      mask = particle_mask(i * group_size, (i + 1) * group_size)[:, :, None] * (
          1.0 / group_size)
      pos = tf.reduce_sum(mask * previous_state.position, axis=1, keepdims=True)
      vel = tf.reduce_sum(mask * previous_state.velocity, axis=1, keepdims=True)
      states.append(pos)
      states.append(vel)
    states = tf.concat(states, axis=2)
    # print('states', states.shape)
    return states

  def __init__(self, sim, previous_state, actuation=None):
    super().__init__(sim)

    self.controller_states = self.get_centroids(previous_state)

    self.actuation = tf.tanh(
        tf.matmul(W1, self.controller_states[0, 0, :, None])[0] + b1) * actuation_strength
    self.actuation = self.actuation[0]
    # print(self.actuation.shape)

    self.t = previous_state.t + dt
    self.grid = tf.zeros(shape=(batch_size, res, res, dim))

    self.get_centroids(previous_state)

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

    # Compute stress tensor (First Piola-Kirchhoff stress)
    self.deformation_gradient = previous_state.deformation_gradient

    if linear:
      self.stress_tensor1 = mu * (
          transpose(self.deformation_gradient) + self.deformation_gradient -
          2 * identity_matrix)
      self.stress_tensor2 = lam * identity_matrix * (
          trace(self.deformation_gradient)[:, :, None, None] - dim)
    else:
      # Corotated elasticity
      r, s = polar_decomposition(self.deformation_gradient)
      j = determinant(self.deformation_gradient)[:, :, None, None]
      self.stress_tensor1 = 2 * mu * (self.deformation_gradient - r)
      self.stress_tensor2 = lam * (
          j - 1) * j * inverse(transpose(self.deformation_gradient))

    self.stress_tensor = self.stress_tensor1 + self.stress_tensor2
    if actuation is not None:
      self.stress_tensor += actuation
    else:
      # TODO make acutation a NN output
      left_actuation = self.actuation[0][None, None]
      right_actuation = self.actuation[1][None, None]
      left_mask = particle_mask(0, group_size)
      right_mask = particle_mask(group_size * (num_groups - 1), particle_count)
      #print(left_actuation.shape)
      #print(right_mask.shape)
      actuation = left_actuation * left_mask + right_actuation * right_mask
      zeros = tf.zeros(shape=(1, particle_count))
      #print(actuation)
      self.stress_tensor += E * make_matrix2d(zeros, zeros, zeros, actuation)
    self.stress_tensor = -1 * self.stress_tensor

    # Rasterize momentum and velocity
    # ... and apply gravity

    self.grid = tf.zeros(shape=(batch_size, res, res, dim))

    self.kernels = self.compute_kernels(previous_state.position)
    assert self.kernels.shape == (batch_size, particle_count, 3, 3, 1)

    self.velocity = previous_state.velocity
    for i in range(3):
      for j in range(3):
        assert batch_size == 1
        delta_indices = np.array([0, i, j])[None, None, :]
        #print((base_indices + delta_indices).shape)
        self.mass = self.mass + tf.scatter_nd(
            shape=(batch_size, res, res, 1),
            indices=base_indices + delta_indices,
            updates=self.kernels[:, :, i, j])

        delta_node_position = np.array([i, j])[None, None, :]
        offset = -(previous_state.position - tf.floor(previous_state.position - 0.5) - \
                 tf.cast(delta_node_position, tf.float32))

        grid_velocity_contributions = self.kernels[:, :, i, j] * (
            self.velocity + matvecmul(self.affine, offset) * 4)
        grid_force_contributions = self.kernels[:, :, i, j] * (
            matvecmul(self.stress_tensor, offset) * (-4 * dt))
        self.grid = self.grid + tf.scatter_nd(
            shape=(batch_size, res, res, dim),
            indices=base_indices + delta_indices,
            updates=grid_velocity_contributions + grid_force_contributions)
    assert self.mass.shape == (batch_size, res, res, 1), 'shape={}'.format(
        self.mass.shape)

    # self.velocity += np.array(gravity)[None, None, :] * dt
    self.grid += self.mass * np.array(gravity)[None, None, None, :] * dt
    self.grid = self.grid / tf.maximum(1e-30, self.mass)

    # Boundary conditions
    if sticky:
      self.grid = self.grid * self.sim.bc
    else:
      # TODO: use sim.bc
      mask = np.zeros((1, res, res, 2))
      mask[:, :, :4, 1] = 1
      self.grid = self.grid * (1 - mask) + mask * tf.maximum(self.grid, 0)
      mask = np.zeros((1, res, res, 2))
      mask[:, 3:res - 3, :res - 3] = 1
      self.grid = self.grid * mask

    # Resample velocity and local affine velocity field
    self.velocity *= 0
    for i in range(3):
      for j in range(3):
        assert batch_size == 1
        delta_indices = np.array([0, i, j])[None, None, :]
        self.velocity = self.velocity + tf.gather_nd(
            params=self.grid,
            indices=base_indices + delta_indices) * self.kernels[:, :, i, j]

        delta_node_position = np.array([i, j])[None, None, :]

        offset = -(previous_state.position - tf.floor(previous_state.position - 0.5) - \
                 tf.cast(delta_node_position, tf.float32))
        assert offset.shape == previous_state.position.shape
        weighted_node_velocity = tf.gather_nd(
            params=self.grid,
            indices=base_indices + delta_indices) * self.kernels[:, :, i, j]
        self.affine = self.affine + outer_product(weighted_node_velocity,
                                                  offset)

    dg_change = identity_matrix - (4 * dt) * self.affine
    #print(dg_change.shape)
    #print(previous_state.deformation_gradient)
    self.deformation_gradient = matmatmul(dg_change,
                                          previous_state.deformation_gradient)

    # Advection
    self.position = previous_state.position + self.velocity * dt


class Simulation:

  def __init__(self, sess):
    self.sess = sess
    self.initial_velocity = tf.placeholder(shape=(2,), dtype=tf.float32)
    self.initial_state = InitialState(
        self, initial_velocity=self.initial_velocity)
    self.updated_states = []

    # Boundary condition
    if sticky:
      self.bc = np.zeros((1, res, res, 2))
      self.bc[:, 4:res - 4, 4:res - 4] = 1
    else:
      self.bc = np.zeros((1, res, res, 2, 2))
      self.bc[:, :, 4:, 1, 0] = 1
      self.bc[:, :, :res - 4, 1, 1] = 1
      self.bc[:, 4:, :, 1, 0] = 1
      self.bc[:, :res - 4, :, 1, 1] = 1

    previous_state = self.initial_state

    for i in range(total_steps):
      new_state = UpdatedState(self, previous_state)
      self.updated_states.append(new_state)
      previous_state = new_state

    self.states = [self.initial_state] + self.updated_states

  def run(self):
    results = [s.get_evaluated() for s in self.states]

    feed_dict = {
        self.initial_state.position: [[[
            random.uniform(0.3, 0.5) * res,
            random.uniform(0.2, 0.4) * res
        ] for i in range(particle_count)]],
        self.initial_velocity: [0, 0],
        self.initial_state.deformation_gradient:
            identity_matrix +
            np.zeros(shape=(batch_size, particle_count, 1, 1))
    }

    results = self.sess.run(results, feed_dict=feed_dict)

    while True:
      for i, r in enumerate(results):
        self.visualize(i, r)

  def optimize(self):
    os.system('cd outputs && rm *.png')
    final_velocity = tf.reduce_mean(
        self.states[-1].velocity[:, :], keepdims=False, axis=(0, 1))
    # Note: taking the first half only
    final_position = tf.reduce_sum(
        self.states[-1].position * particle_mask_from_group(2)[:, :, None], keepdims=False, axis=(0, 1)) / group_size
    loss = (final_position[0] - res * 0.5)**2 + (
        final_position[1] - res * 0.6)**2

    current_velocity = np.array([0, 0], dtype=np.float32)
    results = [s.get_evaluated() for s in self.states]

    # Initial particle samples
    particles = [[]]

    for i, offset in enumerate(group_offsets):
      for x in range(sample_density):
        for y in range(sample_density):
          scale = 0.2
          u = ((x + 0.5) / sample_density + offset[0]) * scale + 0.2
          v = ((y + 0.5) / sample_density + offset[1]) * scale + 0.1
          particles[0].append([res * u, res * v])
    assert len(particles[0]) == particle_count

    counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    opt = ly.optimize_loss(
        loss=loss,
        learning_rate=lr,
        optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9),
        variables=trainables,
        global_step=counter)
    '''
    for i in range(40):
      print('velocity', current_velocity)
      feed_dict = {
          self.initial_state.position:
              particles,
          self.initial_velocity:
              current_velocity,
          self.initial_state.deformation_gradient:
              identity_matrix +
              np.zeros(shape=(batch_size, particle_count, 1, 1))
      }
      l, gradient, evaluated = self.sess.run([loss, grad, results], feed_dict=feed_dict)
      print('    grad', gradient)
      print('    ** loss', l)
      current_velocity -= learning_rate * gradient
      '''
    self.sess.run(tf.global_variables_initializer())

    while True:
      print('velocity', current_velocity)
      feed_dict = {
          self.initial_state.position:
              particles,
          self.initial_velocity:
              current_velocity,
          self.initial_state.deformation_gradient:
              identity_matrix +
              np.zeros(shape=(batch_size, particle_count, 1, 1))
      }
      l, _, evaluated = self.sess.run([loss, opt, results], feed_dict=feed_dict)
      print('    ** loss', l)
      for j, r in enumerate(evaluated):
        frame = i * (total_steps + 1) + j
        self.visualize(
            i=frame, r=r, output_fn='outputs/{:04d}.png'.format(frame))

    os.system('cd outputs && ti video')
    os.system('cp outputs/video.mp4 .')

  def visualize(self, i, r, output_fn=None):
    pos = r['position'][0]
    mass = r['mass'][0]
    grid = r['grid'][0][:, :, 1:2]
    J = determinant(r['deformation_gradient'])[0]
    #5 print(grid.min(), grid.max())
    grid = grid / (1e-5 + np.abs(grid).max()) * 4 + 0.5
    grid = np.clip(grid, 0, 1)
    kernel_sum = np.sum(r['kernels'][0], axis=(1, 2))
    if 0 < i < 3:
      np.testing.assert_array_almost_equal(kernel_sum, 1, decimal=3)
      np.testing.assert_array_almost_equal(
          mass.sum(), particle_count, decimal=3)

    scale = 30

    # Pure-white background
    img = np.ones((scale * res, scale * res, 3), dtype=np.float)

    for i in range(len(pos)):
      p = pos[i]
      x, y = tuple(map(lambda x: math.ceil(x * scale), p))
      #if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
      #  img[x, y] = (0, 0, 1)
      cv2.circle(
          img,
          (y, x),
          radius=1,
          #color=(1, 1, float(J[i])),
          color=(0.2, 0.2, 0.2),
          thickness=-1)

    cv2.line(
        img, (int(res * scale * 0.102), 0),
        (int(res * scale * 0.102), res * scale),
        color=(0, 0, 0))
    cv2.circle(
        img, (int(res * scale * 0.45), int(res * scale * 0.8)),
        radius=8,
        color=(0.5, 0.5, 0.5),
        thickness=-1)

    img = img.swapaxes(0, 1)[::-1, :, ::-1]
    mass = mass.swapaxes(0, 1)[::-1, :, ::-1]
    grid = grid.swapaxes(0, 1)[::-1, :, ::-1]
    #grid = np.concatenate([grid, grid[:, :, 0:1] * 0], axis=2)
    mass = cv2.resize(
        mass, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    grid = cv2.resize(
        grid, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    cv2.imshow('Particles', img)
    cv2.imshow('Mass', mass / 10)
    cv2.imshow('Velocity', grid)
    if output_fn is not None:
      cv2.imwrite(output_fn, img * 255)
    cv2.waitKey(1)
