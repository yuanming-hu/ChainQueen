import tensorflow as tf
import tensorflow.contrib.layers as ly
import cv2
import os
import random
import math
from vector_math import *

from states import InitialState, UpdatedState
lr = 1e-3
sample_density = 20
group_num_particles = sample_density ** 2
E = 4500
if False:
  num_groups = 7
  group_offsets = [(0, 0), (0.5, 0), (0, 1), (1, 1), (2, 1), (2, 0), (2.5, 0)]
  group_sizes = [(0.5, 1), (0.5, 1), (1, 1), (1, 1), (1, 1), (0.5, 1), (0.5, 1)]
  actuations = [0, 1, 5, 6]
else:
  num_groups = 5
  group_offsets = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]
  group_sizes = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
  actuations = [0, 4]

actuation_strength = 0.4
num_particles = group_num_particles * num_groups

def particle_mask(start, end):
  r = tf.range(0, num_particles)
  return tf.cast(tf.logical_and(start <= r, r < end), tf.float32)[None, :]


def particle_mask_from_group(g):
  return particle_mask(g * group_num_particles, (g + 1) * group_num_particles)


# hidden_size = 10
W1 = tf.Variable(0.02 * tf.random_normal(shape=(len(actuations), 6 * len(group_sizes))), trainable=True)
b1 = tf.Variable([[-0.1] * len(actuations)], trainable=True)
#b1 = tf.Variable([[0.1, 0.5]], trainable=True)


class Simulation:
  def __init__(self, sess, res, num_particles, num_steps, gravity=(0, -9.8), dt=0.01, batch_size=1):
    self.E = E
    self.num_steps = num_steps
    self.num_particles = num_particles
    self.scale = 30
    self.res = res
    self.sess = sess
    self.initial_velocity = tf.placeholder(shape=(2,), dtype=tf.float32)
    assert batch_size == 1
    self.batch_size = batch_size
    self.initial_state = InitialState(
        self, initial_velocity=self.initial_velocity)
    self.updated_states = []
    self.gravity = gravity
    self.dt = dt

    # Boundary condition
    previous_state = self.initial_state

    for i in range(num_steps):
      new_state = UpdatedState(self, previous_state)
      self.updated_states.append(new_state)
      previous_state = new_state

    self.states = [self.initial_state] + self.updated_states

  def get_centroids(self, previous_state):
    # return centroid positions and velocities
    states = []
    for i in range(num_groups):
      mask = particle_mask(i * group_num_particles, (i + 1) * group_num_particles)[:, :, None] * (
        1.0 / group_num_particles)
      pos = tf.reduce_sum(mask * previous_state.position, axis=1, keepdims=True)
      vel = tf.reduce_sum(mask * previous_state.velocity, axis=1, keepdims=True)
      states.append(pos)
      states.append(vel)
      states.append(self.initial_state.goal)
    states = tf.concat(states, axis=2)
    # print('states', states.shape)
    return states


  def get_actuation(self, state):
    intermediate = tf.matmul(W1, state.controller_states[0, 0, :, None])
    actuation = tf.tanh(intermediate[:, 0] + b1) * actuation_strength
    actuation = actuation[0]
    state.actuation = actuation
    total_actuation = 0
    zeros = tf.zeros(shape=(1, self.num_particles))
    for i, group in enumerate(actuations):
      act = actuation[i][None, None]
      mask = particle_mask_from_group(group)
      act = act * mask
      # First PK stress here
      act = E * make_matrix2d(zeros, zeros, zeros, act)
      # Convert to Kirchhoff stress
      total_actuation = total_actuation + matmatmul(act, transpose(state['deformation_gradient']))
    return total_actuation


  def visualize(self, i, r):
    pos = r['position'][0]
    mass = r['mass'][0]
    grid = r['grid'][0][:, :, 1:2]
    # J = determinant(r['deformation_gradient'])[0]
    #5 print(grid.min(), grid.max())
    grid = grid / (1e-5 + np.abs(grid).max()) * 4 + 0.5
    grid = np.clip(grid, 0, 1)
    kernel_sum = np.sum(r['kernels'][0], axis=(1, 2))
    if 0 < i < 3:
      np.testing.assert_array_almost_equal(kernel_sum, 1, decimal=3)
      np.testing.assert_array_almost_equal(
          mass.sum(), num_particles, decimal=3)

    scale = self.scale

    # Pure-white background
    img = np.ones((scale * self.res[0], scale * self.res[1], 3), dtype=np.float)

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
        img, (int(self.res[0] * scale * 0.101), 0),
        (int(self.res[0] * scale * 0.101), self.res[1] * scale),
        color=(0, 0, 0))

    try:
      for i in range(len(actuations)):
        act = r['actuation'][i]
        if act < 0:
          color = (255, 0, 0)
        else:
          color = (0, 255, 0)
        x0 = 20 + 25 * i
        x1 = 40 + 25 * i
        y0 = 140
        y1 = int(act * 50 + 140)
        if y0 > y1:
          y0, y1 = y1, y0
        cv2.rectangle(img, (y0, x0), (y1, x1), color, thickness=-1)
    except Exception as e:
      if i != 0:
        print(e)

    try:
      position = [
          r['controller_states'][0, 0][num_groups // 2 * 6], r['controller_states'][0, 0][num_groups // 2 * 6 + 1]
      ]
      cv2.circle(
          img, (int(scale * position[1]), int(scale * position[0])),
          radius=4,
          color=(0.9, 0.0, 0.0),
          thickness=-1)
    except:
      pass

    #mass = mass.swapaxes(0, 1)[::-1, :, ::-1]
    #grid = grid.swapaxes(0, 1)[::-1, :, ::-1]
    #grid = np.concatenate([grid, grid[:, :, 0:1] * 0], axis=2)
    # mass = cv2.resize(
    #     mass, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # grid = cv2.resize(
    #     grid, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    return img
