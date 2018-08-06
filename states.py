import numpy as np
import tensorflow as tf
from vector_math import *

'''
TODO:
dx
'''

sample_density = 20
group_particle_count = sample_density ** 2
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

particle_count = group_particle_count * num_groups
actuation_strength = 0.4

# Lame parameters
E = 4500
nu = 0.3
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))

sticky = False

linear = False

identity_matrix = np.array([[1, 0], [0, 1]])[None, None, :, :]

dim = 2

class State:

  def __init__(self, sim):
    self.sim = sim
    self.affine = tf.zeros(shape=(self.sim.batch_size, particle_count, 2, 2))
    self.actuation = tf.zeros(shape=(len(actuations),))

  def get_evaluated(self):
    # batch, particle, dimension
    assert len(self.position.shape) == 3
    assert len(self.position.shape) == 3
    # batch, particle, matrix dimension1, matrix dimension2
    assert len(self.deformation_gradient.shape) == 4
    # batch, x, y, dimension
    assert len(self.mass.shape) == 4
    assert len(self.grid.shape) == 4

    return {
      'position': self.position,
      'velocity': self.velocity,
      'deformation_gradient': self.deformation_gradient,
      'controller_states': self.controller_states,
      'mass': self.mass,
      'grid': self.grid,
      'kernels': self.kernels,
      'actuation': self.actuation
    }


  def __getitem__(self, item):
    return self.get_evaluated(item)


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

  def get_centroids(self, previous_state):
    # return centroid positions and velocities
    states = []
    for i in range(num_groups):
      mask = particle_mask(i * group_particle_count, (i + 1) * group_particle_count)[:, :, None] * (
        1.0 / group_particle_count)
      pos = tf.reduce_sum(mask * previous_state.position, axis=1, keepdims=True)
      vel = tf.reduce_sum(mask * previous_state.velocity, axis=1, keepdims=True)
      states.append(pos)
      states.append(vel)
      states.append(self.goal)
    states = tf.concat(states, axis=2)
    # print('states', states.shape)
    return states


class InitialState(State):

  def __init__(self, sim, initial_velocity):
    super().__init__(sim)
    self.t = 0
    self.goal = tf.placeholder(tf.float32, [self.sim.batch_size, 1, 2], name='goal')
    self.position = tf.placeholder(
      tf.float32, [self.sim.batch_size, particle_count, dim], name='position')

    broadcaster = [int(i > particle_count // 2) for i in range(particle_count)]
    self.velocity = np.array(broadcaster)[None, :, None] * initial_velocity[
                                                           None, None, :]
    # print(self.velocity.shape)
    '''
    self.velocity = tf.placeholder(
        tf.float32, [batch_size, particle_count, dim], name='velocity')
    '''
    self.deformation_gradient = tf.placeholder(
      tf.float32, [self.sim.batch_size, particle_count, dim, dim], name='dg')
    self.mass = tf.zeros(shape=(self.sim.batch_size, self.sim.res[0], self.sim.res[1], 1))
    self.grid = tf.zeros(shape=(self.sim.batch_size, self.sim.res[0], self.sim.res[1], dim))
    self.kernels = tf.zeros(shape=(self.sim.batch_size, self.sim.res[0], self.sim.res[1], 3, 3))
    '''
    TODO:
    mass, volume, Lame parameters (Young's modulus and Poisson's ratio)
    '''
    self.controller_states = self.get_centroids(self)


def particle_mask(start, end):
  r = tf.range(0, particle_count)
  return tf.cast(tf.logical_and(start <= r, r < end), tf.float32)[None, :]


def particle_mask_from_group(g):
  return particle_mask(g * group_particle_count, (g + 1) * group_particle_count)


# hidden_size = 10
W1 = tf.Variable(0.02 * tf.random_normal(shape=(len(actuations), 6 * len(group_sizes))), trainable=True)
b1 = tf.Variable([[-0.1] * len(actuations)], trainable=True)
#b1 = tf.Variable([[0.1, 0.5]], trainable=True)


class UpdatedState(State):

  def __init__(self, sim, previous_state):
    super().__init__(sim)
    self.goal = previous_state.goal
    self.controller_states = self.get_centroids(previous_state)
    intermediate = tf.matmul(W1, self.controller_states[0, 0, :, None])
    #print(W1.shape)
    #print(self.controller_states[0, 0, :, None].shape)
    #print(intermediate.shape)
    self.actuation = tf.tanh(intermediate[:, 0] + b1) * actuation_strength
    self.actuation = self.actuation[0]
    # print(self.actuation.shape)

    self.t = previous_state.t + self.sim.dt
    self.grid = tf.zeros(shape=(self.sim.batch_size, self.sim.res[0], self.sim.res[1], dim))

    self.get_centroids(previous_state)

    # Rasterize mass and velocity
    base_indices = tf.cast(tf.floor(previous_state.position - 0.5), tf.int32)
    batch_size = self.sim.batch_size
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
    self.mass = tf.zeros(shape=(batch_size, self.sim.res[0], self.sim.res[1], 1))

    # Compute stress tensor (Kirchhoff stress instead of First Piola-Kirchhoff stress)
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
      self.stress_tensor1 = 2 * mu * matmatmul(self.deformation_gradient - r,
                                               transpose(self.deformation_gradient))

      if True:
        zeros = tf.zeros(shape=(1, particle_count))
        for i, group in enumerate(actuations):
          actuation = self.actuation[i][None, None]
          mask = particle_mask_from_group(group)
          actuation = actuation * mask
          # First PK stress here
          actuation = E * make_matrix2d(zeros, zeros, zeros, actuation)
          # Convert to Kirchhoff stress
          actuation = matmatmul(actuation, transpose(self.deformation_gradient))
          self.stress_tensor1 += actuation

      self.stress_tensor2 = lam * (
        j - 1) * j * inverse(transpose(self.deformation_gradient))

    self.stress_tensor = self.stress_tensor1 + self.stress_tensor2
    self.stress_tensor = -1 * self.stress_tensor

    # Rasterize momentum and velocity
    # ... and apply gravity

    self.grid = tf.zeros(shape=(batch_size, self.sim.res[0], self.sim.res[1], dim))

    self.kernels = self.compute_kernels(previous_state.position)
    assert self.kernels.shape == (batch_size, particle_count, 3, 3, 1)

    self.velocity = previous_state.velocity
    for i in range(3):
      for j in range(3):
        assert batch_size == 1
        delta_indices = np.array([0, i, j])[None, None, :]
        #print((base_indices + delta_indices).shape)
        self.mass = self.mass + tf.scatter_nd(
          shape=(batch_size, self.sim.res[0], self.sim.res[1], 1),
          indices=base_indices + delta_indices,
          updates=self.kernels[:, :, i, j])

        delta_node_position = np.array([i, j])[None, None, :]
        offset = -(previous_state.position - tf.floor(previous_state.position - 0.5) - \
                   tf.cast(delta_node_position, tf.float32))

        grid_velocity_contributions = self.kernels[:, :, i, j] * (
          self.velocity + matvecmul(self.affine, offset) * 4)
        grid_force_contributions = self.kernels[:, :, i, j] * (
          matvecmul(self.stress_tensor, offset) * (-4 * self.sim.dt))
        self.grid = self.grid + tf.scatter_nd(
          shape=(batch_size, self.sim.res[0], self.sim.res[1], dim),
          indices=base_indices + delta_indices,
          updates=grid_velocity_contributions + grid_force_contributions)
    assert self.mass.shape == (batch_size, self.sim.res[0], self.sim.res[1], 1), 'shape={}'.format(
      self.mass.shape)

    self.grid += self.mass * np.array(self.sim.gravity)[None, None, None, :] * self.sim.dt
    self.grid = self.grid / tf.maximum(1e-30, self.mass)

    # Boundary conditions
    if sticky:
      self.grid = self.grid * self.sim.bc
    else:
      # TODO: use sim.bc
      mask = np.zeros((1, self.sim.res[0], self.sim.res[1], 2))
      mask_x = np.zeros((1, self.sim.res[0], self.sim.res[1], 2))
      mask_y = np.zeros((1, self.sim.res[0], self.sim.res[1], 2))

      # bottom
      mask[:, :, :3, :] = 1
      mask_x[:, :, :3, 0] = 1
      mask_y[:, :, :3, 1] = 1

      friction = 0.5
      projected_bottom = tf.sign(self.grid) * \
                         tf.maximum(tf.abs(self.grid) + friction * tf.minimum(0.0, self.grid[:, :, :, 1, None]), 0.0)
      self.grid = self.grid * (1 - mask) + (
        mask_x * projected_bottom) + mask_y * tf.maximum(self.grid, 0.0)

      mask = np.zeros((1, self.sim.res[0], self.sim.res[1], 2))
      mask[:, 3:self.sim.res[0] - 3, :self.sim.res[1] - 3] = 1
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

    dg_change = identity_matrix - (4 * self.sim.dt) * self.affine
    #print(dg_change.shape)
    #print(previous_state.deformation_gradient)
    self.deformation_gradient = matmatmul(dg_change,
                                          previous_state.deformation_gradient)

    # Advection
    self.position = previous_state.position + self.velocity * self.sim.dt

