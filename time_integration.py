import numpy as np
import tensorflow as tf
from vector_math import *
'''
TODO:
dx
'''

# Lame parameters
nu = 0.3

sticky = False

linear = False

dim = 2


class State:

  def __init__(self, sim):
    self.sim = sim
    self.affine = tf.zeros(shape=(self.sim.batch_size, sim.num_particles, 2, 2))
    self.position = None
    self.velocity = None
    self.deformation_gradient = None
    self.controller_states = None
    self.mass = None
    self.grid = None
    self.kernels = None
    self.debug = None

  def get_evaluated(self):
    # # batch, particle, dimension
    # assert len(self.position.shape) == 3
    # assert len(self.position.shape) == 3
    # # batch, particle, matrix dimension1, matrix dimension2
    # assert len(self.deformation_gradient.shape) == 4
    # # batch, x, y, dimension
    # assert len(self.mass.shape) == 4
    # assert len(self.grid.shape) == 4

    ret = {
        'position': self.position,
        'velocity': self.velocity,
        'deformation_gradient': self.deformation_gradient,
        'controller_states': self.controller_states,
        'mass': self.mass,
        'grid': self.grid,
        'kernels': self.kernels,
        'debug': self.debug
    }
    ret_filtered = {}
    for k, v in ret.items():
      if v is not None:
        ret_filtered[k] = v
    return ret_filtered

  def __getitem__(self, item):
    return self.get_evaluated()[item]

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


class InitialState(State):

  def __init__(self, sim, initial_velocity):
    super().__init__(sim)
    self.t = 0
    num_particles = sim.num_particles
    self.position = tf.placeholder(
        tf.float32, [self.sim.batch_size, num_particles, dim], name='position')

    broadcaster = [int(i > num_particles // 2) for i in range(num_particles)]
    self.velocity = np.array(broadcaster)[None, :, None] * initial_velocity[
        None, None, :]
    # print(self.velocity.shape)
    '''
    self.velocity = tf.placeholder(
        tf.float32, [batch_size, num_particles, dim], name='velocity')
    '''
    self.deformation_gradient = tf.placeholder(
        tf.float32, [self.sim.batch_size, num_particles, dim, dim], name='dg')
    self.mass = tf.zeros(
        shape=(self.sim.batch_size, self.sim.res[0], self.sim.res[1], 1))
    self.grid = tf.zeros(
        shape=(self.sim.batch_size, self.sim.res[0], self.sim.res[1], dim))
    self.kernels = tf.zeros(
        shape=(self.sim.batch_size, self.sim.res[0], self.sim.res[1], 3, 3))
    '''
    TODO:
    mass, volume, Lame parameters (Young's modulus and Poisson's ratio)
    '''


class UpdatedState(State):

  def __init__(self, sim, previous_state, controller):
    super().__init__(sim)

    self.t = previous_state.t + self.sim.dt
    self.grid = tf.zeros(
        shape=(self.sim.batch_size, self.sim.res[0], self.sim.res[1], dim))

    # Rasterize mass and velocity
    base_indices = tf.cast(tf.floor(previous_state.position - 0.5), tf.int32)
    batch_size = self.sim.batch_size
    assert batch_size == 1
    # print('base indices', base_indices.shape)
    # Add the batch size indices
    num_particles = sim.num_particles
    base_indices = tf.concat(
        [
            tf.zeros(shape=(batch_size, num_particles, 1), dtype=tf.int32),
            base_indices
        ],
        axis=2)
    # print('base indices', base_indices.shape)
    self.mass = tf.zeros(
        shape=(batch_size, self.sim.res[0], self.sim.res[1], 1))

    # Compute stress tensor (Kirchhoff stress instead of First Piola-Kirchhoff stress)
    self.deformation_gradient = previous_state.deformation_gradient

    mu = self.sim.E / (2 * (1 + nu))
    lam = self.sim.E * nu / ((1 + nu) * (1 - 2 * nu))
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
      self.stress_tensor1 = 2 * mu * matmatmul(
          self.deformation_gradient - r, transpose(self.deformation_gradient))

      self.stress_tensor1 += controller(previous_state, self)

      self.stress_tensor2 = lam * (
          j - 1) * j * inverse(transpose(self.deformation_gradient))

    self.stress_tensor = self.stress_tensor1 + self.stress_tensor2
    self.stress_tensor = -1 * self.stress_tensor

    # Rasterize momentum and velocity
    # ... and apply gravity

    self.grid = tf.zeros(
        shape=(batch_size, self.sim.res[0], self.sim.res[1], dim))

    self.kernels = self.compute_kernels(previous_state.position)
    assert self.kernels.shape == (batch_size, num_particles, 3, 3, 1)

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
    assert self.mass.shape == (batch_size, self.sim.res[0], self.sim.res[1],
                               1), 'shape={}'.format(self.mass.shape)

    self.grid += self.mass * np.array(
        self.sim.gravity)[None, None, None, :] * self.sim.dt
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
